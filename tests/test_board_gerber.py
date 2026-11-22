"""Сверка уровня P с выгрузкой KiCad: медь и маска каждого слоя, контур платы, отверстия.

Эталон — Gerber X2 и Excellon, которые выгружает kicad-cli из той же платы; Gerber переводится
в многоугольники независимым читателем (gerber_reference.py). Тексты на меди и маске перед
выгрузкой переносятся на слой Cmts.User: модель их не строит (нужны шрифты KiCad).

Критерий совпадения слоя — в симметрической разности двух изображений нет участков шире
10 мкм (gerber_reference.compare). Мелкие расхождения неизбежны: обе стороны заменяют дуги
ломаными, а KiCad ещё и строит произвольные площадки многоугольниками с допуском до 5 мкм.

Платы: синтетическая (synthetic_board.py — падстеки, удаление неиспользуемых площадок,
отступы маски, вырезы в контуре) и часть демо-плат KiCad 10. Все 16 демо-плат (около 10 мин):
переменная окружения DFM_GERBER_ALL=1.
"""
import os
from pathlib import Path

import pytest
import shapely
from shapely.geometry import LineString, MultiLineString

import gerber_reference as gr
import kicad_reference as ref
import synthetic_board
from dfm import kicad_cli
from dfm.board import load
from dfm.geometry.polygons import contour_coords, layer_geometry

pytestmark = pytest.mark.kicad

DEMOS = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'KiCad' / '10.0' / 'share' / 'kicad' / 'demos'
# по одной плате на каждую особенность: отступ маски платы, контур в компоненте, повёрнутые пазы,
# произвольные площадки и текст, «капли», отступы маски компонентов, 8 и 4 слоя
QUICK = ('microwave', 'RoyalBlue54L-NFC-Antenna', 'ecc83-pp_v2', 'pic_programmer', 'complex_hierarchy',
         'StickHub', 'RoyalBlue54L-Feather', 'One-Air-Max')
_ALL = sorted((p for p in DEMOS.rglob('*.kicad_pcb') if p.stat().st_size < 7e6), key=lambda p: p.stat().st_size) \
    if DEMOS.exists() else []
DEMO_BOARDS = _ALL if os.environ.get('DFM_GERBER_ALL') else [p for p in _ALL if p.stem in QUICK]
BOARDS = ['synthetic'] + DEMO_BOARDS
ids = lambda p: p if isinstance(p, str) else p.stem

GAP = 0.005          # полуширина допустимого расхождения, мм (участки уже 10 мкм не считаются)


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    """Плата без текстов, её модель и выгрузки KiCad — один раз на плату."""
    cache = {}

    def get(src):
        if src not in cache:
            try:
                kicad_cli.find()
            except FileNotFoundError:
                pytest.skip('KiCad не установлен')
            d = tmp_path_factory.mktemp(ids(src).replace(' ', '_'))
            pcb = d / 'board.kicad_pcb'
            if src == 'synthetic':
                synthetic_board.write(pcb)
            else:
                gr.strip_texts(src, pcb)
            kicad_cli.export_gerbers(pcb, d / 'g')
            kicad_cli.export_drill(pcb, d / 'g')
            b = load(pcb)
            cache[src] = (b, gr.layer_files(d / 'g', b.copper), d / 'g')
        return cache[src]
    return get


@pytest.mark.parametrize('src', BOARDS, ids=ids)
def test_layers_match_gerber(src, prepared):
    """Медь каждого слоя и окна маски совпадают с Gerber KiCad до 10 мкм."""
    b, files, _ = prepared(src)
    bad = []
    for lay in b.copper + ['F.Mask', 'B.Mask']:
        ours = layer_geometry(b.layers[lay])
        theirs = gr.image(gr.read(files[lay]))
        _, thick = gr.compare(ours, theirs, GAP)
        if not thick.is_empty:
            side = 'лишнее у нас' if ours.contains(thick.representative_point()) else 'нет у нас'
            bad.append(f'{lay}: {thick.area:.4f} мм² ({side}) в {[round(v, 3) for v in thick.bounds]}')
        assert ours.area > 0 or theirs.area == 0
    assert not bad, '; '.join(bad)


@pytest.mark.parametrize('src', BOARDS, ids=ids)
def test_outline_matches_gerber(src, prepared):
    """Контур модели (все внешние контуры и вырезы) совпадает с линиями слоя Edge.Cuts в Gerber:
    расстояние Хаусдорфа между ними не больше 5 мкм."""
    b, files, _ = prepared(src)
    lines = [o.line for o in gr.read(files['Edge.Cuts']) if o.line is not None]
    rings = []
    for r in b.outline:
        for c in (r.outer, *r.holes):
            pts = contour_coords(c, 0.0002)
            rings.append(LineString([*map(tuple, pts), tuple(pts[0])]))
    assert rings
    d = shapely.hausdorff_distance(MultiLineString(rings), shapely.union_all(lines), densify=0.1)
    assert d <= GAP, f'контур расходится с Gerber на {d * 1000:.1f} мкм'


def test_synthetic_holes_match_excellon(prepared):
    """Отверстия синтетической платы (глухое, паз со смещённой площадкой, неметаллизированные)
    совпадают со сверловкой. Демо-платы проверяет test_board_kicad.test_holes_match_excellon."""
    b, _, g = prepared('synthetic')
    missing, extra = ref.unmatched_holes(b.holes, [h for f in sorted(g.glob('*.drl')) for h in ref.excellon(f)])
    assert not missing and not extra, (missing, extra)


# ---------- адаптер Gerber: второй путь к той же модели ----------

def _match_holes(kb, gb, tol=6e-4):
    """Пары (отверстие KiCad, отверстие Gerber); в Excellon координаты и диаметры округлены до 1 мкм."""
    free = list(gb.holes)
    pairs, missing = [], []
    near = lambda a, b: a is not None and b is not None and abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
    for h in kb.holes:
        for i, g in enumerate(free):
            if g.plated != h.plated or abs(g.diameter - h.diameter) > 1.5e-3 or (g.slot_end is None) != (h.slot_end is None):
                continue
            if near(h.pos, g.pos) and (h.slot_end is None or near(h.slot_end, g.slot_end)) or \
                    h.slot_end is not None and near(h.pos, g.slot_end) and near(h.slot_end, g.pos):
                pairs.append((h, free.pop(i)))
                break
        else:
            missing.append(h)
    return pairs, missing, free


@pytest.mark.parametrize('src', BOARDS, ids=ids)
def test_gerber_path_matches_kicad_path(src, prepared):
    """Модель из Gerber + Excellon + job-файла = модель из .kicad_pcb: медь и маска каждого слоя,
    медь каждой цепи, отверстия (диапазон слоёв, падстек, цепь, роль), стек, контур."""
    from dfm.geometry.copper import BoardGeometry
    kb, _, g = prepared(src)
    gb = load(g)
    assert gb.source == 'gerber' and gb.copper == kb.copper
    kg, gg = BoardGeometry(kb), BoardGeometry(gb)
    bad = []
    for lay in kb.copper + ['F.Mask', 'B.Mask']:
        a, b = kg.layer(lay), gg.layer(lay)
        if not gr.compare(a.geom, b.geom, GAP)[1].is_empty:
            bad.append(f'{lay}: медь слоя')
        if lay in kb.copper:
            for net in set(a.nets) | set(b.nets):
                if net not in a.nets or net not in b.nets:
                    bad.append(f'{lay}: цепь {net!r} есть только в одной модели')
                elif not gr.compare(a.nets[net], b.nets[net], GAP)[1].is_empty:
                    bad.append(f'{lay}: медь цепи {net!r}')
    assert not bad, bad[:10]

    pairs, missing, extra = _match_holes(kb, gb)
    assert not missing and not extra, (missing[:3], extra[:3])
    for h, x in pairs:
        assert (x.span, sorted(x.padstack), x.net, x.role) == (h.span, sorted(h.padstack), h.net, h.role), h.pos

    kinds = lambda b: [(i.kind, round(i.thickness or 0, 4), i.material) for i in b.stackup.items]
    assert kinds(gb) == kinds(kb)
    assert gb.thickness == pytest.approx(kb.thickness, abs=1e-4) and gb.stackup.finish == kb.stackup.finish
    assert [len(r.holes) for r in gb.outline] == [len(r.holes) for r in kb.outline]
    assert gb.nets <= kb.nets


X1_BOARDS = [p for p in DEMO_BOARDS if p.stem in ('pic_programmer', 'StickHub', 'One-Air-Max')]


@pytest.mark.parametrize('src', X1_BOARDS, ids=ids)
def test_gerber_without_attributes(src, tmp_path):
    """Gerber без атрибутов X2 и без job-файла (как от старых САПР): слои определяются по
    расширениям файлов, цепи восстанавливаются по геометрии. Разбиение меди на цепи совпадает
    со связностью модели KiCad: каждая геометрическая цепь = компонента связности KiCad-модели
    (на One-Air-Max это четыре известные пары: заливка поверх площадки и три перемычки)."""
    from collections import defaultdict
    from dfm.geometry.connectivity import connectivity
    from dfm.geometry.copper import BoardGeometry
    pcb = tmp_path / 'board.kicad_pcb'
    gr.strip_texts(src, pcb)
    out = tmp_path / 'x1'
    kicad_cli._run('pcb', 'export', 'gerbers', '--no-x2', '--no-netlist', '--precision', '6', '-o', f'{out}{os.sep}', pcb)
    kicad_cli.export_drill(pcb, out)
    for j in out.glob('*.gbrjob'):
        j.unlink()
    gb = load(out)
    kb = load(pcb)
    assert gb.copper == kb.copper and gb.stackup is None
    assert any('восстановлены по геометрии' in w for w in gb.warnings)
    kg, gg = BoardGeometry(kb), BoardGeometry(gb)
    groups = defaultdict(set)              # геометрическая цепь → цепи KiCad
    for lay in kb.copper:
        a, b = kg.layer(lay), gg.layer(lay)
        assert gr.compare(a.geom, b.geom, GAP)[1].is_empty, lay
        for net, geom in a.nets.items():
            if net is None:
                continue
            for gnet in [n for n, h in b.nets.items() if h.intersects(geom) and h.intersection(geom).area > 1e-6]:
                groups[gnet].add(net)
    c = connectivity(kg)
    comp_nets = defaultdict(set)
    for (lay, k), comp in zip(c.nodes, c.component):
        comp_nets[int(comp)] |= {n for n in kg.layer(lay).island_nets[k] if n is not None}
    assert sorted(map(sorted, groups.values())) == sorted(map(sorted, (s for s in comp_nets.values() if s)))
