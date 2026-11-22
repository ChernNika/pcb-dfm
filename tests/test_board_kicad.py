"""Модель платы на демо-платах KiCad 10 и сверка с тем, что выгружает сам KiCad.

Эталон (kicad-cli): вспышки площадок в Gerber X2 (компонент, вывод, цепь, координаты),
отверстия в Excellon, позиции компонентов, стек в job-файле. См. kicad_reference.py.
"""
import math
import os
from collections import defaultdict
from pathlib import Path

import pytest

import kicad_reference as ref
from dfm import kicad_cli
from dfm.board import load
from dfm.board.sexpr import load as load_tree
from dfm.board.shapes import transform

DEMOS = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'KiCad' / '10.0' / 'share' / 'kicad' / 'demos'
SMALL = sorted((p for p in DEMOS.rglob('*.kicad_pcb') if p.stat().st_size < 7e6), key=lambda p: p.stat().st_size) \
    if DEMOS.exists() else []
pytestmark = pytest.mark.skipif(not SMALL, reason='нет демо-плат KiCad 10')
ids = lambda p: p.stem

POS_TOL = 1e-5      # 10 нм: KiCad считает в целых нанометрах, Gerber выгружается с точностью 1 нм
DRILL_TOL = 6e-4    # в Excellon координаты округлены до 1 мкм


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    """Выгрузки KiCad для каждой платы — один раз на весь модуль."""
    cache = {}

    def get(board):
        if board not in cache:
            try:
                kicad_cli.find()
            except FileNotFoundError:
                pytest.skip('KiCad не установлен')
            d = tmp_path_factory.mktemp(board.stem.replace(' ', '_'))
            kicad_cli.export_gerbers(board, d / 'g')
            kicad_cli.export_drill(board, d / 'g')
            kicad_cli._run('pcb', 'export', 'pos', '--format', 'csv', '--units', 'mm', '--side', 'both',
                           '-o', d / 'pos.csv', board)
            cache[board] = d
        return cache[board]
    return get


# ---------- без KiCad: модель читается полностью и согласована ----------

@pytest.mark.parametrize('path', SMALL, ids=ids)
def test_counts_match_file(path):
    """Число объектов модели = число соответствующих узлов в файле (ничего не потеряно)."""
    b = load(path)
    tree = load_tree(path)
    d = b.design
    fps = tree.all('footprint')
    assert len(d.footprints) == len(fps)
    assert len(d.pads) == sum(len(fp.all('pad')) for fp in fps)
    assert len(d.tracks) == len(tree.all('segment'))
    assert len(d.arcs) == len(tree.all('arc'))
    assert len(d.vias) == len(tree.all('via'))
    assert len(d.all_zones()) == len(tree.all('zone')) + sum(len(fp.all('zone')) for fp in fps)
    assert sum(len(cs) for z in d.all_zones() for cs in z.filled.values()) == \
        sum(len(z.all('filled_polygon')) for z in tree.all('zone'))
    assert len(b.copper) >= 2 and b.stackup is not None
    assert not any('несуществующ' in p or 'неизвестн' in p for p in b.validate())


def test_royalblue_degenerate_vias():
    """В RoyalBlue54L-Feather 183 «глухих» отверстия с отверстием 10 нм через все 8 слоёв."""
    b = load(DEMOS / 'royalblue54L_feather' / 'RoyalBlue54L-Feather.kicad_pcb')
    probs = b.validate()
    assert any('вырожденное' in p and '183 шт.' in p for p in probs)
    assert any('фактически сквозное' in p and '183 шт.' in p for p in probs)


# ---------- сверка с выгрузками KiCad ----------

@pytest.mark.kicad
@pytest.mark.parametrize('path', SMALL, ids=ids)
def test_pads_match_gerber(path, exported):
    """Каждая вспышка площадки в Gerber (наружные слои) = площадка модели: тот же компонент и
    вывод, координаты центра формы с точностью 10 нм, та же цепь. И наоборот: каждая площадка
    модели есть в Gerber (вспышкой или регионом)."""
    b = load(path)
    cu = b.copper
    files = ref.copper_files(sorted((exported(path) / 'g').iterdir()), cu)
    total = 0
    for layer in (cu[0], cu[-1]):
        flashes, regions = ref.gerber_pads(files[layer])
        expected = defaultdict(list)
        for p in b.design.pads:
            if layer in p.layers:
                expected[(p.footprint, p.number)].append(p)
        for f in flashes:
            cands = [p for p in expected.get((f.ref, f.pin), [])
                     if math.dist(transform(p.shape_on(layer, cu).offset, p.pos, p.angle), (f.x, f.y)) <= POS_TOL]
            assert cands, f'{layer}: вспышка {f.ref}.{f.pin} в ({f.x}, {f.y}) не найдена в модели'
            # площадка без цепи в Gerber X2 обозначается «N/C» (not connected)
            gerber_net = None if f.net in ('', 'N/C') else f.net
            assert cands[0].net == gerber_net, f'{f.ref}.{f.pin}: цепь {cands[0].net!r} ≠ {f.net!r}'
        drawn = {(f.ref, f.pin) for f in flashes} | regions
        missing = [k for k, ps in expected.items() if k not in drawn and ps[0].type != 'np_thru_hole']
        assert not missing, f'{layer}: площадок модели нет в Gerber: {missing[:5]}'
        total += len(flashes)
    assert total > 0


@pytest.mark.kicad
@pytest.mark.parametrize('path', SMALL, ids=ids)
def test_holes_match_excellon(path, exported):
    """Отверстия модели (уровень P: переходные и выводов, включая повёрнутые пазы) один к одному
    совпадают с файлами сверловки: место, диаметр, металлизация."""
    b = load(path)
    holes = [h for f in sorted((exported(path) / 'g').glob('*.drl')) for h in ref.excellon(f)]
    missing, extra = ref.unmatched_holes(b.holes, holes)
    assert not missing, f'нет в сверловке: {[(h.pos, h.slot_end, h.diameter, h.plated) for h in missing[:3]]}'
    assert not extra, f'в сверловке лишние отверстия: {extra[:3]}'


@pytest.mark.kicad
@pytest.mark.parametrize('path', SMALL, ids=ids)
def test_footprints_match_position_file(path, exported):
    b = load(path)
    by_ref = defaultdict(list)
    for fp in b.design.footprints:
        by_ref[fp.ref].append(fp)
    rows = ref.positions(exported(path) / 'pos.csv')
    for r, x, y, rot, side in rows:
        cands = [fp for fp in by_ref[r] if math.dist(fp.pos, (x, y)) <= POS_TOL]
        assert cands, f'компонент {r} в ({x}, {y}) не найден'
        fp = cands[0]
        assert fp.side == side
        assert abs((fp.angle - rot + 180) % 360 - 180) < 1e-6, f'{r}: угол {fp.angle} ≠ {rot}'


def _num(v):
    """Числа в job-файле KiCad: в русской Windows εr записывается строкой с запятой («4,45»)."""
    return float(v.replace(',', '.')) if isinstance(v, str) else v


@pytest.mark.kicad
@pytest.mark.parametrize('path', SMALL, ids=ids)
def test_stackup_matches_job_file(path, exported):
    """Стек модели = стек job-файла (толщины там округлены до 4 знаков). Для плат без стека
    в файле модель строит стек по умолчанию так же, как KiCad."""
    b = load(path)
    general, items = ref.job_stackup(next((exported(path) / 'g').glob('*.gbrjob')))
    kinds = {'copper': 'Copper', 'dielectric': 'Dielectric', 'mask': 'SolderMask', 'paste': 'SolderPaste',
             'silk': 'Legend'}
    assert [kinds[i.kind] for i in b.stackup.items] == [m['Type'] for m in items]
    for ours, m in zip(b.stackup.items, items):
        if 'Thickness' in m:
            assert abs(ours.thickness - m['Thickness']) <= 6e-5, ours.name
        if 'Material' in m:
            assert ours.material == m['Material'], ours.name
        for ours_v, key in ((ours.epsilon_r, 'DielectricConstant'), (ours.loss_tangent, 'LossTangent')):
            if key in m:
                assert math.isclose(ours_v, _num(m[key])), (ours.name, key)
    assert abs(b.thickness - general['BoardThickness']) <= 6e-5
    assert (b.stackup.finish or 'None') == general.get('Finish', 'None')
