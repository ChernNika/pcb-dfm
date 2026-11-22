"""Шаг 1.6: писатель плат KiCad, свои тестовые платы и команда compare.

Без KiCad — писатель и генератор плат (то, что записано, читается обратно в ту же модель),
метрики сверки. С KiCad (kicad) — compare на своих тестовых платах и демо-платах документов:
оба пути совпадают, нарушения зазоров совпадают с DRC KiCad."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import shapely
from shapely.geometry import Point, box

from dfm import kicad_cli
from dfm.board import Role, load
from dfm.board.kicad_writer import KicadWriter
from dfm.compare import compare_board, drc_diff, max_deviation, single_clearance
from dfm.geometry.polygons import to_geometry

ROOT = Path(__file__).resolve().parents[1]
BOARDS = sorted((ROOT / 'boards' / 'test').glob('*.kicad_pcb')) + sorted((ROOT / 'boards' / 'demo').glob('*.kicad_pcb'))


def _make_boards():
    spec = importlib.util.spec_from_file_location('make_boards', ROOT / 'boards' / 'make_boards.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['make_boards'] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- писатель плат ----------

def test_writer_round_trip(tmp_path):
    """Что записано писателем (ось Y вверх, углы против часовой), то и читается в модель."""
    w = KicadWriter('rt', copper=4, clearance=0.18, mask_margin=0.04)
    w.outline_rect(0, 0, 20, 10)
    fp = w.footprint('U1', (10, 5), angle=30, mask_margin=0.06)
    fp.pad('1', 'smd', 'rect', (2, 0), (1, 0.5), net='A', angle=15)
    fp.pad('2', 'thru_hole', 'circle', (-2, 0), 1.4, drill=0.7, net='B',
           padstack=('front_inner_back', {'Inner': dict(shape='circle', size=1.0)}))
    w.track([(1, 1), (5, 1), (5, 3)], 0.2, 'In1.Cu', 'A')
    w.arc((6, 1), (7, 2), (8, 1), 0.25, 'B.Cu', 'B')
    w.via((3, 8), 0.6, 0.3, layers=('F.Cu', 'In2.Cu'), kind='blind', net='A', remove_unused=True, keep_end=False)
    p = w.write(tmp_path / 'rt.kicad_pcb')
    b = load(p)
    assert b.copper == ['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'] and not b.validate()
    pads = {q.number: q for q in b.design.pads}
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    assert pads['1'].pos == pytest.approx((10 + 2 * c, 5 + 2 * s))
    assert pads['1'].angle == pytest.approx(45)
    assert pads['2'].shape_on('In1.Cu', b.copper).size == (1.0, 1.0)
    assert b.design.rules.netclass_of('A').clearance == 0.18 and b.design.rules.pad_to_mask_clearance == 0.04
    assert [(t.start, t.end) for t in b.design.tracks] == [((1, 1), (5, 1)), ((5, 1), (5, 3))]
    v = b.design.vias[0]
    assert v.layers == ('F.Cu', 'In2.Cu') and v.remove_unused_layers and not v.keep_end_layers
    assert len(b.outline) == 1 and to_geometry(b.outline[0]).area == pytest.approx(200)


def test_generator_boards_are_consistent(tmp_path, monkeypatch):
    """Каждая тестовая плата генератора читается без замечаний и содержит то, ради чего сделана."""
    mb = _make_boards()
    monkeypatch.setattr(mb, 'TEST', tmp_path / 'test')
    monkeypatch.setattr(mb, 'DEMO', tmp_path / 'demo')
    made = {p.stem: load(p) for p in mb.main(refill=False)}
    for name, b in made.items():
        assert not b.validate(), name
        assert not [x for x in b.warnings if 'текст' not in x and 'контур' in x], name
    assert {fp.angle for fp in made['t01_pads37'].design.footprints} == {37, -110}
    assert {v.via_type for v in made['t03_vias'].design.vias} == {'through', 'blind', 'micro'}
    assert len(made['t07_outline'].outline) == 2
    assert any(p.padstack_mode == 'custom' for p in made['t04_padstacks'].design.pads)
    assert set(made['b1_demo'].nets) >= {'S1', 'S7', 'HV1', 'HV2', 'GND'}


# ---------- метрики сверки ----------

def test_max_deviation():
    """Полоса 3 мкм и круг Ø 20 мкм: наибольшая ширина — 20 мкм, точка — в круге."""
    g = shapely.union_all([box(0, 0, 10, 0.003), Point(5, 1).buffer(0.010, quad_segs=64)])
    w, at = max_deviation(g)
    assert w == pytest.approx(0.020, abs=2e-5) and math.dist(at, (5, 1)) < 0.005
    assert max_deviation(shapely.Polygon()) == (0.0, None)


def test_drc_diff_rules():
    """Касание поглощает зазор той же пары; медь без цепи и известные особенности — отдельно."""
    ours = {'зазор': {('A', 'B'), ('<no net>', 'C')}, 'касание': {('D', 'E')}, 'край': set()}
    theirs = {'зазор': {('A', 'B'), ('D', 'E'), ('F', 'G')}, 'касание': {('D', 'E')}, 'край': set()}
    res, unnamed = drc_diff(ours, theirs, {'зазор': [['G', 'F']]})
    assert res['зазор'] == {'оба': 1, 'только мы': [], 'только KiCad': [], 'известные особенности KiCad': [('F', 'G')]}
    assert res['касание']['оба'] == 1 and unnamed['зазор'] == [('<no net>', 'C')]


def test_single_clearance(tmp_path):
    w = KicadWriter('cl', clearance=0.17)
    w.outline_rect(0, 0, 5, 5)
    assert single_clearance(load(w.write(tmp_path / 'cl.kicad_pcb'))) == 0.17
    w.footprint('R1', (2, 2), clearance=0.3).pad('1', 'smd', 'rect', (0, 0), 1, net='A')
    assert single_clearance(load(w.write(tmp_path / 'cl2.kicad_pcb'))) is None     # местный зазор


# ---------- compare на своих платах ----------

@pytest.mark.kicad
@pytest.mark.parametrize('pcb', BOARDS, ids=lambda p: p.stem)
def test_compare_own_boards(pcb, tmp_path):
    """Оба пути совпадают (отклонение края ≤ 10 мкм, цепи, отверстия, стек, контур, связность),
    нарушения зазоров совпадают с DRC KiCad."""
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    r = compare_board(pcb, tmp_path / pcb.stem, pictures=pcb.stem == 't01_pads37')
    assert r.ok, (tmp_path / pcb.stem / 'report.md').read_text(encoding='utf-8')
    if pcb.stem == 't08_drc':
        assert {k: d['оба'] for k, d in r.drc.items()} == {'зазор': 5, 'касание': 3, 'край': 1}
        assert r.drc_unnamed['зазор'] == [('<no net>', 'S19')]
    if pcb.stem == 't01_pads37':
        assert len(list((tmp_path / pcb.stem / 'diff').glob('*.png'))) == 4


@pytest.mark.skipif(not BOARDS, reason='нет своих тестовых плат')
def test_padstack_and_mask_rules_on_t04():
    """Правила KiCad 10, найденные на t04_padstacks: слой падстека ничего не наследует от основной
    формы; у custom окно маски — круг по ширине основной площадки в центре отверстия."""
    b = load(ROOT / 'boards' / 'test' / 't04_padstacks.kicad_pcb')
    pads = {p.number: p for p in b.design.pads}
    assert pads['3'].shape_on('In1.Cu', b.copper).shape == 'circle'           # только размер → круг
    assert pads['4'].shape_on('In1.Cu', b.copper).size == (0.0, 0.0)          # только форма → нет площадки
    h4 = next(h for h in b.holes if h.pin == '4')
    assert 'In1.Cu' not in h4.padstack
    assert pads['9'].shape_on('B.Cu', b.copper).size == (1.1, 1.1)            # B.Cu берёт Inner
    win = next(f for f in b.layers['F.Mask'].features if f.role is Role.PAD and f.pin == '8')
    assert win.shape.radius == pytest.approx(0.8 + 0.05) and win.shape.center == pytest.approx(pads['8'].pos)
