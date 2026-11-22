"""Класс A3 (шаг 2.4): бюджет σ и формулы против таблиц документа A3, демо-плата A3, сверка с DRC."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely

from dfm import kicad_cli
from dfm.board import load
from dfm.board.kicad_writer import KicadWriter
from dfm.rules.a3 import nominal_rings, p_capture_miss, track_breakout_mc
from dfm.rules.engine import Context, run
from dfm.rules.findings import Severity
from dfm.rules.profile import load_profile
from dfm.rules.registration import d_min_mm, hole_sigmas, layer_depths, p_board, p_breakout

ROOT = Path(__file__).resolve().parents[1]
A3 = ROOT / 'boards' / 'demo' / 'a3_demo.kicad_pcb'
SIGMA = {'F.Cu': 17.0, 'In1.Cu': 29.4, 'In2.Cu': 30.2, 'B.Cu': 18.8}


@pytest.fixture(scope='module')
def board():
    return load(A3)


def test_sigma_budget_by_layer(board):
    """Документ A3, раздел 2.6: 17,0 / 29,4 / 30,2 / 18,8 мкм на глубинах 0; 0,2; 1,4; 1,6 мм."""
    assert layer_depths(board) == pytest.approx({'F.Cu': 0, 'In1.Cu': 0.2, 'In2.Cu': 1.4, 'B.Cu': 1.6})
    h = next(h for h in board.holes if h.net == 'V1')
    sig = hole_sigmas(board, load_profile(), h)
    assert {k: round(v.total, 1) for k, v in sig.items()} == SIGMA
    assert sig['F.Cu'].common == 0 and sig['In1.Cu'].common == pytest.approx(np.hypot(25, 3.75))


@pytest.mark.parametrize('rn, rreq, layer, expected', [
    (75, 0, 'In1.Cu', 0.0387), (75, 0, 'In2.Cu', 0.0459), (75, 0, 'B.Cu', 3.47e-4), (75, 50, 'F.Cu', 0.339),
    (75, 50, 'B.Cu', 0.413), (40, 0, 'In2.Cu', 0.416), (100, 50, 'B.Cu', 0.029), (150, 25, 'In2.Cu', 1.92e-4)])
def test_rayleigh_table(rn, rreq, layer, expected):
    """Таблица раздела 3.2 (V1–V4, классы 2 и 3); формула совпала там с моделированием."""
    assert p_breakout(rn - rreq, SIGMA[layer] if layer != 'In2.Cu' else 30.2169) == pytest.approx(expected, rel=0.02)
    assert p_breakout(0, 20) == p_breakout(-5, 20) == 1.0


def test_rice_systematic_shift():
    """Раздел 3.3: V2 на L2, σ = 29,2 мкм — 3,7 / 4,4 / 6,7 / 11 % при сдвиге 0 / 10 / 20 / 30 мкм."""
    s = np.sqrt(15 ** 2 + 1 ** 2 + 25 ** 2)
    got = [p_breakout(75, s, b) for b in (0, 10, 20, 30)]
    assert got == pytest.approx([0.037, 0.044, 0.067, 0.11], abs=0.002)


def test_board_common_and_local():
    """Раздел 3.5: N переходных V2 на L2 — 3,9 / 16,6 / 39,1 / 57,0 % (а не 98 % у независимых)."""
    loc, com = np.hypot(15, 1), np.hypot(25, 3.75)
    for n, expected in ((1, 0.039), (10, 0.166), (100, 0.391), (500, 0.570)):
        assert p_board(np.full(n, 75.0), np.full(n, loc), com) == pytest.approx(expected, abs=0.012)


def test_d_min_table():
    """Раздел 3.6, отверстие 0,30: P* = 10⁻³ → 0,526 / 0,569 / 0,519 мм."""
    assert d_min_mm(0.3, 50, 17.0, 1e-3) == pytest.approx(0.526, abs=0.001)
    assert d_min_mm(0.3, 25, 29.41, 1e-3) == pytest.approx(0.569, abs=0.001)
    assert d_min_mm(0.3, 0, 29.41, 1e-3) == pytest.approx(0.519, abs=0.001)


def test_microvia_capture():
    """К7: дно 0,08 мм, σ = 25 мкм — 0,31 / 2,0 / 5,6 % при площадке 0,25 / 0,22 / 0,20 мм."""
    assert [p_capture_miss(D, 0.08, 25) for D in (0.25, 0.22, 0.20)] == pytest.approx([0.0031, 0.020, 0.056], rel=0.03)
    assert p_capture_miss(0.20, 0.08, 15) == pytest.approx(3e-4, rel=0.15)


def test_teardrop_simulation():
    """Раздел 4.3: 0,45/0,30, трасса 0,10, σ = 29,4. Без капли выход где угодно ≈ 3,9 %, у трассы
    0,87 % (20 000 испытаний дают разброс ±0,1 %, поэтому 200 000); с каплей у трассы — 0."""
    pad = shapely.Point(0, 0).buffer(0.225, quad_segs=300)
    td = shapely.union_all([pad, shapely.box(0, -0.05, 0.4, 0.05)]).convex_hull
    p_any, p_tr = track_breakout_mc(pad, np.zeros(2), 0.30, [((1, 0), 0.10)], 29.4)
    assert p_any == pytest.approx(0.0386, abs=0.002) and p_tr == pytest.approx(0.0087, abs=0.0006)
    p_any, p_tr = track_breakout_mc(td, np.zeros(2), 0.30, [((1, 0), 0.10)], 29.4)
    assert p_any == pytest.approx(0.030, abs=0.002) and p_tr < 3e-5


@pytest.fixture(scope='module')
def demo(board):
    return run(board, classes=['A3'])


def _by(fs, code):
    return [f for f in fs if f.code == code]


def test_a3_demo_class2(demo):
    """Демо-плата документа, класс 2: V3 — 41,6 %, V2/V5/V6 — 4,6 %, V7 (внутренние 0,45) — 4,6 %,
    V4 — 0,42 % (информация), J2 — offset, V5 — выход у трассы без капли, M1 — промах 0,31 %.
    V1, V8 (внутренние площадки удалены), J1 и капля V6 находок не дают."""
    k5 = {f.title.split(':')[0]: f for f in _by(demo, 'К5')}
    assert set(k5) == {'переходные 0.38/0.30 мм', 'переходные 0.45/0.30 мм — 3 шт.', 'переходные 0.50/0.30 мм',
                       'переходные 0.55/0.30 мм (внутренние площадки 0.45)'}
    assert k5['переходные 0.38/0.30 мм'].probability == pytest.approx(0.416, abs=0.001)
    assert k5['переходные 0.38/0.30 мм'].severity is Severity.ERROR
    assert set(k5['переходные 0.45/0.30 мм — 3 шт.'].nets) == {'V2', 'V5', 'V6'}
    assert k5['переходные 0.45/0.30 мм — 3 шт.'].probability == pytest.approx(0.0459, abs=0.0002)
    assert k5['переходные 0.50/0.30 мм'].severity is Severity.INFO
    k4, = _by(demo, 'К4')
    assert k4.nets == ('J2',) and k4.measured['поясок_мкм']['F.Cu'] == pytest.approx(50, abs=0.01)
    # К3 по критерию стыка: V5 (0,45/0,30, трасса 0,10) — 9,8 %; V4 (0,50/0,30, трасса 0,15) — 0,6 %, хотя
    # вероятность выхода отверстия ≈ 0: стык нарушается раньше выхода (Коробков, с. 92)
    k3 = {f.nets: f for f in _by(demo, 'К3')}
    assert set(k3) == {('V5',), ('V4',)}
    assert k3[('V5',)].severity is Severity.WARNING and not k3[('V5',)].measured['капля']
    assert k3[('V5',)].probability == pytest.approx(0.0984, abs=0.002)
    assert k3[('V4',)].severity is Severity.INFO and k3[('V4',)].probability == pytest.approx(0.0062, abs=0.001)
    k7, = _by(demo, 'К7')
    assert k7.probability == pytest.approx(0.0031, rel=0.03)
    nets = {n for f in demo for n in f.nets}
    assert not nets & {'V1', 'V8', 'J1_1', 'J1_2', 'J1_3', 'J1_4'}
    board_level = [f for f in _by(demo, 'К2') if f.title.startswith('плата')]
    assert len(board_level) == 1 and board_level[0].severity is Severity.ERROR


def test_a3_demo_class3(board):
    """Класс 3 (поясок 50 / 25 мкм): V3 — К1 (40 < 50 мкм на любой плате); V2 решает нижний
    наружный слой — 41 % (документ A3, раздел 3.2)."""
    prof = load_profile()
    prof['etch']['product_class'] = 3
    fs = run(board, ctx=Context(board, prof), classes=['A3'])
    k1 = _by(fs, 'К1')
    assert any('0.38/0.30' in f.title for f in k1)
    v2 = next(f for f in _by(fs, 'К2') if 'V2' in f.nets)
    assert v2.measured['худший_слой'] == 'B.Cu' and v2.probability == pytest.approx(0.413, abs=0.002)


def test_systematic_shift_k6(board):
    """К6: некомпенсированный масштаб 100 ppm на 200 мм (μ = 20 мкм) меняет вердикт у V4."""
    prof = load_profile()
    prof['drilling'].update(systematic_scale_ppm=100, scale_radius_mm=200)
    k6 = _by(run(board, ctx=Context(board, prof), classes=['A3']), 'К6')
    assert k6 and all(f.probability > f.measured['p_выход'][f.measured['худший_слой']] * 0 + 0 for f in k6)
    assert any('0.50/0.30' in f.title for f in k6)


def test_thick_board_k8(tmp_path):
    """К8: V4 (0,50/0,30), класс 3, плата 3,2 мм — σ низа 23,3 мкм, поясок < 50 мкм у 10,1 % плат."""
    w = KicadWriter('thick', thickness=3.235, stackup=[('F.Cu', 'copper', 0.035), ('dielectric 1', 'core', 3.165),
                                                       ('B.Cu', 'copper', 0.035)])     # низ на глубине 3,2 мм
    w.outline_rect(0, 0, 3, 3)
    w.via((1.5, 1.5), 0.50, 0.30, net='V4')
    b = load(w.write(tmp_path / 'thick.kicad_pcb'))
    prof = load_profile()
    prof['etch']['product_class'] = 3
    k8, = _by(run(b, ctx=Context(b, prof), classes=['A3']), 'К8')
    assert k8.layer == 'B.Cu' and k8.measured['σ_мкм']['B.Cu'] == pytest.approx(23.3, abs=0.1)
    assert k8.probability == pytest.approx(0.101, abs=0.002)


def test_injected_defects(tmp_path):
    """Инъекция дефектов (раздел 8.4): правильная плата чиста; уменьшенная площадка, уменьшенные
    внутренние площадки и offset находятся."""
    def make(**kw):
        w = KicadWriter('inj', copper=4)
        w.outline_rect(0, 0, 4, 3)
        w.via((1, 1.5), kw.get('D', 0.60), 0.30, net='A', padstack=kw.get('padstack'))
        fp = w.footprint('J', (3, 1.5), attrs='through_hole')
        fp.pad('1', 'thru_hole', 'circle', (0, 0), 1.6, drill=0.8, net='B', offset=kw.get('offset'))
        return load(w.write(tmp_path / f'inj{len(list(tmp_path.iterdir()))}.kicad_pcb'))
    codes = lambda b: sorted(f.code for f in run(b, classes=['A3']) if not f.title.startswith('плата'))
    assert codes(make()) == []
    assert codes(make(D=0.45)) == ['К5']
    assert codes(make(padstack=('front_inner_back', {'Inner': 0.45, 'B.Cu': 0.60}))) == ['К5']
    assert codes(make(offset=(0.33, 0))) == ['К4']


@pytest.mark.kicad
def test_nominal_ring_matches_kicad_drc(tmp_path):
    """Номинальный поясок совпадает с проверкой «ширина пояска» DRC KiCad (с её порогом 80 мкм):
    те же отверстия и те же величины, в том числе J2 со сдвинутой площадкой."""
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    sys.path.insert(0, str(ROOT / 'boards'))
    import make_boards
    w, _ = make_boards.a3_demo()
    w.rules['min_via_annular_width'] = 0.08
    p = w.write(tmp_path / 'a3_drc.kicad_pcb')
    rep = kicad_cli.drc(p, tmp_path / 'drc.json')
    kicad = {}
    for v in rep['violations']:
        if v['type'] == 'annular_width':
            pos = v['items'][0]['pos']
            actual = float(re.findall(r'(\d+[.,]\d+)\s*mm\)', v['description'])[-1].replace(',', '.'))
            kicad[(round(pos['x'], 3), round(-pos['y'], 3))] = actual * 1000
    ours = {(round(h.pos[0], 3), round(h.pos[1], 3)): min(r.values()) for h, r in nominal_rings(load(p))
            if min(r.values()) < 80 - 0.5}
    assert set(ours) == set(kicad)
    for k in ours:
        assert ours[k] == pytest.approx(kicad[k], abs=0.5)


@pytest.mark.kicad
def test_gerber_path_same_a3_findings(tmp_path, demo):
    """Через Gerber (стек из job-файла, площадки по слоям из меди) находки те же."""
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    kicad_cli.export_gerbers(A3, tmp_path / 'g')
    kicad_cli.export_drill(A3, tmp_path / 'g')
    fs = run(load(tmp_path / 'g'), classes=['A3'])
    key = lambda fs: sorted((f.code, f.severity.ru, round(f.probability or 0, 4)) for f in fs)
    assert key(fs) == key(demo)


def test_junction_criterion_korobkov():
    """К3 по Коробкову (2025, ф. (37), (41)): перемычки меди до углов стыка ≥ (1 − k)·w. Круглая площадка без
    смещения — перемычки 2·поясок; капля убирает сужение; D_пл < (1 − k)·w + d — брак уже без смещения (с. 93)."""
    from dfm.rules.a3 import junction_fail
    pad = shapely.Point(0, 0).buffer(0.225, quad_segs=300)
    td = shapely.union_all([pad, shapely.box(0, -0.05, 0.4, 0.05)]).convex_hull
    p, b0 = junction_fail(pad, np.zeros(2), 0.30, [((1, 0), 0.10)], 29.4, 0.2)
    assert b0 == pytest.approx(0.15, abs=1e-4) and 0.05 < p < 0.15
    assert junction_fail(td, np.zeros(2), 0.30, [((1, 0), 0.10)], 29.4, 0.2)[0] < 1e-4
    small = shapely.Point(0, 0).buffer(0.175, quad_segs=300)          # 0,35 < 0,15 + 0,25
    assert junction_fail(small, np.zeros(2), 0.25, [((1, 0), 0.15)], 5, 0.2)[0] > 0.9


def test_junction_author_point():
    """Коробков, с. 94: площадка 0,40, сверло 0,20, трасса 0,15 (k = 0), σ = 10 мкм → P годного ≈ 0,997;
    интеграл по эллипсу критерия даёт 0,295 %, Монте-Карло — 0,31 %."""
    import shapely
    from dfm.rules.a3 import junction_fail
    land = shapely.Point(0, 0).buffer(0.20, quad_segs=256)
    p, _ = junction_fail(land, (0.0, 0.0), 0.20, [((1.0, 0.0), 0.15)], 10.0, 0.0)
    assert p == pytest.approx(0.003, abs=0.0004)
