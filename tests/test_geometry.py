"""Геометрия (шаг 1.4): медь по слоям и цепям, острова, связность, поиск соседей, растр.

Без KiCad — маленькие платы в тексте теста и демо-платы; сверка связности с DRC KiCad —
test_connectivity_matches_drc (помечен kicad)."""
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from dfm import kicad_cli
from dfm.board import load
from dfm.geometry.connectivity import connectivity
from dfm.geometry.copper import BoardGeometry
from dfm.geometry.neighbors import Neighbors
from dfm.geometry.raster import Grid, rasterize
from test_board_model import board

DEMOS = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'KiCad' / '10.0' / 'share' / 'kicad' / 'demos'
SMALL = sorted((p for p in DEMOS.rglob('*.kicad_pcb') if p.stat().st_size < 7e6), key=lambda p: p.stat().st_size) \
    if DEMOS.exists() else []
by_name = {p.stem: p for p in SMALL}
need_demos = pytest.mark.skipif(not SMALL, reason='нет демо-плат KiCad 10')


# ---------- растр ----------

def test_raster_exact_pixels():
    """Квадрат 1×1 при шаге 0,25: ровно 16 пикселей, строка 0 — верх окна."""
    img = rasterize(box(0, 0, 1, 1), Grid(-0.5, -0.5, 0.25, 8, 8))
    assert img.sum() == 16
    assert img[2:6, 2:6].all()
    tri = shapely.Polygon([(0, 0), (2, 0), (0, 1.8)])          # центры пикселей не на границе
    img = rasterize(tri, Grid(0, 0, 1, 2, 2))
    assert img.tolist() == [[False, False], [True, False]]     # верхняя строка — y ∈ [1, 2]


def test_raster_matches_shapely():
    """Многоугольник с вырезом: пиксель закрашен ⇔ его центр внутри (проверка случайных пикселей)."""
    g = box(0, 0, 10, 6).difference(Point(5, 3).buffer(2)).union(Point(12, 3).buffer(1.5))
    grid = Grid.around(g.bounds, 0.01, margin=0.5)
    img = rasterize(g, grid)
    x, y = grid.centers()
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, grid.ny, 50000), rng.integers(0, grid.nx, 50000)
    assert (shapely.contains_xy(g, x[j], y[i]) == img[i, j]).all()
    assert math.isclose(img.sum() * 1e-4, g.area, rel_tol=1e-3)


def test_raster_neighbours_tile_without_gaps():
    """Два квадрата с общей стороной: каждый пиксель закрашен ровно одним (правило полуинтервала)."""
    grid = Grid(0, 0, 0.1, 20, 10)
    a, b = rasterize(box(0, 0, 1, 1), grid), rasterize(box(1, 0, 2, 1), grid)
    assert not (a & b).any() and (a | b).all()


def test_raster_coverage():
    """Доля площади пикселя: сумма × площадь пикселя ≈ площади многоугольника."""
    g = Point(0, 0).buffer(1.234, quad_segs=64)
    grid = Grid.around(g.bounds, 0.1, margin=0.2)
    cov = rasterize(g, grid, supersample=8)
    assert cov.dtype == np.float32 and cov.max() == 1 and cov.min() == 0
    assert math.isclose(cov.sum() * 0.01, g.area, rel_tol=2e-3)


# ---------- медь, острова и связность на маленьких платах ----------

def _seg(a, b, net, layer='F.Cu', w=0.25):
    return f'(segment (start {a}) (end {b}) (width {w}) (layer "{layer}") (net "{net}"))'


def test_islands_and_shorts(tmp_path):
    """Дорожки одной цепи, которые касаются, — один остров; касание дорожки другой цепи — замыкание."""
    b = board(tmp_path, _seg('0 0', '5 0', 'A') + _seg('5 0', '5 5', 'A') + _seg('5 5', '8 5', 'B')
              + _seg('20 0', '25 0', 'C'))
    geo = BoardGeometry(b)
    lc = geo.layer('F.Cu')
    assert len(lc.islands) == 2
    assert sorted(map(sorted, lc.island_nets)) == [['A', 'B'], ['C']]
    c = connectivity(geo)
    assert [s.nets for s in c.shorts] == [('A', 'B')]
    s = c.shorts[0]
    assert s.layer == 'F.Cu' and math.dist(s.pos, (5, -5)) < 0.2


def test_via_joins_layers_and_open(tmp_path):
    """Переходное отверстие соединяет медь F.Cu и B.Cu; цепь, распавшаяся на два куска, — разрыв."""
    body = (_seg('0 0', '5 0', 'A') + _seg('5 0', '5 5', 'A', 'B.Cu')
            + '(via (at 5 0) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net "A"))'
            + _seg('10 0', '15 0', 'D') + _seg('10 3', '15 3', 'D'))
    b = board(tmp_path, body)
    c = connectivity(BoardGeometry(b))
    comps = {c.component_of(lay, k) for lay, k in c.nodes}
    assert len(comps) == c.count == 3                 # A (два слоя), D, D
    assert [o.net for o in c.opens] == ['D'] and c.missing_connections == 1
    assert not c.shorts


def test_zone_only_island_is_not_open(tmp_path):
    """Кусок заливки зоны без площадок и дорожек — изолированный островок, а не разрыв."""
    zone = ('(zone (net "GND") (layer "F.Cu") (polygon (pts (xy 0 0) (xy 30 0) (xy 30 10) (xy 0 10)))'
            ' (filled_polygon (layer "F.Cu") (pts (xy 0 0) (xy 10 0) (xy 10 10) (xy 0 10)))'
            ' (filled_polygon (layer "F.Cu") (pts (xy 20 0) (xy 30 0) (xy 30 10) (xy 20 10))))')
    b = board(tmp_path, zone + _seg('2 2', '8 2', 'GND'))
    c = connectivity(BoardGeometry(b))
    assert not c.opens
    assert len(c.islands) == 1 and c.islands[0].net == 'GND' and math.isclose(c.islands[0].area, 100)


def test_bridge_and_unnamed_copper(tmp_path):
    """Графика без цепи, соединяющая две цепи, — перемычка; касание одной цепи — отдельный список."""
    poly = '(gr_poly (pts (xy 4 -1) (xy 11 -1) (xy 11 1) (xy 4 1)) (stroke (width 0)) (fill yes) (layer "F.Cu"))'
    b = board(tmp_path, _seg('0 0', '5 0', 'A') + _seg('10 0', '15 0', 'B') + poly)
    c = connectivity(BoardGeometry(b))
    assert not c.shorts
    assert [x[0] for x in c.bridges] == [('A', 'B')]
    assert sorted(x[0] for x in c.unnamed_touch) == ['A', 'B']


def test_features_near(tmp_path):
    b = board(tmp_path, _seg('0 0', '5 0', 'A') + _seg('0 1', '5 1', 'B'))
    lc = BoardGeometry(b).layer('F.Cu')
    near = lc.features_near(Point(2, -0.5), 0.4)
    assert sorted(f.net for f in near) == ['A', 'B']
    assert [f.net for f in lc.features_near(Point(2, 0))] == ['A']


# ---------- поиск соседей ----------

@need_demos
def test_neighbors_match_brute_force():
    """Пары кусков разных цепей ближе 0,3 мм — те же, что при полном переборе."""
    b = load(by_name['pic_programmer'])
    nb = Neighbors(BoardGeometry(b).layer('B.Cu'))
    fast = {(g.pieces, round(g.distance, 9)) for g in nb.pairs_within(0.3)}
    slow = set()
    n = len(nb)
    for i in range(n):
        for j in range(i + 1, n):
            if nb.nets[i] == nb.nets[j] and nb.nets[i] is not None:
                continue
            d = nb.pieces[i].distance(nb.pieces[j])
            if d < 0.3:
                slow.add(((i, j), round(d, 9)))
    assert fast == slow and len(fast) > 10
    g = nb.pairs_within(0.3)[0]
    assert math.isclose(math.dist(g.a, g.b), g.distance, abs_tol=1e-9)


@need_demos
@pytest.mark.parametrize('name', ['pic_programmer', 'multichannel_mixer'])
def test_routed_demo_has_no_opens_or_shorts(name):
    if name not in by_name:
        pytest.skip(name)
    c = connectivity(BoardGeometry(load(by_name[name])))
    assert not c.opens and not c.shorts


# ---------- сверка связности с DRC KiCad ----------

# платы с разрывами, замыканиями, перемычками и островками + несколько обычных
DRC_QUICK = ('pic_programmer', 'StickHub', 'multichannel_mixer-unrouted', 'RoyalBlue54L-Feather',
             'One-Air-Max', 'CM5_MINIMA_3', 'tinytapeout-demo')
DRC_BOARDS = SMALL if os.environ.get('DFM_DRC_ALL') else [p for p in SMALL if p.stem in DRC_QUICK]
_NET = re.compile(r'\[([^\]]*)\]')
_ACTUAL = re.compile(r'(\d+[.,]\d+) mm\)\s*$')


def _kicad_shorts(report):
    """Пары именованных цепей, медь которых касается, по отчёту DRC: замыкания (shorting_items)
    и нарушения зазора с фактическим зазором 0 (так KiCad сообщает о заливке зоны поверх
    площадки чужой цепи)."""
    out = set()
    for v in report['violations']:
        nets = tuple(sorted({_NET.search(i['description']).group(1) for i in v['items']
                             if _NET.search(i['description'])}))
        if len(nets) != 2 or '<no net>' in nets:
            continue
        if v['type'] == 'shorting_items':
            out.add(nets)
        elif v['type'] == 'clearance':
            m = _ACTUAL.search(v['description'])
            if m and float(m.group(1).replace(',', '.')) == 0:
                out.add(nets)
    return out


@pytest.mark.kicad
@need_demos
@pytest.mark.parametrize('path', DRC_BOARDS, ids=lambda p: p.stem)
def test_connectivity_matches_drc(path, tmp_path):
    """Связность по геометрии = связность KiCad: число недостающих соединений равно числу
    «несоединённых элементов» DRC, пары замкнутых цепей совпадают, изолированные островки
    заливки есть тогда и только тогда, когда DRC сообщает isolated_copper."""
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    report = kicad_cli.drc(path, tmp_path / 'drc.json')
    c = connectivity(BoardGeometry(load(path)))
    assert c.missing_connections == len(report['unconnected_items'])
    assert {s.nets for s in c.shorts} == _kicad_shorts(report)
    assert bool(c.islands) == any(v['type'] == 'isolated_copper' for v in report['violations'])
