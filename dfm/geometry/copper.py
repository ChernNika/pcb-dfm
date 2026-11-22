"""Медь платы как многоугольники: слой целиком, по цепям, связные куски (острова).

    geo = BoardGeometry(board)          ленивое построение по слоям, с кешем
    lc = geo.layer('F.Cu')              LayerCopper
    lc.geom                             вся медь слоя (MultiPolygon)
    lc.nets['GND']                      медь цепи GND на слое (None — объекты без цепи)
    lc.islands, lc.island_nets          связные куски меди слоя и цепи объектов в каждом
    lc.island_roles                     роли объектов в каждом острове (площадка, зона …)
    lc.features_near(g, d)              объекты слоя (Feature) не дальше d от геометрии g

Объекты слоя применяются с учётом полярности (dark=False вычитает уже нарисованное, как в
Gerber), поэтому медь цепи — это объединение её объектов, пересечённое с итоговой медью слоя.
Островом называется связный многоугольник итоговой меди: куски разных цепей, которые касаются
друг друга, попадают в один остров — это и есть замыкание.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import shapely

from .polygons import DEFAULT_TOL, EMPTY, to_geometry

__all__ = ['LayerCopper', 'BoardGeometry', 'parts']


def parts(g) -> list:
    """Многоугольники геометрии (Polygon → [Polygon], MultiPolygon → его части)."""
    if g is None or g.is_empty:
        return []
    return [p for p in shapely.get_parts(g) if p.geom_type == 'Polygon' and not p.is_empty]


def _disjoint_union(geoms):
    """Объединение меди цепей слоя. Медь разных цепей почти никогда не касается (иначе это
    замыкание), поэтому слой собирается из кусков как есть, а объединяются только группы
    касающихся кусков. На больших платах это в разы быстрее общего union_all."""
    ps = [p for g in geoms for p in parts(g)]
    if not ps:
        return EMPTY
    # кандидаты — по прямоугольникам; точная проверка — с подготовленным БОЛЬШИМ куском первым
    # (запрос STRtree с условием подготовил бы маленькие куски, и каждый из них проверялся бы
    # против плоскости в десятки тысяч вершин без индекса — на Jetson это 17 с на слой)
    arr = np.asarray(ps, dtype=object)
    a, b = shapely.STRtree(ps).query(ps)
    keep = a < b
    a, b = a[keep], b[keep]
    size = shapely.get_num_coordinates(arr)
    swap = size[a] < size[b]
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    shapely.prepare(arr)
    ok = shapely.intersects(arr[a], arr[b])
    a, b = a[ok], b[ok]
    if not len(a):
        return shapely.MultiPolygon(ps)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(ps)
    _, label = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    groups = defaultdict(list)
    for i, lab in enumerate(label):
        groups[int(lab)].append(ps[i])
    out = []
    for g in groups.values():
        out += g if len(g) == 1 else parts(shapely.union_all(g))
    return shapely.MultiPolygon(out)


def _subdivide(poly, max_vertices: int, depth: int = 0) -> list:
    """Многоугольник → куски не больше max_vertices вершин (деление рамки пополам, рекурсивно;
    глубина ограничена — кусок, который делением не упрощается, остаётся как есть)."""
    if shapely.get_num_coordinates(poly) <= max_vertices or depth >= 24:
        return [poly]
    x0, y0, x1, y1 = poly.bounds
    if x1 - x0 >= y1 - y0:
        xm = (x0 + x1) / 2
        boxes = (shapely.box(x0 - 1, y0 - 1, xm, y1 + 1), shapely.box(xm, y0 - 1, x1 + 1, y1 + 1))
    else:
        ym = (y0 + y1) / 2
        boxes = (shapely.box(x0 - 1, y0 - 1, x1 + 1, ym), shapely.box(x0 - 1, ym, x1 + 1, y1 + 1))
    out = []
    for bx in boxes:
        for part in parts(poly.intersection(bx)):
            out += _subdivide(part, max_vertices, depth + 1)
    return out


class LayerCopper:
    """Медь одного слоя (или любого слоя модели, например маски)."""

    def __init__(self, layer, tol: float = DEFAULT_TOL):
        self.name = layer.name
        self.features = layer.features
        self.tol = tol
        self.feature_geoms = [to_geometry(f.shape, tol) for f in self.features]
        groups = defaultdict(list)
        for f, g in zip(self.features, self.feature_geoms):
            if f.dark:
                groups[f.net].append(g)
        by_net = {net: shapely.union_all(gs) for net, gs in groups.items()}
        if all(f.dark for f in self.features):
            self.geom = _disjoint_union(list(by_net.values()))
        else:
            from .polygons import features_geometry
            self.geom = features_geometry(self.features, tol)
            by_net = {n: g.intersection(self.geom) for n, g in by_net.items()}
        self.nets: dict[str | None, object] = {n: g for n, g in by_net.items() if not g.is_empty}
        self.islands: list = parts(self.geom)
        self.island_tree = shapely.STRtree(self.islands)
        self._prepared = False
        self._tiles = None
        self._tile_tree = None
        self.island_nets: list[frozenset] = self._island_nets()
        self.island_roles: list[frozenset] = self._island_roles()
        self._feature_tree = None

    def locate(self, pts):
        """Острова, в которые попадают точки pts → (номера точек, номера островов).

        Острова-плоскости бывают огромными (десятки тысяч вершин и тысячи вырезов). Запрос
        STRtree с условием подготавливает точки, а не острова, и тогда каждая проверка «точка
        в многоугольнике» проходит весь остров. Поэтому кандидаты берутся по прямоугольникам,
        а точная проверка идёт по подготовленным (индексированным) островам."""
        src, dst = self.island_tree.query(pts)
        if not len(src):
            return src, dst
        if not self._prepared:
            shapely.prepare(self.islands)
            self._prepared = True
        ok = shapely.intersects(np.asarray(self.islands, dtype=object)[dst], np.asarray(pts, dtype=object)[src])
        return src[ok], dst[ok]

    def _island_nets(self) -> list[frozenset]:
        """Цепи объектов, из которых состоит каждый остров. Кусок меди цепи целиком лежит в одном
        острове, поэтому остров ищется по внутренней точке куска. Попутно запоминаются сами куски:
        island_pieces[k][цепь] — медь цепи в острове k (без дорогих пересечений с большими
        многоугольниками цепей)."""
        nets = [set() for _ in self.islands]
        pieces = [defaultdict(list) for _ in self.islands]
        for net, g in self.nets.items():
            ps = parts(g)
            if not ps:
                continue
            pts = shapely.point_on_surface(ps)
            src, dst = self.locate(pts)
            for a, i in zip(src, dst):
                nets[int(i)].add(net)
                pieces[int(i)][net].append(ps[int(a)])
        self.island_pieces = [{n: (v[0] if len(v) == 1 else shapely.union_all(v)) for n, v in d.items()}
                              for d in pieces]
        return [frozenset(s) for s in nets]

    def _island_roles(self) -> list[frozenset]:
        """Роли объектов (площадка, дорожка, зона …), из которых состоит каждый остров. Объект
        может состоять из несвязанных кусков (произвольная площадка с отдельным примитивом) —
        роль получает остров каждого куска."""
        roles = [set() for _ in self.islands]
        dark = [k for k, f in enumerate(self.features) if f.dark and not self.feature_geoms[k].is_empty]
        if dark and self.islands:
            pieces, owner = shapely.get_parts([self.feature_geoms[k] for k in dark], return_index=True)
            src, dst = self.locate(shapely.point_on_surface(pieces))
            for a, i in zip(src, dst):
                roles[int(i)].add(self.features[dark[int(owner[int(a)])]].role)
        return [frozenset(r) for r in roles]

    @property
    def area(self) -> float:
        return self.geom.area

    def island_at(self, x: float, y: float) -> int | None:
        """Номер острова, в который попадает точка (с границей), или None."""
        idx = self.island_tree.query(shapely.points(x, y), predicate='intersects')
        return int(idx[0]) if len(idx) else None

    def tiles(self, max_vertices: int = 256):
        """Острова, нарезанные на плитки: (массив многоугольников, номер острова каждой плитки).

        Большой остров (плоскость земли — десятки тысяч вершин и тысячи вырезов) режется пополам
        по длинной стороне рамки, пока в куске не останется ≤ max_vertices вершин. Поиск соседей
        и расстояний по плиткам в десятки раз быстрее, чем по целым островам: каждая проверка
        касается только маленького куска. Расстояние между островами = минимум по их плиткам."""
        if self._tiles is None:
            geoms, owner = [], []
            for k, isl in enumerate(self.islands):
                for t in _subdivide(isl, max_vertices):
                    geoms.append(t)
                    owner.append(k)
            self._tiles = (np.asarray(geoms, dtype=object), np.asarray(owner, dtype=np.int64))
        return self._tiles

    def tile_tree(self):
        if self._tile_tree is None:
            self._tile_tree = shapely.STRtree(self.tiles()[0])
        return self._tile_tree

    def island_pairs_within(self, distance: float):
        """Пары островов ближе distance: {(i, j): (расстояние, точка на i, точка на j)}, i < j."""
        geoms, owner = self.tiles()
        if len(geoms) < 2:
            return {}
        a, b = self.tile_tree().query(geoms, predicate='dwithin', distance=distance)
        keep = owner[a] < owner[b]
        a, b = a[keep], b[keep]
        if not len(a):
            return {}
        d = shapely.distance(geoms[a], geoms[b])
        best = {}
        for ta, tb, dd in zip(a, b, d):
            key = (int(owner[ta]), int(owner[tb]))
            if dd < distance and (key not in best or dd < best[key][0]):
                best[key] = (float(dd), int(ta), int(tb))
        out = {}
        for key, (dd, ta, tb) in best.items():
            (ax, ay), (bx, by) = shapely.shortest_line(geoms[ta], geoms[tb]).coords
            out[key] = (dd, (ax, ay), (bx, by))
        return out

    def features_near(self, g, distance: float = 0.0) -> list:
        """Объекты слоя, форма которых не дальше distance от геометрии g (0 — касаются)."""
        if self._feature_tree is None:
            self._feature_tree = shapely.STRtree(self.feature_geoms)
        if distance > 0:
            idx = self._feature_tree.query(g, predicate='dwithin', distance=distance)
        else:
            idx = self._feature_tree.query(g, predicate='intersects')
        return [self.features[int(i)] for i in np.sort(idx)]


class BoardGeometry:
    """Геометрия платы: медь слоёв строится при первом обращении и запоминается."""

    def __init__(self, board, tol: float = DEFAULT_TOL):
        self.board = board
        self.tol = tol
        self._layers: dict[str, LayerCopper] = {}
        self._holes = None

    @property
    def copper(self) -> list[str]:
        return self.board.copper

    def layer(self, name: str) -> LayerCopper:
        if name not in self._layers:
            self._layers[name] = LayerCopper(self.board.layers[name], self.tol)
        return self._layers[name]

    def hole_geoms(self) -> list:
        """Отверстия платы как многоугольники (круг или паз готового диаметра)."""
        if self._holes is None:
            from ..board.shapes import Circle, Stroke
            out = []
            for h in self.board.holes:
                if h.slot_end is not None:
                    s = Stroke(np.array([h.pos, h.slot_end]), h.diameter)
                else:
                    s = Circle(h.pos, h.diameter / 2)
                out.append(to_geometry(s, self.tol))
            self._holes = out
        return self._holes

    def outline_geom(self):
        """Плата (контур с вырезами) как многоугольник."""
        return shapely.union_all([to_geometry(r, self.tol) for r in self.board.outline]) \
            if self.board.outline else EMPTY
