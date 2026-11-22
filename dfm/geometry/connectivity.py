"""Связность меди по геометрии и сверка с цепями из файла.

Цепь из файла — это замысел проектировщика; связность по геометрии — то, что реально получится
на плате. Узлы графа — острова меди на каждом слое (LayerCopper.islands); рёбра — металлизированные
отверстия: отверстие соединяет все острова слоёв своего диапазона, которые касаются его стенки
(пересекаются с кругом или пазом готового диаметра). Неметаллизированные отверстия ничего не
соединяют. Связные компоненты графа — электрические цепи по геометрии.

Сверка с файлом даёт:
  * замыкания — медь двух разных именованных цепей касается на слое или соединена отверстием;
  * разрывы — медь одной цепи распалась на несколько компонент (в KiCad это «несоединённые
    элементы»: число недостающих соединений цепи = компонент − 1);
  * изолированные островки заливки — компоненты только из заливки зон (не разрыв: KiCad
    сообщает о них отдельно, isolated copper; для класса B2 это прямые кандидаты);
  * медь без цепи, которая касается цепи (KiCad тоже считает это замыканием; в DFM это чаще
    замысел, например полоски СВЧ, нарисованные графикой) — отдельный список;
  * перемычки — две цепи соединены только медью без цепи (например, замкнутая паяльная
    перемычка «Bridged», где площадки соединены графикой компонента) — тоже отдельно.
Для Gerber без атрибутов цепей те же компоненты восстанавливают цепи (шаг 1.5).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..board.model import Role
from ..board.shapes import Pt
from .copper import BoardGeometry

__all__ = ['Short', 'Open', 'Island', 'Connectivity', 'connectivity', 'assign_nets']


@dataclass(slots=True)
class Short:
    """Замыкание двух цепей: медь касается на слое (via=None) или их соединяет отверстие."""
    nets: tuple[str, str]
    layer: str | None          # слой касания; для отверстия — None
    pos: Pt                    # точка касания или центр отверстия
    via_hole: bool = False


@dataclass(slots=True)
class Open:
    """Разрыв цепи: её медь распалась на fragments кусков (каждый — (слой, точка внутри))."""
    net: str
    fragments: list[tuple[str, Pt]] = field(default_factory=list)

    @property
    def missing(self) -> int:
        """Число недостающих соединений (как «несоединённые элементы» в KiCad)."""
        return len(self.fragments) - 1


@dataclass(slots=True)
class Island:
    """Изолированный островок заливки: медь зоны, ни с чем не соединённая (ни площадки, ни
    переходного отверстия, ни дорожки). KiCad оставляет такие островки, если они больше
    island_area_min (режим 2) или если удаление островков выключено."""
    net: str
    layer: str
    pos: Pt
    area: float


@dataclass(slots=True)
class Connectivity:
    nodes: list[tuple[str, int]]          # (слой, номер острова)
    component: np.ndarray                 # номер компоненты для каждого узла
    count: int                            # число компонент
    shorts: list[Short]
    opens: list[Open]
    unnamed_touch: list[tuple[str, str, Pt]]   # (цепь, слой, точка): медь без цепи касается цепи
    bridges: list[tuple[tuple[str, str], str, Pt]]   # (цепи, слой, точка): соединены медью без цепи
    islands: list[Island]                 # изолированные островки заливки зон

    @property
    def missing_connections(self) -> int:
        return sum(o.missing for o in self.opens)

    def component_of(self, layer: str, island: int) -> int:
        return int(self.component[self.nodes.index((layer, island))])


def _touch_point(a, b) -> Pt:
    g = a.intersection(b)
    if g.is_empty:                        # касание по границе с погрешностью — ближайшие точки
        p, _ = shapely.shortest_line(a, b).coords
        return (p[0], p[1])
    p = g.representative_point() if g.geom_type not in ('Point',) else g
    return (p.x, p.y)


def connectivity(geo: BoardGeometry) -> Connectivity:
    copper = geo.copper
    layers = {name: geo.layer(name) for name in copper}
    nodes, offset = [], {}
    for name in copper:
        offset[name] = len(nodes)
        nodes += [(name, i) for i in range(len(layers[name].islands))]

    # рёбра: металлизированные отверстия соединяют острова, касающиеся их стенки
    rows, cols, hole_links = [], [], []
    holes, hgeoms = [], []
    for h, hg in zip(geo.board.holes, geo.hole_geoms()):
        if h.plated and h.span[0] in copper and h.span[1] in copper:
            holes.append(h)
            hgeoms.append(hg)
    touched = [[] for _ in holes]
    if holes:
        # Запрос «острова слоя × дерево отверстий»: каждый (возможно огромный) остров
        # подготавливается один раз, а не проверяется заново для каждого отверстия.
        tree = shapely.STRtree(hgeoms)
        span = np.array([(copper.index(h.span[0]), copper.index(h.span[1])) for h in holes])
        for li, name in enumerate(copper):
            isl = layers[name].islands
            if not isl:
                continue
            src, dst = tree.query(isl, predicate='intersects')
            for k, j in sorted(zip(src.tolist(), dst.tolist()), key=lambda t: t[1]):
                if span[j, 0] <= li <= span[j, 1]:
                    touched[j].append((name, k))
    for h, hg, t in zip(holes, hgeoms, touched):
        ids = [offset[n] + k for n, k in t]
        for a, b in zip(ids, ids[1:]):
            rows.append(a)
            cols.append(b)
        hole_links.append((h, hg, t))
    n = len(nodes)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    count, comp = connected_components(graph, directed=False) if n else (0, np.zeros(0, int))

    shorts, unnamed, bridges = [], [], []
    for name in copper:
        lc = layers[name]
        for k, nets in enumerate(lc.island_nets):
            if len(nets) < 2:
                continue
            named = sorted(x for x in nets if x is not None)
            island = lc.islands[k]
            piece = lc.island_pieces[k]
            if None in nets:
                for net in named:
                    if piece[net].intersects(piece[None]):
                        unnamed.append((net, name, _touch_point(piece[net], piece[None])))
            for i, a in enumerate(named):
                for b in named[i + 1:]:
                    if piece[a].intersects(piece[b]):
                        # замыкание: медь двух цепей касается на слое
                        shorts.append(Short((a, b), name, _touch_point(piece[a], piece[b])))
                    else:
                        # цепи соединены только медью без цепи (например, перемычка в компоненте)
                        p = piece[None].representative_point() if None in piece else island.representative_point()
                        bridges.append(((a, b), name, (p.x, p.y)))
    # замыкания через отверстие: оно касается РАЗНЫХ островов с разными цепями. Берутся только
    # цепи, чья медь сама касается стенки отверстия (а не все цепи острова: остров может
    # содержать и чужую цепь, замкнутую на слое). Пары, уже лежащие вместе в одном острове,
    # объяснены на слое (замыкание или перемычка).
    seen = {s.nets for s in shorts} | {nets for nets, _, _ in bridges}
    for h, hg, touched in hole_links:
        sets = [frozenset(net for net, piece in layers[name].island_pieces[k].items() if piece.intersects(hg))
                for name, k in touched]
        pairs = set()
        for i, si in enumerate(sets):
            for sj in sets[i + 1:]:
                for a in si:
                    for b in sj:
                        if a is not None and b is not None and a != b:
                            pairs.add(tuple(sorted((a, b))))
        for pair in sorted(pairs):
            if pair not in seen:
                seen.add(pair)
                shorts.append(Short(pair, None, h.pos, via_hole=True))

    # разрывы: медь цепи лежит в нескольких компонентах. Компонента, в которой нет ничего, кроме
    # заливки зон, — изолированный островок заливки (KiCad: isolated copper), а не разрыв.
    zone_only = np.ones(count, dtype=bool)
    for idx, (name, k) in enumerate(nodes):
        if layers[name].island_roles[k] - {Role.ZONE}:
            zone_only[comp[idx]] = False
    where = defaultdict(dict)          # цепь → {компонента: (слой, точка)}
    islands = []
    for idx, (name, k) in enumerate(nodes):
        lc = layers[name]
        c = int(comp[idx])
        for net in lc.island_nets[k]:
            if net is None:
                continue
            if not zone_only[c] and c in where[net]:
                continue
            piece = lc.island_pieces[k][net]
            p = piece.representative_point()
            if zone_only[c]:
                islands.append(Island(net, name, (p.x, p.y), piece.area))
            elif c not in where[net]:
                where[net][c] = (name, (p.x, p.y))
    opens = [Open(net, list(frag.values())) for net, frag in sorted(where.items()) if len(frag) > 1]
    return Connectivity(nodes, comp, count, shorts, opens, unnamed, bridges, islands)


def assign_nets(board, prefix: str = 'N$') -> int:
    """Цепи по геометрии для платы без атрибутов цепей (Gerber X1, выгрузка без netlist).

    Каждая связная компонента меди (острова слоёв + металлизированные отверстия) — своя цепь
    prefix + номер; номера идут в порядке компонент (сверху вниз по слоям, затем по островам).
    Цепь получают объекты меди и отверстия (по площадкам падстека). Возвращает число цепей."""
    geo = BoardGeometry(board)
    c = connectivity(geo)
    names, order = {}, 0
    for comp in c.component:
        if int(comp) not in names:
            order += 1
            names[int(comp)] = f'{prefix}{order}'
    node_index = {node: i for i, node in enumerate(c.nodes)}
    for name in geo.copper:
        lc = geo.layer(name)
        dark = [k for k, f in enumerate(lc.features) if f.dark and not lc.feature_geoms[k].is_empty]
        if not dark or not lc.islands:
            continue
        pts = shapely.point_on_surface([lc.feature_geoms[k] for k in dark])
        src, dst = lc.locate(pts)
        for a, isl in zip(src, dst):
            f = lc.features[dark[int(a)]]
            f.net = names[int(c.component[node_index[(name, int(isl))]])]
    for h in board.holes:
        nets = [f.net for f in h.padstack.values() if f.net]
        h.net = nets[0] if nets and h.plated else None
    board.nets = {f.net for name in geo.copper for f in board.layers[name].features if f.net}
    return len(names)
