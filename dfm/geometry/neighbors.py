"""Быстрый поиск соседей: куски меди разных цепей, которые ближе заданного расстояния.

Без пространственного индекса проверка зазоров сравнивала бы каждый кусок меди с каждым
(на плате Jetson это десятки миллионов пар). Здесь куски меди слоя кладутся в R-дерево
(shapely.STRtree); запрос «какие пары ближе d» сначала отбирает пары по ограничивающим
прямоугольникам, расширенным на d, и только для них считает точное расстояние.

    nb = Neighbors(geo.layer('F.Cu'))
    for gap in nb.pairs_within(0.2):     # все пары разных цепей с зазором < 0,2 мм
        gap.nets, gap.distance, gap.a, gap.b

Кусок — связный многоугольник меди одной цепи (часть LayerCopper.nets[цепь]). Медь без цепи:
каждый её кусок считается отдельной «цепью» (так же поступает KiCad), то есть зазор между
двумя кусками без цепи тоже проверяется.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from ..board.shapes import Pt
from .copper import LayerCopper, parts

__all__ = ['Gap', 'Neighbors']


@dataclass(slots=True)
class Gap:
    """Зазор между двумя кусками меди: расстояние и ближайшие точки a (на первом) и b."""
    layer: str
    nets: tuple[str | None, str | None]
    distance: float
    a: Pt
    b: Pt
    pieces: tuple[int, int]          # номера кусков в Neighbors.pieces


class Neighbors:
    def __init__(self, lc: LayerCopper):
        self.layer = lc.name
        self.pieces = []             # многоугольники
        self.nets = []               # цепь каждого куска
        for net, g in lc.nets.items():
            for p in parts(g):
                self.pieces.append(p)
                self.nets.append(net)
        self.tree = shapely.STRtree(self.pieces)

    def __len__(self):
        return len(self.pieces)

    def pairs_within(self, distance: float, same_net: bool = False) -> list[Gap]:
        """Пары кусков с расстоянием < distance (касающиеся — расстояние 0). По умолчанию только
        пары разных цепей; same_net=True — только пары кусков одной цепи (например, узкие щели
        внутри цепи)."""
        if not self.pieces:
            return []
        i, j = self.tree.query(self.pieces, predicate='dwithin', distance=distance)
        keep = i < j
        i, j = i[keep], j[keep]
        nets = np.array(self.nets, dtype=object)
        same = (nets[i] == nets[j]) & (nets[i] != None)     # noqa: E711 — None — отдельные куски
        keep = same if same_net else ~same
        i, j = i[keep], j[keep]
        if not len(i):
            return []
        a = np.array(self.pieces, dtype=object)
        d = shapely.distance(a[i], a[j])
        keep = d < distance
        i, j, d = i[keep], j[keep], d[keep]
        lines = shapely.shortest_line(a[i], a[j])
        out = []
        for ii, jj, dd, ln in zip(i, j, d, lines):
            (ax, ay), (bx, by) = ln.coords
            out.append(Gap(self.layer, (self.nets[ii], self.nets[jj]), float(dd), (ax, ay), (bx, by),
                           (int(ii), int(jj))))
        out.sort(key=lambda g: g.distance)
        return out

    def near(self, g, distance: float) -> list[int]:
        """Номера кусков не дальше distance от геометрии g."""
        return [int(k) for k in np.sort(self.tree.query(g, predicate='dwithin', distance=distance))]
