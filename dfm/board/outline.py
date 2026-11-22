"""Контур платы: сшивка отдельных линий и дуг в замкнутые контуры и разбор вложенности.

    stitch(closed, pieces, tol) → (области платы с вырезами, замечания)

closed — уже замкнутые контуры (окружности, многоугольники), pieces — незамкнутые куски
(точки, дуги) в виде (массив n×2, {номер ребра: точка на дуге}). Куски сшиваются по концам,
совпадающим с точностью tol; чётная вложенность контура — плата, нечётная — вырез.
Общий для адаптеров KiCad (Edge.Cuts) и Gerber (файл Profile).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import Polygon

from ..geometry.polygons import contour_coords
from .shapes import Contour, Region

__all__ = ['stitch', 'CHAIN_TOL']

CHAIN_TOL = 0.02        # мм: концы соседних линий контура, которые считаются одной точкой
_GEOM_TOL = 0.001


def _reversed(piece):
    pts, arcs = piece
    n = len(pts)
    return pts[::-1].copy(), {n - 2 - i: m for i, m in arcs.items()}


def _join(path) -> Contour:
    """Цепочка кусков (точки, дуги) → замкнутый контур. Совпадающие точки стыков не повторяются;
    если концы разошлись (в пределах допуска), между ними остаётся крошечный отрезок."""
    pts, arcs = [], {}
    for p, a in path:
        start = 0
        if pts and np.array_equal(pts[-1], p[0]):
            start = 1
        base = len(pts) - start
        pts.extend(p[start:])
        for i, m in a.items():
            arcs[base + i] = m
    if len(pts) > 1 and np.array_equal(pts[-1], pts[0]):
        pts.pop()
        arcs = {i: m for i, m in arcs.items() if i < len(pts)}
    return Contour(np.array(pts, dtype=float), arcs)


def _chain(pieces, tol):
    """Сшивает незамкнутые куски в замкнутые контуры. → (контуры, число незамкнутых цепочек)."""
    n = len(pieces)
    if not n:
        return [], 0
    starts = np.array([p[0][0] for p in pieces])
    ends = np.array([p[0][-1] for p in pieces])
    used = np.zeros(n, dtype=bool)
    contours, broken = [], 0

    def extend(path):
        """Наращивает цепочку с хвоста, пока находится продолжение. True — цепочка замкнулась."""
        while True:
            head, tail = path[0][0][0], path[-1][0][-1]
            if len(path) > 1 and math.dist(head, tail) <= tol:
                return True
            ds = np.hypot(*(starts - tail).T)
            de = np.hypot(*(ends - tail).T)
            ds[used] = np.inf
            de[used] = np.inf
            js, je = int(np.argmin(ds)), int(np.argmin(de))
            if min(ds[js], de[je]) > tol:
                return False
            if ds[js] <= de[je]:
                used[js] = True
                path.append(pieces[js])
            else:
                used[je] = True
                path.append(_reversed(pieces[je]))

    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        path = [pieces[i]]
        if not extend(path):
            path = [_reversed(p) for p in reversed(path)]     # продолжить с другого конца
            if not extend(path):
                broken += 1
                continue
        contours.append(_join(path))
    return contours, broken


def stitch(closed: list[Contour], pieces: list, tol: float = CHAIN_TOL) -> tuple[list[Region], list[str]]:
    """Замкнутые контуры и незамкнутые куски → (области платы с вырезами, замечания)."""
    chained, broken = _chain(pieces, tol)
    problems = []
    if broken:
        problems.append(f'контур платы: {broken} незамкнутых цепочек линий — они пропущены')
    contours = closed + chained
    if not contours:
        if not pieces and not closed:
            problems.append('линий контура платы нет — контур не построен')
        return [], problems

    polys = []
    for c in contours:
        p = Polygon(contour_coords(c, _GEOM_TOL))
        if not p.is_valid:
            problems.append('контур платы пересекает сам себя')
            p = p.buffer(0)
        polys.append(p)
    keep = [i for i, p in enumerate(polys) if p.area > 0]
    contours = [contours[i] for i in keep]
    polys = [polys[i] for i in keep]

    # вложенность: сколько других контуров содержат данный (по его внутренней точке)
    areas = np.array([p.area for p in polys])
    pts = [p.representative_point() for p in polys]
    tree = shapely.STRtree(polys)
    inside, container = tree.query(pts, predicate='within')
    parents = defaultdict(list)
    for i, j in zip(inside, container):
        if i != j and areas[j] > areas[i]:
            parents[int(i)].append(int(j))
    depth = {i: len(parents[i]) for i in range(len(polys))}
    holes = defaultdict(list)
    for i, d in depth.items():
        if d % 2:
            parent = min(parents[i], key=lambda j: areas[j])      # ближайший охватывающий контур
            holes[parent].append(contours[i])
    regions = [Region(contours[i], tuple(holes[i])) for i in range(len(polys)) if depth[i] % 2 == 0]
    return regions, problems
