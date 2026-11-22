"""Перевод точных форм модели платы в многоугольники shapely.

Это единственное место, где дуги заменяются ломаными. Точность задаётся параметром tol —
наибольшим отклонением ломаной от дуги, мм. Вершины ломаной лежат на самой дуге, поэтому
многоугольник круга чуть меньше круга (не больше чем на tol по радиусу).

    to_geometry(shape, tol)        одна форма → Polygon / MultiPolygon
    features_geometry(features)    список Feature → объединение (с учётом полярности)
    layer_geometry(layer)          то же для слоя
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

from ..board.shapes import Circle, Compound, Contour, Region, Stroke, arc_points

__all__ = ['DEFAULT_TOL', 'to_geometry', 'features_geometry', 'layer_geometry', 'contour_coords',
           'stroke_coords', 'polygonal', 'EMPTY', 'from_geometry']

DEFAULT_TOL = 0.0005        # 0,5 мкм
EMPTY = Polygon()


def _quad_segs(radius: float, tol: float) -> int:
    """Число отрезков на четверть окружности, при котором стрелка прогиба не больше tol."""
    if radius <= tol:
        return 2
    step = 2 * math.acos(1 - tol / radius)
    return max(int(math.ceil((math.pi / 2) / step)), 2)


def contour_coords(c: Contour, tol: float = DEFAULT_TOL) -> np.ndarray:
    """Вершины замкнутого контура, в котором дуги заменены ломаными."""
    if not c.arcs:
        return c.pts
    n = len(c.pts)
    out = []
    for i in range(n):
        p = c.pts[i]
        out.append(p[None, :])
        if i in c.arcs:
            q = c.pts[(i + 1) % n]
            out.append(arc_points(tuple(p), c.arcs[i], tuple(q), tol)[:-1])
    return np.vstack(out)


def stroke_coords(s: Stroke, tol: float = DEFAULT_TOL) -> np.ndarray:
    """Вершины осевой линии Stroke, в которой дуги заменены ломаными."""
    if not s.arcs:
        return s.pts
    out = [s.pts[:1]]
    for i in range(len(s.pts) - 1):
        if i in s.arcs:
            out.append(arc_points(tuple(s.pts[i]), s.arcs[i], tuple(s.pts[i + 1]), tol))
        else:
            out.append(s.pts[i + 1:i + 2])
    return np.vstack(out)


def polygonal(g):
    """Оставляет от геометрии только многоугольники (после make_valid могут появиться линии)."""
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    parts = [p for p in shapely.get_parts(g) if p.geom_type in ('Polygon', 'MultiPolygon')]
    return shapely.union_all(parts) if parts else EMPTY


def _region(r: Region, tol: float):
    outer = contour_coords(r.outer, tol)
    if len(outer) < 3:
        return EMPTY
    # число вершин проверяется после разбиения дуг: окружность — это контур из двух точек и двух дуг
    holes = [h for h in (contour_coords(c, tol) for c in r.holes) if len(h) >= 3]
    p = Polygon(outer, holes)
    if not p.is_valid:
        # KiCad хранит заливку зон «разрезанной»: вырезы соединены с внешним краем разрезами
        # нулевой ширины. Такой контур касается сам себя; make_valid превращает его в обычный
        # многоугольник с вырезами (разрезы пропадают, площадь не меняется).
        # Метод structure в 10 раз быстрее linework на заливках с десятками тысяч вершин, а
        # площадь та же (проверено на плате Jetson).
        p = polygonal(shapely.make_valid(p, method='structure', keep_collapsed=False))
    return p


def to_geometry(shape, tol: float = DEFAULT_TOL):
    """Форма модели → многоугольник shapely."""
    if isinstance(shape, Circle):
        return Point(shape.center).buffer(shape.radius, quad_segs=_quad_segs(shape.radius, tol))
    if isinstance(shape, Stroke):
        pts = stroke_coords(shape, tol)
        r = shape.width / 2
        if len(pts) > 1:
            keep = np.ones(len(pts), dtype=bool)
            keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)      # повторы подряд идущих точек
            pts = pts[keep]
        if len(pts) == 1:
            return Point(pts[0]).buffer(r, quad_segs=_quad_segs(r, tol))
        return LineString(pts).buffer(r, quad_segs=_quad_segs(r, tol), cap_style='round', join_style='round')
    if isinstance(shape, Region):
        return _region(shape, tol)
    if isinstance(shape, Compound):
        return shapely.union_all([to_geometry(p, tol) for p in shape.parts])
    raise TypeError(f'неизвестная форма: {type(shape).__name__}')


def features_geometry(features, tol: float = DEFAULT_TOL):
    """Объединение объектов слоя. Объекты применяются по порядку: тёмные добавляются,
    светлые (dark=False) вычитаются из уже нарисованного — как в Gerber."""
    result = None
    batch = []
    for f in features:
        g = to_geometry(f.shape, tol)
        if f.dark:
            batch.append(g)
            continue
        if batch:
            result = shapely.union_all(batch if result is None else [result, *batch])
            batch = []
        if result is not None:
            result = result.difference(g)
    if batch:
        result = shapely.union_all(batch if result is None else [result, *batch])
    return result if result is not None else EMPTY


def layer_geometry(layer, tol: float = DEFAULT_TOL):
    return features_geometry(layer.features, tol)


def from_geometry(g):
    """Многоугольник shapely → форма модели (Region или Compound из Region); пустой → None.
    Нужен там, где точной формы нет (уменьшенный многоугольник, макрос Gerber с вычитанием)."""
    from ..board.shapes import Compound, Contour, Region
    polys = [p for p in shapely.get_parts(polygonal(g)) if not p.is_empty]
    if not polys:
        return None
    regions = tuple(Region(Contour(np.asarray(p.exterior.coords)[:-1]),
                           tuple(Contour(np.asarray(r.coords)[:-1]) for r in p.interiors)) for p in polys)
    return regions[0] if len(regions) == 1 else Compound(regions)
