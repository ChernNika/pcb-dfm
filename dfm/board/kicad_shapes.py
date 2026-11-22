"""Формы объектов KiCad → точные примитивы модели (Circle, Stroke, Region, Compound).

    pad_shape(ps, pos, angle, margin)    форма площадки на плате (margin — расширение для окна маски)
    graphic_shape(g, f, grow)            линия, дуга, окружность, многоугольник, кривая Безье
    hole_ends(drill, pos, angle)         круглое отверстие или паз

Все формы площадок строятся в системе самой площадки (центр формы — начало координат,
ось Y вверх, без поворота), затем поворачиваются на угол площадки и переносятся в точку
pos + поворот(offset).

Расширение для маски повторяет то, что KiCad делает при выводе слоя маски (проверено по Gerber):
  * круг, овал — размер увеличивается на 2·margin (это точное расширение);
  * прямоугольник — размер + 2·margin и скругление углов радиусом margin (точное расширение);
    при отрицательном margin — просто меньший прямоугольник;
  * скруглённый прямоугольник — размер + 2·margin, а ДОЛЯ скругления сохраняется
    (радиус = доля × меньшая сторона нового размера), то есть это не точное расширение;
  * трапеция, срезанные углы, произвольная площадка — точное расширение многоугольника.
"""
from __future__ import annotations

import math

import numpy as np

from .design import Drill, Graphic, PadShape
from .shapes import (Circle, Compound, Contour, Pt, Region, Shape, Stroke, closed_stroke, ring_stroke,
                     rotate)

__all__ = ['pad_shape', 'graphic_shape', 'hole_ends', 'bezier_points', 'is_chamfered', 'box_contour', 'grow']

_S45 = math.sqrt(0.5)
# углы в системе площадки (ось Y вверх); «top» в KiCad — меньший y файла, то есть больший y здесь
_CORNERS = (('bottom_left', -1, -1), ('bottom_right', 1, -1), ('top_right', 1, 1), ('top_left', -1, 1))


def is_chamfered(ps: PadShape) -> bool:
    return ps.shape in ('roundrect', 'rect') and ps.chamfer_ratio > 0 and bool(ps.chamfer)


def _placer(pos: Pt, angle: float, offset: Pt):
    """Перевод точки из системы формы площадки в систему платы."""
    cx, cy = rotate(offset, angle)
    ox, oy = pos[0] + cx, pos[1] + cy
    if not angle:
        return lambda p: (ox + p[0], oy + p[1])
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    if angle % 90 == 0:
        c, s = round(c), round(s)
    return lambda p: (ox + p[0] * c - p[1] * s, oy + p[0] * s + p[1] * c)


def box_contour(w: float, h: float, radius: float = 0.0, chamfer: float = 0.0,
                 chamfered: frozenset = frozenset()) -> Contour:
    """Прямоугольник w×h с центром в начале координат, обход против часовой стрелки.
    Углы из chamfered срезаны на chamfer, остальные скруглены радиусом radius (0 — острые)."""
    hw, hh = w / 2, h / 2
    radius = min(radius, hw, hh)
    chamfer = min(chamfer, hw, hh) if chamfered else 0.0
    pts, arcs = [], {}
    for name, sx, sy in _CORNERS:
        x, y = sx * hw, sy * hh
        cut = chamfer if name in chamfered else radius
        if cut <= 0:
            pts.append((x, y))
            continue
        # при обходе против часовой в нижний левый и верхний правый углы приходят по вертикальной
        # стороне, в два других — по горизонтальной: a — точка входа в угол, b — точка выхода
        if sx * sy > 0:
            a, b = (x, y - sy * cut), (x - sx * cut, y)
        else:
            a, b = (x - sx * cut, y), (x, y - sy * cut)
        pts.append(a)
        if name not in chamfered:
            cxr, cyr = x - sx * cut, y - sy * cut            # центр скругления
            arcs[len(pts) - 1] = (cxr + sx * cut * _S45, cyr + sy * cut * _S45)
        pts.append(b)
    return Contour(np.array(pts, dtype=float), arcs)


def _trapezoid(w: float, h: float, delta: tuple[float, float]) -> Contour:
    """Трапеция KiCad: rect_delta = (dx, dy). dy сужает/расширяет левую и правую стороны,
    dx — верхнюю и нижнюю. Углы — как в PAD::BuildPadPolygon, переведённые в ось Y вверх."""
    hw, hh = w / 2, h / 2
    dx, dy = delta[0] / 2, delta[1] / 2
    # KiCad (ось Y вниз): (−hw−dy, +hh+dx), (−hw+dy, −hh−dx), (+hw−dy, −hh+dx), (+hw+dy, +hh−dx)
    kicad = [(-hw - dy, hh + dx), (hw + dy, hh - dx), (hw - dy, -hh + dx), (-hw + dy, -hh - dx)]
    return Contour(np.array([(x, -y) for x, y in kicad], dtype=float))


def _grown(c: Contour, margin: float) -> Shape:
    """Многоугольник, расширенный на margin > 0: сам многоугольник плюс обводка пером 2·margin."""
    return Compound((Region(c), closed_stroke(c, 2 * margin))) if margin > 0 else Region(c)


def _shrunk(shape: Shape, margin: float, tol: float) -> Shape | None:
    """Форма, уменьшенная на |margin| (margin < 0). Точного представления у уменьшенного
    многоугольника в общем случае нет, поэтому здесь результат — многоугольник (shapely)."""
    from ..geometry.polygons import to_geometry
    g = to_geometry(shape, tol).buffer(margin, quad_segs=16, join_style='mitre')
    if g.is_empty:
        return None
    polys = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    regions = tuple(Region(Contour(np.asarray(p.exterior.coords)[:-1]),
                           tuple(Contour(np.asarray(r.coords)[:-1]) for r in p.interiors)) for p in polys)
    return regions[0] if len(regions) == 1 else Compound(regions)


def pad_shape(ps: PadShape, pos: Pt, angle: float, margin: float = 0.0, tol: float = 0.0005) -> Shape | None:
    """Форма площадки на плате. margin — расширение (окно маски), 0 — сама медь.
    Возвращает None, если после расширения от формы ничего не осталось."""
    put = _placer(pos, angle, ps.offset)
    w, h = ps.size[0] + 2 * margin, ps.size[1] + 2 * margin
    kind = ps.shape
    if kind == 'custom':
        return _custom(ps, put, margin, tol)
    if w <= 0 or h <= 0:
        return None
    if kind == 'circle':
        return Circle(put((0.0, 0.0)), w / 2)
    if kind == 'oval':
        if w == h:
            return Circle(put((0.0, 0.0)), w / 2)
        e = ((w - h) / 2, 0.0) if w > h else (0.0, (h - w) / 2)
        return Stroke(np.array([put((-e[0], -e[1])), put(e)]), min(w, h))
    if kind == 'trapezoid':
        c = _trapezoid(ps.size[0], ps.size[1], ps.rect_delta).mapped(put)
        return _grown(c, margin) if margin >= 0 else _shrunk(Region(c), margin, tol)
    if is_chamfered(ps):
        m = min(ps.size)
        c = box_contour(ps.size[0], ps.size[1], ps.roundrect_rratio * m, ps.chamfer_ratio * m, ps.chamfer).mapped(put)
        if margin > 0:
            return Compound((Region(c), closed_stroke(c, 2 * margin)))
        return Region(c) if margin == 0 else _shrunk(Region(c), margin, tol)
    if kind == 'roundrect':
        return Region(box_contour(w, h, ps.roundrect_rratio * min(w, h)).mapped(put))
    if kind == 'rect':
        return Region(box_contour(w, h, margin if margin > 0 else 0.0).mapped(put))
    raise ValueError(f'неизвестная форма площадки: {kind}')


def _custom(ps: PadShape, put, margin: float, tol: float) -> Shape | None:
    """Произвольная площадка: опорная форма (круг диаметром size.x или прямоугольник size)
    плюс примитивы. Всё задано в системе площадки."""
    parts = []
    w, h = ps.size
    if ps.anchor == 'rect':
        if w > 0 and h > 0:
            parts.append(Region(box_contour(w, h).mapped(put)))
    elif w > 0:
        parts.append(Circle(put((0.0, 0.0)), w / 2))
    for g in ps.primitives:
        s = graphic_shape(g, put)
        if s is not None:
            parts.append(s)
    if not parts:
        return None
    shape = parts[0] if len(parts) == 1 else Compound(tuple(parts))
    if margin > 0:
        return grow(shape, margin)
    return shape if margin == 0 else _shrunk(shape, margin, tol)


def grow(shape: Shape, margin: float) -> Shape:
    """Точное расширение формы на margin > 0."""
    if isinstance(shape, Circle):
        return Circle(shape.center, shape.radius + margin)
    if isinstance(shape, Stroke):
        return Stroke(shape.pts, shape.width + 2 * margin, shape.arcs)
    if isinstance(shape, Region):
        parts = [shape, closed_stroke(shape.outer, 2 * margin)]
        parts += [closed_stroke(hc, 2 * margin) for hc in shape.holes]
        return Compound(tuple(parts))
    return Compound(tuple(grow(p, margin) for p in shape.parts))


def bezier_points(p0: Pt, p1: Pt, p2: Pt, p3: Pt, step: float = 0.02) -> np.ndarray:
    """Кубическая кривая Безье → точки ломаной (шаг по длине около step, мм)."""
    length = math.dist(p0, p1) + math.dist(p1, p2) + math.dist(p2, p3)
    n = min(max(int(math.ceil(length / step)), 16), 1000)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    a, b, c, d = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d


def graphic_shape(g: Graphic, f=None, grow_by: float = 0.0) -> Shape | None:
    """Графический объект → форма. f — перевод точек (для примитивов произвольной площадки);
    grow_by — расширение (для окна маски над медной графикой).
    Линия без толщины и без заливки формы не даёт (None)."""
    mapped = f is not None
    f = f or (lambda p: p)
    width = g.width + 2 * grow_by if g.width > 0 else (2 * grow_by if grow_by > 0 else 0.0)
    kind = g.kind
    if kind == 'line':
        if width <= 0:
            return None
        a, b = f(g.points[0]), f(g.points[1])
        return Stroke(np.array([a] if a == b else [a, b]), width)
    if kind == 'arc':
        if width <= 0:
            return None
        s, m, e = (f(p) for p in g.points)
        if s == e:                 # вырожденная дуга нулевой длины (не окружность)
            return Stroke(np.array([s]), width)
        return Stroke(np.array([s, e]), width, {0: m})
    if kind == 'curve':
        if width <= 0 or len(g.points) != 4:
            return None
        return Stroke(np.array([f(tuple(p)) for p in bezier_points(*g.points)]), width)
    if kind == 'circle':
        c, e = f(g.points[0]), f(g.points[1])
        r = math.dist(c, e)
        if g.filled:
            return Circle(c, r + width / 2)
        return ring_stroke(c, r, width) if width > 0 else None
    if kind == 'poly':
        c = g.contour.mapped(f) if mapped else g.contour
        if len(c) < 2:
            return None
        outline = closed_stroke(c, width) if width > 0 else None
        if g.filled and (len(c) >= 3 or c.arcs):
            return Compound((Region(c), outline)) if outline is not None else Region(c)
        return outline
    return None


def hole_ends(drill: Drill, pos: Pt, angle: float) -> tuple[Pt, Pt | None, float]:
    """Отверстие площадки → (центр или первый конец паза, второй конец паза или None, диаметр/ширина).
    Овальное отверстие — паз вдоль большей оси, повёрнутый вместе с площадкой."""
    sx, sy = drill.size
    if not drill.oval or abs(sx - sy) < 1e-9:
        return pos, None, sx
    half = ((sx - sy) / 2, 0.0) if sx > sy else (0.0, (sy - sx) / 2)
    ex, ey = rotate(half, angle)
    return (pos[0] - ex, pos[1] - ey), (pos[0] + ex, pos[1] + ey), min(sx, sy)
