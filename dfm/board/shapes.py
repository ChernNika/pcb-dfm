"""Точные геометрические примитивы модели платы.

Любой объект меди или маски из KiCad и из Gerber представим ТОЧНО тремя примитивами
и их объединением:

    Circle   — круг (переходное отверстие, круглая площадка, вспышка круглой апертуры);
    Stroke   — линия с круглыми концами (дорожка, дуга-дорожка, овальная площадка,
               линия графики, рисование круглой апертурой в Gerber);
    Region   — область: внешний контур и вырезы (прямоугольные, скруглённые, трапециевидные
               площадки, заливки зон, многоугольники, регионы Gerber);
    Compound — объединение нескольких форм, если один объект составлен из частей
               (произвольная площадка = опорная форма + примитивы; многоугольник с обводкой;
               макро-апертура Gerber).

Контуры хранят дуги как дуги (точка на дуге), а не ломаные. Дуги разбиваются на отрезки
в одном месте — при переводе в многоугольники (dfm.geometry), с заданной точностью.

Система координат всей модели: миллиметры, ось Y вверх (как в Gerber и в математике),
углы в градусах, положительные — против часовой стрелки.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ['Pt', 'Contour', 'Circle', 'Stroke', 'Region', 'Compound', 'Shape', 'rotate', 'transform',
           'arc_geometry', 'arc_points', 'closed_stroke', 'ring_stroke', 'polygon', 'outline_of',
           'mapped', 'moved']

Pt = tuple[float, float]


def rotate(p: Pt, angle_deg: float) -> Pt:
    """Поворот точки вокруг начала координат против часовой стрелки."""
    if not angle_deg:
        return p
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    # точные значения для углов, кратных 90°, чтобы не появлялись хвосты вида 1e-17
    if angle_deg % 90 == 0:
        c, s = round(c), round(s)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def transform(p: Pt, origin: Pt, angle_deg: float) -> Pt:
    """Перевод точки из локальной системы (компонента, площадки) в систему платы."""
    x, y = rotate(p, angle_deg)
    return (origin[0] + x, origin[1] + y)


def _points(pts) -> np.ndarray:
    a = np.asarray(pts, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f'ожидался массив точек (n, 2), получено {a.shape}')
    return a


@dataclass(frozen=True, slots=True, eq=False)
class Contour:
    """Замкнутый контур: вершины pts (n×2) и дуги.

    arcs[i] = точка на дуге, если ребро от pts[i] к pts[i+1] (последнее — к pts[0]) — дуга.
    Остальные рёбра — отрезки."""
    pts: np.ndarray
    arcs: dict[int, Pt] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'pts', _points(self.pts))

    def __len__(self):
        return len(self.pts)

    def mapped(self, f) -> Contour:
        """Новый контур, у которого каждая точка (и точки дуг) переведена функцией f."""
        return Contour(np.array([f((x, y)) for x, y in self.pts]),
                       {i: f(m) for i, m in self.arcs.items()})


@dataclass(frozen=True, slots=True, eq=False)
class Circle:
    """Сплошной круг."""
    center: Pt
    radius: float


@dataclass(frozen=True, slots=True, eq=False)
class Stroke:
    """Незамкнутая линия pts (n×2, n ≥ 1), проведённая круглым пером диаметра width.
    arcs — как у Contour, но для рёбер i → i+1 (замыкающего ребра нет).
    Одна точка — это круг диаметра width (вспышка круглой апертурой в Gerber рисуется Circle,
    а Stroke из одной точки встречается у вырожденных дорожек нулевой длины)."""
    pts: np.ndarray
    width: float
    arcs: dict[int, Pt] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'pts', _points(self.pts))


@dataclass(frozen=True, slots=True, eq=False)
class Region:
    """Область: внешний контур минус вырезы holes."""
    outer: Contour
    holes: tuple[Contour, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class Compound:
    """Объединение форм parts: один объект, составленный из нескольких частей."""
    parts: tuple


Shape = Circle | Stroke | Region | Compound


# ---------- дуги ----------

def arc_geometry(start: Pt, mid: Pt, end: Pt):
    """Дуга по трём точкам → (центр, радиус, начальный угол, угол поворота) в радианах.
    Угол поворота положительный — против часовой стрелки. Если start == end, дуга — полная
    окружность, а mid — диаметрально противоположная точка. Возвращает None, если три точки
    лежат на одной прямой (дуги нет — это отрезок)."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    if (x1, y1) == (x3, y3):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = math.hypot(x1 - cx, y1 - cy)
        return ((cx, cy), r, math.atan2(y1 - cy, x1 - cx), 2 * math.pi) if r > 0 else None
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    scale = max(abs(x1 - x3), abs(y1 - y3), abs(x1 - x2), abs(y1 - y2))
    if abs(d) < 1e-12 * max(scale * scale, 1e-30):
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    r = math.hypot(x1 - cx, y1 - cy)
    a1 = math.atan2(y1 - cy, x1 - cx)
    a2 = math.atan2(y2 - cy, x2 - cx)
    a3 = math.atan2(y3 - cy, x3 - cx)
    two_pi = 2 * math.pi
    ccw_mid = (a2 - a1) % two_pi          # путь против часовой до середины и до конца
    ccw_end = (a3 - a1) % two_pi
    sweep = ccw_end if ccw_mid < ccw_end else ccw_end - two_pi
    return (cx, cy), r, a1, sweep


def arc_points(start: Pt, mid: Pt, end: Pt, tol: float) -> np.ndarray:
    """Точки ломаной, заменяющей дугу (без начальной точки, с конечной). Отклонение ломаной от
    дуги не больше tol; вершины лежат на дуге."""
    g = arc_geometry(start, mid, end)
    if g is None:
        return np.array([end], dtype=float)
    (cx, cy), r, a1, sweep = g
    step = 2 * math.acos(max(1 - tol / r, -1.0)) if r > tol else math.pi / 2
    n = max(int(math.ceil(abs(sweep) / step)), 2)
    a = a1 + sweep * np.arange(1, n + 1) / n
    pts = np.column_stack((cx + r * np.cos(a), cy + r * np.sin(a)))
    pts[-1] = end
    return pts


# ---------- готовые формы ----------

def polygon(pts) -> Region:
    """Область без вырезов и дуг по списку вершин."""
    return Region(Contour(pts))


def closed_stroke(c: Contour, width: float) -> Stroke:
    """Обводка замкнутого контура круглым пером: линия проходит по всем рёбрам и возвращается
    в первую вершину."""
    return Stroke(np.vstack((c.pts, c.pts[:1])), width, dict(c.arcs))


def ring_stroke(center: Pt, radius: float, width: float) -> Stroke:
    """Окружность, проведённая пером width (кольцо)."""
    cx, cy = center
    a, b = (cx + radius, cy), (cx - radius, cy)
    return Stroke(np.array([a, b, a]), width, {0: (cx, cy + radius), 1: (cx, cy - radius)})


def outline_of(shape) -> list[Contour]:
    """Внешние контуры областей формы (для Region и Compound из областей); у остальных — пусто."""
    if isinstance(shape, Region):
        return [shape.outer]
    if isinstance(shape, Compound):
        return [c for p in shape.parts for c in outline_of(p)]
    return []


def mapped(shape, f):
    """Новая форма, у которой каждая точка переведена функцией f (сдвиг, поворот). Ширина линий и
    радиусы не меняются, поэтому f должна сохранять расстояния (движение плоскости)."""
    if isinstance(shape, Circle):
        return Circle(f(shape.center), shape.radius)
    if isinstance(shape, Stroke):
        return Stroke(np.array([f((x, y)) for x, y in shape.pts]), shape.width,
                      {i: f(m) for i, m in shape.arcs.items()})
    if isinstance(shape, Region):
        return Region(shape.outer.mapped(f), tuple(h.mapped(f) for h in shape.holes))
    return Compound(tuple(mapped(p, f) for p in shape.parts))


def moved(shape, dx: float, dy: float):
    """Форма, сдвинутая на (dx, dy)."""
    if not dx and not dy:
        return shape
    return mapped(shape, lambda p: (p[0] + dx, p[1] + dy))
