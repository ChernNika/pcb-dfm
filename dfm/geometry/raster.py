"""Растр: перевод многоугольников в картинку с заданным шагом.

Нужен нейросетям B-классов (вход — изображение участка слоя) и расчёту плотности меди (B3).

    grid = Grid.around(bounds, step=0.005)       # 5 мкм на пиксель, окно по прямоугольнику
    img = rasterize(geom, grid)                  # bool: центр пикселя внутри меди
    cov = rasterize(geom, grid, supersample=4)   # float32: доля площади пикселя, занятая медью

Строка 0 изображения — верх окна (наибольший y), как в обычных картинках; столбец 0 — левый край.
Пиксель закрашен, если его ЦЕНТР лежит внутри многоугольника (правило чёт-нечет по пересечениям
горизонтального луча с рёбрами). Заполнение построчное и векторизованное (numpy): для каждого ребра
считаются пересечения со всеми строками сразу, поэтому время пропорционально числу пересечений,
а не числу пикселей × рёбер. Многоугольник предварительно обрезается окном.

Доля площади (supersample=s) — среднее по s×s подпикселям; погрешность площади — около
периметр × шаг/(2s) в худшем случае, в среднем много меньше.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import box

__all__ = ['Grid', 'rasterize']


@dataclass(frozen=True, slots=True)
class Grid:
    """Окно растра: левый нижний угол (x0, y0), шаг, размер в пикселях."""
    x0: float
    y0: float
    step: float
    nx: int
    ny: int

    @classmethod
    def around(cls, bounds, step: float, margin: float = 0.0) -> Grid:
        """Окно, накрывающее прямоугольник bounds = (xmin, ymin, xmax, ymax) с полями margin;
        левый нижний угол — на bounds, размер округлён вверх до целого числа пикселей."""
        xmin, ymin, xmax, ymax = bounds
        xmin, ymin, xmax, ymax = xmin - margin, ymin - margin, xmax + margin, ymax + margin
        return cls(xmin, ymin, step, max(int(math.ceil((xmax - xmin) / step - 1e-9)), 1),
                   max(int(math.ceil((ymax - ymin) / step - 1e-9)), 1))

    @property
    def x1(self) -> float:
        return self.x0 + self.nx * self.step

    @property
    def y1(self) -> float:
        return self.y0 + self.ny * self.step

    @property
    def bounds(self):
        return self.x0, self.y0, self.x1, self.y1

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    def centers(self):
        """Координаты центров пикселей: (x по столбцам, y по строкам — сверху вниз)."""
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.step
        y = self.y1 - (np.arange(self.ny) + 0.5) * self.step
        return x, y

    def refined(self, s: int) -> Grid:
        return Grid(self.x0, self.y0, self.step / s, self.nx * s, self.ny * s)


def _edges(geom) -> np.ndarray:
    """Все рёбра всех колец многоугольников → массив (n, 4): x1, y1, x2, y2."""
    rings = []
    for p in shapely.get_parts(geom):
        if p.geom_type != 'Polygon' or p.is_empty:
            continue
        rings.append(np.asarray(p.exterior.coords))
        rings += [np.asarray(r.coords) for r in p.interiors]
    if not rings:
        return np.zeros((0, 4))
    return np.vstack([np.hstack((r[:-1], r[1:])) for r in rings if len(r) > 1])


def _fill(geom, g: Grid) -> np.ndarray:
    e = _edges(geom)
    img = np.zeros(g.shape, dtype=bool)
    if not len(e):
        return img
    x1, y1, x2, y2 = e.T
    ylo, yhi = np.minimum(y1, y2), np.maximum(y1, y2)
    # строка i содержит центр y = y1_окна − (i + 0,5)·шаг; ребро пересекает строки с центром в [ylo, yhi)
    t_top, h = g.y1, g.step
    i_lo = np.floor((t_top - yhi) / h - 0.5).astype(np.int64) + 1
    i_hi = np.floor((t_top - ylo) / h - 0.5).astype(np.int64)
    # центр ровно на ylo входит, ровно на yhi — нет (полуинтервал): поправка для точных совпадений
    yc_hi = t_top - (i_hi + 0.5) * h
    i_hi = np.where(yc_hi < ylo, i_hi - 1, i_hi)
    yc_lo = t_top - (i_lo + 0.5) * h
    i_lo = np.where(yc_lo >= yhi, i_lo + 1, i_lo)
    i_lo = np.clip(i_lo, 0, g.ny)
    i_hi = np.clip(i_hi, -1, g.ny - 1)
    n = np.maximum(i_hi - i_lo + 1, 0)
    keep = n > 0
    if not keep.any():
        return img
    x1, y1, x2, y2, i_lo, n = x1[keep], y1[keep], x2[keep], y2[keep], i_lo[keep], n[keep]
    edge = np.repeat(np.arange(len(n)), n)
    first = np.repeat(np.cumsum(n) - n, n)
    rows = i_lo[edge] + (np.arange(len(edge)) - first)
    yc = t_top - (rows + 0.5) * h
    xs = x1[edge] + (yc - y1[edge]) * (x2[edge] - x1[edge]) / (y2[edge] - y1[edge])
    # пересечение левее центра пикселя j переключает его состояние: j = первый столбец с центром > x
    cols = np.floor((xs - g.x0) / h - 0.5).astype(np.int64) + 1
    cols = np.clip(cols, 0, g.nx)
    w = g.nx + 1
    counts = np.bincount(rows * w + cols, minlength=g.ny * w).reshape(g.ny, w)
    img[:] = (np.cumsum(counts[:, :g.nx], axis=1) & 1).astype(bool)
    return img


def rasterize(geom, grid: Grid, supersample: int = 1) -> np.ndarray:
    """Многоугольник (Polygon/MultiPolygon без перекрытий частей) → изображение окна grid.
    supersample=1 — bool по центрам пикселей; s > 1 — float32, доля площади пикселя."""
    if geom is None or geom.is_empty:
        return np.zeros(grid.shape, dtype=bool if supersample == 1 else np.float32)
    clipped = geom.intersection(box(*grid.bounds).buffer(grid.step, join_style='mitre'))
    if supersample == 1:
        return _fill(clipped, grid)
    s = int(supersample)
    fine = _fill(clipped, grid.refined(s))
    return fine.reshape(grid.ny, s, grid.nx, s).mean(axis=(1, 3), dtype=np.float32)
