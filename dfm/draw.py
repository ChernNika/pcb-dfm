"""Картинки слоёв платы и карты расхождений (matplotlib, PNG).

    draw_layers(board, geo, папка)        медь каждого слоя, окна маски, контур
    draw_compare(kg, gg, diffs, папка)    медь слоя и места расхождения двух моделей

Оформление как у рисунков документов: шрифт DejaVu Sans, медь #D95B30. Ось Y вверх, мм.
Расхождения обычно субмикронные и на масштабе платы не видны, поэтому участки шире 1 мкм
обводятся кружками, а самый широкий подписывается.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import shapely                                      # noqa: E402
from matplotlib.patches import PathPatch            # noqa: E402
from matplotlib.path import Path as MPath           # noqa: E402

__all__ = ['draw_layers', 'draw_compare', 'geom_patch']

COPPER = '#D95B30'
MASK = '#2E7D4F'
EDGE = '#333333'
plt.rcParams['font.family'] = 'DejaVu Sans'


def geom_patch(g, **kw):
    """Многоугольники shapely (с вырезами) → один PathPatch."""
    verts, codes = [], []
    for p in shapely.get_parts(g):
        if p.geom_type != 'Polygon' or p.is_empty:
            continue
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
    if not verts:
        return None
    return PathPatch(MPath(np.vstack(verts), codes), **kw)


def _frame(geo, title):
    outline = geo.outline_geom()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    if not outline.is_empty:
        ax.add_patch(geom_patch(outline, facecolor='#F4F1EA', edgecolor=EDGE, lw=0.6))
        x0, y0, x1, y1 = outline.bounds
    else:
        x0, y0, x1, y1 = 0, 0, 1, 1
    m = 0.03 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x, мм')
    ax.set_ylabel('y, мм')
    ax.tick_params(labelsize=7)
    return fig, ax


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_layers(board, geo, out_dir) -> list[Path]:
    out_dir = Path(out_dir)
    paths = []
    for name in board.copper + ['F.Mask', 'B.Mask']:
        lc = geo.layer(name)
        color = COPPER if name in board.copper else MASK
        what = 'медь' if name in board.copper else 'окна маски'
        fig, ax = _frame(geo, f'{board.name}: {name} — {what}, {lc.area:.1f} мм²')
        patch = geom_patch(lc.geom, facecolor=color, edgecolor='none', alpha=0.9)
        if patch is not None:
            ax.add_patch(patch)
        for h in board.holes:
            ax.add_patch(plt.Circle(h.pos, h.diameter / 2, facecolor='white', edgecolor='none'))
        p = out_dir / f'{name.replace(".", "_")}.png'
        _save(fig, p)
        paths.append(p)
    return paths


def draw_compare(kg, gg, diffs, out_dir, threshold: float = 0.001) -> list[Path]:
    """Карта расхождений слоя: медь (модель KiCad) и кружки вокруг участков расхождения шире
    threshold (мм); подпись — наибольшее отклонение края и его место."""
    out_dir = Path(out_dir)
    paths = []
    for d in diffs:
        a, b = kg.layer(d.layer), gg.layer(d.layer)
        diff = a.geom.symmetric_difference(b.geom)
        fig, ax = _frame(kg, f'{d.layer}: расхождение {d.diff_area:.4f} мм², '
                             f'наибольшее отклонение края {d.max_dev * 1000:.2f} мкм')
        patch = geom_patch(a.geom, facecolor='#C9C2B8', edgecolor='none')
        if patch is not None:
            ax.add_patch(patch)
        big = [p for p in shapely.get_parts(diff)
               if p.geom_type == 'Polygon' and 2 * np.sqrt(p.area / np.pi) > threshold]
        if big:
            wide = [p for p in big if not p.buffer(-threshold / 2).is_empty]
            pts = np.array([p.representative_point().coords[0] for p in wide]) if wide else np.zeros((0, 2))
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=40, facecolors='none', edgecolors='#C0392B', lw=1.0,
                           label=f'участки шире {threshold * 1000:g} мкм: {len(pts)}')
                ax.legend(loc='upper right', fontsize=7)
        if d.max_dev_at is not None and d.max_dev > threshold:
            ax.annotate(f'{d.max_dev * 1000:.1f} мкм', d.max_dev_at, xytext=(10, 10), textcoords='offset points',
                        fontsize=7, color='#C0392B', arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.6))
        p = out_dir / f'{d.layer.replace(".", "_")}.png'
        _save(fig, p)
        paths.append(p)
    return paths
