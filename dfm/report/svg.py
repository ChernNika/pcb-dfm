"""Обзор платы и фрагмент меди у каждой находки в координатах SVG."""
from __future__ import annotations

from html import escape

import numpy as np
import shapely

from ..geometry.copper import parts
from ..rules.findings import Severity

SEV_CLASS = {Severity.ERROR: 'err', Severity.WARNING: 'warn', Severity.INFO: 'info', Severity.NOT_CHECKED: 'nc',
             Severity.OK: 'ok'}


class BoardScene:
    """Кэш геометрии для картинок мест: плитки меди по слоям, отверстия, контур."""

    def __init__(self, ctx):
        self.ctx, self.board = ctx, ctx.board
        self._tiles = {}
        holes = ctx.geo.hole_geoms()
        self.holes = np.asarray(holes, dtype=object)
        self.hole_tree = shapely.STRtree(holes) if holes else None
        self.outline = ctx.geo.outline_geom()

    def tiles(self, layer):
        if layer not in self._tiles:
            if layer not in self.board.layers:
                self._tiles[layer] = None
            else:
                lc = self.ctx.geo.layer(layer)
                g, _ = lc.tiles()
                self._tiles[layer] = (g, lc.tile_tree()) if len(g) else None
        return self._tiles[layer]

    def clip(self, layer, box):
        t = self.tiles(layer)
        if t is None:
            return shapely.Polygon()
        g, tree = t
        idx = tree.query(box)
        if not len(idx):
            return shapely.Polygon()
        return shapely.union_all(shapely.intersection(g[idx], box))

    def holes_in(self, box):
        if self.hole_tree is None:
            return []
        return [self.holes[i] for i in self.hole_tree.query(box)]


def _d(g, nd=3) -> str:
    """Многоугольники → атрибут d пути SVG (ось Y перевёрнута)."""
    out = []
    for p in parts(g):
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)[:-1]
            if len(c) < 3:
                continue
            pts = ' '.join(f'{x:.{nd}f},{-y:.{nd}f}' for x, y in c)
            out.append(f'M{pts}Z')
    return ''.join(out)


def _points(f) -> list:
    pts = [tuple(f.pos)] if f.pos is not None else []
    m = f.measured
    for k in ('точка_a', 'точка_b'):
        if isinstance(m.get(k), (list, tuple)):
            pts.append(tuple(m[k]))
    for k in ('позиции', 'точки'):
        for p in (m.get(k) or [])[:20]:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pts.append(tuple(p))
    return pts


def _side_copper(board, layer):
    if layer == 'F.Mask':
        return board.copper[0]
    if layer == 'B.Mask':
        return board.copper[-1]
    return layer if layer in board.copper else board.copper[0]


def crop_svg(scene: BoardScene, f) -> str | None:
    """Картинка места находки (SVG) или None, если у находки нет места."""
    pts = _points(f)
    if not pts:
        return None
    a = np.asarray(pts, dtype=float)
    x0, y0 = a.min(axis=0)
    x1, y1 = a.max(axis=0)
    size = max(x1 - x0, y1 - y0)
    if size > 5.0:                                      # группа по всей плате — окно вокруг первой
        cx, cy = pts[0]
        half = 1.5
    else:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(0.6, size / 2 + max(0.4, 0.3 * size))
    box = shapely.box(cx - half, cy - half, cx + half, cy + half)
    layer = f.layer or scene.board.copper[0]
    cu = _side_copper(scene.board, layer)
    body = []
    if not scene.outline.is_empty:
        off = box.difference(scene.outline)
        if not off.is_empty:
            body.append(f'<path class="off" d="{_d(off)}"/>')
    copper = scene.clip(cu, box)
    if not copper.is_empty:
        body.append(f'<path class="cu" d="{_d(copper)}"/>')
    if layer in ('F.Mask', 'B.Mask'):
        win = scene.clip(layer, box)
        if not win.is_empty:
            body.append(f'<path class="win" d="{_d(win)}"/>')
    for h in scene.holes_in(box):
        body.append(f'<path class="hole" d="{_d(h.intersection(box))}"/>')
    sw = half / 60
    r = half / 9
    mk = []
    for (px, py) in pts[1:]:
        mk.append(f'<circle cx="{px:.4f}" cy="{-py:.4f}" r="{r * 0.55:.4f}" class="mk2" stroke-width="{sw:.4f}"/>')
    m = f.measured
    if isinstance(m.get('точка_a'), (list, tuple)) and isinstance(m.get('точка_b'), (list, tuple)):
        (ax, ay), (bx, by) = m['точка_a'], m['точка_b']
        mk.append(f'<line x1="{ax:.4f}" y1="{-ay:.4f}" x2="{bx:.4f}" y2="{-by:.4f}" class="mkl" '
                  f'stroke-width="{2 * sw:.4f}"/>')
    px, py = pts[0]
    mk.append(f'<circle cx="{px:.4f}" cy="{-py:.4f}" r="{r:.4f}" class="mk {SEV_CLASS[f.severity]}" '
              f'stroke-width="{2 * sw:.4f}"/>')
    scale = f'{2 * half:.2f} мм'
    return (f'<svg class="crop" viewBox="{cx - half:.4f} {-cy - half:.4f} {2 * half:.4f} {2 * half:.4f}" '
            f'role="img" aria-label="место находки, окно {scale}">'
            f'<rect class="bg" x="{cx - half:.4f}" y="{-cy - half:.4f}" width="{2 * half:.4f}" height="{2 * half:.4f}"/>'
            + ''.join(body) + ''.join(mk) + '</svg>'
            f'<div class="scale">окно {scale} · {escape(cu if layer == cu else f"{layer} над {cu}")}</div>')


def overview_svg(scene: BoardScene, findings) -> str:
    b = scene.board
    g = scene.outline
    if g.is_empty:
        return ''
    x0, y0, x1, y1 = g.bounds
    w, h = x1 - x0, y1 - y0
    pad = 0.02 * max(w, h)
    r = 0.012 * max(w, h)
    body = [f'<path class="board" d="{_d(g, 2)}"/>']
    if len(b.holes) <= 6000:
        pts = ''.join(f'M{hx:.2f},{-hy:.2f}h0' for hx, hy in (hh.pos for hh in b.holes))
        body.append(f'<path class="holes" d="{pts}" stroke-width="{r / 3:.3f}"/>')
    marks = []
    for i, f in sorted(enumerate(findings), key=lambda t: int(t[1].severity)):
        if f.pos is None:
            continue
        px, py = f.pos
        rr = r * (0.6 if f.severity is Severity.INFO else 1.0)       # информация — мельче, ошибки — поверх
        marks.append(f'<a href="#f{i}" data-i="{i}"><circle cx="{px:.3f}" cy="{-py:.3f}" r="{rr:.3f}" '
                     f'class="om {SEV_CLASS[f.severity]}"><title>{escape(f.cls + " " + f.code + ": " + f.title)}</title>'
                     f'</circle></a>')
    return (f'<svg class="overview" viewBox="{x0 - pad:.3f} {-y1 - pad:.3f} {w + 2 * pad:.3f} {h + 2 * pad:.3f}" '
            f'role="img" aria-label="плата с метками находок">' + ''.join(body) + ''.join(marks) + '</svg>')
