"""Маленькие платы прямо в модели (уровень P, без файла KiCad) — для регрессионных тестов правил."""
from __future__ import annotations

import numpy as np
import shapely

from dfm.board.model import Board, Feature, Hole, Role
from dfm.board.shapes import Circle, Contour, Region, Stroke


def board(copper=('F.Cu', 'B.Cu'), mask=True, outline=(0, 0, 20, 20), name='t'):
    """Плата с медными слоями copper (сверху вниз), масками (если mask) и прямоугольным контуром."""
    b = Board(name, 'model')
    for c in copper:
        b.add_layer(c)
    if mask:
        b.add_layer('F.Mask')
        b.add_layer('B.Mask')
    b.add_layer('Edge.Cuts')
    x0, y0, x1, y1 = outline
    b.outline = [Region(Contour(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)))]
    b.thickness = 1.6
    return b


def track(b, layer, p0, p1, w, net):
    f = Feature(Stroke(np.array([p0, p1], float), w), net, Role.TRACK)
    b.layers[layer].features.append(f)
    b.nets.add(net)
    return f


def pad(b, layer, center, d, net, role=Role.PAD, component=None, pin=None):
    f = Feature(Circle(center, d / 2), net, role, component=component, pin=pin)
    b.layers[layer].features.append(f)
    if net:
        b.nets.add(net)
    return f


def via(b, pos, drill, span, pads: dict, net, microvia=None):
    """Отверстие с площадками {слой: диаметр} (площадки кладутся и на слои, и в padstack)."""
    stack = {L: pad(b, L, pos, d, net, Role.VIA) for L, d in pads.items()}
    h = Hole(pos, drill, True, span, net, Role.VIA, padstack=stack, microvia=microvia)
    b.holes.append(h)
    return h


def mask_window(b, layer, center, d):
    f = Feature(Circle(center, d / 2), None, Role.PAD)
    b.layers[layer].features.append(f)
    return f


def region(b, layer, pts, net):
    f = Feature(Region(Contour(np.array(pts, float))), net, Role.ZONE)
    b.layers[layer].features.append(f)
    b.nets.add(net)
    return f


__all__ = ['board', 'track', 'pad', 'via', 'mask_window', 'region', 'shapely']
