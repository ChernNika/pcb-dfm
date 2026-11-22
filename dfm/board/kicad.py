"""Адаптер KiCad: .kicad_pcb (+ .kicad_pro) → модель платы.

Здесь читается уровень D целиком (компоненты, площадки, дорожки, переходные отверстия, зоны,
графика, тексты на меди и маске, правила проекта) и общая часть уровня P (слои, стек, цепи,
толщина платы). Медь, окна маски, отверстия с падстеками и контур уровня P затем строятся
из уровня D в kicad_build.py.

Поддерживаются форматы KiCad 8.99 … 10 (проверено на демо-платах: 20241030, 20241229,
20250513, 20250907, 20260206). Различия, которые здесь учтены:
  * цепи: KiCad ≤ 9 — таблица (net 3 "GND") в начале файла и ссылки (net 3) в объектах;
    KiCad 10 — только имя (net "GND");
  * номера слоёв в KiCad 9 поменялись — порядок слоёв определяется по именам (layers.py).

Перевод координат: x_модели = x_KiCad, y_модели = −y_KiCad. Угол не меняется: поворот,
который в KiCad виден на экране против часовой стрелки, в системе с осью Y вверх тоже
против часовой стрелки. Объекты компонента записаны в файле относительно компонента и
без его поворота: на плате = позиция компонента + поворот(угол компонента)·(x, −y).
Угол площадки в файле — уже полный (включает поворот компонента), второй раз не поворачивается.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

from operator import itemgetter

import numpy as np

from . import sexpr
from .design import (ArcTrack, Design, DesignRules, Drill, Footprint, Graphic, NetClass, Pad, PadShape,
                     PadType, Text, Track, Via, ViaType, Zone, ZoneConnection)
from .kicad_build import build_physical
from .kicad_shapes import box_contour
from .layers import LayerKind, copper_order, expand, kind_of
from .model import Board, StackItem, StackKind, Stackup
from .sexpr import Node, Sym
from .shapes import Contour, Pt, transform

__all__ = ['load_kicad', 'TESTED_VERSIONS']

TESTED_VERSIONS = (20241030, 20260206)   # самая старая и самая новая проверенные версии формата

_FP_GRAPHICS = frozenset({'fp_line', 'fp_arc', 'fp_circle', 'fp_rect', 'fp_poly', 'fp_curve'})
_GR_GRAPHICS = frozenset({'gr_line', 'gr_arc', 'gr_circle', 'gr_rect', 'gr_poly', 'gr_curve'})
_GRAPHIC_KINDS = frozenset({LayerKind.COPPER, LayerKind.MASK, LayerKind.PASTE, LayerKind.OUTLINE})


# ---------- мелкие помощники ----------

def _xy(n: Node | None) -> Pt:
    """(at x y [угол]) / (start x y) → точка модели (ось Y вверх)."""
    return (float(n[1]), -float(n[2]) + 0.0)


def _local(n: Node) -> Pt:
    """Точка в системе компонента, ось Y вверх (ещё без поворота и сдвига)."""
    return (float(n[1]), -float(n[2]) + 0.0)


def _angle(n: Node | None) -> float:
    return float(n[3]) if n is not None and len(n) > 3 and not isinstance(n[3], str) else 0.0


def _yes(node: Node, name: str, default: bool = False) -> bool:
    """Флаг в любой из форм KiCad: «(name yes)», «(name no)», «(name)» или слово name среди аргументов."""
    if node.has(name):
        return True
    sub = node.get(name)
    if sub is None:
        return default
    if len(sub) == 1:
        return True
    return sub[1] in ('yes', 'true')


def _opt_bool(v) -> bool | None:
    if v in ('yes', 'true'):
        return True
    if v in ('no', 'false'):
        return False
    return None    # none / нет значения — как задано на уровне платы


def _sides(node: Node | None) -> dict[str, bool | None]:
    """(tenting front back) | (tenting (front yes) (back none)) → {'front': …, 'back': …}."""
    if node is None:
        return {}
    out = {}
    for s in ('front', 'back'):
        if node.has(s):
            out[s] = True
        elif node.get(s) is not None:
            out[s] = _opt_bool(node.val(s))
    if not out and len(node) > 1 and node[1] in ('none', 'no'):
        out = {'front': False, 'back': False} if node[1] == 'no' else {}
    return out


def _num(v, default=None):
    return float(v) if isinstance(v, (int, float)) else default


class _F:
    """Узел с быстрым доступом к полям: словарь «имя → первый вложенный узел» строится за один
    проход, тогда как Node.get каждый раз просматривает все дочерние узлы. У площадки около
    25 обращений к полям, поэтому для площадок, отверстий и зон это в разы быстрее.
    Остальное (индексы, срезы, has, all) передаётся исходному узлу."""
    __slots__ = ('n', 'd')

    def __init__(self, n: Node):
        self.n = n
        d = {}
        for x in n:
            if type(x) is Node and x and x[0] not in d:
                d[x[0]] = x
        self.d = d

    def __getitem__(self, i):
        return self.n[i]

    def __len__(self):
        return len(self.n)

    def __iter__(self):
        return iter(self.n)

    def get(self, tag, default=None):
        return self.d.get(tag, default)

    def val(self, tag, default=None):
        x = self.d.get(tag)
        return x[1] if x is not None and len(x) > 1 else default

    def vals(self, tag):
        x = self.d.get(tag)
        return x[1:] if x is not None else []

    def has(self, word):
        return self.n.has(word)

    def all(self, tag):
        return self.n.all(tag)

    def children(self):
        return self.n.children()


_TAG = itemgetter(0)
_XY = itemgetter(1, 2)


def _pts(node: Node, f=None) -> Contour:
    """(pts (xy …) (arc (start) (mid) (end)) …) → Contour. f — перевод точки (по умолчанию только y → −y)."""
    items = node[1:]
    if set(map(_TAG, items)) <= {'xy'}:     # быстрый путь (без дуг): выборка значений на скорости C
        a = np.array(list(map(_XY, items)), dtype=float).reshape(-1, 2)
        if f is None:
            a[:, 1] *= -1
            a[:, 1] += 0.0         # убрать «−0.0»
        else:
            a = np.array([f((x, -y + 0.0)) for x, y in a], dtype=float).reshape(-1, 2)
        return Contour(a)
    f = f or (lambda p: p)
    pts, arcs = [], {}
    for i in items:
        if type(i) is not Node:
            continue
        if i[0] == 'xy':
            pts.append(f(_local(i)))
        elif i[0] == 'arc':
            s, m, e = (f(_local(i.get(k))) for k in ('start', 'mid', 'end'))
            if not pts or pts[-1] != s:
                pts.append(s)
            arcs[len(pts) - 1] = m
            pts.append(e)
    if len(pts) > 1 and pts[-1] == pts[0]:     # контур замкнут явно — убрать повтор
        pts.pop()
        arcs = {k: v for k, v in arcs.items() if k < len(pts)}
    return Contour(np.array(pts, dtype=float).reshape(-1, 2), arcs)


# ---------- контекст чтения ----------

class _Reader:
    def __init__(self, tree: Node, warnings: list[str]):
        self.t = tree
        self.w = warnings
        self.layer_names: list[str] = []
        self.layer_types: dict[str, str] = {}
        self.user_names: dict[str, str] = {}
        for e in (tree.get('layers') or [])[1:]:
            if type(e) is Node and len(e) >= 3:
                name = str(e[1])
                self.layer_names.append(name)
                self.layer_types[name] = str(e[2])
                if len(e) > 3 and isinstance(e[3], str):
                    self.user_names[name] = str(e[3])
        self.copper = copper_order(self.layer_names)
        self.net_table = {}          # KiCad ≤ 9: номер → имя
        for n in tree.all('net'):
            if len(n) > 2:
                self.net_table[n[1]] = str(n[2])
        self.nets: set[str] = {v for v in self.net_table.values() if v}
        self.unknown_layers: set[str] = set()
        self.texts: list[Text] = []

    def wanted(self, n: Node) -> bool:
        """Нужна ли графика с этого слоя: медь, маска, паста и контур платы.
        Шелкография, сборочные слои и контуры корпусов в DFM-анализе не участвуют, а на больших
        платах их сотни тысяч линий (Jetson — 330 тыс.), поэтому они не читаются."""
        lay = n.val('layer')
        return lay is not None and kind_of(str(lay)) in _GRAPHIC_KINDS

    # цепь объекта в любой из форм
    def net(self, node: Node) -> str | None:
        n = node.get('net')
        name = None
        if n is not None and len(n) > 1:
            if isinstance(n[-1], str) and type(n[-1]) is not Sym:
                name = str(n[-1])
            elif isinstance(n[1], int):
                name = self.net_table.get(n[1])
                if name is None:
                    self.w.append(f'ссылка на неизвестную цепь номер {n[1]}')
        nn = node.val('net_name')
        if nn is not None:
            name = str(nn)
        if name:
            self.nets.add(name)
            return name
        return None

    def layers(self, names) -> tuple[str, ...]:
        out = expand([str(x) for x in names], self.layer_names)
        for x in out:
            if x not in self.layer_names and x not in ('Inner',):
                self.unknown_layers.add(x)
        return tuple(out)

    # ---------- графика ----------
    def graphic(self, n: Node, f, footprint: str | None = None) -> Graphic | None:
        n = _F(n)
        kind = str(n[0]).split('_', 1)[1]     # gr_line → line, fp_poly → poly
        layer = n.val('layer')
        stroke = n.get('stroke')
        width = _num(stroke.val('width') if stroke is not None else n.val('width'), 0.0)
        fill = n.get('fill')
        filled = fill is not None and (any(x in ('yes', 'solid') for x in fill[1:])
                                       or fill.val('type') in ('solid', 'yes'))
        uid = n.val('uuid')
        g = Graphic(kind, str(layer) if layer is not None else '', width, filled=filled,
                    net=self.net(n), footprint=footprint, uid=uid)
        if kind in ('line',):
            g.points = (f(_local(n.get('start'))), f(_local(n.get('end'))))
        elif kind == 'arc':
            g.points = tuple(f(_local(n.get(k))) for k in ('start', 'mid', 'end'))
        elif kind == 'circle':
            g.points = (f(_local(n.get('center'))), f(_local(n.get('end'))))
        elif kind == 'rect':
            # прямоугольник компонента после поворота уже не параллелен осям — храним как многоугольник
            (x0, y0), (x1, y1) = _local(n.get('start')), _local(n.get('end'))
            g.kind = 'poly'
            radius = _num(n.val('radius'), 0.0)
            if radius > 0:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                g.contour = box_contour(abs(x1 - x0), abs(y1 - y0), radius).mapped(
                    lambda p: f((cx + p[0], cy + p[1])))
            else:
                g.contour = Contour(np.array([f(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]))
        elif kind == 'poly':
            g.contour = _pts(n.get('pts'), f)
        elif kind in ('curve', 'bezier'):
            g.kind = 'curve'
            g.points = tuple(f(_local(x)) for x in n.get('pts')[1:] if type(x) is Node)
        else:
            return None
        return g

    def text(self, n: Node, f, footprint=None) -> Text | None:
        lay = n.get('layer')
        if lay is None or kind_of(str(lay[1])) not in (LayerKind.COPPER, LayerKind.MASK):
            return None
        if _yes(n, 'hide') or (n.get('effects') is not None and _yes(n.get('effects'), 'hide')):
            return None
        if n[0] == 'property':
            s = str(n[2]) if len(n) > 2 else ''
        elif n[0] == 'fp_text':
            s = str(n[2]) if len(n) > 2 else ''
        else:
            s = str(n[1]) if len(n) > 1 else ''
        at = n.get('at')
        pos = f(_local(at)) if at is not None else (0.0, 0.0)
        eff = n.get('effects')
        font = eff.get('font') if eff is not None else None
        size = thickness = None
        bold = False
        if font is not None:
            sz = font.vals('size')
            if sz:
                size = (float(sz[0]), float(sz[1] if len(sz) > 1 else sz[0]))
            thickness = _num(font.val('thickness'))
            bold = _yes(font, 'bold')
        return Text(s, str(lay[1]), pos, footprint, knockout=lay.has('knockout'), uid=n.val('uuid'),
                    size=size, thickness=thickness, bold=bold)

    # ---------- площадки ----------
    def pad_shape(self, n: Node, base: PadShape | None = None) -> PadShape:
        """Форма площадки из узла pad или из (layer …) падстека.

        Слой падстека от основной формы НИЧЕГО не наследует: без формы — круг, без размера —
        нулевой размер (площадки на слое нет), без доли скругления — 0, без смещения — 0. Так
        делает KiCad 10.0; проверено по Gerber на тестовой плате t04_padstacks (в шаге 1.2
        предполагалось наследование — это было неверно). Из base берутся только настройки
        подключения к зонам и опорная форма произвольной площадки."""
        if not isinstance(n, _F):
            n = _F(n)
        b = base
        shape = n[3] if n[0] == 'pad' else n.val('shape', 'circle')
        size = n.vals('size')
        size = (float(size[0]), float(size[1] if len(size) > 1 else size[0])) if size else (0.0, 0.0)
        offset = (0.0, 0.0)
        drill = n.get('drill')
        off = (drill.get('offset') if drill is not None else None) or n.get('offset')
        if off is not None:
            offset = _local(off)
        ch = n.get('chamfer')
        corners = frozenset(str(x) for x in ch[1:]) if ch is not None else frozenset()
        rd = n.vals('rect_delta')
        opts = n.get('options')
        prims = n.get('primitives')
        ps = PadShape(
            shape=str(shape),
            size=size,
            offset=offset,
            roundrect_rratio=_num(n.val('roundrect_rratio'), 0.0),
            chamfer_ratio=_num(n.val('chamfer_ratio'), 0.0),
            chamfer=corners,
            rect_delta=(float(rd[0]), float(rd[1])) if rd else (0.0, 0.0),
            anchor=str(opts.val('anchor', 'circle')) if opts is not None else (b.anchor if b else 'circle'),
            thermal_bridge_width=_num(n.val('thermal_bridge_width'), b.thermal_bridge_width if b else None),
            thermal_gap=_num(n.val('thermal_gap'), b.thermal_gap if b else None),
            thermal_bridge_angle=_num(n.val('thermal_bridge_angle'), b.thermal_bridge_angle if b else None),
            zone_connect=ZoneConnection(n.val('zone_connect')) if n.val('zone_connect') is not None
            else (b.zone_connect if b else ZoneConnection.INHERITED),
        )
        if prims is not None:
            # примитивы произвольной площадки — в системе площадки, только ось Y вверх
            ps.primitives = [g for g in (self.graphic(p, lambda p: p) for p in prims.children()) if g]
        return ps

    def pad(self, n: Node, fp: Footprint, fpos: Pt, fangle: float) -> Pad:
        n = _F(n)
        at = n.get('at')
        local = _local(at)
        shape = self.pad_shape(n)
        d = n.get('drill')
        drill = None
        if d is not None:
            nums = [float(x) for x in d[1:] if isinstance(x, (int, float))]
            if nums and nums[0] > 0:
                drill = Drill((nums[0], nums[1] if len(nums) > 1 else nums[0]), oval=d.has('oval'))
        p = Pad(
            number=str(n[1]),
            type=PadType(str(n[2])),
            pos=transform(local, fpos, fangle),
            angle=_angle(at),
            shape=shape,
            layers=self.layers(n.vals('layers')),
            net=self.net(n),
            footprint=fp.ref,
            drill=drill,
            local_pos=(local[0], -local[1] + 0.0),
            solder_mask_margin=_num(n.val('solder_mask_margin')),
            solder_paste_margin=_num(n.val('solder_paste_margin')),
            solder_paste_margin_ratio=_num(n.val('solder_paste_margin_ratio')),
            clearance=_num(n.val('clearance')),
            remove_unused_layers=_yes(n, 'remove_unused_layers'),
            keep_end_layers=_yes(n, 'keep_end_layers'),
            zone_layer_connections=tuple(str(x) for x in n.vals('zone_layer_connections')),
            tenting=_sides(n.get('tenting')),
            pintype=n.val('pintype'),
            pinfunction=n.val('pinfunction'),
            uid=n.val('uuid'),
        )
        ps = n.get('padstack')
        if ps is not None:
            p.padstack_mode = str(ps.val('mode', 'normal'))
            for lay in ps.all('layer'):
                p.layer_shapes[str(lay[1])] = self.pad_shape(lay, shape)
        return p

    # ---------- компонент ----------
    def footprint(self, n: Node) -> Footprint:
        n = _F(n)
        at = n.get('at')
        fpos, fangle = _xy(at), _angle(at)
        f = lambda p: transform(p, fpos, fangle)
        ref = value = ''
        for prop in n.all('property'):
            if prop[1] == 'Reference':
                ref = str(prop[2])
            elif prop[1] == 'Value':
                value = str(prop[2])
        for t in n.all('fp_text'):              # формат до KiCad 8
            if t[1] == 'reference' and not ref:
                ref = str(t[2])
            elif t[1] == 'value' and not value:
                value = str(t[2])
        attr = n.get('attr')
        fp = Footprint(
            ref=ref, value=value, lib_id=str(n[1]) if len(n) > 1 and isinstance(n[1], str) else '',
            pos=fpos, angle=fangle,
            side='bottom' if n.val('layer') == 'B.Cu' else 'top',
            attrs=frozenset(str(x) for x in attr[1:]) if attr is not None else frozenset(),
            solder_mask_margin=_num(n.val('solder_mask_margin')),
            solder_paste_margin=_num(n.val('solder_paste_margin')),
            clearance=_num(n.val('clearance')),
            zone_connect=ZoneConnection(n.val('zone_connect')) if n.val('zone_connect') is not None
            else ZoneConnection.INHERITED,
            uid=n.val('uuid'),
            locked=_yes(n, 'locked'),
        )
        for c in n[1:]:
            if type(c) is not Node:
                continue
            tag = c[0]
            if tag == 'pad':
                fp.pads.append(self.pad(c, fp, fpos, fangle))
            elif tag in _FP_GRAPHICS and self.wanted(c):
                g = self.graphic(c, f, ref)
                if g:
                    fp.graphics.append(g)
            elif tag == 'zone':
                fp.zones.append(self.zone(c, ref))
            elif tag in ('fp_text', 'property'):
                t = self.text(c, f, ref)
                if t:
                    self.texts.append(t)
        return fp

    # ---------- трассировка ----------
    def track(self, n: Node) -> Track:
        n = _F(n)
        return Track(_xy(n.get('start')), _xy(n.get('end')), float(n.val('width')), str(n.val('layer')),
                     self.net(n), n.val('uuid'), _yes(n, 'locked'))

    def arc(self, n: Node) -> ArcTrack:
        n = _F(n)
        return ArcTrack(_xy(n.get('start')), _xy(n.get('mid')), _xy(n.get('end')), float(n.val('width')),
                        str(n.val('layer')), self.net(n), n.val('uuid'), _yes(n, 'locked'))

    def via(self, n: Node) -> Via:
        n = _F(n)
        vt = ViaType.THROUGH
        for word in ('blind', 'micro', 'buried'):
            if n.has(word):
                vt = ViaType.BLIND if word == 'buried' else ViaType(word)
        lay = [str(x) for x in n.vals('layers')]
        lay = [x for x in self.copper if x in lay] or lay      # сверху вниз
        span = (lay[0], lay[-1]) if lay else ('F.Cu', 'B.Cu')
        v = Via(
            pos=_xy(n.get('at')), size=float(n.val('size')), drill=float(n.val('drill')),
            layers=span, via_type=vt, net=self.net(n), uid=n.val('uuid'),
            remove_unused_layers=_yes(n, 'remove_unused_layers'),
            keep_end_layers=_yes(n, 'keep_end_layers'),
            zone_layer_connections=tuple(str(x) for x in n.vals('zone_layer_connections')),
            tenting=_sides(n.get('tenting')), covering=_sides(n.get('covering')),
            plugging=_sides(n.get('plugging')),
            filling=_opt_bool(n.val('filling')), capping=_opt_bool(n.val('capping')),
            free=_yes(n, 'free'), locked=_yes(n, 'locked'),
        )
        ps = n.get('padstack')
        if ps is not None:
            v.padstack_mode = str(ps.val('mode', 'normal'))
            for lay in ps.all('layer'):
                if lay.get('size') is not None:
                    v.sizes[str(lay[1])] = float(lay.vals('size')[0])
        return v

    def zone(self, n: Node, footprint: str | None = None) -> Zone:
        n = _F(n)
        lays = n.vals('layers') or n.vals('layer')
        cp = n.get('connect_pads')
        conn = ZoneConnection.THERMAL
        if cp is not None:
            if cp.has('yes'):
                conn = ZoneConnection.FULL
            elif cp.has('no'):
                conn = ZoneConnection.NONE
            elif cp.has('thru_hole_only'):
                conn = ZoneConnection.THT_THERMAL
        fill = n.get('fill')
        ko = n.get('keepout')
        attr = n.get('attr')
        z = Zone(
            net=self.net(n),
            layers=self.layers(lays),
            outline=[_pts(p.get('pts')) for p in n.all('polygon')],
            name=str(n.val('name', '')),
            priority=int(n.val('priority', 0)),
            clearance=_num(cp.val('clearance')) if cp is not None else None,
            min_thickness=_num(n.val('min_thickness')),
            connect_pads=conn,
            thermal_gap=_num(fill.val('thermal_gap')) if fill is not None else None,
            thermal_bridge_width=_num(fill.val('thermal_bridge_width')) if fill is not None else None,
            fill_mode=str(fill.val('mode', 'solid')) if fill is not None else 'solid',
            island_removal_mode=int(fill.val('island_removal_mode', 0)) if fill is not None else 0,
            island_area_min=_num(fill.val('island_area_min')) if fill is not None else None,
            keepout={str(k[0]): k[1] == 'not_allowed' for k in ko.children()} if ko is not None else None,
            teardrop=attr is not None and attr.get('teardrop') is not None,
            footprint=footprint,
            uid=n.val('uuid'),
        )
        for fpoly in n.all('filled_polygon'):
            layer = str(fpoly.val('layer', z.layers[0] if z.layers else ''))
            z.filled.setdefault(layer, []).append(_pts(fpoly.get('pts')))
        return z

    # ---------- стек ----------
    def stackup(self, thickness: float | None) -> Stackup | None:
        setup = self.t.get('setup')
        st = setup.get('stackup') if setup is not None else None
        if st is None:
            return _default_stackup(self.copper, thickness if thickness else 1.6)   # 1,6 мм — по умолчанию KiCad
        items = []
        for lay in st.all('layer'):
            name = str(lay[1])
            typ = str(lay.val('type', ''))
            low = typ.lower()
            if low == 'copper':
                kind = StackKind.COPPER
            elif 'mask' in low:
                kind = StackKind.MASK
            elif 'paste' in low:
                kind = StackKind.PASTE
            elif 'silk' in low:
                kind = StackKind.SILK
            else:
                kind = StackKind.DIELECTRIC
            # диэлектрик может состоять из нескольких подслоёв: «… addsublayer (thickness …) …»
            subs, cur = [], {}
            for x in lay[2:]:
                if type(x) is Sym and x == 'addsublayer':
                    subs.append(cur)
                    cur = {}
                elif type(x) is Node and x[0] in ('thickness', 'material', 'epsilon_r', 'loss_tangent', 'color'):
                    cur[str(x[0])] = x[1]
            subs.append(cur)
            for k, s in enumerate(subs):
                items.append(StackItem(
                    name=name if len(subs) == 1 else f'{name} ({k + 1})',
                    kind=kind,
                    thickness=_num(s.get('thickness')),
                    material=str(s['material']) if 'material' in s else None,
                    epsilon_r=_num(s.get('epsilon_r')),
                    loss_tangent=_num(s.get('loss_tangent')),
                    dielectric_type=typ if kind is StackKind.DIELECTRIC else None,
                    color=str(s['color']) if 'color' in s else None,
                ))
        return Stackup(items, finish=str(st.val("copper_finish")) if st.val("copper_finish") not in (None, "None", "") else None)


def _default_stackup(copper: list[str], thickness: float) -> Stackup:
    """Стек, который KiCad строит сам, если в файле его нет (так же он попадает в .gbrjob):
    маска 0,01 мм, медь 0,035 мм, FR4 (εr 4,5, tg δ 0,02), а диэлектрик делит остаток толщины
    поровну: (толщина − медь·n − 2·маска)/(n − 1). Для 2 слоёв и 1,6 мм это 1,51 мм — сверено
    с job-файлом KiCad 10 (демо multichannel_mixer). Тип диэлектрика (core/prepreg) KiCad
    в job-файл не пишет; здесь, как в KiCad, нечётные — core, чётные — prepreg."""
    cu, mask = 0.035, 0.01
    n = len(copper)
    diel = (thickness - cu * n - 2 * mask) / (n - 1) if n > 1 else None
    items = [StackItem('F.SilkS', StackKind.SILK), StackItem('F.Paste', StackKind.PASTE),
             StackItem('F.Mask', StackKind.MASK, mask)]
    for i, name in enumerate(copper):
        items.append(StackItem(name, StackKind.COPPER, cu))
        if i < n - 1:
            items.append(StackItem(f'dielectric {i + 1}', StackKind.DIELECTRIC, round(diel, 6), 'FR4', 4.5, 0.02,
                                   'core' if (i + 1) % 2 else 'prepreg'))
    items += [StackItem('B.Mask', StackKind.MASK, mask), StackItem('B.Paste', StackKind.PASTE),
              StackItem('B.SilkS', StackKind.SILK)]
    return Stackup(items, synthetic=True)


# ---------- правила проекта ----------

_RULE_KEYS = ('min_clearance', 'min_track_width', 'min_connection', 'min_via_diameter', 'min_via_annular_width',
              'min_through_hole_diameter', 'min_hole_to_hole', 'min_hole_clearance', 'min_copper_edge_clearance',
              'min_microvia_diameter', 'min_microvia_drill', 'min_groove_width', 'solder_mask_to_copper_clearance',
              'max_error', 'allow_blind_buried_vias', 'allow_microvias')


def _read_rules(tree: Node, pcb_path: Path, warnings: list[str]) -> DesignRules:
    r = DesignRules()
    setup = tree.get('setup')
    if setup is not None:
        for k in ('pad_to_mask_clearance', 'solder_mask_min_width', 'pad_to_paste_clearance',
                  'pad_to_paste_clearance_ratio'):
            if setup.val(k) is not None:
                setattr(r, k, float(setup.val(k)))
        r.allow_soldermask_bridges_in_footprints = _yes(setup, 'allow_soldermask_bridges_in_footprints')
        t = _sides(setup.get('tenting'))
        if setup.get('tenting') is not None:
            r.tenting = {s: bool(t.get(s)) for s in ('front', 'back')}
    pro = pcb_path.with_suffix('.kicad_pro')
    r.has_custom_rules = pcb_path.with_suffix('.kicad_dru').exists()
    if r.has_custom_rules:
        warnings.append('есть пользовательские правила .kicad_dru — они не учитываются')
    if not pro.exists():
        warnings.append('нет файла проекта .kicad_pro — правила и классы цепей по умолчанию')
        r.netclasses['Default'] = NetClass('Default')
        return r
    r.has_project_file = True
    d = json.loads(pro.read_text(encoding='utf-8'))
    ds = d.get('board', {}).get('design_settings', {})
    for k in _RULE_KEYS:
        v = ds.get('rules', {}).get(k)
        if v is not None:
            setattr(r, k, bool(v) if isinstance(getattr(r, k), bool) else float(v))
    zmin = ds.get('defaults', {}).get('zones', {}).get('min_clearance')
    r.zone_min_clearance = float(zmin) if zmin is not None else None
    ns = d.get('net_settings', {})
    for c in ns.get('classes', []):
        nc = NetClass(c['name'], int(c.get('priority', 2 ** 31 - 1)))
        for f in NetClass.FIELDS:
            if c.get(f) is not None:
                setattr(nc, f, float(c[f]))
        r.netclasses[nc.name] = nc
    if 'Default' not in r.netclasses:
        r.netclasses['Default'] = NetClass('Default')
    r.netclass_patterns = [(p['pattern'], p['netclass']) for p in (ns.get('netclass_patterns') or [])]
    for net, cls in (ns.get('netclass_assignments') or {}).items():
        r.netclass_assignments[net] = [cls] if isinstance(cls, str) else list(cls)
    return r


# ---------- главная функция ----------

def load_kicad(path) -> Board:
    """Читает .kicad_pcb (и .kicad_pro рядом) в модель платы."""
    path = Path(path)
    warnings: list[str] = []
    tree = sexpr.load(path, warnings)
    if tree.tag != 'kicad_pcb':
        raise ValueError(f'{path.name}: это не файл платы KiCad (корневой узел {tree.tag!r})')
    version = int(tree.val('version', 0))
    if version < TESTED_VERSIONS[0]:
        warnings.append(f'формат {version} старше проверенных ({TESTED_VERSIONS[0]}+); возможны ошибки чтения')
    elif version > TESTED_VERSIONS[1]:
        warnings.append(f'формат {version} новее проверенных (до {TESTED_VERSIONS[1]})')

    # сборщик мусора выключен по той же причине, что и при разборе (см. sexpr.parse)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        return _build(tree, path, version, warnings)
    finally:
        if gc_was_enabled:
            gc.enable()


def _build(tree: Node, path: Path, version: int, warnings: list[str]) -> Board:
    rd = _Reader(tree, warnings)
    design = Design(version=version, generator_version=str(tree.val('generator_version', '')),
                    layer_types={k: v for k, v in rd.layer_types.items() if k in rd.copper},
                    layer_user_names=rd.user_names)
    same = lambda p: p          # точки уже переведены в систему модели в _local
    for n in tree[1:]:
        if type(n) is not Node:
            continue
        tag = n[0]
        if tag == 'segment':
            design.tracks.append(rd.track(n))
        elif tag == 'arc':
            design.arcs.append(rd.arc(n))
        elif tag == 'via':
            design.vias.append(rd.via(n))
        elif tag == 'footprint':
            design.footprints.append(rd.footprint(n))
        elif tag == 'zone':
            design.zones.append(rd.zone(n))
        elif tag in _GR_GRAPHICS and rd.wanted(n):
            g = rd.graphic(n, same)
            if g:
                design.graphics.append(g)
        elif tag == 'gr_text':
            t = rd.text(n, same)
            if t:
                rd.texts.append(t)
        elif tag in ('module', 'fp_text_box', 'gr_text_box', 'gr_bbox'):
            warnings.append(f'объект {tag} не поддерживается и пропущен')
    design.texts = rd.texts
    design.rules = _read_rules(tree, path, warnings)
    if rd.unknown_layers:
        warnings.append('ссылки на слои, которых нет в таблице слоёв: ' + ', '.join(sorted(rd.unknown_layers)))

    general = tree.get('general')
    thickness = _num(general.val('thickness')) if general is not None else None
    board = Board(name=path.stem, source='kicad', thickness=thickness, design=design, warnings=warnings)
    for name in rd.copper + [n for n in rd.layer_names if n not in rd.copper]:
        board.add_layer(name)
    board.stackup = rd.stackup(thickness)
    board.nets = rd.nets
    build_physical(board)
    return board
