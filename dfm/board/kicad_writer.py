"""Запись платы KiCad 10 (.kicad_pcb + .kicad_pro) из простого описания на Python.

Нужна для своих тестовых плат (шаг 1.6), демо-плат из документов, плат с заложенными
нарушениями для проверки правил (пункт 2) и генератора данных нейросети (пункт 4).

    w = KicadWriter('t1', copper=4, clearance=0.2)
    w.outline_rect(0, 0, 40, 30)
    fp = w.footprint('U1', (20, 15), angle=37)
    fp.pad('1', 'smd', 'roundrect', (-2, 0), (1, 1.5), net='A', rratio=0.25)
    w.track([(0, 0), (5, 0)], 0.25, 'F.Cu', 'A')
    w.write('boards/test/t1.kicad_pcb')

Координаты — как во всей модели: мм, ось Y вверх, углы против часовой стрелки. В файл они
пишутся в системе KiCad (y → −y). Координаты внутри компонента — относительно компонента и без
его поворота; угол площадки задаётся относительно компонента (в файл пишется полный угол).
uuid детерминированные (от имени платы и номера объекта): повторная генерация даёт тот же файл.
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

__all__ = ['KicadWriter', 'Footprint']

_NS = uuid.UUID('6f1d3a2e-4b7c-4e1a-9a55-1c2d3e4f5a6b')
_INNER_IDS = {f'In{k}.Cu': 2 + 2 * k for k in range(1, 31)}     # номера слоёв KiCad 9+


def _n(v: float) -> str:
    """Число без экспоненты и лишних нулей (как пишет KiCad)."""
    if v == 0:
        return '0'
    s = f'{v:.6f}'.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def _q(s: str) -> str:
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') + '"'


def _xy(p, tag='xy') -> str:
    return f'({tag} {_n(p[0])} {_n(-p[1])})'


def _yn(v: bool) -> str:
    return 'yes' if v else 'no'


class Footprint:
    """Компонент: площадки и графика в системе компонента (ось Y вверх)."""

    def __init__(self, w: KicadWriter, ref: str, pos, angle: float, side: str, mask_margin, clearance, attrs):
        self.w, self.ref, self.pos, self.angle, self.side = w, ref, pos, angle, side
        self.mask_margin, self.clearance, self.attrs = mask_margin, clearance, attrs
        self.items: list[str] = []

    def _side(self, layer: str) -> str:
        """Слой «F.*» для компонента на нижней стороне становится «B.*»."""
        if self.side == 'bottom' and layer.startswith('F.'):
            return 'B.' + layer[2:]
        return layer

    def pad(self, number, kind: str, shape: str, pos, size, *, net=None, angle: float = 0.0, layers=None,
            drill=None, offset=None, rratio=None, chamfer=None, rect_delta=None, mask_margin=None,
            clearance=None, padstack=None, remove_unused=None, keep_end=True, zone_layer_connections=(),
            primitives=None, anchor='rect', zone_connect=None):
        """Площадка. kind: smd | thru_hole | np_thru_hole | connect; shape: circle | rect | oval |
        roundrect | trapezoid | custom. drill — диаметр или (ширина, высота) паза. chamfer —
        (доля, углы); padstack — ('front_inner_back' | 'custom', {слой: dict(shape=…, size=…, …)});
        primitives — для custom: [('poly', точки, толщина, заливка), ('circle', центр, r, толщина,
        заливка), ('line', a, b, толщина), ('arc', начало, середина, конец, толщина)]."""
        if layers is None:
            layers = {'smd': ['F.Cu', 'F.Mask', 'F.Paste'], 'connect': ['F.Cu', 'F.Mask'],
                      'thru_hole': ['*.Cu', '*.Mask'], 'np_thru_hole': ['*.Cu', '*.Mask']}[kind]
        layers = [self._side(x) for x in layers]
        sx, sy = (size, size) if isinstance(size, (int, float)) else size
        parts = [f'(pad {_q(number)} {kind} {shape}',
                 f'(at {_n(pos[0])} {_n(-pos[1])} {_n(self.angle + angle)})', f'(size {_n(sx)} {_n(sy)})']
        if drill is not None or offset is not None:
            d = ''
            if isinstance(drill, (tuple, list)):
                d = f' oval {_n(drill[0])} {_n(drill[1])}'
            elif drill is not None:
                d = f' {_n(drill)}'
            o = f' (offset {_n(offset[0])} {_n(-offset[1])})' if offset is not None else ''
            parts.append(f'(drill{d}{o})')
        parts.append('(layers ' + ' '.join(_q(x) for x in layers) + ')')
        if remove_unused is not None:
            parts.append(f'(remove_unused_layers {_yn(remove_unused)})')
            if remove_unused:
                parts.append(f'(keep_end_layers {_yn(keep_end)})')
        if zone_layer_connections:
            parts.append('(zone_layer_connections ' + ' '.join(_q(x) for x in zone_layer_connections) + ')')
        if rratio is not None:
            parts.append(f'(roundrect_rratio {_n(rratio)})')
        if chamfer is not None:
            parts.append(f'(chamfer_ratio {_n(chamfer[0])}) (chamfer {" ".join(chamfer[1])})')
        if rect_delta is not None:
            parts.append(f'(rect_delta {_n(rect_delta[0])} {_n(rect_delta[1])})')
        if mask_margin is not None:
            parts.append(f'(solder_mask_margin {_n(mask_margin)})')
        if clearance is not None:
            parts.append(f'(clearance {_n(clearance)})')
        if zone_connect is not None:
            parts.append(f'(zone_connect {zone_connect})')
        if net is not None:
            parts.append(f'(net {_q(net)})')
            self.w.nets.add(net)
        if shape == 'custom':
            parts.append(f'(options (clearance outline) (anchor {anchor}))')
            parts.append('(primitives ' + ' '.join(_primitive(p) for p in primitives or []) + ')')
        if padstack is not None:
            mode, per_layer = padstack
            ls = []
            for lay, d in per_layer.items():
                body = ''
                if 'shape' in d:
                    body += f' (shape {d["shape"]})'
                if 'size' in d:
                    s = d['size']
                    s = (s, s) if isinstance(s, (int, float)) else s
                    body += f' (size {_n(s[0])} {_n(s[1])})'
                if 'rratio' in d:
                    body += f' (roundrect_rratio {_n(d["rratio"])})'
                if 'offset' in d:
                    body += f' (offset {_n(d["offset"][0])} {_n(-d["offset"][1])})'
                ls.append(f'(layer {_q(lay)}{body})')
            parts.append(f'(padstack (mode {mode}) ' + ' '.join(ls) + ')')
        parts.append(self.w.uid())
        self.items.append(' '.join(parts) + ')')

    def graphic(self, kind: str, layer: str, *points, width: float = 0.1, fill: bool = False, radius=None):
        """Графика компонента: line (a, b), arc (начало, середина, конец), circle (центр, точка),
        rect (угол, угол), poly (точки…)."""
        self.items.append(_graphic('fp_' + kind, self._side(layer), points, width, fill, None, radius, self.w.uid()))

    def zone(self, net, layers, outline, **kw):
        """Зона внутри компонента (координаты — абсолютные, как KiCad хранит контуры зон)."""
        self.items.append(self.w._zone(net, layers, outline, **kw))

    def text(self):
        x, y = self.pos
        a = _n(self.angle)
        silk = self._side('F.SilkS')
        return (f'(footprint "dfm:{self.ref}" (layer {_q(self._side("F.Cu"))}) {self.w.uid()} '
                f'(at {_n(x)} {_n(-y)} {a}) '
                f'(property "Reference" {_q(self.ref)} (at 0 0 {a}) (layer {_q(silk)}) (hide yes) {self.w.uid()} '
                f'(effects (font (size 1 1) (thickness 0.15)))) '
                f'(property "Value" {_q(self.ref)} (at 0 0 {a}) (layer {_q(self._side("F.Fab"))}) (hide yes) '
                f'{self.w.uid()} (effects (font (size 1 1) (thickness 0.15)))) '
                + (f'(solder_mask_margin {_n(self.mask_margin)}) ' if self.mask_margin is not None else '')
                + (f'(clearance {_n(self.clearance)}) ' if self.clearance is not None else '')
                + f'(attr {self.attrs}) ' + ' '.join(self.items) + ')')


def _primitive(p) -> str:
    kind = p[0]
    if kind == 'poly':
        _, pts, width, fill = p
        return f'(gr_poly (pts {" ".join(_xy(q) for q in pts)}) (width {_n(width)}) (fill {_yn(fill)}))'
    if kind == 'circle':
        _, c, r, width, fill = p
        return f'(gr_circle {_xy(c, "center")} {_xy((c[0] + r, c[1]), "end")} (width {_n(width)}) (fill {_yn(fill)}))'
    if kind == 'line':
        _, a, b, width = p
        return f'(gr_line {_xy(a, "start")} {_xy(b, "end")} (width {_n(width)}))'
    if kind == 'arc':
        _, s, m, e, width = p
        return f'(gr_arc {_xy(s, "start")} {_xy(m, "mid")} {_xy(e, "end")} (width {_n(width)}))'
    raise ValueError(kind)


def _graphic(tag, layer, points, width, fill, net, radius, uid) -> str:
    kind = tag.split('_', 1)[1]
    if kind == 'line':
        geom = f'{_xy(points[0], "start")} {_xy(points[1], "end")}'
    elif kind == 'arc':
        geom = f'{_xy(points[0], "start")} {_xy(points[1], "mid")} {_xy(points[2], "end")}'
    elif kind == 'circle':
        geom = f'{_xy(points[0], "center")} {_xy(points[1], "end")}'
    elif kind == 'rect':
        geom = f'{_xy(points[0], "start")} {_xy(points[1], "end")}'
    elif kind == 'poly':
        geom = f'(pts {" ".join(_xy(p) for p in points)})'
    else:
        raise ValueError(kind)
    fill_s = f' (fill {_yn(fill)})' if kind in ('circle', 'rect', 'poly') else ''
    rad = f' (radius {_n(radius)})' if radius else ''
    net_s = f' (net {_q(net)})' if net is not None else ''
    return f'({tag} {geom} (stroke (width {_n(width)}) (type solid)){fill_s}{rad} (layer {_q(layer)}){net_s} {uid})'


class KicadWriter:
    def __init__(self, name: str, copper: int = 2, thickness: float = 1.6, *, clearance: float = 0.2,
                 track_width: float = 0.2, edge_clearance: float = 0.0, mask_margin: float = 0.0,
                 mask_min_width: float = 0.0, tenting=(True, True), rules: dict | None = None,
                 stackup: list | None = None):
        """stackup — [(слой, тип, толщина мм)] сверху вниз: тип copper | core | prepreg | Top Solder Mask …;
        без него KiCad строит стек сам (диэлектрик поровну)."""
        self.name = name
        self.stackup = stackup
        self.dru: list[str] = []          # правила .kicad_dru: '(rule "…" (condition "…") (constraint …))'
        self.netclasses: dict[str, dict] = {}   # класс цепей → {'clearance': …, 'pattern': 'S*'}
        self.copper = ['F.Cu'] + [f'In{k}.Cu' for k in range(1, copper - 1)] + ['B.Cu']
        self.thickness = thickness
        self.clearance, self.track_width, self.edge_clearance = clearance, track_width, edge_clearance
        self.mask_margin, self.mask_min_width, self.tenting = mask_margin, mask_min_width, tenting
        self.rules = rules or {}
        self.items: list[str] = []
        self.footprints: list[Footprint] = []
        self.nets: set[str] = set()
        self._uid = 0

    def uid(self) -> str:
        self._uid += 1
        return f'(uuid "{uuid.uuid5(_NS, f"{self.name}/{self._uid}")}")'

    def _net(self, net):
        if net is None:
            return ''
        self.nets.add(net)
        return f' (net {_q(net)})'

    # ----- трассировка -----
    def track(self, pts, width: float, layer: str = 'F.Cu', net=None):
        """Ломаная из отрезков дорожки."""
        for a, b in zip(pts, pts[1:]):
            self.items.append(f'(segment {_xy(a, "start")} {_xy(b, "end")} (width {_n(width)}) '
                              f'(layer {_q(layer)}){self._net(net)} {self.uid()})')

    def arc(self, start, mid, end, width: float, layer: str = 'F.Cu', net=None):
        self.items.append(f'(arc {_xy(start, "start")} {_xy(mid, "mid")} {_xy(end, "end")} (width {_n(width)}) '
                          f'(layer {_q(layer)}){self._net(net)} {self.uid()})')

    def via(self, pos, size: float, drill: float, *, layers=('F.Cu', 'B.Cu'), net=None, kind: str | None = None,
            remove_unused=None, keep_end=True, zone_layer_connections=(), padstack=None, tenting=None,
            teardrops=False):
        """Переходное отверстие. kind: None (сквозное) | blind | buried | micro; padstack —
        ('front_inner_back' | 'custom', {слой: диаметр}); tenting — (сверху, снизу) или None."""
        s = f'(via{" " + kind if kind else ""} {_xy(pos, "at")} (size {_n(size)}) (drill {_n(drill)}) ' \
            f'(layers {_q(layers[0])} {_q(layers[1])})'
        if remove_unused is not None:
            s += f' (remove_unused_layers {_yn(remove_unused)})'
            if remove_unused:
                s += f' (keep_end_layers {_yn(keep_end)})'
        if zone_layer_connections:
            s += ' (zone_layer_connections ' + ' '.join(_q(x) for x in zone_layer_connections) + ')'
        if tenting is not None:
            s += f' (tenting (front {_yn(tenting[0])}) (back {_yn(tenting[1])}))'
        if padstack is not None:
            mode, sizes = padstack
            s += f' (padstack (mode {mode}) ' + ' '.join(f'(layer {_q(k)} (size {_n(v)}))' for k, v in sizes.items()) + ')'
        if teardrops:                    # капли строит сам KiCad при перезаливке (параметры по умолчанию)
            s += (' (teardrops (best_length_ratio 0.5) (max_length 1) (best_width_ratio 1) (max_width 2) '
                  '(curved_edges no) (filter_ratio 0.9) (enabled yes) (allow_two_segments yes) '
                  '(prefer_zone_connections yes))')
        self.items.append(s + self._net(net) + f' {self.uid()})')

    # ----- компоненты, графика, зоны -----
    def footprint(self, ref: str, pos, angle: float = 0.0, side: str = 'top', *, mask_margin=None, clearance=None,
                  attrs: str = 'smd') -> Footprint:
        fp = Footprint(self, ref, pos, angle, side, mask_margin, clearance, attrs)
        self.footprints.append(fp)
        return fp

    def graphic(self, kind: str, layer: str, *points, width: float = 0.1, fill: bool = False, net=None, radius=None):
        """Графика платы: line, arc, circle, rect, poly (точки — как у Footprint.graphic)."""
        if net is not None:
            self.nets.add(net)
        self.items.append(_graphic('gr_' + kind, layer, points, width, fill, net, radius, self.uid()))

    def text(self, s: str, pos, layer: str = 'F.Cu', size: float = 1.0, thickness: float = 0.15):
        self.items.append(f'(gr_text {_q(s)} {_xy(pos, "at")} (layer {_q(layer)}) {self.uid()} '
                          f'(effects (font (size {_n(size)} {_n(size)}) (thickness {_n(thickness)}))))')

    def _zone(self, net, layers, outline, *, priority=0, clearance=0.2, min_thickness=0.2, thermal_gap=0.3,
              thermal_width=0.3, connect='thermal', hatch=None, filled=None, keepout=None, island_removal=0,
              name=None, teardrop=False) -> str:
        """Зона. connect: thermal | solid | none | thru_hole_only; hatch — (толщина, шаг) для
        сетчатой заливки; filled — {слой: [контуры]} готовой заливки (иначе KiCad зальёт сам при
        refill_zones); keepout — dict запретов (tracks, vias, pads, copperpour, footprints)."""
        lays = [layers] if isinstance(layers, str) else list(layers)
        cp = {'thermal': '', 'solid': ' yes', 'none': ' no', 'thru_hole_only': ' thru_hole_only'}[connect]
        s = f'(zone{self._net(net)} (layers {" ".join(_q(x) for x in lays)}) {self.uid()}'
        if name:
            s += f' (name {_q(name)})'
        s += f' (hatch edge 0.5) (priority {priority})'
        if teardrop:
            s += ' (attr (teardrop (type padvia)))'
        if keepout is not None:
            s += ' (keepout ' + ' '.join(f'({k} {"not_allowed" if v else "allowed"})' for k, v in keepout.items()) + ')'
        else:
            s += f' (connect_pads{cp} (clearance {_n(clearance)})) (min_thickness {_n(min_thickness)})'
            fill = f' (fill yes (thermal_gap {_n(thermal_gap)}) (thermal_bridge_width {_n(thermal_width)})'
            if hatch is not None:
                fill += f' (mode hatch) (hatch_thickness {_n(hatch[0])}) (hatch_gap {_n(hatch[1])})'
            fill += f' (island_removal_mode {island_removal}))'
            s += fill
        s += f' (polygon (pts {" ".join(_xy(p) for p in outline)}))'
        for lay, contours in (filled or {}).items():
            for c in contours:
                s += f' (filled_polygon (layer {_q(lay)}) (pts {" ".join(_xy(p) for p in c)}))'
        return s + ')'

    def zone(self, net, layers, outline, **kw):
        self.items.append(self._zone(net, layers, outline, **kw))

    # ----- контур платы -----
    def outline_rect(self, x0, y0, x1, y1, radius: float = 0.0, width: float = 0.05):
        if radius:
            self.graphic('rect', 'Edge.Cuts', (x0, y0), (x1, y1), width=width, radius=radius)
        else:
            self.outline_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], width=width)

    def outline_polygon(self, pts, width: float = 0.05):
        """Контур отдельными отрезками (как его рисуют вручную)."""
        for a, b in zip(pts, list(pts[1:]) + [pts[0]]):
            self.graphic('line', 'Edge.Cuts', a, b, width=width)

    def outline_circle(self, center, r: float, width: float = 0.05):
        self.graphic('circle', 'Edge.Cuts', center, (center[0] + r, center[1]), width=width)

    # ----- запись -----
    def _layers(self) -> str:
        cu = ['(0 "F.Cu" signal)'] + [f'({_INNER_IDS[x]} {_q(x)} signal)' for x in self.copper[1:-1]] + ['(2 "B.Cu" signal)']
        tech = ['(9 "F.Adhes" user "F.Adhesive")', '(11 "B.Adhes" user "B.Adhesive")', '(13 "F.Paste" user)',
                '(15 "B.Paste" user)', '(5 "F.SilkS" user "F.Silkscreen")', '(7 "B.SilkS" user "B.Silkscreen")',
                '(1 "F.Mask" user)', '(3 "B.Mask" user)', '(17 "Dwgs.User" user "User.Drawings")',
                '(19 "Cmts.User" user "User.Comments")', '(25 "Edge.Cuts" user)', '(27 "Margin" user)',
                '(31 "F.CrtYd" user "F.Courtyard")', '(29 "B.CrtYd" user "B.Courtyard")', '(35 "F.Fab" user)',
                '(33 "B.Fab" user)']
        return '(layers ' + ' '.join(cu + tech) + ')'

    def text_pcb(self) -> str:
        head = (f'(kicad_pcb (version 20260206) (generator "dfm") (generator_version "10.0") '
                f'(general (thickness {_n(self.thickness)}) (legacy_teardrops no)) (paper "A4") {self._layers()} '
                f'(setup (pad_to_mask_clearance {_n(self.mask_margin)}) (solder_mask_min_width {_n(self.mask_min_width)}) '
                f'(allow_soldermask_bridges_in_footprints no) '
                f'(tenting (front {_yn(self.tenting[0])}) (back {_yn(self.tenting[1])})){self._stackup()})')
        body = [fp.text() for fp in self.footprints] + self.items
        return head + '\n' + '\n'.join(body) + '\n)\n'

    def _stackup(self) -> str:
        if not self.stackup:
            return ''
        # KiCad принимает стек только полным (с шелкографией и пастой): иначе в job-файл Gerber
        # толщины слоёв не попадают
        ls = ['(layer "F.SilkS" (type "Top Silk Screen"))', '(layer "F.Paste" (type "Top Solder Paste"))']
        for name, typ, t in self.stackup:
            extra = ' (material "FR4") (epsilon_r 4.5) (loss_tangent 0.02)' if typ in ('core', 'prepreg') else ''
            ls.append(f'(layer {_q(name)} (type {_q(typ)}) (thickness {_n(t)}){extra})')
        ls += ['(layer "B.Paste" (type "Bottom Solder Paste"))', '(layer "B.SilkS" (type "Bottom Silk Screen"))']
        return ' (stackup ' + ' '.join(ls) + ' (copper_finish "None") (dielectric_constraints no))'

    def project(self) -> dict:
        rules = {'min_clearance': 0.0, 'min_track_width': 0.0, 'min_connection': 0.0, 'min_via_diameter': 0.0,
                 'min_via_annular_width': 0.0, 'min_through_hole_diameter': 0.0, 'min_hole_to_hole': 0.0,
                 'min_hole_clearance': 0.0, 'min_copper_edge_clearance': self.edge_clearance,
                 'min_microvia_diameter': 0.0, 'min_microvia_drill': 0.0, 'min_groove_width': 0.0,
                 'solder_mask_to_copper_clearance': 0.0, 'allow_blind_buried_vias': True, 'allow_microvias': True,
                 'max_error': 0.005}
        rules.update(self.rules)
        default = {'name': 'Default', 'priority': 2147483647, 'clearance': self.clearance,
                   'track_width': self.track_width, 'via_diameter': 0.6, 'via_drill': 0.3, 'microvia_diameter': 0.3,
                   'microvia_drill': 0.1, 'diff_pair_gap': 0.25, 'diff_pair_via_gap': 0.25, 'diff_pair_width': 0.2,
                   'wire_width': 6, 'bus_width': 12, 'line_style': 0, 'schematic_color': 'rgba(0, 0, 0, 0.000)',
                   'pcb_color': 'rgba(0, 0, 0, 0.000)'}
        classes, patterns = [default], []
        for k, (name, nc) in enumerate(self.netclasses.items()):
            classes.append({**default, 'name': name, 'priority': k,
                            **{key: v for key, v in nc.items() if key != 'pattern'}})
            patterns.append({'netclass': name, 'pattern': nc['pattern']})
        return {'board': {'design_settings': {'rules': rules, 'defaults': {'zones': {'min_clearance': self.clearance}}}},
                'meta': {'filename': f'{self.name}.kicad_pro', 'version': 3},
                'net_settings': {'classes': classes, 'meta': {'version': 4}, 'net_colors': None,
                                 'netclass_assignments': None, 'netclass_patterns': patterns}}

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text_pcb(), encoding='utf-8', newline='\n')
        path.with_suffix('.kicad_pro').write_text(json.dumps(self.project(), indent=2, ensure_ascii=False) + '\n',
                                                  encoding='utf-8', newline='\n')
        if self.dru:                     # пользовательские правила DRC (их учитывает и заливка зон)
            path.with_suffix('.kicad_dru').write_text('(version 1)\n' + '\n'.join(self.dru) + '\n',
                                                      encoding='utf-8', newline='\n')
        return path


def polar(p, angle_deg: float, length: float):
    """Точка на расстоянии length от p в направлении angle_deg (против часовой от оси X)."""
    a = math.radians(angle_deg)
    return (p[0] + length * math.cos(a), p[1] + length * math.sin(a))
