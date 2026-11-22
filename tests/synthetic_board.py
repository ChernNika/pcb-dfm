"""Синтетическая плата KiCad 10, на которой есть всё, чего нет на демо-платах.

Плата 4-слойная (F.Cu, In1.Cu, In2.Cu, B.Cu), компонент U1 повёрнут на 30°. На ней:
  * площадки всех форм: прямоугольник, скруглённый, со срезанными углами (и со срезом, и со
    скруглением), трапеция, овал, круг, произвольная (опорный прямоугольник + многоугольник,
    круг, линия, дуга);
  * отступы маски: свой у площадки (в том числе отрицательный), у компонента, у платы;
  * падстеки: front_inner_back и custom — у площадок и у переходного отверстия;
  * удаление неиспользуемых площадок (remove_unused_layers) у выводов и переходных отверстий:
    подключение дорожкой (конец дорожки в площадке), дорожкой «насквозь» (центр отверстия
    на дорожке), зоной (zone_layer_connections), с сохранением крайних слоёв и без;
  * глухое отверстие F.Cu–In1.Cu, закрытие переходных отверстий маской: по плате и своё;
  * неметаллизированные отверстия без меди и с медным пояском, паз со смещённой площадкой;
  * графика на меди и маске (многоугольник, окружность, дуга, кривая Безье, прямоугольник);
  * контур из линий и дуги, перечисленных вразнобой и в разных направлениях; вырезы —
    окружность, прямоугольник и паз из линий и дуг внутри компонента.

write(path) записывает плату и возвращает ожидаемые слои площадок (FLASH) — то, что должно
получиться по правилам KiCad независимо от выгрузки Gerber.
"""
import math
import uuid

FP_AT = (30.0, 25.0, 30.0)       # компонент U1: x, y, угол (координаты KiCad, ось Y вниз)

# ожидаемые слои меди у объектов с удалением неиспользуемых площадок и с падстеками
FLASH = {
    'U1.6': ['F.Cu', 'In1.Cu', 'B.Cu'],       # крайние слои сохранены + дорожка на In1
    'U1.13': ['In2.Cu', 'B.Cu'],              # зона на In2 + дорожка на B.Cu, крайние не сохраняются
    'via1': ['F.Cu', 'In2.Cu', 'B.Cu'],       # крайние + дорожка, оканчивающаяся в отверстии
    'via2': ['In1.Cu'],                       # дорожка проходит через центр отверстия
    'via3': ['In2.Cu'],                       # подключение зоной
    'via4': ['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'],
    'via5': ['F.Cu', 'In1.Cu'],               # глухое
}


def _abs(x, y):
    """Точка компонента U1 → координаты платы KiCad (ось Y вниз, поворот против часовой)."""
    X, Y, a = FP_AT
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return round(X + x * c + y * s, 6), round(Y - x * s + y * c, 6)


def _u():
    return f'(uuid "{uuid.uuid4()}")'


def _pad(num, kind, shape, at, extra_angle=0.0, body=''):
    x, y = at
    return f'(pad "{num}" {kind} {shape} (at {x} {y} {FP_AT[2] + extra_angle:g}) {body} {_u()})'


def _footprint():
    pads = [
        _pad('1', 'smd', 'rect', (-6, -4), body='(size 1 2) (layers "F.Cu" "F.Mask" "F.Paste") '
             '(solder_mask_margin 0.1) (net "A")'),
        _pad('2', 'smd', 'roundrect', (-3, -4), body='(size 1.2 2) (layers "F.Cu" "F.Mask") '
             '(roundrect_rratio 0.25) (net "A")'),
        _pad('3', 'smd', 'roundrect', (0, -4), body='(size 1.5 1.5) (layers "F.Cu" "F.Mask") '
             '(roundrect_rratio 0) (chamfer_ratio 0.2) (chamfer top_left bottom_right) '
             '(solder_mask_margin 0.1) (net "B")'),
        _pad('3b', 'smd', 'roundrect', (0, -7), body='(size 2 1.4) (layers "F.Cu" "F.Mask") '
             '(roundrect_rratio 0.15) (chamfer_ratio 0.25) (chamfer top_right bottom_left) (net "B")'),
        _pad('4', 'smd', 'trapezoid', (3, -4), body='(size 1.5 2) (rect_delta 0 0.4) '
             '(layers "F.Cu" "F.Mask") (solder_mask_margin 0.08) (net "B")'),
        _pad('5', 'smd', 'oval', (6, -4), extra_angle=45, body='(size 1 2.5) (layers "F.Cu" "F.Mask") (net "C")'),
        _pad('6', 'thru_hole', 'circle', (-6, 0), body='(size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask") '
             '(remove_unused_layers yes) (keep_end_layers yes) (net "D")'),
        _pad('7', 'thru_hole', 'oval', (-3, 0), body='(size 1.4 2.4) (drill oval 0.6 1.4 (offset 0 0.2)) '
             '(layers "*.Cu" "*.Mask") (net "D")'),
        _pad('8', 'smd', 'custom', (0, 0), body='(size 0.8 0.8) (layers "F.Cu" "F.Mask") '
             '(solder_mask_margin 0.1) (net "E") (options (clearance outline) (anchor rect)) '
             '(primitives (gr_poly (pts (xy 0 0) (xy 1.5 -0.5) (xy 1.5 0.5)) (width 0.1) (fill yes)) '
             '(gr_circle (center -0.6 0) (end -0.2 0) (width 0) (fill yes)) '
             '(gr_line (start 0 0) (end 0 1.2) (width 0.3)) '
             '(gr_arc (start 0.5 1) (mid 0.8 1.3) (end 1.1 1) (width 0.2)))'),
        _pad('', 'np_thru_hole', 'circle', (3, 0), body='(size 1.2 1.2) (drill 1.2) (layers "*.Cu" "*.Mask")'),
        _pad('9', 'np_thru_hole', 'circle', (3, 3), body='(size 2 2) (drill 1) (layers "*.Cu" "*.Mask")'),
        _pad('10', 'smd', 'rect', (6, 0), body='(size 1.5 1.5) (layers "F.Cu" "F.Mask") '
             '(solder_mask_margin -0.1) (net "E")'),
        _pad('11', 'thru_hole', 'circle', (-6, 4), body='(size 1.8 1.8) (drill 0.9) (layers "*.Cu" "*.Mask") '
             '(net "F") (padstack (mode front_inner_back) (layer "Inner" (shape circle) (size 1.2 1.2)) '
             '(layer "B.Cu" (shape rect) (size 2 1.5)))'),
        _pad('12', 'thru_hole', 'circle', (-2, 4), body='(size 1.7 1.7) (drill 0.8) (layers "*.Cu" "*.Mask") '
             '(net "F") (padstack (mode custom) '
             '(layer "In1.Cu" (shape roundrect) (size 1.4 1.1) (roundrect_rratio 0.25)) '
             '(layer "In2.Cu" (shape oval) (size 1.2 1.9)) '
             '(layer "B.Cu" (shape circle) (size 2.1 2.1)))'),
        _pad('13', 'thru_hole', 'rect', (2, 6), body='(size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask") '
             '(remove_unused_layers yes) (keep_end_layers no) (zone_layer_connections "In2.Cu") (net "G")'),
        _pad('14', 'smd', 'roundrect', (6, 4), body='(size 1.2 1.6) (layers "B.Cu" "B.Mask") '
             '(roundrect_rratio 0.2) (net "C")'),
    ]
    graphics = [
        # паз из двух отрезков и двух дуг (в системе компонента)
        '(fp_line (start -1 8) (end 1 8) (stroke (width 0.05) (type solid)) (layer "Edge.Cuts") ' + _u() + ')',
        '(fp_arc (start 1 8) (mid 1.5 8.5) (end 1 9) (stroke (width 0.05) (type solid)) (layer "Edge.Cuts") ' + _u() + ')',
        '(fp_line (start 1 9) (end -1 9) (stroke (width 0.05) (type solid)) (layer "Edge.Cuts") ' + _u() + ')',
        '(fp_arc (start -1 9) (mid -1.5 8.5) (end -1 8) (stroke (width 0.05) (type solid)) (layer "Edge.Cuts") ' + _u() + ')',
        '(fp_rect (start 4 7) (end 6 8) (stroke (width 0.1) (type solid)) (fill yes) (layer "F.Cu") ' + _u() + ')',
        '(fp_poly (pts (xy -7 7) (xy -5 7) (xy -6 8.5)) (stroke (width 0) (type solid)) (fill yes) (layer "F.Mask") ' + _u() + ')',
    ]
    x, y, a = FP_AT
    return (f'(footprint "Test:All" (layer "F.Cu") {_u()} (at {x} {y} {a:g}) '
            f'(property "Reference" "U1" (at 0 -10 {a:g}) (layer "F.SilkS") {_u()} '
            f'(effects (font (size 1 1) (thickness 0.15)))) '
            f'(property "Value" "ALL" (at 0 10 {a:g}) (layer "F.Fab") {_u()} '
            f'(effects (font (size 1 1) (thickness 0.15)))) '
            f'(solder_mask_margin 0.07) (attr smd) ' + ' '.join(pads + graphics) + ')')


def _seg(a, b, w, layer, net):
    return f'(segment (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (width {w}) (layer "{layer}") (net "{net}") {_u()})'


def _board():
    p6, p13 = _abs(-6, 0), _abs(2, 6)
    items = [
        # дорожки
        _seg(p6, (8, 30), 0.25, 'In1.Cu', 'D'),                    # оканчивается в выводе 6
        _seg(p13, (40, 40), 0.3, 'B.Cu', 'G'),                     # оканчивается в выводе 13
        _seg((10, 44), (10, 48), 0.25, 'In2.Cu', 'D'),             # оканчивается в via1
        _seg((13, 46), (17, 46), 0.2, 'In1.Cu', 'D'),              # проходит через via2
        f'(arc (start 40 8) (mid 44 6) (end 48 8) (width 0.4) (layer "B.Cu") (net "A") {_u()})',
        # переходные отверстия
        f'(via (at 10 44) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (remove_unused_layers yes) '
        f'(keep_end_layers yes) (net "D") {_u()})',
        f'(via (at 15 46) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (remove_unused_layers yes) '
        f'(keep_end_layers no) (net "D") {_u()})',
        f'(via (at 20 46) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (remove_unused_layers yes) '
        f'(keep_end_layers no) (zone_layer_connections "In2.Cu") (net "G") {_u()})',
        f'(via (at 25 46) (size 0.8) (drill 0.3) (layers "F.Cu" "B.Cu") (net "G") '
        f'(padstack (mode front_inner_back) (layer "Inner" (size 0.5)) (layer "B.Cu" (size 1)))  {_u()})',
        f'(via blind (at 30 46) (size 0.6) (drill 0.3) (layers "F.Cu" "In1.Cu") (net "D") {_u()})',
        f'(via (at 35 46) (size 0.9) (drill 0.4) (layers "F.Cu" "B.Cu") (tenting (front yes) (back no)) '
        f'(net "G") {_u()})',
        # зона на In2.Cu с готовой заливкой (с вырезом, соединённым разрезом, как пишет KiCad)
        f'(zone (net "G") (layer "In2.Cu") {_u()} (hatch edge 0.5) (connect_pads (clearance 0.2)) '
        f'(min_thickness 0.2) (fill yes (thermal_gap 0.3) (thermal_bridge_width 0.3)) '
        f'(polygon (pts (xy 36 30) (xy 56 30) (xy 56 44) (xy 36 44))) '
        f'(filled_polygon (layer "In2.Cu") (pts (xy 36.1 30.1) (xy 55.9 30.1) (xy 55.9 43.9) (xy 46 43.9) '
        f'(xy 46 40) (xy 48 40) (xy 48 34) (xy 44 34) (xy 44 40) (xy 46 40) (xy 46 43.9) (xy 36.1 43.9))))',
        # графика на меди и маске
        f'(gr_poly (pts (xy 4 4) (xy 9 4) (xy 9 6) (xy 6 8)) (stroke (width 0.2) (type solid)) (fill yes) '
        f'(layer "F.Cu") {_u()})',
        f'(gr_circle (center 6 14) (end 7.5 14) (stroke (width 0.3) (type solid)) (fill no) (layer "B.Cu") {_u()})',
        f'(gr_arc (start 4 20) (mid 6 18) (end 8 20) (stroke (width 0.3) (type solid)) (layer "In2.Cu") {_u()})',
        f'(gr_curve (pts (xy 4 24) (xy 6 21) (xy 8 27) (xy 10 24)) (stroke (width 0.25) (type solid)) '
        f'(layer "F.Cu") {_u()})',
        f'(gr_rect (start 12 4) (end 16 7) (stroke (width 0.1) (type solid)) (fill yes) (layer "B.Cu") {_u()})',
        f'(gr_line (start 50 4) (end 54 6) (stroke (width 0.5) (type solid)) (layer "F.Mask") {_u()})',
        f'(gr_poly (pts (xy 50 10) (xy 54 10) (xy 52 13)) (stroke (width 0.1) (type solid)) (fill yes) '
        f'(layer "B.Mask") {_u()})',
        # контур: отрезки вразнобой, один записан в обратном направлении, угол скруглён дугой
        f'(gr_line (start 60 50) (end 0 50) (stroke (width 0.1) (type solid)) (layer "Edge.Cuts") {_u()})',
        f'(gr_line (start 0 0) (end 57 0) (stroke (width 0.1) (type solid)) (layer "Edge.Cuts") {_u()})',
        f'(gr_line (start 0 50) (end 0 0) (stroke (width 0.1) (type solid)) (layer "Edge.Cuts") {_u()})',
        f'(gr_line (start 60 50) (end 60 3) (stroke (width 0.1) (type solid)) (layer "Edge.Cuts") {_u()})',
        f'(gr_arc (start 57 0) (mid 59.12132 0.87868) (end 60 3) (stroke (width 0.1) (type solid)) '
        f'(layer "Edge.Cuts") {_u()})',
        # вырезы: окружность и прямоугольник
        f'(gr_circle (center 52 22) (end 54 22) (stroke (width 0.1) (type solid)) (fill no) (layer "Edge.Cuts") {_u()})',
        f'(gr_rect (start 20 3) (end 26 6) (stroke (width 0.1) (type solid)) (fill no) (layer "Edge.Cuts") {_u()})',
    ]
    return items


HEADER = '''(kicad_pcb (version 20260206) (generator "pcbnew") (generator_version "10.0")
  (general (thickness 1.6) (legacy_teardrops no))
  (paper "A4")
  (layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal)
    (9 "F.Adhes" user "F.Adhesive") (11 "B.Adhes" user "B.Adhesive") (13 "F.Paste" user) (15 "B.Paste" user)
    (5 "F.SilkS" user "F.Silkscreen") (7 "B.SilkS" user "B.Silkscreen") (1 "F.Mask" user) (3 "B.Mask" user)
    (17 "Dwgs.User" user "User.Drawings") (19 "Cmts.User" user "User.Comments") (25 "Edge.Cuts" user)
    (27 "Margin" user) (31 "F.CrtYd" user "F.Courtyard") (29 "B.CrtYd" user "B.Courtyard")
    (35 "F.Fab" user) (33 "B.Fab" user))
  (setup (pad_to_mask_clearance 0.05) (allow_soldermask_bridges_in_footprints no)
    (tenting (front no) (back yes)))
'''


def write(path):
    text = HEADER + '\n'.join(['  ' + _footprint()] + ['  ' + s for s in _board()]) + '\n)\n'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return FLASH
