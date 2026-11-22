"""Модель платы и адаптер KiCad на маленьких платах, записанных прямо в тесте (KiCad не нужен)."""
import math

import pytest

from dfm.board import load
from dfm.board.design import DesignRules, NetClass, ZoneConnection
from dfm.board.kicad import _default_stackup, _pts
from dfm.board.layers import LayerKind, Side, copper_order, expand, kind_of, side_of
from dfm.board.sexpr import parse
from dfm.board.shapes import rotate, transform

LAYERS4 = '''(layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" power "GND") (2 "B.Cu" signal)
    (1 "F.Mask" user) (3 "B.Mask" user) (25 "Edge.Cuts" user) (5 "F.SilkS" user "F.Silkscreen"))'''


def board(tmp_path, body, layers=LAYERS4, version=20260206, pro=None, name='t'):
    p = tmp_path / f'{name}.kicad_pcb'
    p.write_text(f'(kicad_pcb (version {version}) (generator "pcbnew") (generator_version "10.0")'
                 f' (general (thickness 1.6)) {layers} (setup (pad_to_mask_clearance 0.05)) {body})',
                 encoding='utf-8')
    if pro is not None:
        p.with_suffix('.kicad_pro').write_text(pro, encoding='utf-8')
    return load(p)


# ---------- слои и геометрия ----------

def test_copper_order_by_name():
    names = ['B.Cu', 'In10.Cu', 'F.Mask', 'In2.Cu', 'F.Cu', 'In1.Cu']
    assert copper_order(names) == ['F.Cu', 'In1.Cu', 'In2.Cu', 'In10.Cu', 'B.Cu']


def test_layer_kinds():
    assert kind_of('In3.Cu') is LayerKind.COPPER and side_of('In3.Cu') is Side.INNER
    assert kind_of('B.Mask') is LayerKind.MASK and side_of('B.Mask') is Side.BOTTOM
    assert kind_of('Edge.Cuts') is LayerKind.OUTLINE and side_of('Edge.Cuts') is None
    assert kind_of('F.Fab') is LayerKind.OTHER


def test_expand_wildcards():
    lay = ['F.Cu', 'In1.Cu', 'B.Cu', 'F.Mask', 'B.Mask', 'F.Paste', 'B.Paste']
    assert expand(['*.Cu', '*.Mask'], lay) == ['F.Cu', 'In1.Cu', 'B.Cu', 'F.Mask', 'B.Mask']
    assert expand(['F&B.Cu'], lay) == ['F.Cu', 'B.Cu']
    assert expand(['F.Cu', 'F.Paste', 'F.Cu'], lay) == ['F.Cu', 'F.Paste']


def test_rotation_exact_and_ccw():
    assert rotate((1.0, 0.0), 90) == (0, 1)          # против часовой, без хвостов 1e-17
    assert rotate((1.0, 2.0), 180) == (-1, -2)
    x, y = rotate((1.0, 0.0), 30)
    assert math.isclose(x, math.sqrt(3) / 2) and math.isclose(y, 0.5)
    assert transform((1.0, 0.0), (10.0, 5.0), 90) == (10, 6)


def test_pts_with_arc():
    c = _pts(parse('(pts (xy 0 0) (arc (start 1 0) (mid 1.7071 -0.7071) (end 2 -1)) (xy 2 -3))'))
    assert c.pts.tolist() == [[0, 0], [1, 0], [2, 1], [2, 3]]       # ось Y перевёрнута
    assert c.arcs == {1: (1.7071, 0.7071)}


# ---------- компонент: координаты, поворот, сторона ----------

FP = '''(footprint "Lib:Part" (layer "{side}") (at 10 20 {angle}) (uuid "fp1")
    (property "Reference" "U1" (at 0 0 0) (layer "F.SilkS")) (property "Value" "X" (at 0 0 0) (layer "F.Fab"))
    (attr smd)
    (pad "1" smd rect (at 1 0 {padangle}) (size 0.5 0.8) (layers "{cu}" "{mask}") (net "N1") (uuid "p1"))
    (pad "2" thru_hole oval (at -1 2 {padangle}) (size 1.5 2) (drill oval 0.6 1.0 (offset 0 0.2))
        (layers "*.Cu" "*.Mask") (remove_unused_layers yes) (keep_end_layers yes) (net "N2") (uuid "p2"))
    (fp_line (start 0 0) (end 2 0) (stroke (width 0.1) (type solid)) (layer "{cu}") (uuid "g1"))
    (fp_line (start 0 0) (end 2 0) (stroke (width 0.1) (type solid)) (layer "F.SilkS") (uuid "g2")))'''


def test_footprint_transform(tmp_path):
    b = board(tmp_path, FP.format(side='F.Cu', angle=90, padangle=90, cu='F.Cu', mask='F.Mask'))
    fp = b.design.footprints[0]
    assert (fp.ref, fp.value, fp.side, fp.angle) == ('U1', 'X', 'top', 90)
    assert fp.pos == (10, -20)
    p1, p2 = fp.pads
    # локальная точка (1, 0) повёрнута на 90° против часовой: (0, 1) → (10, −19)
    assert p1.pos == (10, -19) and p1.angle == 90
    # (−1, 2) в системе KiCad = (−1, −2) с осью Y вверх; поворот на 90°: (2, −1)
    assert p2.pos == (12, -21)
    assert p2.layers == ('F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu', 'F.Mask', 'B.Mask')
    assert p2.drill.oval and p2.drill.size == (0.6, 1.0)
    assert p2.shape.offset == (0, -0.2)              # смещение формы — в системе площадки, Y вверх
    assert p2.remove_unused_layers and p2.keep_end_layers
    # графика компонента: только медь (шелкография не читается), координаты платы
    assert [g.points for g in fp.graphics] == [((10, -20), (10, -18))]
    assert p1.net == 'N1' and {'N1', 'N2'} <= b.nets


def test_footprint_on_back(tmp_path):
    b = board(tmp_path, FP.format(side='B.Cu', angle=0, padangle=0, cu='B.Cu', mask='B.Mask'))
    fp = b.design.footprints[0]
    assert fp.side == 'bottom' and fp.pads[0].layers == ('B.Cu', 'B.Mask')


# ---------- KiCad 9 и KiCad 10: запись цепей ----------

def test_nets_kicad9_table(tmp_path):
    body = '''(net 0 "") (net 1 "GND") (net 2 "/sig")
        (segment (start 0 0) (end 1 0) (width 0.2) (layer "F.Cu") (net 1) (uuid "s1"))
        (segment (start 0 1) (end 1 1) (width 0.2) (layer "B.Cu") (net 0) (uuid "s2"))
        (zone (net 2) (net_name "/sig") (layer "In1.Cu") (uuid "z1") (connect_pads yes (clearance 0.3))
            (min_thickness 0.2) (fill yes (thermal_gap 0.5) (thermal_bridge_width 0.4))
            (polygon (pts (xy 0 0) (xy 5 0) (xy 5 5))))'''
    b = board(tmp_path, body, version=20241229)
    t1, t2 = b.design.tracks
    assert t1.net == 'GND' and t2.net is None
    z = b.design.zones[0]
    assert z.net == '/sig' and z.layers == ('In1.Cu',) and z.connect_pads is ZoneConnection.FULL
    assert z.clearance == 0.3 and not z.filled
    assert b.nets == {'GND', '/sig'}


def test_nets_kicad10_names(tmp_path):
    body = '(segment (start 0 0) (end 1 0) (width 0.2) (layer "F.Cu") (net "GND") (uuid "s1"))'
    assert board(tmp_path, body).design.tracks[0].net == 'GND'


# ---------- переходные отверстия и падстеки ----------

def test_via_types_and_padstack(tmp_path):
    body = '''(via (at 1 1) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net "A") (uuid "v1"))
        (via blind (at 2 2) (size 0.5) (drill 0.2) (layers "In1.Cu" "F.Cu") (net "A") (uuid "v2")
            (tenting (front yes) (back none)))
        (via micro (at 3 3) (size 0.3) (drill 0.1) (layers "F.Cu" "In1.Cu") (uuid "v3"))
        (via (at 4 4) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (uuid "v4")
            (padstack (mode front_inner_back) (layer "Inner" (size 0.6)) (layer "B.Cu" (size 0.9))))'''
    b = board(tmp_path, body)
    v1, v2, v3, v4 = b.design.vias
    assert v1.via_type == 'through' and v2.via_type == 'blind' and v3.via_type == 'micro'
    assert v2.layers == ('F.Cu', 'In1.Cu')          # упорядочено сверху вниз
    assert v2.tenting == {'front': True, 'back': None}
    cu = b.copper
    assert [v4.size_on(l, cu) for l in cu] == [0.8, 0.6, 0.6, 0.9]
    assert [p for p in b.validate() if not p.startswith("контур платы")] == []      # контура в этой плате нет


def test_pad_padstack_shapes(tmp_path):
    body = '''(footprint "L:P" (layer "F.Cu") (at 0 0) (property "Reference" "J1" (at 0 0) (layer "F.SilkS"))
        (pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (drill 1.0) (layers "*.Cu" "*.Mask") (uuid "p")
            (padstack (mode front_inner_back) (layer "Inner" (shape circle) (size 1.2 1.2))
                      (layer "B.Cu" (shape rect) (size 2 2)))))'''
    b = board(tmp_path, body)
    p = b.design.pads[0]
    shapes = [p.shape_on(l, b.copper) for l in b.copper]
    assert [(s.shape, s.size) for s in shapes] == [('circle', (1.6, 1.6)), ('circle', (1.2, 1.2)),
                                                   ('circle', (1.2, 1.2)), ('rect', (2, 2))]


# ---------- правила, классы цепей, стек ----------

PRO = '''{"board": {"design_settings": {"rules": {"min_clearance": 0.15, "min_track_width": 0.1,
            "min_via_annular_width": 0.1, "allow_microvias": false},
          "defaults": {"zones": {"min_clearance": 0.3}}}},
  "net_settings": {"classes": [
      {"name": "Default", "priority": 2147483647, "clearance": 0.2, "track_width": 0.25, "via_diameter": 0.6},
      {"name": "Power", "priority": 0, "track_width": 0.8},
      {"name": "HV", "priority": 1, "clearance": 1.0, "track_width": 0.5}],
    "netclass_patterns": [{"netclass": "Power", "pattern": "+*V"}, {"netclass": "HV", "pattern": "/^HV_\\\\d+$/"}],
    "netclass_assignments": {"+12V": ["HV"]}}}'''


def test_rules_from_project(tmp_path):
    b = board(tmp_path, '', pro=PRO)
    r = b.design.rules
    assert r.has_project_file and not r.has_custom_rules
    assert (r.min_clearance, r.min_track_width, r.allow_microvias) == (0.15, 0.1, False)
    assert r.pad_to_mask_clearance == 0.05 and r.zone_min_clearance == 0.3
    assert r.netclass_of('GND').name == 'Default'
    assert r.netclass_of('+5V').track_width == 0.8 and r.netclass_of('+5V').clearance == 0.2  # из Default
    hv = r.netclass_of('HV_12')                       # регулярное выражение /…/
    assert hv.name == 'HV' and hv.clearance == 1.0
    both = r.netclass_of('+12V')                       # шаблон Power + явное назначение HV
    assert both.track_width == 0.8 and both.clearance == 1.0 and both.name == 'Power,HV'


def test_no_project_file(tmp_path):
    b = board(tmp_path, '')
    assert not b.design.rules.has_project_file
    assert any('.kicad_pro' in w for w in b.warnings)


def test_default_stackup_like_kicad():
    st = _default_stackup(['F.Cu', 'B.Cu'], 1.6)
    assert st.synthetic and [i.kind for i in st.items].count('copper') == 2
    diel = [i for i in st.items if i.kind == 'dielectric']
    assert len(diel) == 1 and math.isclose(diel[0].thickness, 1.51)
    assert math.isclose(st.thickness, 1.6)
    st4 = _default_stackup(['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'], 1.6)
    assert [d.dielectric_type for d in st4.items if d.kind == 'dielectric'] == ['core', 'prepreg', 'core']
    assert math.isclose(st4.thickness, 1.6)


# ---------- проверка модели ----------

def test_validate_reports_problems(tmp_path):
    body = '''(via blind (at 1 1) (size 0.45) (drill 0.00001) (layers "F.Cu" "B.Cu") (uuid "v1"))
        (via blind (at 2 1) (size 0.45) (drill 0.00001) (layers "F.Cu" "B.Cu") (uuid "v2"))
        (segment (start 0 0) (end 1 0) (width 0.2) (layer "In7.Cu") (uuid "s1"))'''
    b = board(tmp_path, body)
    probs = b.validate()
    assert any('вырожденное' in p and '2 шт.' in p for p in probs)
    assert any('фактически сквозное' in p for p in probs)
    assert any('дорожка на несуществующем слое In7.Cu' in p for p in probs)


def test_netclass_default_fallback():
    r = DesignRules(netclasses={'Default': NetClass('Default', clearance=0.2)})
    assert r.netclass_of(None).clearance == 0.2 and r.netclass_of('X').name == 'Default'
