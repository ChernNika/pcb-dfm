"""Уровень P из .kicad_pcb (kicad_build.py) без запуска KiCad: формы, маска, падстеки, контур.

Сверка с Gerber и сверловкой от KiCad — в test_board_gerber.py."""
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import synthetic_board
from dfm.board import Role, load
from dfm.board.design import PadShape
from dfm.board.kicad_shapes import pad_shape
from dfm.geometry.polygons import layer_geometry, to_geometry
from test_board_model import board

TOL = 1e-4          # точность ломаных в проверках площадей, мм
AREA = 2e-3         # допуск по площади, мм² (ломаные чуть меньше дуг)


def area(shape):
    return to_geometry(shape, TOL).area


# ---------- формы площадок ----------

def test_pad_shape_areas():
    pos = (10.0, 5.0)
    assert math.isclose(area(pad_shape(PadShape('rect', (1, 2)), pos, 30)), 2, abs_tol=1e-9)
    r = 0.25 * 1.2
    assert math.isclose(area(pad_shape(PadShape('roundrect', (1.2, 2), roundrect_rratio=0.25), pos, 0)),
                        2.4 - (4 - math.pi) * r * r, abs_tol=AREA)
    assert math.isclose(area(pad_shape(PadShape('oval', (1, 2.5)), pos, 45)),
                        math.pi * 0.25 + 1 * 1.5, abs_tol=AREA)
    assert math.isclose(area(pad_shape(PadShape('circle', (1.6, 1.6)), pos, 0)), math.pi * 0.64, abs_tol=AREA)
    # два угла срезаны на 0,2 × 1,5 = 0,3 мм
    ch = PadShape('roundrect', (1.5, 1.5), chamfer_ratio=0.2, chamfer=frozenset({'top_left', 'bottom_right'}))
    assert math.isclose(area(pad_shape(ch, pos, 30)), 2.25 - 2 * 0.3 * 0.3 / 2, abs_tol=1e-9)
    # трапеция с симметричным rect_delta сохраняет площадь w·h
    assert math.isclose(area(pad_shape(PadShape('trapezoid', (1.5, 2), rect_delta=(0, 0.4)), pos, 0)), 3, abs_tol=1e-9)


def test_pad_shape_offset_and_rotation():
    """Центр формы = позиция отверстия + поворот(offset); угол площадки — полный."""
    s = pad_shape(PadShape('circle', (1, 1), offset=(1.0, 0.0)), (10.0, 5.0), 90)
    assert np.allclose(s.center, (10, 6))
    g = to_geometry(pad_shape(PadShape('rect', (2, 1)), (0.0, 0.0), 90), TOL)
    assert np.allclose(g.bounds, (-0.5, -1, 0.5, 1))


def test_mask_expansion_rules():
    """Окно маски: прямоугольник — точное расширение (углы скругляются радиусом отступа);
    скруглённый прямоугольник — сохраняет долю скругления, как в KiCad; отрицательный отступ
    уменьшает прямоугольник без скругления."""
    m = 0.1
    rect = area(pad_shape(PadShape('rect', (1, 2)), (0.0, 0.0), 0, m))
    assert math.isclose(rect, 1.2 * 2.2 - (4 - math.pi) * m * m, abs_tol=AREA)
    rr = area(pad_shape(PadShape('roundrect', (1, 2), roundrect_rratio=0.25), (0.0, 0.0), 0, m))
    r = 0.25 * 1.2
    assert math.isclose(rr, 1.2 * 2.2 - (4 - math.pi) * r * r, abs_tol=AREA)
    assert math.isclose(area(pad_shape(PadShape('rect', (1.5, 1.5)), (0.0, 0.0), 0, -0.1)), 1.3 ** 2, abs_tol=1e-9)
    assert pad_shape(PadShape('rect', (1, 1)), (0.0, 0.0), 0, -0.5) is None


# ---------- синтетическая плата: всё, чего нет на демо-платах ----------

@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    p = tmp_path_factory.mktemp('synth') / 'synth.kicad_pcb'
    synthetic_board.write(p)
    return load(p)


def _stacks(b):
    out, vias = {}, 0
    for h in b.holes:
        if h.role is Role.VIA:
            vias += 1
            out[f'via{vias}'] = list(h.padstack)
        else:
            out[f'{h.component}.{h.pin}'] = list(h.padstack)
    return out


def test_unused_pads_removed(synth):
    """Площадки на слоях без подключения удалены так, как это делает KiCad: подключение — конец
    дорожки в площадке, центр отверстия на дорожке или зона (zone_layer_connections)."""
    stacks = _stacks(synth)
    for key, layers in synthetic_board.FLASH.items():
        assert stacks[key] == layers, key
    # медь слоя и падстек — одни и те же объекты
    for h in synth.holes:
        for lay, f in h.padstack.items():
            assert any(f is g for g in synth.layers[lay].features)


def test_padstack_shapes(synth):
    """Вывод 11 (front_inner_back): круг 1,8 сверху, 1,2 внутри, прямоугольник 2×1,5 снизу.
    Вывод 12 (custom): своя форма на каждом слое."""
    h = {f'{x.component}.{x.pin}': x for x in synth.holes if x.component}
    p11 = {lay: area(f.shape) for lay, f in h['U1.11'].padstack.items()}
    assert math.isclose(p11['F.Cu'], math.pi * 0.81, abs_tol=AREA)
    assert math.isclose(p11['In1.Cu'], math.pi * 0.36, abs_tol=AREA)
    assert math.isclose(p11['In2.Cu'], math.pi * 0.36, abs_tol=AREA)
    assert math.isclose(p11['B.Cu'], 3.0, abs_tol=1e-9)
    p12 = {lay: area(f.shape) for lay, f in h['U1.12'].padstack.items()}
    assert math.isclose(p12['In2.Cu'], math.pi * 0.36 + 1.2 * 0.7, abs_tol=AREA)
    assert math.isclose(p12['B.Cu'], math.pi * 1.05 ** 2, abs_tol=AREA)
    # падстек переходного отверстия: 0,8 снаружи, 0,5 внутри, 1,0 снизу
    via4 = next(x for x in synth.holes if x.role is Role.VIA and x.pos == (25.0, -46.0))
    radii = {lay: f.shape.radius for lay, f in via4.padstack.items()}
    assert radii == {'F.Cu': 0.4, 'In1.Cu': 0.25, 'In2.Cu': 0.25, 'B.Cu': 0.5}


def test_mask_windows(synth):
    """Отступ маски: свой у площадки → у компонента (0,07) → у платы (0,05). Переходные
    отверстия: снизу закрыты маской (настройка платы), сверху открыты, кроме via6 со своей
    настройкой; у глухого F.Cu–In1.Cu окна снизу нет."""
    fm = {(f.component, f.pin): f for f in synth.layers['F.Mask'].features if f.role is Role.PAD}
    cu = {(f.component, f.pin): f for f in synth.layers['F.Cu'].features if f.role is Role.PAD}
    # вывод 6 — круг 1,6; отступ компонента 0,07
    assert math.isclose(fm[('U1', '6')].shape.radius, 0.8 + 0.07)
    # вывод 10 — свой отрицательный отступ: квадрат 1,5 → 1,3
    assert math.isclose(area(fm[('U1', '10')].shape), 1.3 ** 2, abs_tol=1e-9)
    assert area(fm[('U1', '1')].shape) > area(cu[('U1', '1')].shape)
    # неметаллизированное отверстие без меди: меди нет, окно маски есть
    assert ('U1', '') not in cu and ('U1', '') in fm
    vias_f = sorted(f.shape.center for f in synth.layers['F.Mask'].features if f.role is Role.VIA)
    vias_b = sorted(f.shape.center for f in synth.layers['B.Mask'].features if f.role is Role.VIA)
    assert vias_f == [(10.0, -44.0), (15.0, -46.0), (20.0, -46.0), (25.0, -46.0), (30.0, -46.0)]
    assert vias_b == [(35.0, -46.0)]
    v = next(f for f in synth.layers['F.Mask'].features if f.role is Role.VIA)
    assert math.isclose(v.shape.radius, 0.4 + 0.05)


def test_outline_with_cutouts(synth):
    """Контур из отрезков вразнобой и дуги + три выреза: окружность, прямоугольник, паз
    в компоненте. Площадь платы считается вручную."""
    assert len(synth.outline) == 1
    r = synth.outline[0]
    assert len(r.holes) == 3
    corner = 9 - math.pi * 9 / 4                   # угол 3×3, скруглённый радиусом 3
    expected = 60 * 50 - corner - math.pi * 4 - 18 - (2 * 1 + math.pi * 0.25)
    assert math.isclose(to_geometry(r, TOL).area, expected, abs_tol=AREA)


def test_outline_open_chain(tmp_path):
    body = ''.join(f'(gr_line (start {a}) (end {b}) (stroke (width 0.1)) (layer "Edge.Cuts"))'
                   for a, b in (('0 0', '10 0'), ('10 0', '10 10'), ('10 10', '0 10')))
    b = board(tmp_path, body)
    assert b.outline == []
    assert any('незамкнутых цепочек' in w for w in b.warnings)


def test_outline_small_gap_is_closed(tmp_path):
    """Концы, разошедшиеся меньше чем на 0,02 мм, сшиваются."""
    body = ''.join(f'(gr_line (start {a}) (end {b}) (stroke (width 0.1)) (layer "Edge.Cuts"))'
                   for a, b in (('0 0', '10 0'), ('10.01 0', '10 10'), ('10 10', '0 10'), ('0 10', '0 0.005')))
    b = board(tmp_path, body)
    assert len(b.outline) == 1 and not any('Edge.Cuts' in w for w in b.warnings)
    assert math.isclose(to_geometry(b.outline[0], TOL).area, 100, abs_tol=0.1)


def test_copper_graphics_and_texts(synth, tmp_path):
    """Графика на меди становится медью, текст — нет (предупреждение)."""
    roles = Counter(f.role for f in synth.layers['F.Cu'].features)
    assert roles[Role.GRAPHIC] == 3           # многоугольник, кривая Безье, прямоугольник компонента
    b = board(tmp_path, '(gr_text "X" (at 1 1) (layer "F.Cu") (effects (font (size 1 1) (thickness 0.2))))')
    assert any('не переведён в геометрию' in w for w in b.warnings)


# ---------- демо-платы: ничего не потеряно при переходе D → P ----------

DEMOS = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'KiCad' / '10.0' / 'share' / 'kicad' / 'demos'
SMALL = sorted((p for p in DEMOS.rglob('*.kicad_pcb') if p.stat().st_size < 7e6), key=lambda p: p.stat().st_size) \
    if DEMOS.exists() else []


@pytest.mark.skipif(not SMALL, reason='нет демо-плат KiCad 10')
@pytest.mark.parametrize('path', SMALL, ids=lambda p: p.stem)
def test_demo_physical_counts(path):
    b = load(path)
    d = b.design
    feats = Counter((lay.kind.value, f.role) for lay in b.layers.values() for f in lay.features)
    assert feats[('copper', Role.TRACK)] == len(d.tracks) + len(d.arcs)
    assert feats[('copper', Role.ZONE)] == sum(len(cs) for z in d.all_zones() for lay, cs in z.filled.items()
                                               if b.layers[lay].kind.value == 'copper')
    assert len(b.holes) == len(d.vias) + sum(p.drill is not None for p in d.pads)
    assert len(b.outline) == 1
    # площадь меди наружного слоя не меньше площади его площадок
    assert layer_geometry(b.layers[b.copper[0]]).area > 0
