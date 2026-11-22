"""Генератор своих тестовых плат (шаг 1.6) и демо-плат из документов.

    python boards/make_boards.py            # все платы → boards/test/, boards/demo/ + заливка зон KiCad

Тестовые платы — каждая на одну тонкость модели, чтобы при расхождении сразу было видно, где ошибка:
  t01_pads37      все формы площадок на компоненте, повёрнутом на 37°, сверху и снизу; отступы маски
  t02_arcs        дорожки-дуги разных углов и радиусов, дуги графики и контура
  t03_vias        6 слоёв: сквозные, глухие, скрытые, микроотверстия; закрытие маской
  t04_padstacks   падстеки front_inner_back и custom, в том числе заданные не полностью
  t05_unused      удаление неиспользуемых площадок: дорожкой, зоной (KiCad сам определит), без подключения
  t06_zones       термобарьеры, сплошное подключение, без подключения, сетка, приоритеты, островки,
                  область запрета, зона внутри компонента
  t07_outline     контур с дугами, вырезы (круг, скруглённый прямоугольник, многоугольник, паз),
                  вторая плата в той же заготовке, паз-отверстие
  t09_a1          дефекты класса A1 с известными размерами (Д1–Д5, Д7, Д8) — для проверки правил
  t08_drc         заложенные нарушения зазоров (дорожка–дорожка, дорожка–площадка, площадка–площадка,
                  переходное–дорожка, дуга, графика без цепи, до края платы, замыкание) и пары рядом
                  с порогом без нарушения
Демо-платы документов (координаты и ситуации — как в research/b1/geom.py и research/a7/a7geom.py):
  b1_demo         кислотные ловушки и шипы Л1–Л8 (класс B1); ситуации — в b1_demo.json
  a7_demo         окна маски: шаг выводов, BGA, переходные у площадок, край платы (класс A7)

Зоны заливает сам KiCad (kicad-cli, один раз: перезаливка недетерминирована, см. dfm/kicad_cli.py).
uuid детерминированные, поэтому повторный запуск меняет файлы только там, где меняется заливка.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from dfm import kicad_cli                                  # noqa: E402
from dfm.board.kicad_writer import KicadWriter, polar      # noqa: E402

TEST, DEMO = ROOT / 'test', ROOT / 'demo'


def all_pad_shapes(fp, nets=('A', 'B', 'C', 'D')):
    """Все формы площадок на одном компоненте (координаты компонента, мм)."""
    a, b, c, d = nets
    fp.pad('1', 'smd', 'rect', (-6, 4), (1.0, 2.0), net=a, mask_margin=0.1)
    fp.pad('2', 'smd', 'roundrect', (-4, 4), (1.2, 2.0), net=a, rratio=0.25)
    fp.pad('3', 'smd', 'roundrect', (-2, 4), (1.5, 1.5), net=b, rratio=0, chamfer=(0.2, ('top_left', 'bottom_right')),
           mask_margin=0.08)
    fp.pad('4', 'smd', 'roundrect', (0, 4), (2.0, 1.4), net=b, rratio=0.15, chamfer=(0.25, ('top_right',)))
    fp.pad('5', 'smd', 'trapezoid', (2.2, 4), (1.5, 2.0), net=c, rect_delta=(0, 0.4), mask_margin=0.06)
    fp.pad('6', 'smd', 'trapezoid', (4.4, 4), (1.6, 1.6), net=c, rect_delta=(0.3, 0))
    fp.pad('7', 'smd', 'oval', (6.4, 4), (1.0, 2.4), net=d, angle=25)
    fp.pad('8', 'smd', 'circle', (-6, 0.5), 1.2, net=d, mask_margin=-0.1)
    fp.pad('9', 'smd', 'custom', (-3.5, 0.5), (0.6, 0.6), net=a, mask_margin=0.05, anchor='rect',
           primitives=[('poly', [(0, 0), (1.3, -0.5), (1.3, 0.5)], 0.1, True), ('circle', (-0.5, 0), 0.35, 0, True),
                       ('line', (0, 0), (0, 1.0), 0.3), ('arc', (0.4, 0.8), (0.7, 1.1), (1.0, 0.8), 0.2)])
    fp.pad('10', 'smd', 'custom', (-0.5, 0.5), (0.8, 0.8), net=b, anchor='circle',
           primitives=[('poly', [(0, -0.2), (1.2, -0.2), (1.2, 0.2), (0, 0.2)], 0, True)])
    fp.pad('11', 'thru_hole', 'circle', (2, 0.5), 1.7, drill=0.9, net=c)
    fp.pad('12', 'thru_hole', 'rect', (4.2, 0.5), 1.7, drill=0.9, net=c)
    fp.pad('13', 'thru_hole', 'oval', (6.5, 0.5), (1.4, 2.6), drill=(0.7, 1.8), offset=(0, 0.2), net=d)
    fp.pad('14', 'thru_hole', 'roundrect', (-6, -3), (1.8, 1.4), drill=0.8, rratio=0.2, net=a, angle=90)
    fp.pad('', 'np_thru_hole', 'circle', (-3.5, -3), 1.2, drill=1.2)
    fp.pad('', 'np_thru_hole', 'oval', (-1, -3), (1.0, 2.0), drill=(1.0, 2.0))
    fp.pad('15', 'np_thru_hole', 'circle', (1.5, -3), 2.0, drill=1.0)
    fp.pad('16', 'connect', 'rect', (4.5, -3), (1.2, 2.2), net=b)
    fp.pad('17', 'smd', 'rect', (6.8, -3), (1.0, 1.0), layers=['F.Cu'], net=d)          # без окна маски
    fp.pad('18', 'smd', 'rect', (6.8, -5), (0.8, 0.8), layers=['F.Mask'])              # только окно маски


def t01_pads37():
    w = KicadWriter('t01_pads37', clearance=0.15, mask_margin=0.05)
    w.outline_rect(0, 0, 40, 46)
    all_pad_shapes(w.footprint('U1', (20, 34), angle=37, mask_margin=0.07))
    all_pad_shapes(w.footprint('U2', (20, 12), angle=-110, side='bottom'), nets=('E', 'F', 'G', 'H'))
    return w


def t02_arcs():
    w = KicadWriter('t02_arcs', clearance=0.15)
    w.graphic('line', 'Edge.Cuts', (3, 0), (37, 0), width=0.05)
    w.graphic('arc', 'Edge.Cuts', (37, 0), polar((37, 3), -45, 3), (40, 3), width=0.05)
    w.graphic('line', 'Edge.Cuts', (40, 3), (40, 27), width=0.05)
    w.graphic('arc', 'Edge.Cuts', (40, 27), polar((37, 27), 45, 3), (37, 30), width=0.05)
    w.graphic('line', 'Edge.Cuts', (37, 30), (3, 30), width=0.05)
    w.graphic('arc', 'Edge.Cuts', (3, 30), polar((3, 27), 135, 3), (0, 27), width=0.05)
    w.graphic('line', 'Edge.Cuts', (0, 27), (0, 3), width=0.05)
    w.graphic('arc', 'Edge.Cuts', (0, 3), polar((3, 3), 225, 3), (3, 0), width=0.05)
    fp = w.footprint('J1', (5, 15))
    for i in range(6):
        fp.pad(str(i + 1), 'smd', 'circle', (0, -6 + 2.4 * i), 1.0, net=f'N{i}')
    for i, (sweep, r, width, layer) in enumerate([(90, 4, 0.25, 'F.Cu'), (180, 2.5, 0.3, 'F.Cu'), (270, 1.6, 0.2, 'B.Cu'),
                                                    (30, 10, 0.5, 'F.Cu'), (120, 0.6, 0.15, 'B.Cu'), (200, 3, 1.0, 'B.Cu')]):
        start = (5, 9 + 2.4 * i)
        cx, cy = start[0] + r, start[1]
        mid = polar((cx, cy), 180 - sweep / 2, r)
        end = polar((cx, cy), 180 - sweep, r)
        if layer == 'B.Cu':
            w.via(start, 0.6, 0.3, net=f'N{i}')
        w.arc(start, mid, end, width, layer, f'N{i}')
        w.track([end, (end[0] + 2, end[1])], width, layer, f'N{i}')
    # цепочка «отрезок – дуга – отрезок» по касательной
    w.track([(20, 26), (26, 26)], 0.3, 'F.Cu', 'N0')
    w.arc((26, 26), polar((26, 24), 45, 2), (28, 24), 0.3, 'F.Cu', 'N0')
    w.track([(28, 24), (28, 18)], 0.3, 'F.Cu', 'N0')
    w.graphic('arc', 'F.Cu', (30, 5), (33, 8), (36, 5), width=0.4, net='N1')
    w.graphic('arc', 'F.Mask', (30, 12), (33, 15), (36, 12), width=0.6)
    w.graphic('circle', 'B.Cu', (33, 20), (34.5, 20), width=0.3)
    return w


def t03_vias():
    w = KicadWriter('t03_vias', copper=6, clearance=0.15, mask_margin=0.05, tenting=(False, True))
    w.outline_rect(0, 0, 40, 24)
    kinds = [(None, ('F.Cu', 'B.Cu'), 0.6, 0.3), ('blind', ('F.Cu', 'In2.Cu'), 0.5, 0.2), ('blind', ('In3.Cu', 'B.Cu'), 0.5, 0.2),
             ('blind', ('In1.Cu', 'In4.Cu'), 0.5, 0.2), ('micro', ('F.Cu', 'In1.Cu'), 0.3, 0.1),
             ('micro', ('In4.Cu', 'B.Cu'), 0.3, 0.1)]
    fp = w.footprint('J1', (4, 12))
    for i, (kind, lay, size, drill) in enumerate(kinds):
        net = f'V{i}'
        fp.pad(str(i + 1), 'thru_hole', 'circle', (0, -7.5 + 3 * i), 1.5, drill=0.8, net=net)
        x, y = 12 + 4 * i, 4.5 + 3 * i
        w.via((x, y), size, drill, layers=lay, kind=kind, net=net, tenting=(True, False) if i == 3 else None)
        w.track([(4, y), (x, y)], 0.2, lay[0] if lay[0] in ('F.Cu', 'B.Cu') else lay[0], net)
        w.track([(x, y), (x, y + 2), (36, y + 2)], 0.2, lay[1], net)
    w.via((36, 20), 0.6, 0.3, net='V0')           # одиночное сквозное, открытое сверху
    return w


def t04_padstacks():
    w = KicadWriter('t04_padstacks', copper=4, clearance=0.15, mask_margin=0.05)
    w.outline_rect(0, 0, 36, 20)
    fp = w.footprint('U1', (18, 10), angle=15, attrs='through_hole')
    fp.pad('1', 'thru_hole', 'circle', (-12, 4), 1.8, drill=0.9, net='A',
           padstack=('front_inner_back', {'Inner': dict(shape='circle', size=1.2), 'B.Cu': dict(shape='rect', size=(2.0, 1.5))}))
    fp.pad('2', 'thru_hole', 'circle', (-8, 4), 1.7, drill=0.8, net='A',
           padstack=('custom', {'In1.Cu': dict(shape='roundrect', size=(1.4, 1.1), rratio=0.25),
                                'In2.Cu': dict(shape='oval', size=(1.2, 1.9)), 'B.Cu': dict(shape='circle', size=2.1)}))
    # заданные не полностью: только размер (форма — от основной?) и только форма (размер — от основной?)
    fp.pad('3', 'thru_hole', 'rect', (-4, 4), (1.6, 1.6), drill=0.8, net='B',
           padstack=('front_inner_back', {'Inner': dict(size=1.0), 'B.Cu': dict(size=(2.2, 1.4))}))
    fp.pad('4', 'thru_hole', 'roundrect', (0, 4), (1.8, 1.4), drill=0.8, rratio=0.3, net='B',
           padstack=('custom', {'In1.Cu': dict(shape='circle'), 'In2.Cu': dict(size=(1.2, 1.2)),
                                'B.Cu': dict(shape='oval', size=(1.4, 2.2))}))
    # доля скругления и срез углов слоя не заданы: берутся ли они от основной формы?
    fp.pad('6', 'thru_hole', 'circle', (8, 4), 1.6, drill=0.7, net='C',
           padstack=('custom', {'In1.Cu': dict(shape='roundrect', size=(1.6, 1.2)),
                                'In2.Cu': dict(shape='roundrect', size=(1.2, 1.6), rratio=0.4),
                                'B.Cu': dict(shape='rect', size=(1.8, 1.8))}))
    fp.pad('7', 'thru_hole', 'roundrect', (12, 4), (1.8, 1.4), drill=0.7, rratio=0.1, net='C',
           padstack=('custom', {'In1.Cu': dict(shape='roundrect', size=(1.6, 1.2)),
                                'In2.Cu': dict(shape='circle', size=1.4), 'B.Cu': dict(shape='roundrect', size=(1.8, 1.4))}))
    fp.pad('5', 'thru_hole', 'oval', (4, 4), (1.2, 2.0), drill=(0.6, 1.2), net='C',
           padstack=('front_inner_back', {'Inner': dict(shape='oval', size=(1.0, 1.6)), 'B.Cu': dict(shape='oval', size=(1.5, 2.4))}))
    # смещение формы: основное задано, у слоёв — нет (In1) и своё (B.Cu)
    fp.pad('8', 'thru_hole', 'circle', (-12, -2), 1.6, drill=0.7, offset=(0.3, 0), net='A',
           padstack=('custom', {'In1.Cu': dict(shape='circle', size=1.2), 'In2.Cu': dict(shape='circle', size=1.3),
                                'B.Cu': dict(shape='circle', size=1.6, offset=(-0.3, 0))}))
    fp.pad('9', 'thru_hole', 'rect', (-8, -2), (1.6, 1.6), drill=0.7, net='A',
           padstack=('front_inner_back', {'Inner': dict(shape='circle', size=1.1)}))      # B.Cu не задан
    # переходные: падстек задан не полностью (какой размер получат незаданные слои?)
    for i, (mode, sizes) in enumerate([('front_inner_back', {'Inner': 0.45, 'B.Cu': 0.9}),
                                       ('custom', {'In1.Cu': 0.4, 'In2.Cu': 0.55, 'B.Cu': 0.7}),
                                       ('front_inner_back', {'Inner': 0.5}),
                                       ('front_inner_back', {'B.Cu': 0.9}),
                                       ('custom', {'In1.Cu': 0.4})]):
        w.via((6 + 5 * i, 3), 0.7, 0.3, net='C', padstack=(mode, sizes))
    return w


def t05_unused():
    w = KicadWriter('t05_unused', copper=4, clearance=0.15)
    w.outline_rect(0, 0, 40, 24)
    fp = w.footprint('J1', (3, 12))
    for i, net in enumerate(('D', 'E', 'G', 'H')):
        fp.pad(str(i + 1), 'thru_hole', 'circle', (0, -6 + 4 * i), 1.6, drill=0.8, net=net)
    fp2 = w.footprint('U1', (14, 12), attrs='through_hole')
    fp2.pad('1', 'thru_hole', 'circle', (0, 4), 1.6, drill=0.8, net='D', remove_unused=True, keep_end=True)
    fp2.pad('2', 'thru_hole', 'circle', (0, 0), 1.6, drill=0.8, net='E', remove_unused=True, keep_end=False)
    fp2.pad('3', 'thru_hole', 'rect', (0, -4), 1.6, drill=0.8, net='G', remove_unused=True, keep_end=False)
    w.track([(3, 6), (14, 16)], 0.25, 'In1.Cu', 'D')                 # конец дорожки в выводе 1 на In1
    w.track([(3, 10), (14, 12)], 0.25, 'B.Cu', 'E')                  # вывод 2: только B.Cu
    # переходные: конец дорожки, дорожка через центр, зона (подключение определит KiCad), ничего
    w.via((24, 18), 0.8, 0.4, net='D', remove_unused=True, keep_end=True)
    w.track([(24, 18), (24, 22)], 0.25, 'In2.Cu', 'D')
    w.track([(3, 6), (24, 18)], 0.25, 'F.Cu', 'D')
    w.via((28, 18), 0.8, 0.4, net='D', remove_unused=True, keep_end=False)
    w.track([(26, 18), (30, 18)], 0.2, 'In1.Cu', 'D')
    w.via((24, 6), 0.8, 0.4, net='G', remove_unused=True, keep_end=False)
    w.via((28, 6), 0.8, 0.4, net='G', remove_unused=True, keep_end=True)
    w.via((32, 6), 0.8, 0.4, net='H', remove_unused=True, keep_end=False)   # ни к чему не подключено
    w.track([(3, 14), (20, 14)], 0.25, 'F.Cu', 'G')
    w.zone('G', ['In2.Cu'], [(20, 1), (39, 1), (39, 10), (20, 10)], clearance=0.3)
    w.zone('G', ['In2.Cu'], [(8, 1), (19, 1), (19, 10), (8, 10)], clearance=0.3)  # подключит вывод 3
    return w


def t06_zones():
    w = KicadWriter('t06_zones', copper=4, clearance=0.2)
    w.outline_rect(0, 0, 50, 34)
    fp = w.footprint('J1', (3, 17))
    for i, net in enumerate(('GND', 'P1', 'P2', 'P3', 'S')):
        fp.pad(str(i + 1), 'thru_hole', 'circle', (0, -8 + 4 * i), 1.6, drill=0.8, net=net)
    fp2 = w.footprint('U1', (14, 24), attrs='through_hole')
    fp2.pad('1', 'thru_hole', 'circle', (-3, 0), 1.6, drill=0.8, net='GND')                     # термобарьер
    fp2.pad('2', 'thru_hole', 'circle', (0, 0), 1.6, drill=0.8, net='GND', zone_connect=2)      # сплошное
    fp2.pad('3', 'thru_hole', 'circle', (3, 0), 1.6, drill=0.8, net='GND', zone_connect=0)      # без подключения
    fp2.pad('4', 'smd', 'rect', (0, 3), (1.2, 0.8), net='GND')
    fp2.pad('5', 'thru_hole', 'circle', (6, 0), 1.6, drill=0.8, net='S')
    w.zone('GND', ['F.Cu'], [(8, 18), (26, 18), (26, 31), (8, 31)], connect='thermal', thermal_gap=0.4, thermal_width=0.35)
    w.zone('GND', ['In1.Cu'], [(1, 1), (49, 1), (49, 33), (1, 33)], connect='solid')
    w.zone('P1', ['In2.Cu'], [(1, 1), (24, 1), (24, 33), (1, 33)], priority=1, connect='thermal')
    w.zone('P2', ['In2.Cu'], [(20, 1), (49, 1), (49, 33), (20, 33)], priority=0)                # уступает P1
    w.zone('P3', ['B.Cu'], [(28, 2), (48, 2), (48, 16), (28, 16)], hatch=(0.4, 1.0))            # сетка
    w.zone(None, ['B.Cu'], [(30, 20), (47, 20), (47, 32), (30, 32)], keepout=None, island_removal=1)
    w.zone(None, ['B.Cu'], [(36, 24), (42, 24), (42, 28), (36, 28)],
           keepout=dict(tracks=True, vias=True, pads=True, copperpour=True, footprints=False))  # вырез
    # островок: заливка цепи S, рассечённая дорожкой GND, островки оставлены
    w.zone('S', ['F.Cu'], [(30, 4), (46, 4), (46, 14), (30, 14)], island_removal=1)
    w.track([(38, 3), (38, 15)], 0.6, 'F.Cu', 'P1')
    w.track([(3, 13), (24, 13), (38, 3)], 0.25, 'F.Cu', 'P1')
    w.track([(9, 17), (30, 9)], 0.3, 'F.Cu', 'S')
    w.via((44, 6), 0.6, 0.3, net='P3')
    fp3 = w.footprint('Q1', (40, 26), attrs='smd')
    fp3.pad('1', 'smd', 'rect', (0, 0), (3, 3), net='P2')
    fp3.zone('P2', ['F.Cu'], [(37, 23), (43, 23), (43, 29), (37, 29)])                          # зона в компоненте
    return w


def t07_outline():
    w = KicadWriter('t07_outline', clearance=0.2)
    pts = [(0, 0), (30, 0), (30, 12), (26, 12), (26, 16), (30, 16), (30, 28), (0, 28)]
    w.outline_polygon(pts)
    w.outline_circle((8, 8), 2.5)
    w.graphic('rect', 'Edge.Cuts', (14, 4), (22, 9), width=0.05, radius=1.5)
    w.graphic('poly', 'Edge.Cuts', (6, 18), (12, 18), (9, 23), width=0.05)
    fp = w.footprint('J1', (20, 21), angle=30)
    fp.graphic('line', 'Edge.Cuts', (-2, 1), (2, 1), width=0.05)
    fp.graphic('arc', 'Edge.Cuts', (2, 1), (3, 0), (2, -1), width=0.05)
    fp.graphic('line', 'Edge.Cuts', (2, -1), (-2, -1), width=0.05)
    fp.graphic('arc', 'Edge.Cuts', (-2, -1), (-3, 0), (-2, 1), width=0.05)
    fp.pad('1', 'thru_hole', 'oval', (0, 4), (1.6, 3.0), drill=(1.0, 2.2), net='A')
    fp.pad('2', 'smd', 'rect', (0, -4), (1.0, 1.0), net='A')
    w.graphic('rect', 'Edge.Cuts', (34, 0), (44, 10), width=0.05)          # вторая плата в заготовке
    f2 = w.footprint('R1', (39, 5))
    f2.pad('1', 'smd', 'rect', (-1, 0), (1, 1), net='B')
    f2.pad('2', 'smd', 'rect', (1, 0), (1, 1), net='B')
    w.track([(38, 5), (40, 5)], 0.3, 'F.Cu', 'B')
    return w


def t08_drc():
    """Нарушения зазоров при пороге 0,2 мм (класс Default) и 0,3 мм до края платы. Каждый случай —
    в своём ряду (ряд i: площадка разъёма J1 и цепь S<i>), чтобы случаи не задевали друг друга:
      A  S0–S1   дорожка–дорожка 0,15 (нарушение)        B  S2–S3   дорожка–дорожка 0,25 (нет)
      C  S4–S5   площадка–дорожка 0,10                     D  S6–S7   площадка–площадка 0,12
      E  S8–S9   переходное–дорожка 0,18                   F  S10–S11 дуга–дорожка 0,10
      G  S12     до края платы 0,20                         H  S13–S14 дорожка заходит на край площадки
      I  S15–S16 конец дорожки внутри чужой площадки     J  S17–S18 пересечение дорожек
      K  S19     графика без цепи в 0,10 от дорожки      L  S2–S3   переходные на B.Cu, 0,22 (нет)
    K DRC KiCad 10 не проверяет (медная графика без цепи) — см. t08_drc.json."""
    W, H = 40, 42
    w = KicadWriter('t08_drc', clearance=0.2, edge_clearance=0.3)
    w.outline_rect(0, 0, W, H)
    y = lambda i: 2 + 2 * i
    fp = w.footprint('J1', (2, 21))
    for i in range(20):
        fp.pad(str(i + 1), 'smd', 'rect', (0, y(i) - 21), (1.0, 1.0), net=f'S{i}')
        w.track([(2.5, y(i)), (6, y(i))], 0.25, 'F.Cu', f'S{i}')
    t = lambda i, *pts: w.track([(6, y(i)), *pts], 0.25, 'F.Cu', f'S{i}')
    # A, B
    t(0, (16, y(0)))
    t(1, (8, y(0) + 0.40), (16, y(0) + 0.40))
    t(2, (16, y(2)))
    t(3, (8, y(2) + 0.50), (16, y(2) + 0.50))
    # C: площадка R1 (S4) 1,2 × 1,2, дорожка S5 над ней с зазором 0,1
    w.footprint('R1', (20, y(4))).pad('1', 'smd', 'rect', (0, 0), 1.2, net='S4')
    t(4, (20, y(4)))
    t(5, (15, y(5)), (15, y(4) + 0.825), (25, y(4) + 0.825))
    # D: площадки разных компонентов с зазором 0,12
    w.footprint('R2', (28, y(6) + 1)).pad('1', 'smd', 'rect', (0, 0), 1.2, net='S6')
    w.footprint('R3', (29.32, y(6) + 1)).pad('1', 'smd', 'rect', (0, 0), 1.2, net='S7')
    t(6, (28, y(6)), (28, y(6) + 1))
    t(7, (29.32, y(7)), (29.32, y(6) + 1))
    # E: переходное S8 Ø0,6 и дорожка S9 в 0,18 от него
    w.via((12, y(8)), 0.6, 0.3, net='S8')
    t(8, (12, y(8)))
    t(9, (10, y(9)), (10, y(8) + 0.605), (14, y(8) + 0.605))
    # F: дуга S10 (стрелка 0,8 над хордой 4) и дорожка S11 в 0,1 от её вершины
    w.arc((6, y(10)), (8, y(10) + 0.8), (10, y(10)), 0.25, 'F.Cu', 'S10')
    t(11, (12, y(11)), (12, y(10) + 1.15), (7, y(10) + 1.15))
    # G: 0,2 до правого края
    t(12, (W - 0.2 - 0.125, y(12)))
    # H: дорожка S13 заходит на 0,025 мм на площадку R4 (S14)
    w.footprint('R4', (14, y(13) + 0.6)).pad('1', 'smd', 'rect', (0, 0), 1.0, net='S14')
    t(13, (20, y(13)))
    t(14, (14, y(14)), (14, y(13) + 0.6))
    # I: конец дорожки S15 внутри площадки R5 (S16)
    w.footprint('R5', (20, y(15) + 1)).pad('1', 'smd', 'rect', (0, 0), 1.2, net='S16')
    t(15, (19.8, y(15)), (19.8, y(15) + 0.5))
    t(16, (25, y(16)), (25, y(15) + 1), (20, y(15) + 1))
    # J: пересечение дорожек
    t(17, (30, y(17)))
    t(18, (20, y(18)), (20, y(17) - 0.8))
    # K: графика без цепи в 0,1 от дорожки S19
    t(19, (20, y(19)))
    w.graphic('poly', 'F.Cu', (8, y(19) + 0.225), (14, y(19) + 0.225), (14, y(19) + 1.0), (8, y(19) + 1.0), width=0, fill=True)
    # L: переходные на B.Cu с зазором 0,22 — нарушения нет
    w.via((32, 4), 0.6, 0.3, net='S2')
    w.via((32, 4 + 0.6 + 0.22), 0.6, 0.3, net='S3')
    return w


T08_KNOWN = {
    'drc_known': {},
    'notes': {
        'графика без цепи': 'DRC KiCad 10.0 не проверяет зазоры и касания медной графики без цепи на плате '
                            '(случай K); в сверку такие пары не входят и показываются отдельно',
    },
}


# ---------- демо-платы документов ----------

def b1_demo():
    """Демо-плата B1 (16 × 11 мм): та же геометрия, что в research/b1/geom.py, но объектами KiCad."""
    w = KicadWriter('b1_demo', clearance=0.15, track_width=0.15)
    w.outline_rect(0, 0, 16, 11)
    W = 0.15
    labels = {}
    n_fp = [0]

    def pad_rect(net, cx, cy, sx, sy):
        n_fp[0] += 1
        w.footprint(f'P{n_fp[0]}', (cx, cy)).pad('1', 'smd', 'rect', (0, 0), (sx, sy), net=net)

    def pad_circ(net, cx, cy, d):
        n_fp[0] += 1
        w.footprint(f'P{n_fp[0]}', (cx, cy)).pad('1', 'smd', 'circle', (0, 0), d, net=net)

    def poly_pad(net, pts):
        n_fp[0] += 1
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        w.footprint(f'P{n_fp[0]}', (cx, cy)).pad('1', 'smd', 'custom', (0, 0), (0.05, 0.05), net=net, anchor='rect',
                                                 primitives=[('poly', [(x - cx, y - cy) for x, y in pts], 0, True)])

    # Л1: трасса к прямоугольной площадке под острым углом (25°)
    pad_rect('S1', 2.0, 9.8, 1.6, 1.0)
    E = (1.5, 9.38)
    S = polar(E, -18, 2.4)
    w.track([E, S, (S[0], 7.6)], W, 'F.Cu', 'S1')
    labels['Л1'] = (2.09, 9.3)
    # Л2: ответвление под острым углом (30°)
    w.track([(4.4, 1.0), (8.0, 1.0)], W, 'F.Cu', 'S2')
    pad_rect('S2', 8.0, 1.0, 0.8, 0.8)
    b0 = (5.2, 1.0)
    b1 = polar(b0, 30, 2.0)
    w.track([b0, b1, (8.0, b1[1])], W, 'F.Cu', 'S2')
    labels['Л2'] = b0
    # Л3: «хвост» за точкой стыка
    w.track([(10.6, 10.4), (13.6, 10.4)], W, 'F.Cu', 'S3')
    j = (13.2, 10.4)
    k = polar(j, -45, 1.2)
    w.track([j, k, (k[0], 8.6)], W, 'F.Cu', 'S3')
    pad_rect('S3', k[0], 8.3, 0.9, 0.9)
    labels['Л3'] = j
    # заливка GND с чужими переходными (зазор 0,25)
    foreign = [(12.0, 1.6), (13.4, 1.6), (14.6, 3.2), (12.2, 4.6)]
    for x, y in foreign:
        w.via((x, y), 0.6, 0.3, net='S4')
    w.track([foreign[0], foreign[1]], 0.15, 'F.Cu', 'S4')
    w.zone('GND', ['F.Cu'], [(10.4, 0.5), (15.5, 0.5), (15.5, 6.2), (10.4, 6.2)], clearance=0.25, min_thickness=0.1,
           connect='solid')
    # Л4: вход трассы в заливку под острым углом (25°)
    e = (10.4, 2.2)
    s = polar(e, 245, 1.8)
    w.track([(10.7, 2.2), e, s, (9.3, s[1])], 0.3, 'F.Cu', 'GND')
    pad_circ('GND', 9.3, s[1], 0.6)
    labels['Л4'] = e
    # Л5: две трассы из одной точки (веер), угол 22°
    v = (7.4, 6.4)
    pad_circ('S5', *v, 0.6)
    a1 = polar(v, 172, 2.0)
    a2 = polar(v, 187, 1.8)
    w.track([v, a1], W, 'F.Cu', 'S5')
    w.track([v, a2, (a2[0], 5.0)], W, 'F.Cu', 'S5')
    labels['Л5'] = v
    # Л6: трасса к круглой площадке со смещением (ложный острый угол)
    for i, (cx, cy, dirn) in enumerate([(9.2, 5.2, -1), (11.4, 8.6, 1)]):
        pad_circ(f'S6{i}', cx, cy, 0.9)
        w.track([(cx, cy - 0.3), (cx + dirn * 1.6, cy - 0.3)], W, 'F.Cu', f'S6{i}')
    labels['Л6'] = (9.2, 5.2)
    # Л7: шип — медный многоугольник с острым выступом (28°)
    poly_pad('S7', [(2.2, 5.4), (3.6, 5.4), (3.6, 6.2), (2.2, 6.2), (0.6, 5.85)])
    w.track([(3.6, 5.8), (4.3, 5.8)], W, 'F.Cu', 'S7')
    labels['Л7'] = (0.6, 5.85)
    # Л8: шип, обращённый к чужой высоковольтной цепи
    poly_pad('HV1', [(0.6, 2.4), (2.0, 2.4), (2.0, 2.55), (3.1, 2.8), (2.0, 3.05), (2.0, 3.2), (0.6, 3.2)])
    w.track([(3.75, 1.6), (3.75, 4.0)], 0.4, 'F.Cu', 'HV2')
    pad_circ('HV2', 3.75, 1.6, 0.6)
    labels['Л8'] = (3.1, 2.8)
    # фон: безвредная разводка
    for i in range(4):
        pad_rect(f'U{i}', 5.2 + i * 1.27, 9.8, 0.6, 1.5)
    w.track([(5.2, 9.8), (5.2, 8.6), (4.8, 8.2)], W, 'F.Cu', 'U0')
    w.track([(6.47, 9.8), (6.47, 8.4)], W, 'F.Cu', 'U1')
    w.track([(7.74, 9.8), (7.74, 8.4)], W, 'F.Cu', 'U2')
    w.track([(9.01, 9.8), (9.01, 8.7), (9.5, 8.2)], W, 'F.Cu', 'U3')
    w.track([(4.6, 3.6), (8.0, 3.6)], W, 'F.Cu', 'S9')                # Т-стык 90°
    w.track([(6.0, 3.6), (6.0, 4.3)], W, 'F.Cu', 'S9')
    pad_circ('S9', 8.0, 3.6, 0.6)
    pad_circ('V1', 15.0, 7.0, 0.6)                                     # переходное с трассой по центру
    w.track([(15.0, 7.0), (14.0, 7.0)], W, 'F.Cu', 'V1')
    for name, (x, y) in labels.items():
        w.text(name, (x, y + 0.35), 'Cmts.User', size=0.3, thickness=0.05)
    return w, {'labels': {k: list(v) for k, v in labels.items()},
               'source': 'research/b1/geom.py:demo_board', 'units': 'мм, ось Y вверх'}


def a7_demo():
    """Демо-плата A7 (14 × 9 мм): research/a7/a7geom.py:demo(), окна маски — средствами KiCad."""
    w = KicadWriter('a7_demo', clearance=0.1, track_width=0.12, mask_margin=0.05)
    w.outline_rect(0, 0, 14, 9)
    W = 0.15
    fps = {}

    def pad(ref, net, shape, cx, cy, sx, sy=None, e=None):
        sy = sx if sy is None else sy
        if ref not in fps:
            fps[ref] = (w.footprint(ref, (0, 0)), [0])
        fp, n = fps[ref]
        n[0] += 1
        if shape == 'rect':
            fp.pad(str(n[0]), 'smd', 'rect', (cx, cy), (sx, sy), net=net, mask_margin=e)
        else:
            fp.pad(str(n[0]), 'smd', 'circle', (cx, cy), sx, net=net, mask_margin=e)

    situations = {}
    for i in range(8):
        pad('U1', f'U1_{i}', 'rect', 1.0 + i * 0.65, 7.6, 0.30, 1.2)
        pad('U2', f'U2_{i}', 'rect', 1.0 + i * 0.40, 5.9, 0.20, 0.7)
        pad('U3', f'U3_{i}', 'rect', 1.0 + i * 0.35, 4.5, 0.18, 0.6)
    situations.update({'шаг 0,65': (1.0, 7.6), 'шаг 0,40': (1.0, 5.9), 'шаг 0,35': (1.0, 4.5)})
    bx, by = 8.0, 4.4
    for r in range(4):
        for c in range(4):
            x, y = bx + c * 0.8, by + r * 0.8
            if (r, c) == (1, 2):
                pad('U4', f'B{r}{c}', 'circ', x, y, 0.50, e=-0.05)          # SMD-площадка BGA
            else:
                pad('U4', f'B{r}{c}', 'circ', x, y, 0.40)
    situations['BGA'] = (bx, by)
    for r, c, tent, od, off in [(0, 0, 'full', 0.55, (0, 0)), (0, 1, 'full', 0.55, (0, 0)), (1, 0, 'full', 0.55, (0, 0)),
                                (2, 1, 'open', 0.55, (0, 0)), (2, 2, 'partial', 0.35, (0.12, 0.0)), (3, 3, 'full', 0.55, (0, 0))]:
        x, y = bx + c * 0.8 + 0.4, by + r * 0.8 + 0.4
        w.via((x, y), 0.45, 0.2, net=f'B{r}{c}', tenting=(tent != 'open', True))
        if tent == 'partial':                    # частичное окно — отдельной графикой на маске
            w.graphic('circle', 'F.Mask', (x + off[0], y + off[1]), (x + off[0] + od / 2, y + off[1]), width=0, fill=True)
        w.track([(bx + c * 0.8, by + r * 0.8), (x, y)], 0.12, 'F.Cu', f'B{r}{c}')
    pad('C1', 'N1', 'rect', 1.55, 2.0, 0.5, 0.55)
    pad('C1', 'GND', 'rect', 2.45, 2.0, 0.5, 0.55)
    w.via((3.0, 2.0), 0.45, 0.2, net='GND', tenting=(False, True))
    w.track([(2.45, 2.0), (3.0, 2.0)], 0.2, 'F.Cu', 'GND')
    w.track([(1.55, 2.0), (0.8, 2.0), (0.8, 1.0)], W, 'F.Cu', 'N1')
    situations['C1: переходное вплотную, открыто'] = (3.0, 2.0)
    pad('C2', 'N2', 'rect', 4.55, 2.0, 0.5, 0.55)
    pad('C2', 'GND', 'rect', 5.45, 2.0, 0.5, 0.55)
    w.via((6.25, 2.0), 0.45, 0.2, net='GND', tenting=(True, True))
    w.track([(5.45, 2.0), (6.25, 2.0)], 0.2, 'F.Cu', 'GND')
    w.track([(4.55, 2.0), (3.9, 2.0), (3.9, 1.0)], W, 'F.Cu', 'N2')
    situations['C2: правильно'] = (6.25, 2.0)
    pad('J1', 'CONN', 'rect', 12.2, 7.5, 1.6, 1.6, e=0.15)
    w.track([(13.175, 6.0), (13.175, 8.7)], W, 'F.Cu', 'SIG')
    situations['J1: окно 0,15 и чужая трасса'] = (12.2, 7.5)
    pad('R1', 'N3', 'rect', 6.6, 7.6, 0.8, 0.9, e=0.0)
    pad('R1', 'N4', 'rect', 7.9, 7.6, 0.8, 0.9, e=0.0)
    situations['R1: без расширения'] = (6.6, 7.6)
    pad('TP1', 'TP', 'circ', 13.4, 1.0, 1.0)
    situations['TP1: у края'] = (13.4, 1.0)
    w.via((9.0, 1.4), 0.45, 0.2, net='N5', tenting=(False, True))
    w.via((9.6, 1.4), 0.45, 0.2, net='N6', tenting=(False, True))
    w.track([(9.0, 1.4), (9.0, 0.5)], W, 'F.Cu', 'N5')
    w.track([(9.6, 1.4), (9.6, 0.5)], W, 'F.Cu', 'N6')
    situations['пара открытых переходных'] = (9.3, 1.4)
    return w, {'situations': {k: list(v) for k, v in situations.items()},
               'source': 'research/a7/a7geom.py:demo', 'units': 'мм, ось Y вверх'}


def a3_demo():
    """Демо-плата A3 (14 × 9 мм, 4 слоя): research/a3/figs_a3.py:VIAS. Стек подобран так, чтобы
    глубины слоёв были как в документе: 0; 0,2; 1,4; 1,6 мм."""
    stack = [('F.Mask', 'Top Solder Mask', 0.01), ('F.Cu', 'copper', 0.035), ('dielectric 1', 'prepreg', 0.165),
             ('In1.Cu', 'copper', 0.035), ('dielectric 2', 'core', 1.165), ('In2.Cu', 'copper', 0.035),
             ('dielectric 3', 'prepreg', 0.165), ('B.Cu', 'copper', 0.035), ('B.Mask', 'Bottom Solder Mask', 0.01)]
    w = KicadWriter('a3_demo', copper=4, thickness=1.655, clearance=0.1, track_width=0.1, stackup=stack)
    w.outline_rect(0, 0, 14, 9)
    situations = {}
    for name, x, y, D in (('V1', 2.0, 7.0, 0.60), ('V2', 4.0, 7.0, 0.45), ('V3', 6.0, 7.0, 0.38),
                          ('V5', 8.0, 7.0, 0.45), ('V6', 10.0, 7.0, 0.45), ('V4', 2.0, 4.5, 0.50)):
        w.via((x, y), D, 0.30, net=name, teardrops=name == 'V6')
        situations[f'{name} {D:.2f}/0,30'] = (x, y)
    # V5 и V6: трасса 0,10 мм на внутреннем слое In1.Cu; у V6 включены капли — их строит сам KiCad
    # при перезаливке (по умолчанию: длина 0,5 D, ширина 1 D; в документе — упрощённая выпуклая оболочка)
    for name, x in (('V5', 8.0), ('V6', 10.0)):
        w.track([(x, 7.0), (x + 1.2, 7.0)], 0.10, 'In1.Cu', name)
    w.track([(2.0, 4.5), (2.0, 3.2)], 0.15, 'F.Cu', 'V4')
    # V7: наружные площадки 0,55, внутренние 0,45 (стек площадки); V8: внутренние удалены
    w.via((4.0, 4.5), 0.55, 0.30, net='V7', padstack=('front_inner_back', {'Inner': 0.45, 'B.Cu': 0.55}))
    w.via((6.0, 4.5), 0.55, 0.30, net='V8', remove_unused=True, keep_end=True)
    situations.update({'V7 0,55/0,45 внутри': (4.0, 4.5), 'V8 без внутренних': (6.0, 4.5)})
    # M1: микроотверстие F.Cu → In1.Cu, 0,10 мм, целевая площадка 0,25 мм
    w.via((9.0, 4.5), 0.25, 0.10, layers=('F.Cu', 'In1.Cu'), net='M1', kind='micro')
    situations['M1 микроотверстие'] = (9.0, 4.5)
    # J2: квадрат 1,2 мм, отверстие 0,8 мм, площадка сдвинута на 0,15 мм (offset); внутренние
    # площадки не подключены и удалены — поясок только на наружных слоях, как в документе
    fp = w.footprint('J2', (12.3, 7.0), attrs='through_hole')
    fp.pad('1', 'thru_hole', 'rect', (0, 0), 1.2, drill=0.8, offset=(0.15, 0.0), net='J2',
           remove_unused=True, keep_end=True)
    situations['J2 offset 0,15'] = (12.3, 7.0)
    fp = w.footprint('J1', (2.0, 2.0), attrs='through_hole')
    for i in range(4):
        fp.pad(str(i + 1), 'thru_hole', 'circle', (i * 2.54, 0), 1.70, drill=1.00, net=f'J1_{i + 1}')
        w.track([(2.0 + i * 2.54, 2.0), (2.0 + i * 2.54, 0.8)], 0.3, 'F.Cu', f'J1_{i + 1}')
    situations['J1 1,70/1,00'] = (2.0, 2.0)
    return w, {'situations': {k: list(v) for k, v in situations.items()},
               'source': 'research/a3/figs_a3.py:VIAS', 'units': 'мм, ось Y вверх'}


def a5_demo():
    """Демо-плата A5 (14 × 9 мм, 4 слоя): research/a5/figs_a5.py. L2 (In1.Cu) — земля, L3 (In2.Cu) —
    +3V3. Вскрытия строит сам KiCad по правилам (от ГОТОВОГО отверстия 0,30 мм: D_a = 0,30 + 2·зазор),
    V6 сдвигается на 0,15 мм ПОСЛЕ заливки (a5_after_refill) — устаревшая заливка."""
    stack = [('F.Mask', 'Top Solder Mask', 0.01), ('F.Cu', 'copper', 0.035), ('dielectric 1', 'prepreg', 0.165),
             ('In1.Cu', 'copper', 0.035), ('dielectric 2', 'core', 1.165), ('In2.Cu', 'copper', 0.035),
             ('dielectric 3', 'prepreg', 0.165), ('B.Cu', 'copper', 0.035), ('B.Mask', 'Bottom Solder Mask', 0.01)]
    w = KicadWriter('a5_demo', copper=4, thickness=1.655, clearance=0.1, track_width=0.15, stackup=stack,
                    rules={'max_error': 0.001})
    w.outline_rect(0, 0, 14, 9)
    plane = [(0.3, 0.3), (13.7, 0.3), (13.7, 8.7), (0.3, 8.7)]
    w.zone('GND', 'In1.Cu', plane, clearance=0.1, min_thickness=0.05, connect='solid', island_removal=1)
    w.zone('+3V3', 'In2.Cu', plane, clearance=0.1, min_thickness=0.05, connect='solid', island_removal=1)
    w.netclasses['BGA'] = {'clearance': 0.1, 'pattern': 'B*'}
    # зазор от готового отверстия: 0,275 → вскрытие 0,85; правила ниже по файлу важнее
    w.dru += ["(rule \"вскрытие\" (condition \"A.Type == 'Via'\") (constraint clearance (min 0.275mm)))",
              "(rule \"V2\" (condition \"A.NetName == 'S2'\") (constraint clearance (min 0.175mm)))",
              "(rule \"V3\" (condition \"A.NetName == 'S3'\") (constraint clearance (min 0.125mm)))",
              "(rule \"V4\" (condition \"A.NetName == 'S4'\") (constraint clearance (min 0.125mm)))",
              "(rule \"поле L3\" (layer \"In2.Cu\") (condition \"A.NetClass == 'BGA'\") "
              "(constraint clearance (min 0.2mm)))"]
    situations = {}
    for name, net, x, kw in (('V1', 'S1', 1.5, {}), ('V2', 'S2', 3.2, {}), ('V3', 'S3', 4.9, {}),
                             ('V5', 'HV', 8.3, {}), ('V6', 'S6', 9.85, {})):
        w.via((x, 7.2), 0.45, 0.30, net=net, remove_unused=True, keep_end=True, **kw)
        situations[name] = (x if name != 'V6' else 10.0, 7.2)
    w.via((6.6, 7.2), 0.60, 0.30, net='S4', remove_unused=False)          # V4: площадки на всех слоях
    situations['V4 с площадкой 0,60'] = (6.6, 7.2)
    for i in range(5):
        for j in range(5):
            x, y = 7.6 + i * 0.8, 1.4 + j * 0.8
            if j in (0, 4):                    # земля поля: глухие F.Cu → In1.Cu, через L3 не проходят
                w.via((x, y), 0.45, 0.30, net='GND', layers=('F.Cu', 'In1.Cu'), kind='blind')
            else:
                w.via((x, y), 0.45, 0.30, net=f'B{i}{j}', remove_unused=True, keep_end=True)
    situations['поле BGA'] = (9.2, 3.0)
    for x, y in ((1.5, 1.5), (3.0, 1.5), (12.8, 7.2)):
        w.via((x, y), 0.45, 0.30, net='GND')
    return w, {'situations': {k: list(v) for k, v in situations.items()},
               'source': 'research/a5/figs_a5.py', 'units': 'мм, ось Y вверх',
               'after_refill': 'V6 сдвинуто с (9,85; 7,2) на (10,0; 7,2)'}


def a5_after_refill(path):
    """V6 сдвигается после заливки: вскрытие остаётся на старом месте (З8)."""
    t = path.read_text(encoding='utf-8')
    assert t.count('(at 9.85 -7.2)') == 1, 'V6 не найдено'
    path.write_text(t.replace('(at 9.85 -7.2)', '(at 10 -7.2)'), encoding='utf-8', newline='\n')


AFTER_REFILL = {'a5_demo': a5_after_refill}


def t09_a1():
    """Дефекты класса A1 с известными размерами (условный завод: зазор 90 мкм снаружи и 73,3 мкм
    внутри, ширина 79,4 и 77,6 мкм). Ряд i — вывод i разъёма J1 и своя цепь:
      Д4  A0–A1 F.Cu 60 мкм; B0–B1 F.Cu 85 мкм (снаружи мало); C0–C1 In1 85 мкм (внутри норма);
          D0–D1 In1 70 мкм; S — два куска ОДНОЙ цепи на F.Cu в 50 мкм (соединены через In1);
      Д1  W F.Cu 0,07 мм;   Д7  PWR F.Cu 0,3 мм (при 2 А нужно ≈ 0,46 мм);
      Д5  HV–LV F.Cu 0,5 мм под маской;   Д8  E0–E1 F.Cu 120 мкм на 33 мм;
      Д3  заливка GND на In2, ряд чужих переходных: шаг 1,0, площадка 0,6, зазор 0,17 → перешейки
          60 мкм; min_thickness зоны 0,05 мм меньше минимума ширины;
      Д2  текст на F.Cu: линии 50 мкм; «88» высотой 0,35 мм при линиях 0,1 мм; линия графики 50 мкм.
    Напряжения, токи и таблицу зазоров для Д5 и Д7 задаёт тестовый профиль (t09_a1_profile.toml)."""
    w = KicadWriter('t09_a1', copper=4, clearance=0.09, track_width=0.07)
    w.outline_rect(0, 0, 50, 37)
    y = lambda i: 2 + 2 * i
    nets = ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1', 'S', 'W', 'PWR', 'HV', 'LV', 'E0', 'E1', 'GND']
    fp = w.footprint('J1', (2, 17), attrs='through_hole')
    for i, net in enumerate(nets):
        fp.pad(str(i + 1), 'thru_hole', 'circle', (0, y(i) - 17), 1.2, drill=0.6, net=net)
    t = lambda net, layer, width, *pts: w.track(list(pts), width, layer, net)
    # Д4
    t('A0', 'F.Cu', 0.2, (2, y(0)), (30, y(0)))
    t('A1', 'F.Cu', 0.2, (2, y(1)), (5, y(1)), (7, y(0) + 0.26), (30, y(0) + 0.26))
    t('B0', 'F.Cu', 0.2, (2, y(2)), (30, y(2)))
    t('B1', 'F.Cu', 0.2, (2, y(3)), (5, y(3)), (7, y(2) + 0.285), (30, y(2) + 0.285))
    t('C0', 'In1.Cu', 0.2, (2, y(4)), (30, y(4)))
    t('C1', 'In1.Cu', 0.2, (2, y(5)), (5, y(5)), (7, y(4) + 0.285), (30, y(4) + 0.285))
    t('D0', 'In1.Cu', 0.2, (2, y(6)), (30, y(6)))
    t('D1', 'In1.Cu', 0.2, (2, y(7)), (5, y(7)), (7, y(6) + 0.27), (30, y(6) + 0.27))
    t('S', 'F.Cu', 0.2, (2, y(8)), (20, y(8)))                       # кусок 1
    t('S', 'F.Cu', 0.2, (20, y(8) + 0.25), (30, y(8) + 0.25))        # кусок 2: 50 мкм до куска 1
    w.via((30, y(8) + 0.25), 0.6, 0.3, net='S')
    t('S', 'In1.Cu', 0.2, (2, y(8)), (4, y(8) + 1), (30, y(8) + 1), (30, y(8) + 0.25))
    # Д1, Д7, Д5, Д8
    t('W', 'F.Cu', 0.07, (2, y(9)), (30, y(9)))
    t('PWR', 'F.Cu', 0.3, (2, y(10)), (30, y(10)))
    t('HV', 'F.Cu', 0.3, (2, y(11)), (30, y(11)))
    t('LV', 'F.Cu', 0.3, (2, y(12)), (5, y(12)), (7, y(11) + 0.8), (30, y(11) + 0.8))
    t('E0', 'F.Cu', 0.2, (2, y(13)), (40, y(13)))
    t('E1', 'F.Cu', 0.2, (2, y(14)), (5, y(14)), (7, y(13) + 0.32), (40, y(13) + 0.32))
    # Д3: заливка GND на In2 и ряд чужих переходных SIG
    for k in range(16):
        w.via((32.5 + k, 12), 0.6, 0.3, net='SIG')
    t('GND', 'In2.Cu', 0.3, (2, y(15)), (40, y(15)), (40, 21))
    w.zone('GND', ['In2.Cu'], [(32, 2), (48, 2), (48, 22), (32, 22)], clearance=0.17, min_thickness=0.05,
           connect='solid')
    # Д2
    w.text('TEST', (10, 35), 'F.Cu', size=0.3, thickness=0.05)
    w.text('88', (20, 35), 'F.Cu', size=0.35, thickness=0.1)
    w.graphic('line', 'F.Cu', (30, 35), (35, 35), width=0.05)
    return w


T09_PROFILE = '''# Тестовый профиль для t09_a1: напряжения, токи и таблица зазоров — ТЕСТОВЫЕ числа,
# не IPC-2221C Table 6-1 (они нужны только чтобы проверить работу Д5 и Д7).
[nets]
voltages = { HV = 300.0, LV = 0.0 }
currents = { PWR = 2.0 }
temperature_rise_c = 10.0
voltage_table = { B1 = [[400, 0.4]], B2 = [[400, 1.5]], B4 = [[400, 0.8]] }
'''


TEST_BOARDS = [t01_pads37, t02_arcs, t03_vias, t04_padstacks, t05_unused, t06_zones, t07_outline, t08_drc, t09_a1]
DEMO_BOARDS = [b1_demo, a7_demo, a3_demo, a5_demo]


def main(refill: bool = True):
    made = []
    for make in TEST_BOARDS:
        w = make()
        made.append(w.write(TEST / f'{w.name}.kicad_pcb'))
    (TEST / 't09_a1_profile.toml').write_text(T09_PROFILE, encoding='utf-8')
    (TEST / 't08_drc.json').write_text(json.dumps(T08_KNOWN, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for make in DEMO_BOARDS:
        w, info = make()
        p = w.write(DEMO / f'{w.name}.kicad_pcb')
        p.with_suffix('.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        made.append(p)
    if refill:
        for p in made:
            if '(zone' in p.read_text(encoding='utf-8'):
                kicad_cli.refill_zones(p)
                for extra in (p.with_suffix('.kicad_prl'),):
                    extra.unlink(missing_ok=True)
            if p.stem in AFTER_REFILL:
                AFTER_REFILL[p.stem](p)
    for p in made:
        print(p.relative_to(ROOT.parent) if p.is_relative_to(ROOT.parent) else p)
    return made


if __name__ == '__main__':
    main('--no-refill' not in sys.argv)
