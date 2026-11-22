"""Эталонное чтение Gerber X2 от KiCad в многоугольники shapely (только для проверок).

Это независимый от модели путь: KiCad сам рисует плату в Gerber, а здесь его рисунок
переводится в геометрию и сравнивается с тем, что построил адаптер .kicad_pcb.
Разбирается то подмножество Gerber, которое выдаёт KiCad:
  * апертуры C, R, O, P и макросы (примитивы 1, 4, 5, 20, 21, выражения с $n);
  * вспышки D03, рисование D01 круглой апертурой по прямой и по дуге (G01/G02/G03, G75);
  * регионы G36/G37, полярность LPD/LPC;
  * атрибуты X2: функция апертуры (TA.AperFunction), цепь (TO.N), вывод (TO.P), компонент (TO.C).
Всё остальное вызывает ошибку — чтобы незнакомая конструкция не прошла незамеченной.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

TOL = 0.0002          # точность замены дуг ломаными, мм

_ESC = re.compile(r'\\u([0-9A-Fa-f]{4})')
_OP = re.compile(r'(G0?[123]|G3[67]|G7[45])?((?:[XYIJ][-+]?\d+)*)(D0?[123])?\Z')
_COORD = re.compile(r'([XYIJ])([-+]?\d+)')
_EXPR_OK = re.compile(r'[0-9.+\-*/() ]*\Z')


def _unescape(s):
    return _ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def _quad_segs(r, tol=TOL):
    if r <= tol:
        return 2
    return max(int(math.ceil((math.pi / 2) / (2 * math.acos(1 - tol / r)))), 2)


def _disc(x, y, r):
    return Point(x, y).buffer(r, quad_segs=_quad_segs(r))


def _valid(p):
    if p.is_valid:
        return p
    g = shapely.make_valid(p)
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    parts = [q for q in shapely.get_parts(g) if q.geom_type in ('Polygon', 'MultiPolygon')]
    return shapely.union_all(parts) if parts else Polygon()


def _rot(pts, deg):
    if not deg:
        return pts
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


@dataclass
class GObj:
    """Один графический объект Gerber."""
    kind: str                 # flash | draw | region
    geom: object              # многоугольник shapely
    dark: bool
    func: str | None          # TA.AperFunction: Conductor, ViaPad, SMDPad,CuDef, NonConductor, Profile …
    net: str | None           # TO.N
    ref: str | None           # компонент (из TO.P или TO.C)
    pin: str | None           # номер вывода (TO.P)
    line: object = None       # осевая линия для draw (LineString)
    pos: tuple | None = None  # точка вспышки


class _Macro:
    def __init__(self, name, blocks):
        self.name = name
        self.blocks = blocks

    @staticmethod
    def _eval(expr, var):
        # значения подставляются без экспоненты (1e-05 не прошло бы проверку ниже)
        e = re.sub(r'\$(\d+)', lambda m: f'({float(var.get(int(m.group(1)), 0.0)):.15f})', expr)
        e = e.replace('x', '*').replace('X', '*')
        if not _EXPR_OK.match(e):
            raise ValueError(f'выражение макроса не разобрано: {expr!r}')
        return float(eval(e, {'__builtins__': {}}, {}))      # только числа и + − × ÷, см. проверку выше

    def build(self, params):
        var = {i + 1: v for i, v in enumerate(params)}
        geom = Polygon()
        for b in self.blocks:
            b = b.strip()
            if not b or b.startswith('0 ') or b == '0':
                continue
            if b.startswith('$'):
                k, _, expr = b.partition('=')
                var[int(k[1:])] = self._eval(expr, var)
                continue
            v = [self._eval(x, var) for x in b.split(',')]
            code, exposure = int(v[0]), v[1]
            if code == 1:
                g = _disc(*_rot([(v[3], v[4])], v[5] if len(v) > 5 else 0)[0], v[2] / 2)
            elif code == 4:
                n = int(v[2])
                pts = [(v[3 + 2 * i], v[4 + 2 * i]) for i in range(n + 1)]
                g = _valid(Polygon(_rot(pts, v[5 + 2 * n])))
            elif code == 5:
                n, cx, cy, d, rot = int(v[2]), v[3], v[4], v[5], v[6]
                pts = [(cx + d / 2 * math.cos(2 * math.pi * i / n), cy + d / 2 * math.sin(2 * math.pi * i / n))
                       for i in range(n)]
                g = Polygon(_rot(pts, rot))
            elif code == 20:
                w, x1, y1, x2, y2, rot = v[2:8]
                L = math.hypot(x2 - x1, y2 - y1)
                if L == 0:
                    continue
                nx, ny = -(y2 - y1) / L * w / 2, (x2 - x1) / L * w / 2
                g = Polygon(_rot([(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)], rot))
            elif code == 21:
                w, h, cx, cy, rot = v[2:7]
                g = Polygon(_rot([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                                  (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)], rot))
            else:
                raise ValueError(f'примитив макроса {code} не поддерживается (макрос {self.name})')
            geom = geom.union(g) if exposure else geom.difference(g)
        return geom


def _standard(shape, p):
    """Стандартная апертура → (геометрия, диаметр или None)."""
    def hole(g, i):
        return g.difference(_disc(0, 0, p[i] / 2)) if len(p) > i and p[i] > 0 else g
    if shape == 'C':
        return hole(_disc(0, 0, p[0] / 2), 1), p[0]
    if shape == 'R':
        w, h = p[0], p[1]
        return hole(Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]), 2), None
    if shape == 'O':
        w, h = p[0], p[1]
        r = min(w, h) / 2
        if w == h:
            g = _disc(0, 0, r)
        else:
            a = ((w - h) / 2, 0) if w > h else (0, (h - w) / 2)
            g = LineString([(-a[0], -a[1]), a]).buffer(r, quad_segs=_quad_segs(r))
        return hole(g, 2), None
    if shape == 'P':
        d, n = p[0], int(p[1])
        rot = math.radians(p[2]) if len(p) > 2 else 0.0
        g = Polygon([(d / 2 * math.cos(rot + 2 * math.pi * i / n), d / 2 * math.sin(rot + 2 * math.pi * i / n))
                     for i in range(n)])
        return hole(g, 3), None
    raise ValueError(f'апертура {shape} не поддерживается')


def _arc(p0, p1, center, clockwise):
    """Точки дуги от p0 до p1 вокруг center (без p0)."""
    (x0, y0), (x1, y1), (cx, cy) = p0, p1, center
    r = math.hypot(x0 - cx, y0 - cy)
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    two_pi = 2 * math.pi
    sweep = (a0 - a1) % two_pi if clockwise else (a1 - a0) % two_pi
    if sweep < 1e-12 or p0 == p1:
        sweep = two_pi if p0 == p1 else sweep
    if clockwise:
        sweep = -sweep
    step = 2 * math.acos(max(1 - TOL / r, -1.0)) if r > TOL else math.pi / 2
    n = max(int(math.ceil(abs(sweep) / step)), 2)
    a = a0 + sweep * np.arange(1, n + 1) / n
    pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in a]
    pts[-1] = (x1, y1)
    return pts


def read(path) -> list[GObj]:
    """Все графические объекты файла Gerber по порядку."""
    text = Path(path).read_text(encoding='utf-8')
    objs: list[GObj] = []
    apertures = {}          # номер → (геометрия, диаметр, атрибуты)
    macros = {}
    ta, to = {}, {}
    scale = 1e-6
    dark = True
    interp = 1              # 1 линейно, 2 по часовой, 3 против часовой
    cur = None              # текущая апертура
    x = y = 0.0
    in_region = False
    contour: list = []
    region_attrs = None

    def obj_attrs(func):
        ref = pin = None
        if 'P' in to:
            parts = to['P'].split(',')
            ref, pin = parts[0], parts[1] if len(parts) > 1 else None
        elif 'C' in to:
            ref = to['C']
        net = to.get('N')
        return func, (None if net in (None, '', 'N/C') else net), ref, pin

    def close_contour():
        nonlocal contour
        if len(contour) >= 3:
            f, n, r, p = region_attrs
            objs.append(GObj('region', _valid(Polygon(contour)), dark, f, n, r, p))
        contour = []

    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in '\r\n \t':
            i += 1
            continue
        if ch == '%':
            j = text.index('%', i + 1)
            blocks = [b.strip() for b in text[i + 1:j].replace('\r', '').replace('\n', '').split('*')]
            blocks = [b for b in blocks if b]
            i = j + 1
            head = blocks[0]
            code = head[:2]
            if code == 'FS':
                m = re.match(r'FSLAX(\d)(\d)Y(\d)(\d)', head)
                if not m:
                    raise ValueError(f'формат координат не поддерживается: {head}')
                scale = 10.0 ** -int(m.group(2))
            elif code == 'MO':
                if head != 'MOMM':
                    raise ValueError('ожидались миллиметры (MOMM)')
            elif code == 'LP':
                dark = head[2] == 'D'
            elif code == 'AM':
                macros[head[2:]] = _Macro(head[2:], blocks[1:])
            elif code == 'AD':
                m = re.match(r'ADD(\d+)([A-Za-z_.$][A-Za-z0-9_.$]*)(?:,(.*))?\Z', head)
                num, name, par = int(m.group(1)), m.group(2), m.group(3)
                p = [float(v) for v in par.split('X')] if par else []
                if name in macros:
                    geom, dia = macros[name].build(p), None
                else:
                    geom, dia = _standard(name, p)
                apertures[num] = (geom, dia, dict(ta))
            elif code == 'TF':
                pass
            elif code == 'TA':
                k, _, v = _unescape(head[3:]).partition(',')
                ta[k.lstrip('.')] = v
            elif code == 'TO':
                k, _, v = _unescape(head[3:]).partition(',')
                to[k.lstrip('.')] = v
            elif code == 'TD':
                k = head[2:].lstrip('.')
                if k:
                    ta.pop(k, None)
                    to.pop(k, None)
                else:
                    ta.clear()
                    to.clear()
            else:
                raise ValueError(f'команда Gerber не поддерживается: %{head}%')
            continue
        j = text.index('*', i)
        b = text[i:j].strip()
        i = j + 1
        if not b or b.startswith('G04'):
            continue
        if b in ('M02', 'M00'):
            break
        if b.startswith('G54'):
            b = b[3:]
        m = re.fullmatch(r'D(\d+)', b)
        if m and int(m.group(1)) >= 10:
            cur = int(m.group(1))
            continue
        m = _OP.match(b)
        if not m:
            raise ValueError(f'команда Gerber не поддерживается: {b}')
        g, coords, d = m.groups()
        if g:
            gn = int(g[1:])
            if gn in (1, 2, 3):
                interp = gn
            elif gn == 36:
                in_region, contour = True, []
                region_attrs = obj_attrs(ta.get('AperFunction'))
            elif gn == 37:
                close_contour()
                in_region = False
            elif gn == 74:
                raise ValueError('одноквадрантные дуги (G74) не поддерживаются')
        if not coords and not d:
            continue
        nx, ny, ci, cj = x, y, 0.0, 0.0
        for axis, v in _COORD.findall(coords):
            val = int(v) * scale
            if axis == 'X':
                nx = val
            elif axis == 'Y':
                ny = val
            elif axis == 'I':
                ci = val
            else:
                cj = val
        dn = int(d[1:]) if d else 1
        if dn == 2:
            if in_region:
                close_contour()
                contour = [(nx, ny)]
        elif dn == 1:
            if interp == 1:
                path = [(x, y), (nx, ny)]
            else:
                path = [(x, y)] + _arc((x, y), (nx, ny), (x + ci, y + cj), interp == 2)
            if in_region:
                if not contour:
                    contour = [path[0]]
                contour.extend(path[1:])
            else:
                geom, dia, attrs = apertures[cur]
                if dia is None:
                    raise ValueError('рисование некруглой апертурой не поддерживается')
                r = dia / 2
                if all(p == path[0] for p in path):
                    gm, ln = _disc(x, y, r), LineString([path[0], path[0]])
                else:
                    ln = LineString(path)
                    gm = ln.buffer(r, quad_segs=_quad_segs(r))
                objs.append(GObj('draw', gm, dark, *obj_attrs(attrs.get('AperFunction')), line=ln))
        elif dn == 3:
            geom, dia, attrs = apertures[cur]
            objs.append(GObj('flash', affinity.translate(geom, nx, ny), dark,
                             *obj_attrs(attrs.get('AperFunction')), pos=(nx, ny)))
        x, y = nx, ny
    return objs


def image(objs, keep=None):
    """Итоговое изображение слоя: объединение тёмных объектов минус светлые, по порядку.
    keep(obj) → bool отбирает объекты."""
    result = None
    batch = []
    for o in objs:
        if keep is not None and not keep(o):
            continue
        if o.dark:
            batch.append(o.geom)
            continue
        if batch:
            result = shapely.union_all(batch if result is None else [result, *batch])
            batch = []
        if result is not None:
            result = result.difference(o.geom)
    if batch:
        result = shapely.union_all(batch if result is None else [result, *batch])
    return result if result is not None else Polygon()


def file_function(path):
    """Строка %TF.FileFunction (например 'Copper,L1,Top')."""
    with open(path, encoding='utf-8') as f:
        for _ in range(30):
            line = f.readline()
            if line.startswith('%TF.FileFunction,'):
                return line.strip()[len('%TF.FileFunction,'):-2]
    return None


def layer_files(folder, copper):
    """{имя слоя модели: путь к Gerber} для меди, маски и контура. Имена файлов KiCad берёт из
    пользовательских имён слоёв, поэтому слой определяется по атрибуту FileFunction."""
    out = {}
    for p in sorted(Path(folder).iterdir()):
        if not p.is_file() or p.suffix in ('.gbrjob', '.drl', '.csv'):
            continue
        f = file_function(p)
        if not f:
            continue
        parts = f.split(',')
        if parts[0] == 'Copper':
            out[copper[int(parts[1][1:]) - 1]] = p
        elif parts[0] == 'Soldermask':
            out['F.Mask' if parts[1] == 'Top' else 'B.Mask'] = p
        elif parts[0] == 'Profile':
            out['Edge.Cuts'] = p
    return out


# ---------- подготовка платы и сравнение ----------

from dfm.compare import strip_texts  # noqa: E402,F401 — подготовка платы общая с командой compare


def compare(ours, theirs, tol=0.005):
    """Сравнение двух изображений слоя. → (площадь расхождения, «толстая» часть расхождения).

    Толстая часть — то, что остаётся от симметрической разности после сужения на tol: участки
    шире 2·tol. Мелкие расхождения неизбежны (KiCad заменяет дуги произвольных площадок
    ломаными с допуском до 5 мкм), а толстых быть не должно."""
    diff = ours.symmetric_difference(theirs)
    return diff.area, diff.buffer(-tol)
