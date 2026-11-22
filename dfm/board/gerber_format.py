"""Чтение одного файла Gerber (RS-274X, атрибуты X2) в точные формы модели.

    gf = read_gerber(path)      → GerberFile: функция файла, полярность, список объектов
    gf.objects[i]               → GObject: kind (flash | draw | region), shape, dark, атрибуты

Поддерживается подмножество, которое выдают KiCad и другие распространённые САПР:
  * единицы MOMM / MOIN, формат координат FSLA (ведущие нули опущены, абсолютные);
  * апертуры C, R, O, P (с отверстием и без) и макросы: примитивы 1 (круг), 4 (контур),
    5 (многоугольник), 20 (линия), 21 (прямоугольник по центру), 7 (термобарьер),
    выражения с переменными $n, присваивания; примитивы с экспозицией «выкл.»;
  * вспышки D03, рисование D01 по прямой (G01) и по дуге (G02/G03, G75), регионы G36/G37,
    полярность LPD/LPC;
  * атрибуты X2: файла (TF), апертуры (TA), объекта (TO), удаление (TD).
Неподдерживаемое (пошаговое повторение SR, преобразования апертур LM/LR/LS, блоки AB,
одноквадрантные дуги G74, относительные координаты) вызывает ошибку с понятным текстом:
лучше остановиться, чем молча получить неверную медь.

Формы точные: дуги хранятся как дуги, круглая апертура — круг, рисование круглой апертурой —
линия с круглыми концами (Stroke), макрос — объединение примитивов (Compound). Приближение
(ломаная) появляется только для макросов с вычитанием и термобарьеров.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .shapes import Circle, Compound, Contour, Pt, Region, Shape, Stroke, moved, rotate

__all__ = ['GObject', 'GerberFile', 'read_gerber', 'GerberError', 'parse_function']


class GerberError(ValueError):
    pass


@dataclass(slots=True, eq=False)
class GObject:
    """Графический объект Gerber. attrs — атрибуты апертуры и объекта на момент создания:
    'AperFunction' (строка через запятую), 'N' (цепь), 'P' (компонент, вывод, …), 'C' (компонент)."""
    kind: str                     # flash | draw | region
    shape: Shape
    dark: bool
    attrs: dict
    pos: Pt | None = None         # точка вспышки


@dataclass(slots=True)
class GerberFile:
    path: Path
    function: str | None = None    # %TF.FileFunction: 'Copper,L1,Top', 'Soldermask,Top', 'Profile,NP' …
    polarity: str | None = None    # %TF.FilePolarity: Positive / Negative
    attrs: dict = field(default_factory=dict)
    objects: list[GObject] = field(default_factory=list)


def parse_function(s: str | None) -> list[str]:
    return [x.strip() for x in s.split(',')] if s else []


# ---------- выражения макросов ----------

_TOKEN = re.compile(r'\s*(?:(\d+\.?\d*|\.\d+)|\$(\d+)|([-+xX*/()]))')


def _eval(expr: str, var: dict) -> float:
    """Арифметика макроса Gerber: числа, $n, + − x(×) /, скобки, унарный знак."""
    tokens, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            raise GerberError(f'выражение макроса не разобрано: {expr!r}')
        num, ref, op = m.groups()
        tokens.append(('n', float(num)) if num else ('n', float(var.get(int(ref), 0.0))) if ref
                      else ('o', '*' if op in 'xX' else op))
        pos = m.end()
    i = 0

    def atom():
        nonlocal i
        kind, v = tokens[i]
        if kind == 'o' and v in '+-':
            i += 1
            a = atom()
            return -a if v == '-' else a
        if kind == 'o' and v == '(':
            i += 1
            a = add()
            i += 1                      # ')'
            return a
        i += 1
        return v

    def mul():
        nonlocal i
        a = atom()
        while i < len(tokens) and tokens[i] in (('o', '*'), ('o', '/')):
            op = tokens[i][1]
            i += 1
            b = atom()
            a = a * b if op == '*' else a / b
        return a

    def add():
        nonlocal i
        a = mul()
        while i < len(tokens) and tokens[i] in (('o', '+'), ('o', '-')):
            op = tokens[i][1]
            i += 1
            b = mul()
            a = a + b if op == '+' else a - b
        return a

    if not tokens:
        return 0.0
    return add()


# ---------- апертуры ----------

def _circle_contour(cx, cy, r) -> Contour:
    return Contour(np.array([(cx + r, cy), (cx - r, cy)]), {0: (cx, cy + r), 1: (cx, cy - r)})


def _rect(cx, cy, w, h, rot=0.0) -> Region:
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    return Region(Contour(np.array([rotate(p, rot) for p in pts])))


def _obround_contour(w, h) -> Contour:
    if w >= h:
        r, a = h / 2, (w - h) / 2
        return Contour(np.array([(-a, -r), (a, -r), (a, r), (-a, r)]), {1: (a + r, 0.0), 3: (-a - r, 0.0)})
    r, a = w / 2, (h - w) / 2
    return Contour(np.array([(r, -a), (r, a), (-r, a), (-r, -a)]), {1: (0.0, a + r), 3: (0.0, -a - r)})


def _with_hole(shape: Shape, outer: Contour, p: list, i: int) -> Shape:
    if len(p) > i and p[i] > 0:
        return Region(outer, (_circle_contour(0.0, 0.0, p[i] / 2),))
    return shape


def _standard(name: str, p: list[float]):
    """Стандартная апертура → (форма в точке (0, 0), диаметр, если апертура круглая)."""
    if name == 'C':
        return _with_hole(Circle((0.0, 0.0), p[0] / 2), _circle_contour(0, 0, p[0] / 2), p, 1), p[0]
    if name == 'R':
        r = _rect(0, 0, p[0], p[1])
        return _with_hole(r, r.outer, p, 2), None
    if name == 'O':
        w, h = p[0], p[1]
        if w == h:
            s = Circle((0.0, 0.0), w / 2)
        else:
            e = ((w - h) / 2, 0.0) if w > h else (0.0, (h - w) / 2)
            s = Stroke(np.array([(-e[0], -e[1]), e]), min(w, h))
        return _with_hole(s, _obround_contour(w, h), p, 2), None
    if name == 'P':
        d, n = p[0], int(p[1])
        rot = p[2] if len(p) > 2 else 0.0
        c = Contour(np.array([rotate((d / 2 * math.cos(2 * math.pi * k / n), d / 2 * math.sin(2 * math.pi * k / n)), rot)
                              for k in range(n)]))
        return _with_hole(Region(c), c, p, 3), None
    raise GerberError(f'стандартная апертура {name} не поддерживается')


class _Macro:
    def __init__(self, name: str, blocks: list[str]):
        self.name = name
        self.blocks = [b.strip() for b in blocks if b.strip()]

    def build(self, params: list[float]) -> Shape:
        var = {i + 1: v for i, v in enumerate(params)}
        parts: list[tuple[Shape, bool]] = []      # (форма, экспозиция)
        approx = False
        for b in self.blocks:
            if b.startswith('0'):
                continue                          # комментарий
            if b.startswith('$'):
                k, _, e = b.partition('=')
                var[int(k[1:])] = _eval(e, var)
                continue
            v = [_eval(x, var) for x in b.split(',')]
            code = int(v[0])
            if code == 1:                          # круг: экспозиция, диаметр, x, y [, поворот]
                rot = v[5] if len(v) > 5 else 0.0
                parts.append((Circle(rotate((v[3], v[4]), rot), v[2] / 2), v[1] > 0))
            elif code == 4:                        # контур: экспозиция, n, точки ×(n+1), поворот
                n = int(v[2])
                pts = [(v[3 + 2 * k], v[4 + 2 * k]) for k in range(n + 1)]
                rot = v[5 + 2 * n]
                if pts[-1] == pts[0]:
                    pts = pts[:-1]
                parts.append((Region(Contour(np.array([rotate(p, rot) for p in pts]))), v[1] > 0))
            elif code == 5:                        # правильный многоугольник
                n, cx, cy, d, rot = int(v[2]), v[3], v[4], v[5], v[6]
                pts = [rotate((cx + d / 2 * math.cos(2 * math.pi * k / n), cy + d / 2 * math.sin(2 * math.pi * k / n)), rot)
                       for k in range(n)]
                parts.append((Region(Contour(np.array(pts))), v[1] > 0))
            elif code in (2, 20):                  # линия с плоскими концами
                w, x1, y1, x2, y2, rot = v[2:8]
                L = math.hypot(x2 - x1, y2 - y1)
                if L == 0:
                    continue
                nx, ny = -(y2 - y1) / L * w / 2, (x2 - x1) / L * w / 2
                pts = [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]
                parts.append((Region(Contour(np.array([rotate(p, rot) for p in pts]))), v[1] > 0))
            elif code == 21:                       # прямоугольник по центру
                w, h, cx, cy, rot = v[2:7]
                parts.append((_rect(cx, cy, w, h, rot), v[1] > 0))
            elif code == 7:                        # термобарьер: x, y, D внеш., D внутр., зазор, поворот
                parts.append((self._thermal(*v[1:7]), True))
                approx = True
            else:
                raise GerberError(f'примитив макроса {code} не поддерживается (макрос {self.name})')
        if not parts:
            return Compound(())
        if all(e for _, e in parts) and not approx:
            return parts[0][0] if len(parts) == 1 else Compound(tuple(s for s, _ in parts))
        # вычитание внутри макроса: точного представления нет — многоугольник
        from ..geometry.polygons import from_geometry, to_geometry
        g = None
        for s, e in parts:
            sg = to_geometry(s)
            g = (sg if e else None) if g is None else (g.union(sg) if e else g.difference(sg))
        return from_geometry(g) if g is not None else Compound(())

    @staticmethod
    def _thermal(cx, cy, outer, inner, gap, rot):
        from shapely.geometry import Point, box
        from ..geometry.polygons import from_geometry
        ring = Point(cx, cy).buffer(outer / 2, quad_segs=64).difference(Point(cx, cy).buffer(inner / 2, quad_segs=64))
        cross = box(cx - outer, cy - gap / 2, cx + outer, cy + gap / 2).union(
            box(cx - gap / 2, cy - outer, cx + gap / 2, cy + outer))
        g = ring.difference(cross)
        if rot:
            from shapely import affinity
            g = affinity.rotate(g, rot, origin=(0, 0))
        return from_geometry(g)


# ---------- разбор файла ----------

_OP = re.compile(r'(G0?[123]|G3[67]|G7[45])?((?:[XYIJ][-+]?\d+)*)(D0?[123])?\Z')
_COORD = re.compile(r'([XYIJ])([-+]?\d+)')
_ESC = re.compile(r'\\u([0-9A-Fa-f]{4})')
_IGNORED = ('G04', 'G70', 'G71', 'G90', 'M00', 'M01')
_TOKENS = re.compile(r'%([^%]*)%|([^%*]*)\*')          # расширенная команда или слово до «*»
_DCODE = re.compile(r'D(\d+)')


def _unescape(s: str) -> str:
    return _ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def _arc_mid(start: Pt, end: Pt, center: Pt, clockwise: bool) -> Pt:
    """Точка в середине дуги G02/G03 (многоквадрантный режим G75)."""
    (x0, y0), (x1, y1), (cx, cy) = start, end, center
    r = math.hypot(x0 - cx, y0 - cy)
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    two_pi = 2 * math.pi
    if clockwise:
        sweep = -((a0 - a1) % two_pi)
    else:
        sweep = (a1 - a0) % two_pi
    if start == end or abs(sweep) < 1e-12:
        sweep = -two_pi if clockwise else two_pi       # совпадающие концы в G75 — полная окружность
    am = a0 + sweep / 2
    return (cx + r * math.cos(am), cy + r * math.sin(am))


def read_gerber(path) -> GerberFile:
    path = Path(path)
    text = path.read_text(encoding='utf-8', errors='replace')
    gf = GerberFile(path)
    objs = gf.objects
    apertures: dict[int, tuple[Shape, float | None, dict]] = {}
    macros: dict[str, _Macro] = {}
    ta: dict[str, str] = {}
    to: dict[str, str] = {}
    scale = 1e-6
    unit = 1.0                       # мм на единицу файла
    dark = True
    interp = 1
    quadrant_multi = True
    cur = None
    x = y = 0.0
    in_region = False
    contour: list = []
    arcs: dict = {}
    region_attrs: dict = {}
    last_draw = None                 # для склейки подряд идущих отрезков в одну линию

    def obj_attrs(aperture_attrs: dict) -> dict:
        a = {}
        if 'AperFunction' in aperture_attrs:
            a['AperFunction'] = aperture_attrs['AperFunction']
        for k in ('N', 'P', 'C'):
            if k in to:
                a[k] = to[k]
        return a

    def close_contour():
        nonlocal contour, arcs
        if len(contour) >= 2:
            pts, ar = contour, arcs
            if len(pts) > 1 and pts[-1] == pts[0]:
                pts = pts[:-1]
                ar = {i: m for i, m in ar.items() if i < len(pts)}
            if len(pts) >= 3 or (len(pts) >= 2 and ar):
                objs.append(GObject('region', Region(Contour(np.array(pts, dtype=float), ar)), dark, region_attrs))
        contour, arcs = [], {}

    for tok in _TOKENS.finditer(text):
        ext, word = tok.group(1), tok.group(2)
        if ext is not None:                 # расширенная команда %…%
            blocks = [b.strip() for b in ext.replace('\r', '').replace('\n', '').split('*')]
            blocks = [b for b in blocks if b]
            if not blocks:
                continue
            head = blocks[0]
            code = head[:2]
            if code == 'FS':
                m = re.match(r'FSLAX(\d)(\d)Y(\d)(\d)\Z', head)
                if not m:
                    raise GerberError(f'{path.name}: формат координат не поддерживается: {head}')
                scale = 10.0 ** -int(m.group(2))
            elif code == 'MO':
                if head == 'MOMM':
                    unit = 1.0
                elif head == 'MOIN':
                    unit = 25.4
                else:
                    raise GerberError(f'{path.name}: единицы {head} не поддерживаются')
            elif code == 'LP':
                dark = head[2] == 'D'
                last_draw = None
            elif code == 'AM':
                macros[head[2:]] = _Macro(head[2:], blocks[1:])
            elif code == 'AD':
                m = re.match(r'ADD(\d+)([A-Za-z_.$][A-Za-z0-9_.$\-]*)(?:,(.*))?\Z', head)
                if not m:
                    raise GerberError(f'{path.name}: определение апертуры не разобрано: {head}')
                num, name, par = int(m.group(1)), m.group(2), m.group(3)
                p = [float(v) for v in par.split('X')] if par else []
                if name in macros:
                    shape, dia = macros[name].build(p), None
                else:
                    shape, dia = _standard(name, p)
                if unit != 1.0:
                    shape = _scaled(shape, unit)
                    dia = dia * unit if dia is not None else None
                apertures[num] = (shape, dia, dict(ta))
            elif code == 'TF':
                k, _, v = _unescape(head[3:]).partition(',')
                gf.attrs[k] = v
                if k == 'FileFunction':
                    gf.function = v
                elif k == 'FilePolarity':
                    gf.polarity = v
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
            elif code in ('IP', 'IN', 'LN', 'OF', 'AS', 'IR', 'MI', 'SF'):
                if head not in ('IPPOS', 'OFA0B0', 'ASAXBY', 'IR0', 'MIA0B0', 'SFA1B1') and code not in ('IN', 'LN'):
                    raise GerberError(f'{path.name}: устаревшая команда %{head}% с непустым действием не поддерживается')
            else:
                raise GerberError(f'{path.name}: команда %{head}% не поддерживается')
            continue
        b = word.strip()
        if not b or b.startswith(_IGNORED):
            continue
        if b == 'M02':
            break
        if b.startswith('G54'):
            b = b[3:]
        if b == 'G74':
            raise GerberError(f'{path.name}: одноквадрантные дуги (G74) не поддерживаются')
        if b == 'G91':
            raise GerberError(f'{path.name}: относительные координаты (G91) не поддерживаются')
        m = re.fullmatch(r'D(\d+)', b)
        if m and int(m.group(1)) >= 10:
            cur = int(m.group(1))
            if cur not in apertures:
                raise GerberError(f'{path.name}: апертура D{cur} не определена')
            last_draw = None
            continue
        m = _OP.match(b)
        if not m:
            raise GerberError(f'{path.name}: команда {b!r} не поддерживается')
        g, coords, d = m.groups()
        if g:
            gn = int(g[1:])
            if gn in (1, 2, 3):
                interp = gn
            elif gn == 36:
                in_region, contour, arcs = True, [], {}
                region_attrs = obj_attrs(ta)
                last_draw = None
            elif gn == 37:
                close_contour()
                in_region = False
            elif gn == 75:
                quadrant_multi = True
        if not coords and not d:
            continue
        nx, ny, ci, cj = x, y, 0.0, 0.0
        for axis, v in _COORD.findall(coords):
            val = int(v) * scale * unit
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
            last_draw = None
        elif dn == 1:
            mid = None
            if interp != 1:
                if not quadrant_multi:
                    raise GerberError(f'{path.name}: одноквадрантные дуги не поддерживаются')
                mid = _arc_mid((x, y), (nx, ny), (x + ci, y + cj), interp == 2)
            if in_region:
                if not contour:
                    contour = [(x, y)]
                if mid is not None:
                    arcs[len(contour) - 1] = mid
                contour.append((nx, ny))
            else:
                if cur is None:
                    raise GerberError(f'{path.name}: рисование без выбранной апертуры')
                shape, dia, aattrs = apertures[cur]
                attrs = obj_attrs(aattrs)
                if dia is None:
                    objs.append(GObject('draw', _rect_draw(shape, (x, y), (nx, ny), mid, path), dark, attrs))
                    last_draw = None
                elif (last_draw is not None and last_draw[1] == cur and last_draw[2] == attrs
                      and last_draw[0].dark == dark):
                    # продолжение той же линии: добавить точку к уже начатому Stroke
                    o, _, _, pts, ar = last_draw
                    if mid is not None:
                        ar[len(pts) - 1] = mid
                    pts.append((nx, ny))
                    o.shape = Stroke(np.array(pts), dia, dict(ar))
                else:
                    pts = [(x, y)] if (x, y) == (nx, ny) and mid is None else [(x, y), (nx, ny)]
                    ar = {0: mid} if mid is not None else {}
                    o = GObject('draw', Stroke(np.array(pts), dia, dict(ar)), dark, attrs)
                    objs.append(o)
                    last_draw = (o, cur, attrs, pts, ar)
        elif dn == 3:
            if cur is None:
                raise GerberError(f'{path.name}: вспышка без выбранной апертуры')
            shape, dia, aattrs = apertures[cur]
            objs.append(GObject('flash', moved(shape, nx, ny), dark, obj_attrs(aattrs), (nx, ny)))
            last_draw = None
        x, y = nx, ny
    return gf


def _scaled(shape: Shape, k: float) -> Shape:
    """Форма в дюймах → мм (апертуры задаются в единицах файла)."""
    from .shapes import mapped
    if isinstance(shape, Circle):
        return Circle((shape.center[0] * k, shape.center[1] * k), shape.radius * k)
    if isinstance(shape, Stroke):
        return Stroke(shape.pts * k, shape.width * k, {i: (m[0] * k, m[1] * k) for i, m in shape.arcs.items()})
    if isinstance(shape, Region):
        return mapped(shape, lambda p: (p[0] * k, p[1] * k))
    return Compound(tuple(_scaled(p, k) for p in shape.parts))


def _rect_draw(shape: Shape, a: Pt, b: Pt, mid, path) -> Shape:
    """Рисование некруглой апертурой (прямоугольник, многоугольник): по прямой это выпуклая
    оболочка апертуры в начальной и конечной точках (точно для выпуклых апертур)."""
    if mid is not None or not isinstance(shape, Region) or shape.holes or shape.outer.arcs:
        raise GerberError(f'{path.name}: рисование дугой или невыпуклой апертурой не поддерживается')
    from shapely import MultiPoint
    pts = np.vstack((shape.outer.pts + a, shape.outer.pts + b))
    hull = MultiPoint(pts).convex_hull
    return Region(Contour(np.asarray(hull.exterior.coords)[:-1]))
