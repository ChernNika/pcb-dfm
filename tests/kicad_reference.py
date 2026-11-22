"""Эталон для проверки модели: то, что выгружает сам KiCad (kicad-cli).

Здесь не адаптер Gerber (он будет в шаге 1.5), а минимальное чтение ровно того, что нужно
для сверки уровня D:
  * вспышки площадок в Gerber X2: позиционное обозначение, номер вывода, цепь, координаты
    (атрибуты %TO.P / %TO.N, которые KiCad ставит перед каждой вспышкой D03);
  * отверстия из Excellon: координаты, диаметр, металлизация, тип (переходное / вывода);
  * позиции компонентов (файл позиций CSV);
  * стек слоёв из job-файла .gbrjob.
Все координаты KiCad выгружает с осью Y вверх — так же, как в нашей модели.
"""
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

_ESC = re.compile(r'\\u([0-9A-Fa-f]{4})')
_COORD = re.compile(r'([XYIJ])([-+]?\d+)')


def _unescape(s):
    return _ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


@dataclass
class PadFlash:
    ref: str
    pin: str
    net: str
    x: float
    y: float


def gerber_pads(path):
    """Вспышки площадок (D03 с атрибутом .P) и обозначения площадок, нарисованных регионами (G36)."""
    flashes, regions = [], set()
    attrs = {}
    dec = 6
    x = y = 0.0
    in_region = False
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line.startswith('%FSLA'):
            dec = int(re.search(r'X\d(\d)', line).group(1))
        elif line.startswith('%TO.'):
            body = _unescape(line[4:-2])
            key, _, val = body.partition(',')
            attrs[key] = val
        elif line.startswith('%TD'):
            key = line[3:-2].lstrip('.')
            if key:
                attrs.pop(key, None)
            else:
                attrs.clear()
        elif line.startswith('G36'):
            in_region = True
            if 'P' in attrs:
                ref, pin = attrs['P'].split(',')[:2]
                regions.add((ref, pin))
        elif line.startswith('G37'):
            in_region = False
        elif line.startswith(('X', 'Y')) and not in_region:
            for axis, v in _COORD.findall(line):
                if axis == 'X':
                    x = int(v) / 10 ** dec
                elif axis == 'Y':
                    y = int(v) / 10 ** dec
            if line.endswith('D03*') and 'P' in attrs:
                ref, pin = attrs['P'].split(',')[:2]
                flashes.append(PadFlash(ref, pin, attrs.get('N', ''), x, y))
    return flashes, regions


def gerber_function(path):
    """Строка %TF.FileFunction (например 'Copper,L1,Top')."""
    for line in Path(path).read_text(encoding='utf-8').splitlines()[:30]:
        if line.startswith('%TF.FileFunction,'):
            return line[len('%TF.FileFunction,'):-2]
    return None


def copper_files(gerbers, copper):
    """{имя медного слоя модели: путь к Gerber}. Слои KiCad называет по-своему (top_layer…),
    поэтому сопоставляем по атрибуту FileFunction: Copper,L<n>,Top|Inr|Bot."""
    out = {}
    for p in gerbers:
        if p.suffix == '.gbrjob' or not p.is_file():
            continue
        f = gerber_function(p)
        if f and f.startswith('Copper,L'):
            n = int(f.split(',')[1][1:])
            out[copper[n - 1]] = p
    return out


@dataclass
class DrillHole:
    x: float
    y: float
    d: float
    plated: bool
    function: str            # ViaDrill, ComponentDrill, MechanicalDrill …
    x2: float | None = None  # конец паза
    y2: float | None = None


def excellon(path):
    """Отверстия из файла сверловки KiCad (метрический, десятичные координаты)."""
    text = Path(path).read_text(encoding='utf-8').splitlines()
    plated = True
    tools, func, pending = {}, {}, None
    holes = []
    tool = None
    route_start = None
    xy = lambda s: tuple(float(v) for v in re.findall(r'[XY]([-+]?[\d.]+)', s))
    for line in text:
        if line.startswith('; #@! TF.FileFunction,'):
            plated = line.split(',')[1] == 'Plated'
        elif line.startswith('; #@! TA.AperFunction,'):
            pending = line.split(',')[-1]
        elif re.match(r'T\d+C', line):
            t, d = re.match(r'(T\d+)C([\d.]+)', line).groups()
            tools[t] = float(d)
            func[t] = pending
        elif re.fullmatch(r'T\d+', line):
            tool = line
        elif line.startswith('G00X'):
            route_start = xy(line)
        elif line.startswith('G01X') and route_start:
            x2, y2 = xy(line)
            holes.append(DrillHole(*route_start, tools[tool], plated, func[tool], x2, y2))
            route_start = None
        elif line.startswith('X') and tool:
            if 'G85' in line:
                a, b = line.split('G85')
                holes.append(DrillHole(*xy(a), tools[tool], plated, func[tool], *xy(b)))
            else:
                holes.append(DrillHole(*xy(line), tools[tool], plated, func[tool]))
    return holes


def positions(path):
    """Файл позиций CSV: [(ref, x, y, rot, side)]."""
    with open(path, encoding='utf-8') as f:
        return [(r['Ref'], float(r['PosX']), float(r['PosY']), float(r['Rot']), r['Side'])
                for r in csv.DictReader(f)]


def job_stackup(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return d['GeneralSpecs'], d['MaterialStackup']


def unmatched_holes(model_holes, drilled, tol=6e-4, dia_tol=1.5e-3):
    """Сверка отверстий модели (Hole уровня P) с отверстиями из Excellon.
    → (отверстия модели, не найденные в сверловке; лишние отверстия сверловки).
    В Excellon координаты округлены до 1 мкм, поэтому допуск tol."""
    near = lambda a, x, y: abs(a[0] - x) <= tol and abs(a[1] - y) <= tol
    free = list(drilled)
    missing = []
    for h in model_holes:
        start, end = h.pos, h.slot_end
        for i, d in enumerate(free):
            if d.plated != h.plated or abs(d.d - h.diameter) > dia_tol or (end is None) != (d.x2 is None):
                continue
            if end is None and near(start, d.x, d.y) or end is not None and (
                    near(start, d.x, d.y) and near(end, d.x2, d.y2) or near(end, d.x, d.y) and near(start, d.x2, d.y2)):
                free.pop(i)
                break
        else:
            missing.append(h)
    return missing, free
