"""Чтение файла сверловки Excellon (с атрибутами X2 в комментариях, как пишет KiCad).

    df = read_excellon(path)      → DrillFile: металлизация, слои, отверстия (DrillHit)

Что берётся из файла:
  * единицы METRIC / INCH; координаты десятичные (KiCad) или целые с опущенными нулями
    (LZ — ведущие нули сохранены, а хвостовые опущены; TZ — наоборот, так по умолчанию;
    формат по умолчанию 3.3 мм / 2.4 дюйма,
    его можно задать строкой FILE_FORMAT=a:b или комментарием KiCad ;FORMAT={a:b/…});
  * инструменты T<n>C<диаметр>;
  * сверление X…Y…, пазы X…Y…G85X…Y… и в режиме фрезерования G00 … M15 G01 … M16;
  * атрибуты X2 в комментариях: «; #@! TF.FileFunction,Plated,1,4,PTH» (металлизация и пара
    слоёв: 1 — верхний медный, 4 — нижний; Blind/Buried — глухие и скрытые) и
    «; #@! TA.AperFunction,Plated,PTH,ViaDrill» перед инструментом (переходное / вывода).
Если атрибутов нет, металлизация угадывается по имени файла (NPTH — неметаллизированные),
а отверстие считается сквозным.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .shapes import Pt

__all__ = ['DrillHit', 'DrillFile', 'read_excellon', 'is_excellon']


@dataclass(slots=True)
class DrillHit:
    pos: Pt
    diameter: float
    end: Pt | None = None              # второй конец паза
    function: str | None = None        # ViaDrill, ComponentDrill, MechanicalDrill …


@dataclass(slots=True)
class DrillFile:
    path: Path
    plated: bool | None = None         # None — не указано в файле
    layers: tuple[int, int] | None = None   # номера медных слоёв (1 — верхний), None — не указано
    kind: str | None = None            # PTH, NPTH, Blind, Buried …
    hits: list[DrillHit] = field(default_factory=list)
    attributes: bool = False           # есть ли в файле атрибуты X2


def is_excellon(path) -> bool:
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            head = f.read(2000)
    except OSError:
        return False
    return head.lstrip().startswith('M48') or '\nM48' in head[:200]


_NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)'
_XY = re.compile(rf'X({_NUM})|Y({_NUM})')


def read_excellon(path) -> DrillFile:
    path = Path(path)
    df = DrillFile(path)
    unit = 1.0
    zeros = 'TZ'                       # по умолчанию ведущие нули опущены (как в Gerber)
    fmt = None                         # (целых, дробных)
    decimal = False                    # ;FORMAT={…/ decimal} — числа без точки тоже десятичные
    tools: dict[str, float] = {}
    tool_func: dict[str, str | None] = {}
    pending = None
    tool = None
    in_header = True
    route_start = None
    plunged = False
    x = y = 0.0

    def coord(s: str, axis_prev: float) -> float:
        if '.' in s or decimal:
            return float(s) * unit
        sign = -1.0 if s.startswith('-') else 1.0
        digits = s.lstrip('+-')
        ip, dp = fmt or ((3, 3) if unit == 1.0 else (2, 4))
        if zeros == 'LZ':            # ведущие нули сохранены, хвостовые опущены: дописать справа
            digits = digits.ljust(ip + dp, '0')
        val = int(digits) / 10 ** dp if digits else 0.0
        return sign * val * unit

    def xy(line: str):
        nx, ny = x, y
        for mx, my in _XY.findall(line):
            if mx:
                nx = coord(mx, x)
            if my:
                ny = coord(my, y)
        return nx, ny

    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(';'):
            body = line[1:].strip()
            if body.startswith('#@!'):
                df.attributes = True
                body = body[3:].strip()
                if body.startswith('TF.FileFunction,'):
                    f = body.split(',')[1:]
                    df.plated = f[0] == 'Plated'
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        df.layers = (int(f[1]), int(f[2]))
                    df.kind = f[3] if len(f) > 3 else None
                elif body.startswith('TA.AperFunction,'):
                    pending = body.split(',')[-1]
            else:
                m = re.search(r'FORMAT=\{(\d+):(\d+)', body)
                if m:
                    fmt = (int(m.group(1)), int(m.group(2)))
                if 'FORMAT=' in body and 'decimal' in body:
                    decimal = True
            continue
        if in_header:
            if line in ('%', 'M95'):
                in_header = False
                continue
            if line.startswith(('METRIC', 'INCH')):
                unit = 1.0 if line.startswith('METRIC') else 25.4
                if 'TZ' in line:
                    zeros = 'TZ'
                elif 'LZ' in line:
                    zeros = 'LZ'
                m = re.search(r',(0+)\.(0+)', line)
                if m:
                    fmt = (len(m.group(1)), len(m.group(2)))
                continue
            m = re.match(r'FILE_FORMAT=(\d+):(\d+)', line)
            if m:
                fmt = (int(m.group(1)), int(m.group(2)))
                continue
            m = re.match(r'(T\d+)(?:[FS][\d.]+)*C([\d.]+)', line)
            if m:
                t = 'T%d' % int(m.group(1)[1:])
                tools[t] = float(m.group(2)) * unit
                tool_func[t] = pending
                pending = None
            continue
        # тело файла
        if line in ('M30', 'M00'):
            break
        m = re.fullmatch(r'T(\d+)(?:C([\d.]+))?', line)
        if m:
            t = 'T%d' % int(m.group(1))
            if m.group(2):                 # инструмент, заданный прямо в теле
                tools[t] = float(m.group(2)) * unit
                tool_func.setdefault(t, None)
            tool = t if int(m.group(1)) else None
            continue
        if line.startswith(('G90', 'G05', 'G93', 'M71', 'M72', 'G40', 'G41', 'G42', 'M17')):
            if line.startswith('M71'):
                unit = 1.0
            elif line.startswith('M72'):
                unit = 25.4
            continue
        if line.startswith('G00'):
            route_start = xy(line[3:])
            x, y = route_start
            plunged = False
            continue
        if line == 'M15':
            plunged = True
            continue
        if line == 'M16':
            route_start = None
            plunged = False
            continue
        if line.startswith('G01') and route_start is not None:
            end = xy(line[3:])
            df.hits.append(DrillHit(route_start, tools[tool], end, tool_func.get(tool)))
            x, y = end
            route_start = end if plunged else None
            continue
        if line.startswith(('X', 'Y')):
            if tool is None or tool not in tools:
                raise ValueError(f'{path.name}: отверстие без выбранного инструмента: {line}')
            if 'G85' in line:
                a, b = line.split('G85')
                start = xy(a)
                x, y = start
                end = xy(b)
                x, y = end
                df.hits.append(DrillHit(start, tools[tool], end, tool_func.get(tool)))
            else:
                x, y = xy(line)
                df.hits.append(DrillHit((x, y), tools[tool], None, tool_func.get(tool)))
            continue
        if line.startswith(('G', 'M')):
            continue                       # прочие режимы станка на геометрию не влияют
        raise ValueError(f'{path.name}: строка сверловки не разобрана: {line!r}')
    if df.plated is None:
        df.plated = 'NPTH' not in path.name.upper()
    return df
