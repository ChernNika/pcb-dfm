"""Адаптер Gerber: папка с Gerber X2 + Excellon + job-файлом (.gbrjob) → модель платы (уровень P).

    board = load_gerber('папка')      # то же делает dfm.board.load('папка')

Что откуда берётся:
  * назначение файла (медь Lk, маска, контур) — из job-файла, иначе из атрибута %TF.FileFunction,
    иначе по имени файла (расширения Protel: .gtl .gbl .g1 … .gts .gbs .gm1 .gko; имена KiCad
    F_Cu, In1_Cu, B_Mask, Edge_Cuts). Медный слой Lk получает имя модели: L1 → F.Cu, последний —
    B.Cu, остальные — In(k−1).Cu, как в KiCad;
  * медь и маска — объекты Gerber с точными формами (gerber_format.py); роль — из .AperFunction
    (площадка, переходное, проводник; регион-проводник считается заливкой), цепь — из .N
    (N/C — без цепи), компонент и вывод — из .P/.C;
  * окна маски получают роль, цепь, компонент и вывод от медной площадки в той же точке;
  * отверстия — из Excellon: металлизация и пара слоёв из атрибутов X2; площадки отверстия по
    слоям (падстек) — объекты меди, которые накрывают центр отверстия;
  * контур — линии и дуги файла Profile (Edge.Cuts), сшитые как в адаптере KiCad;
  * стек и толщина платы — из job-файла (εr KiCad в русской Windows пишет строкой «4,45»).
Если атрибутов цепей нет (Gerber X1 или выгрузка без netlist), цепи восстанавливаются по
геометрии: каждая связная компонента меди (с учётом металлизированных отверстий) — своя цепь
с именем N$1, N$2 … (connectivity.assign_nets).

Уровня D (объектов KiCad) у такой платы нет: board.design = None.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import shapely

from .excellon import is_excellon, read_excellon
from .gerber_format import GerberError, parse_function, read_gerber
from .model import Board, Feature, Hole, Role, StackItem, StackKind, Stackup
from .outline import stitch
from .shapes import Contour, Stroke

__all__ = ['load_gerber']

_PAD = frozenset({'ComponentPad', 'SMDPad', 'BGAPad', 'ConnectorPad', 'HeatsinkPad', 'WasherPad', 'TestPad',
                  'CastellatedPad', 'FiducialPad', 'ThermalReliefPad', 'OtherPad'})
_GRAPHIC = frozenset({'NonConductor', 'EtchedComponent', 'Other', 'Profile', 'Drawing', 'Material'})

# расширения Protel и имена KiCad → (вид, сторона/номер)
_EXT = {'.gtl': ('Copper', 'Top'), '.gbl': ('Copper', 'Bot'), '.gts': ('Soldermask', 'Top'),
        '.gbs': ('Soldermask', 'Bot'), '.gm1': ('Profile', None), '.gko': ('Profile', None),
        '.gml': ('Profile', None), '.gm': ('Profile', None)}
_KICAD_NAME = [(re.compile(r'[-_]F_Cu\b'), ('Copper', 'Top')), (re.compile(r'[-_]B_Cu\b'), ('Copper', 'Bot')),
               (re.compile(r'[-_]In(\d+)_Cu\b'), ('Copper', 'Inr')), (re.compile(r'[-_]F_Mask\b'), ('Soldermask', 'Top')),
               (re.compile(r'[-_]B_Mask\b'), ('Soldermask', 'Bot')), (re.compile(r'[-_]Edge_Cuts\b'), ('Profile', None))]


# слои, которые модели не нужны (шелкография, паста, клей, сборочные, пользовательские): по имени
_OTHER_NAME = re.compile(r'\.(gto|gbo|gtp|gbp|gta|gba)$|Silk|Paste|Adhes|Fab|Courtyard|CrtYd|User|Margin|'
                         r'Eco\d|Dwgs|Cmts|Legend|Assembly', re.IGNORECASE)


def _role(func: list[str], kind: str) -> Role:
    if not func:
        return Role.OTHER
    f = func[0]
    if f in _PAD:
        return Role.PAD
    if f == 'ViaPad':
        return Role.VIA
    if f == 'Conductor':
        return Role.ZONE if kind == 'region' else Role.TRACK
    if f in _GRAPHIC:
        return Role.GRAPHIC
    return Role.OTHER


def _is_gerber(p: Path) -> bool:
    try:
        with open(p, encoding='utf-8', errors='replace') as f:
            head = f.read(4000)
    except OSError:
        return False
    return '%FS' in head or '%MO' in head


def _header_function(p: Path) -> str | None:
    with open(p, encoding='utf-8', errors='replace') as f:
        for _ in range(40):
            line = f.readline()
            if not line:
                break
            if line.startswith('%TF.FileFunction,'):
                return line.strip()[len('%TF.FileFunction,'):].rstrip('*%')
    return None


def _guess_function(p: Path):
    """Назначение файла по имени → (вид, сторона, номер внутреннего слоя) или None."""
    name, ext = p.stem, p.suffix.lower()
    m = re.fullmatch(r'\.g(\d+)', ext) or re.fullmatch(r'\.gl(\d+)', ext)
    if m:
        return 'Copper', 'Inr', int(m.group(1))
    if ext in _EXT:
        return (*_EXT[ext], None)
    for rx, (kind, side) in _KICAD_NAME:
        m = rx.search(name)
        if m:
            return kind, side, int(m.group(1)) if m.groups() else None
    return None


def _num(v):
    if isinstance(v, str):
        try:
            return float(v.replace(',', '.'))
        except ValueError:
            return None
    return float(v) if isinstance(v, (int, float)) else None


def _stackup(job: dict, copper: list[str]) -> Stackup | None:
    items = job.get('MaterialStackup')
    if not items:
        return None
    kinds = {'Copper': StackKind.COPPER, 'Dielectric': StackKind.DIELECTRIC, 'SolderMask': StackKind.MASK,
             'SolderPaste': StackKind.PASTE, 'Legend': StackKind.SILK}
    out, cu, diel, seen = [], iter(copper), 0, Counter()
    for m in items:
        kind = kinds.get(m.get('Type'))
        if kind is None:
            continue
        if kind is StackKind.COPPER:
            name = next(cu, m.get('Name', 'Cu'))
        elif kind is StackKind.DIELECTRIC:
            diel += 1
            name = f'dielectric {diel}'
        else:
            side = 'F' if seen[kind] == 0 else 'B'
            seen[kind] += 1
            name = side + {StackKind.MASK: '.Mask', StackKind.PASTE: '.Paste', StackKind.SILK: '.SilkS'}[kind]
        out.append(StackItem(name, kind, _num(m.get('Thickness')), m.get('Material'), _num(m.get('DielectricConstant')),
                             _num(m.get('LossTangent')), None, m.get('Color')))
    finish = job.get('GeneralSpecs', {}).get('Finish')
    return Stackup(out, finish=finish if finish not in (None, '', 'None') else None)


def _features(gf, layer_name: str) -> tuple[list[Feature], dict, bool]:
    """Объекты файла Gerber → (Feature, точки вспышек {(x, y): Feature}, есть ли атрибуты цепей .N)."""
    feats, flashes = [], {}
    has_net = False
    for o in gf.objects:
        func = parse_function(o.attrs.get('AperFunction'))
        net = o.attrs.get('N')
        has_net = has_net or net is not None
        net = None if net in (None, '', 'N/C') else net
        comp = pin = None
        if 'P' in o.attrs:
            p = o.attrs['P'].split(',')
            comp, pin = p[0], (p[1] if len(p) > 1 else None)
        elif 'C' in o.attrs:
            comp = o.attrs['C']
        f = Feature(o.shape, net, _role(func, o.kind), o.dark, None, comp, pin)
        feats.append(f)
        if o.kind == 'flash' and o.pos is not None:
            flashes[(round(o.pos[0], 4), round(o.pos[1], 4))] = f
    return feats, flashes, has_net


def _outline(gf) -> tuple[list[Contour], list]:
    """Линии и дуги файла контура → (замкнутые контуры, незамкнутые куски)."""
    closed, pieces = [], []
    for o in gf.objects:
        s = o.shape
        if o.kind != 'draw' or not isinstance(s, Stroke) or len(s.pts) < 2:
            continue
        pts, arcs = s.pts, dict(s.arcs)
        if np.array_equal(pts[0], pts[-1]):          # замкнутая линия (например, окружность одной дугой)
            closed.append(Contour(pts[:-1].copy(), {i: m for i, m in arcs.items() if i < len(pts) - 1}))
        else:
            pieces.append((pts, arcs))
    return closed, pieces


def load_gerber(folder) -> Board:
    folder = Path(folder)
    if folder.is_file():
        folder = folder.parent
    warnings: list[str] = []
    files = sorted(p for p in folder.iterdir() if p.is_file())
    job_path = next((p for p in files if p.suffix.lower() == '.gbrjob'), None)
    job = json.loads(job_path.read_text(encoding='utf-8')) if job_path else {}
    if not job:
        warnings.append('нет job-файла (.gbrjob): стек и толщина платы неизвестны')

    # назначение файлов: (путь, вид, сторона, номер, номер взят из атрибута)
    job_funcs = {(folder / fa['Path']).resolve(): fa.get('FileFunction') for fa in job.get('FilesAttributes', [])}
    found = []
    drills = []
    unknown = []
    for p in files:
        if p == job_path:
            continue
        if is_excellon(p):
            drills.append(p)
            continue
        if not _is_gerber(p):
            continue
        func = job_funcs.get(p.resolve()) or _header_function(p)
        if func:
            f = parse_function(func)
            kind = 'Soldermask' if f[0].lower() == 'soldermask' else f[0]
            if kind == 'Copper':
                found.append((p, 'Copper', f[2] if len(f) > 2 else None, int(f[1][1:]), True))
            elif kind in ('Soldermask', 'Profile'):
                found.append((p, kind, f[1] if len(f) > 1 else None, None, False))
            continue
        g = _guess_function(p)
        if g is None:
            if not _OTHER_NAME.search(p.name):
                unknown.append(p.name)
        else:
            found.append((p, *g, False))
    if unknown:
        warnings.append('назначение файлов не определено, они пропущены: ' + ', '.join(unknown))
    cu_files = [x for x in found if x[1] == 'Copper']
    if not cu_files:
        raise ValueError(f'{folder}: не найдено ни одного файла меди Gerber')

    # число медных слоёв: из job-файла, иначе наибольший номер Lk, иначе 2 + внутренние по именам
    n = int(job.get('GeneralSpecs', {}).get('LayerNumber') or 0)
    if not n:
        nums = [num for _, _, _, num, attr in cu_files if attr]
        n = max(nums) if nums else 2 + sum(1 for x in cu_files if x[2] == 'Inr')
    copper = ['F.Cu'] + [f'In{k}.Cu' for k in range(1, n - 1)] + ['B.Cu'] if n >= 2 else ['F.Cu']

    def name_of(kind, side, num, from_attr):
        if kind == 'Copper':
            if from_attr:
                return copper[num - 1] if 1 <= num <= n else None
            if side == 'Top':
                return 'F.Cu'
            if side == 'Bot':
                return 'B.Cu'
            return f'In{num}.Cu' if num and num <= n - 2 else None
        if kind == 'Soldermask':
            return 'F.Mask' if side == 'Top' else 'B.Mask'
        return 'Edge.Cuts'

    board = Board(name=job.get('GeneralSpecs', {}).get('ProjectId', {}).get('Name') or folder.name,
                  source='gerber', warnings=warnings)
    for name in copper + ['F.Mask', 'B.Mask', 'Edge.Cuts']:
        board.add_layer(name)
    flash_maps = {}
    net_attrs = False
    profile = None
    parsed = {}
    for p, kind, side, num, from_attr in found:
        lay = name_of(kind, side, num, from_attr)
        if lay is None:
            warnings.append(f'файл {p.name}: медный слой вне стека из {n} слоёв — пропущен')
            continue
        if lay in parsed:
            warnings.append(f'для слоя {lay} два файла: {parsed[lay].name} и {p.name}; взят первый')
            continue
        parsed[lay] = p
        try:
            gf = read_gerber(p)
        except GerberError as e:
            raise ValueError(str(e)) from None
        if lay == 'Edge.Cuts':
            profile = gf
            continue
        board.layers[lay].features, flash_maps[lay], has_net = _features(gf, lay)
        net_attrs = net_attrs or (has_net and lay in copper)

    # окна маски: роль, цепь, компонент и вывод — от площадки меди в той же точке
    for mask, cu in (('F.Mask', copper[0]), ('B.Mask', copper[-1])):
        pads = flash_maps.get(cu, {})
        for key, mf in flash_maps.get(mask, {}).items():
            src = pads.get(key)
            if src is not None and src.role in (Role.PAD, Role.VIA):
                mf.role, mf.net, mf.component, mf.pin = src.role, src.net, src.component, src.pin

    # контур
    if profile is not None:
        closed, pieces = _outline(profile)
        board.outline, problems = stitch(closed, pieces)
        warnings.extend(problems)
    else:
        warnings.append('нет файла контура (Profile / Edge.Cuts) — контур платы не построен')

    # отверстия
    if not drills:
        warnings.append('нет файлов сверловки Excellon — отверстий нет')
    _holes(board, drills, copper, warnings)

    # стек
    if job:
        board.stackup = _stackup(job, copper)
        board.thickness = _num(job.get('GeneralSpecs', {}).get('BoardThickness'))
    board.nets = {f.net for lay in board.layers.values() for f in lay.features if f.net is not None}
    if not net_attrs:
        from ..geometry.connectivity import assign_nets
        assign_nets(board)
        warnings.append('в Gerber нет атрибутов цепей — цепи восстановлены по геометрии (N$1, N$2 …)')
    return board


def _holes(board: Board, drills: list[Path], copper: list[str], warnings: list[str]) -> None:
    from ..geometry.polygons import to_geometry
    # площадки, которые могут быть падстеком: вспышки и объекты с ролью площадки / переходного
    trees = {}
    for lay in copper:
        cands = [f for f in board.layers[lay].features if f.dark and f.role in (Role.PAD, Role.VIA, Role.OTHER)]
        geoms = [to_geometry(f.shape, 0.001) for f in cands]
        trees[lay] = (shapely.STRtree(geoms), cands, geoms) if cands else None
    for path in drills:
        df = read_excellon(path)
        if not df.attributes:
            warnings.append(f'{path.name}: нет атрибутов X2 — металлизация по имени файла, отверстия сквозные')
        if df.layers and 1 <= df.layers[0] <= len(copper) and 1 <= df.layers[1] <= len(copper):
            a, b = sorted(df.layers)
            span = (copper[a - 1], copper[b - 1])
        else:
            span = (copper[0], copper[-1])
        i0, i1 = copper.index(span[0]), copper.index(span[1])
        for h in df.hits:
            role = Role.VIA if h.function == 'ViaDrill' else Role.PAD if h.function else None
            center = h.pos if h.end is None else ((h.pos[0] + h.end[0]) / 2, (h.pos[1] + h.end[1]) / 2)
            # площадка на слое — объект меди, накрывающий центр отверстия (у неметаллизированного
            # отверстия тоже бывает медный поясок); из нескольких — с той же ролью и самый маленький
            stack = {}
            for lay in copper[i0:i1 + 1]:
                t = trees.get(lay)
                if t is None:
                    continue
                idx = t[0].query(shapely.points(*center), predicate='intersects')
                if not len(idx):
                    continue
                best = sorted(idx, key=lambda k: (role is not None and t[1][k].role is not role, t[2][k].area))[0]
                stack[lay] = t[1][best]
            if role is None:
                role = Role.VIA if any(f.role is Role.VIA for f in stack.values()) else Role.PAD
            nets = Counter(f.net for f in stack.values() if f.net)
            pad = next((f for f in stack.values() if f.component), None)
            board.holes.append(Hole(h.pos, h.diameter, df.plated, span, nets.most_common(1)[0][0] if nets else None,
                                    role, h.end, stack, None, pad.component if pad else None,
                                    pad.pin if pad else None))
