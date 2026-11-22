"""Сверка ядра: два пути к модели (.kicad_pcb и Gerber от KiCad) и нарушения зазоров против DRC KiCad.

    python -m dfm compare плата.kicad_pcb [ещё …] -o папка     отчёт по каждой плате и общая таблица

Для каждой платы:
  1. копия платы, где тексты на меди и маске перенесены на слой Cmts.User (текст в медь модель
     не переводит — нужны шрифты KiCad; через Gerber он пришёл бы готовой геометрией);
  2. kicad-cli: Gerber X2 + Excellon + job-файл и DRC (отчёт JSON, все нарушения каждой дорожки)
     этой копии;
  3. модель из .kicad_pcb и модель из Gerber; по каждому слою меди и маски:
       площадь расхождения (симметрической разности) и максимальное отклонение края — ширина
       самого толстого участка расхождения (диаметр наибольшей вписанной окружности);
     по каждой цепи — есть ли расхождение шире порога; отверстия (место, диаметр, металлизация,
     диапазон слоёв, падстек, цепь, роль); стек; контур; связность обоих путей;
  4. нарушения зазоров по одному порогу против DRC KiCad (если зазор на плате один: только класс
     Default, без местных зазоров и .kicad_dru; иначе сверка нарушений — в пункте 2, правилами).
     Сравниваются множества пар цепей: зазор меньше порога (clearance), касание (shorting_items,
     tracks_crossing) и медь ближе порога к краю платы (copper_edge_clearance).
"""
from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import shapely

from . import kicad_cli
from .board import load
from .board.layers import LayerKind, kind_of

__all__ = ['strip_texts', 'max_deviation', 'wider_than', 'compare_board', 'BoardReport', 'load_report',
           'summary_table']

GAP = 0.005          # мм: расхождение уже 2·GAP = 10 мкм считается допустимым


# ---------- подготовка ----------

_TEXT_TAGS = frozenset({'gr_text', 'gr_text_box', 'fp_text', 'fp_text_box', 'property'})


def strip_texts(src, dst) -> int:
    """Копия платы, где тексты на меди и маске перенесены на слой Cmts.User. Узлы не удаляются,
    чтобы сохранились позиционные обозначения компонентов (KiCad пишет их в атрибуты Gerber).
    Файл проекта копируется рядом. → число перенесённых текстов."""
    from .board import sexpr
    tree = sexpr.load(src)
    moved = 0

    def visit(node):
        nonlocal moved
        for c in node:
            if type(c) is not sexpr.Node or not c:
                continue
            if c[0] in _TEXT_TAGS:
                lay = c.get('layer')
                if lay is not None and len(lay) > 1 and kind_of(str(lay[1])) in (LayerKind.COPPER, LayerKind.MASK):
                    del lay[1:]
                    lay.append('Cmts.User')
                    moved += 1
            elif c[0] == 'footprint':
                visit(c)

    visit(tree)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sexpr.dump(tree, dst)
    pro = Path(src).with_suffix('.kicad_pro')
    if pro.exists():
        shutil.copy(pro, dst.with_suffix('.kicad_pro'))
    return moved


# ---------- метрики геометрии ----------

def max_deviation(diff, tol: float = 1e-5) -> tuple[float, tuple | None]:
    """Наибольшая ширина участка расхождения (диаметр вписанной окружности) с точностью tol и
    точка внутри самого широкого участка.

    Ширина ищется делением пополам: расхождение, сжатое на w/2, не пусто ⇔ где-то оно шире w.
    Точный поиск вписанной окружности по каждому куску здесь в сотни раз медленнее: расхождения
    вдоль заливок — длинные тонкие полоски. Верхняя граница — 2·√(A/π) для самого большого куска."""
    parts = _parts(diff)
    if not parts:
        return 0.0, None
    lo, hi = 0.0, max(_bound(p) for p in parts)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if wider_than(parts, mid):
            lo = mid
        else:
            hi = mid
    cand = [p for p in parts if _bound(p) > lo]
    core = shapely.union_all(cand).buffer(-lo / 2, quad_segs=4) if lo > 0 and cand else shapely.union_all(parts)
    p = core.representative_point() if not core.is_empty else None
    return hi, ((p.x, p.y) if p is not None else None)


def _parts(g) -> list:
    return [p for p in shapely.get_parts(g) if p.geom_type == 'Polygon' and not p.is_empty]


def _bound(p) -> float:
    """Кусок площади A не может быть шире 2·√(A/π) (шире всего — круг)."""
    return 2 * math.sqrt(p.area / math.pi)


def wider_than(diff, w: float) -> bool:
    """Есть ли участок расхождения шире w. Куски, которые не могут быть шире w по площади,
    отбрасываются ДО отрицательного буфера: на тысячах субмикронных осколков буфер GEOS иногда
    оставляет ложный «остаток» (так было на interf_u: осколки 2·10⁻¹¹ мм², а остаток — 5·10⁻⁹)."""
    parts = diff if isinstance(diff, list) else _parts(diff)
    cand = [p for p in parts if _bound(p) > w]
    if not cand:
        return False
    return not shapely.union_all(cand).buffer(-w / 2, quad_segs=4).is_empty


@dataclass
class LayerDiff:
    layer: str
    area_kicad: float
    area_gerber: float
    diff_area: float
    max_dev: float               # мм
    max_dev_at: tuple | None
    nets_differ: int             # цепей, у которых медь расходится шире порога
    nets_only_one: int           # цепей, которые есть только в одной модели


def layer_diffs(kg, gg, layers, with_nets: bool = True) -> list[LayerDiff]:
    out = []
    for lay in layers:
        a, b = kg.layer(lay), gg.layer(lay)
        diff = a.geom.symmetric_difference(b.geom)
        dev, at = max_deviation(diff)
        differ = only = 0
        if with_nets and kind_of(lay) is LayerKind.COPPER:
            for net in set(a.nets) | set(b.nets):
                if net not in a.nets or net not in b.nets:
                    only += 1
                elif wider_than(a.nets[net].symmetric_difference(b.nets[net]), 2 * GAP):
                    differ += 1
        out.append(LayerDiff(lay, a.area, b.area, diff.area, dev, at, differ, only))
    return out


def match_holes(kb, gb, tol: float = 6e-4):
    """Пары отверстий двух моделей. В Excellon координаты и диаметры округлены до 1 мкм.
    → (пары, только в KiCad, только в Gerber)."""
    free = list(gb.holes)
    pairs, missing = [], []
    near = lambda a, b: a is not None and b is not None and abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
    for h in kb.holes:
        for i, g in enumerate(free):
            if g.plated != h.plated or abs(g.diameter - h.diameter) > 1.5e-3 or (g.slot_end is None) != (h.slot_end is None):
                continue
            if near(h.pos, g.pos) and (h.slot_end is None or near(h.slot_end, g.slot_end)) or \
                    h.slot_end is not None and near(h.pos, g.slot_end) and near(h.slot_end, g.pos):
                pairs.append((h, free.pop(i)))
                break
        else:
            missing.append(h)
    return pairs, missing, free


# ---------- нарушения зазоров ----------

_NET = re.compile(r'\[([^\]]*)\]')
_COPPER_TYPES = {'clearance': 'зазор', 'shorting_items': 'касание', 'tracks_crossing': 'касание',
                 'copper_edge_clearance': 'край'}


def single_clearance(board) -> float | None:
    """Порог зазора, если он на плате один (только класс Default, без местных зазоров и .kicad_dru)."""
    d = board.design
    if d is None or d.rules.has_custom_rules:
        return None
    classes = {c.clearance for c in d.rules.netclasses.values() if c.clearance is not None}
    if len(classes) > 1 or d.rules.netclass_patterns or d.rules.netclass_assignments:
        return None
    if any(p.clearance is not None for p in d.pads) or any(fp.clearance is not None for fp in d.footprints):
        return None
    return max(classes | {d.rules.min_clearance}) if classes else d.rules.min_clearance or None


def _key(nets) -> tuple:
    """Пара цепей как ключ множества (без цепи — '<no net>', как в отчёте KiCad)."""
    return tuple(sorted('<no net>' if n is None else n for n in nets))


NO_NET = '<no net>'
DRC_EPS = 0.0005     # мм: допуск DRC KiCad — нарушение, если зазор меньше нормы больше чем на него


def our_violations(board, geo, clearance: float, edge: float) -> dict[str, set]:
    """Нарушения по геометрии: пары кусков меди разных цепей ближе порога (касание — расстояние 0)
    и медь ближе порога к краю платы. Ключ — пара цепей (или одна цепь для края)."""
    from .geometry.neighbors import Neighbors
    out = {'зазор': set(), 'касание': set(), 'край': set()}
    outline = geo.outline_geom()
    boundary = outline.boundary if not outline.is_empty else None
    # Как в DRC KiCad: нарушение — зазор меньше нормы больше чем на DRC_EPS. Это же поглощает
    # погрешность ломаных (дуга дорожки, заменённая хордами, с вогнутой стороны выпирает до tol):
    # на StickHub пары с зазором 0,1495…0,1500 при норме 0,15 KiCad нарушениями не считает.
    clearance -= DRC_EPS
    edge = edge - DRC_EPS if edge > 0 else edge
    for lay in board.copper:
        lc = geo.layer(lay)
        nb = Neighbors(lc)
        for g in nb.pairs_within(clearance):
            out['касание' if g.distance == 0 else 'зазор'].add(_key(g.nets))
        if boundary is not None and edge > 0:
            for k in nb.near(boundary, edge):
                if nb.pieces[k].distance(boundary) < edge:
                    out['край'].add(_key([nb.nets[k]]))
    return out


_ACTUAL = re.compile(r'(\d+[.,]\d+) mm\)\s*$')


def kicad_violations(report: dict) -> dict[str, set]:
    """Нарушения из отчёта DRC KiCad по видам. Нарушение зазора с фактическим зазором 0 — это
    касание (так KiCad сообщает о заливке зоны поверх площадки чужой цепи)."""
    out = {'зазор': set(), 'касание': set(), 'край': set()}
    for v in report.get('violations', []):
        kind = _COPPER_TYPES.get(v['type'])
        if kind is None:
            continue
        if kind == 'зазор':
            m = _ACTUAL.search(v.get('description', ''))
            if m and float(m.group(1).replace(',', '.')) == 0:
                kind = 'касание'
        nets = [m.group(1) for i in v['items'] for m in [_NET.search(i['description'])] if m]
        if kind == 'край':
            out[kind].add(_key(nets[:1]))
        elif len(nets) == 2:
            out[kind].add(_key(nets))
    return out


def drc_diff(ours: dict, theirs: dict, known: dict | None = None) -> tuple[dict, dict]:
    """Сравнение множеств нарушений. Пары с медью без цепи в сравнение не входят (DRC KiCad 10
    медную графику без цепи не проверяет) — они возвращаются отдельно. known — известные
    особенности KiCad, проверенные отдельно: {вид: [[цепь, цепь], …]}."""
    known = {k: {tuple(sorted(p)) for p in v} for k, v in (known or {}).items()}
    # касание поглощает нарушение зазора той же пары цепей (KiCad сообщает о паре дважды, если
    # одни её объекты касаются, а другие просто близко)
    theirs = dict(theirs)
    theirs['зазор'] = theirs.get('зазор', set()) - theirs.get('касание', set())
    res, unnamed = {}, {}
    for k in ours:
        a = {p for p in ours[k] if NO_NET not in p}
        b = {p for p in theirs.get(k, set()) if NO_NET not in p}
        kn = known.get(k, set())
        res[k] = {'оба': len(a & b), 'только мы': sorted(a - b - kn), 'только KiCad': sorted(b - a - kn),
                  'известные особенности KiCad': sorted((a ^ b) & kn)}
        unnamed[k] = sorted(p for p in ours[k] if NO_NET in p)
    return res, unnamed


# ---------- отчёт ----------

@dataclass
class BoardReport:
    board: str
    copper_layers: int
    t_kicad: float
    t_gerber: float
    t_geometry: float
    layers: list[LayerDiff] = field(default_factory=list)
    holes: int = 0
    holes_missing: int = 0
    holes_extra: int = 0
    holes_attr_diff: int = 0
    stackup_equal: bool = True
    outline_equal: bool = True
    shorts: tuple = (0, 0)                 # (KiCad-путь, Gerber-путь)
    missing_connections: tuple = (0, 0)
    drc: dict | None = None                # {вид: {'оба': n, 'только мы': [...], 'только KiCad': [...], …}}
    drc_unnamed: dict | None = None        # нарушения с медью без цепи (только у нас: KiCad её не проверяет)
    drc_note: str = ''
    texts_moved: int = 0
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)   # объяснения расхождений, которые идут от данных

    @property
    def max_dev(self) -> float:
        return max((x.max_dev for x in self.layers), default=0.0)

    @property
    def ok(self) -> bool:
        return (self.max_dev <= 2 * GAP and not self.holes_missing and not self.holes_extra and not self.holes_attr_diff
                and self.stackup_equal and self.outline_equal and self.shorts[0] == self.shorts[1]
                and self.missing_connections[0] == self.missing_connections[1]
                and all(x.nets_differ == 0 and x.nets_only_one == 0 for x in self.layers)
                and (self.drc is None or all(not d['только мы'] and not d['только KiCad'] for d in self.drc.values())))


def compare_board(pcb, out_dir, pictures: bool = True, with_nets: bool = True) -> BoardReport:
    """Полная сверка одной платы; файлы (выгрузки KiCad, картинки, report.md) — в out_dir."""
    from .geometry.connectivity import connectivity
    from .geometry.copper import BoardGeometry
    pcb, out = Path(pcb), Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copy = out / 'board.kicad_pcb'
    moved = strip_texts(pcb, copy)
    g = out / 'gerber'
    kicad_cli.export_gerbers(copy, g)
    kicad_cli.export_drill(copy, g)
    drc = kicad_cli.drc(copy, out / 'drc.json', all_track_errors=True)

    t0 = time.perf_counter()
    kb = load(copy)
    t1 = time.perf_counter()
    gb = load(g)
    t2 = time.perf_counter()
    kg, gg = BoardGeometry(kb), BoardGeometry(gb)
    layers = kb.copper + ['F.Mask', 'B.Mask']
    rep = BoardReport(pcb.stem, len(kb.copper), t1 - t0, t2 - t1, 0.0, texts_moved=moved,
                      warnings=[f'KiCad: {w}' for w in kb.warnings] + [f'Gerber: {w}' for w in gb.warnings])
    rep.layers = layer_diffs(kg, gg, layers, with_nets)
    pairs, missing, extra = match_holes(kb, gb)
    rep.holes, rep.holes_missing, rep.holes_extra = len(pairs), len(missing), len(extra)
    # цепь отверстия без площадок Gerber узнать неоткуда (у Excellon нет цепей) — её не сравниваем
    attrs = lambda h, net: (h.span, sorted(h.padstack), net, h.role)
    rep.holes_attr_diff = sum(attrs(x, x.net) != attrs(h, h.net if x.padstack else None) for h, x in pairs)
    kinds = lambda b: [(i.kind, round(i.thickness or 0, 4), i.material) for i in b.stackup.items] if b.stackup else []
    rep.stackup_equal = kinds(kb) == kinds(gb) and abs((kb.thickness or 0) - (gb.thickness or 0)) < 1e-4
    rep.outline_equal = sorted(len(r.holes) for r in kb.outline) == sorted(len(r.holes) for r in gb.outline)
    ck, cg = connectivity(kg), connectivity(gg)
    rep.shorts = (len(ck.shorts), len(cg.shorts))
    rep.missing_connections = (ck.missing_connections, cg.missing_connections)
    rep.t_geometry = time.perf_counter() - t2
    zero = sum(1 for h in gb.holes if h.plated and h.diameter <= 0)
    if zero and rep.missing_connections[0] != rep.missing_connections[1]:
        rep.notes.append(f'в Excellon {zero} металлизированных отверстий диаметром 0 (в .kicad_pcb — '
                         f'{min(h.diameter for h in kb.holes if h.plated) * 1000:g} мкм): в модели из Gerber они слоёв не '
                         f'соединяют, отсюда разная связность — это дефект данных платы, а не расхождение моделей')

    thr = single_clearance(kb)
    if thr:
        ours = our_violations(kb, kg, thr, kb.design.rules.min_copper_edge_clearance)
        # замыкание через металлизированное отверстие (дорожка касается стенки отверстия чужой
        # площадки, удалённой на этом слое) — тоже касание; KiCad сообщает о нём как shorting_items
        ours['касание'] |= {_key(s.nets) for s in ck.shorts}
        meta = pcb.with_suffix('.json')
        known = json.loads(meta.read_text(encoding='utf-8')).get('drc_known') if meta.exists() else None
        rep.drc, rep.drc_unnamed = drc_diff(ours, kicad_violations(drc), known)
        rep.drc_note = f'порог зазора {thr:g} мм, до края {kb.design.rules.min_copper_edge_clearance:g} мм'
    else:
        rep.drc_note = 'на плате несколько порогов зазора (классы цепей, местные зазоры) — сверка нарушений в пункте 2'

    if pictures:
        from .draw import draw_compare, draw_layers
        draw_layers(kb, kg, out / 'layers')
        draw_compare(kg, gg, rep.layers, out / 'diff')
    (out / 'report.md').write_text(report_md(rep), encoding='utf-8')
    (out / 'report.json').write_text(json.dumps(asdict(rep), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    return rep


def _um(v: float) -> str:
    return f'{v * 1000:.2f}'


def report_md(r: BoardReport) -> str:
    lines = [f'# Сверка: {r.board}', '',
             f'Итог: **{"совпадает" if r.ok else "есть расхождения"}**. Слоёв меди {r.copper_layers}; '
             f'время: чтение .kicad_pcb {r.t_kicad:.1f} с, Gerber {r.t_gerber:.1f} с, геометрия и связность '
             f'обоих путей {r.t_geometry:.1f} с. Перенесено текстов с меди и маски: {r.texts_moved}.', '',
             '| Слой | Площадь KiCad, мм² | Площадь Gerber, мм² | Расхождение, мм² | Макс. отклонение края, мкм | Цепей с расхождением |',
             '|---|---|---|---|---|---|']
    for x in r.layers:
        nets = '—' if not kind_of(x.layer) is LayerKind.COPPER else f'{x.nets_differ} (+{x.nets_only_one} только в одной)'
        lines.append(f'| {x.layer} | {x.area_kicad:.3f} | {x.area_gerber:.3f} | {x.diff_area:.4f} | {_um(x.max_dev)} | {nets} |')
    lines += ['', f'Отверстия: совпало {r.holes}, только в KiCad {r.holes_missing}, только в Gerber {r.holes_extra}, '
                  f'с разными атрибутами (слои, падстек, цепь, роль) {r.holes_attr_diff}.',
              f'Стек: {"совпадает" if r.stackup_equal else "различается"}. Контур: {"совпадает" if r.outline_equal else "различается"}.',
              f'Связность (KiCad-путь / Gerber-путь): замыканий {r.shorts[0]} / {r.shorts[1]}, '
              f'недостающих соединений {r.missing_connections[0]} / {r.missing_connections[1]}.', '',
              f'Нарушения против DRC KiCad: {r.drc_note}.']
    if r.drc:
        pairs = lambda ps: ', '.join('—'.join(p) for p in ps) or '—'
        lines += ['', '| Вид | Совпало | Только у нас | Только у KiCad | Известные особенности KiCad |', '|---|---|---|---|---|']
        for k, d in r.drc.items():
            lines.append(f'| {k} | {d["оба"]} | {pairs(d["только мы"])} | {pairs(d["только KiCad"])} | '
                         f'{pairs(d["известные особенности KiCad"])} |')
        un = {k: v for k, v in (r.drc_unnamed or {}).items() if v}
        if un:
            lines += ['', 'Медь без цепи (DRC KiCad 10 медную графику без цепи не проверяет, в сверку не входит): '
                      + '; '.join(f'{k}: {pairs(v)}' for k, v in un.items()) + '.']
    if r.notes:
        lines += ['', 'Пояснения:', ''] + [f'- {n}' for n in r.notes]
    if r.warnings:
        lines += ['', 'Предупреждения чтения:', ''] + [f'- {w}' for w in r.warnings]
    return '\n'.join(lines) + '\n'


def load_report(path) -> BoardReport:
    """Отчёт report.json, записанный compare_board, → BoardReport (для общей таблицы)."""
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    d['layers'] = [LayerDiff(**x) for x in d['layers']]
    for k in ('shorts', 'missing_connections'):
        d[k] = tuple(d[k])
    if d.get('drc'):
        for v in d['drc'].values():
            for kk in ('только мы', 'только KiCad', 'известные особенности KiCad'):
                v[kk] = [tuple(x) for x in v.get(kk, [])]
    return BoardReport(**d)


def summary_table(reports: list[BoardReport]) -> str:
    lines = ['| Плата | Слоёв | Макс. отклонение края, мкм | Расхождение меди, мм² | Отверстий (расх.) | '
             'Цепи | Связность | Нарушения vs DRC | Время KiCad / Gerber / геом., с | Итог |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in reports:
        cu = [x for x in r.layers if kind_of(x.layer) is LayerKind.COPPER]
        nets = sum(x.nets_differ + x.nets_only_one for x in cu)
        holes = f'{r.holes} ({r.holes_missing + r.holes_extra + r.holes_attr_diff})'
        conn = 'совпадает' if r.shorts[0] == r.shorts[1] and r.missing_connections[0] == r.missing_connections[1] else 'разная'
        if r.drc is None:
            drc = 'н/п'
        else:
            both = sum(d['оба'] for d in r.drc.values())
            diff = sum(len(d['только мы']) + len(d['только KiCad']) for d in r.drc.values())
            drc = f'{both} совп., {diff} расх.'
        lines.append(f'| {r.board} | {r.copper_layers} | {_um(r.max_dev)} | {sum(x.diff_area for x in cu):.4f} | {holes} | '
                     f'{"совпадают" if nets == 0 else f"{nets} расх."} | {conn} | {drc} | '
                     f'{r.t_kicad:.1f} / {r.t_gerber:.1f} / {r.t_geometry:.1f} | {"да" if r.ok else "нет"}'
                     f'{" (" + "; ".join(r.notes) + ")" if r.notes else ""} |')
    return '\n'.join(lines) + '\n'

