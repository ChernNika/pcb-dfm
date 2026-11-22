"""Шаг 2.6, часть 2: внесённые дефекты — полнота и точность правил A1, A3, A5, A7.

Плата-полигон (4 слоя, In1 — GND, In2 — +3V3, маска с расширением 0,1 мм): строки — виды дефектов,
столбцы — три размера дефекта и контрольный (правильный) образец. Для каждой ячейки известно,
какие коды должны найтись. Все классы запускаются вместе с профилем по умолчанию.

    полнота  = доля ячеек с дефектом, где найден ожидаемый код;
    точность = доля находок (предупреждение и ошибка, с местом), относящихся к ожидаемому коду
               ячейки с дефектом; находки в контрольных ячейках и не того кода — ложные.
Сводные находки без места (плата целиком, настройки, «невыполнено») и «информация» в точность
не входят — они перечисляются отдельно.

    python research/validation/inject.py        → research/validation/inject.md, inject.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))

from dfm import kicad_cli                                    # noqa: E402
from dfm.board import load                                   # noqa: E402
from dfm.board.kicad_writer import KicadWriter               # noqa: E402
from dfm.rules.engine import Context, run                    # noqa: E402
from dfm.rules.findings import Severity                      # noqa: E402
from dfm.rules.profile import load_profile                   # noqa: E402

CW, CH = 6.0, 4.5                                            # ячейка, мм
STACK = [('F.Mask', 'Top Solder Mask', 0.01), ('F.Cu', 'copper', 0.035), ('dielectric 1', 'prepreg', 0.165),
         ('In1.Cu', 'copper', 0.035), ('dielectric 2', 'core', 1.165), ('In2.Cu', 'copper', 0.035),
         ('dielectric 3', 'prepreg', 0.165), ('B.Cu', 'copper', 0.035), ('B.Mask', 'Bottom Solder Mask', 0.01)]


class Cell:
    def __init__(self, kind, col, x0, y0, defect, value, expect, allow=()):
        self.kind, self.col, self.x0, self.y0 = kind, col, x0, y0
        self.defect, self.value = defect, value
        self.expect, self.allow = set(expect) if defect else set(), set(allow)
        self.id = f'{kind}{col}'
        self.refs, self.points = set(), []

    def ref(self, n=''):
        r = f'X{self.kind.upper()}{self.col}{n}'
        self.refs.add(r)
        return r

    def contains(self, p, m=0.3):
        return self.x0 - m <= p[0] <= self.x0 + CW + m and self.y0 - m <= p[1] <= self.y0 + CH + m


# ---------- виды дефектов: (ячейка, писатель) → значение; значения дефектов и контроля ----------

def narrow(c, w, v):                                         # A1 Д1: узкая трасса
    n = f'N{c.id}'
    w.track([(c.x0 + 1, c.y0 + 2.25), (c.x0 + 5, c.y0 + 2.25)], v, 'F.Cu', n)


def gap(c, w, v):                                            # A1 Д4: зазор между цепями
    y = c.y0 + 2
    w.track([(c.x0 + 1, y), (c.x0 + 5, y)], 0.2, 'F.Cu', f'N{c.id}a')
    w.track([(c.x0 + 1, y + 0.2 + v), (c.x0 + 5, y + 0.2 + v)], 0.2, 'F.Cu', f'N{c.id}b')


def small_ring(c, w, v):                                     # A3: малая площадка переходного
    w.via((c.x0 + 3, c.y0 + 2.25), v, 0.30, net=f'N{c.id}', remove_unused=False)


def offset(c, w, v):                                         # A3 К4: площадка со смещением
    fp = w.footprint(c.ref(), (c.x0 + 3, c.y0 + 2.25), attrs='through_hole')
    fp.pad('1', 'thru_hole', 'rect', (0, 0), 1.2, drill=0.8, offset=(v, 0.0) if v else None, net=f'N{c.id}',
           remove_unused=True, keep_end=True)


def inner(c, w, v):                                          # A3 К5: малые внутренние площадки
    w.via((c.x0 + 3, c.y0 + 2.25), 0.55, 0.30, net=f'N{c.id}', remove_unused=False,
          padstack=('front_inner_back', {'Inner': v, 'B.Cu': 0.55}))


def antipad(c, w, v):                                        # A5 З1/З2: малое вскрытие
    n = f'N{c.id}'
    w.via((c.x0 + 3, c.y0 + 2.25), 0.6, 0.30, net=n, remove_unused=True, keep_end=True)
    if v:
        w.dru.append(f"(rule \"{n}\" (condition \"A.NetName == '{n}'\") (constraint clearance (min {v}mm)))")


def nfp(c, w, v):                                            # A5 З3: оставленная площадка близко к слою
    n = f'N{c.id}'
    w.via((c.x0 + 3, c.y0 + 2.25), 0.6, 0.30, net=n, remove_unused=False)
    w.dru.append(f"(rule \"{n}\" (condition \"A.NetName == '{n}'\") (constraint clearance (min {v}mm)))")


def stale(c, w, v):                                          # A5 З8: переходное сдвинуто после заливки
    x, y = c.x0 + 3, c.y0 + 2.25
    w.via((x, y), 0.6, 0.30, net=f'N{c.id}', remove_unused=True, keep_end=True)
    c.move = ((x, y), (round(x + v, 4), y)) if v else None


def zero_mask(c, w, v):                                      # A7 М2: явное нулевое расширение
    fp = w.footprint(c.ref(), (c.x0 + 3, c.y0 + 2.25))
    fp.pad('1', 'smd', 'rect', (-0.9, 0), (1.0, 1.2), net=f'N{c.id}a', mask_margin=v)
    fp.pad('2', 'smd', 'rect', (0.9, 0), (1.0, 1.2), net=f'N{c.id}b', mask_margin=v)


def foreign(c, w, v):                                        # A7 М3: чужая трасса у крупного окна
    fp = w.footprint(c.ref(), (c.x0 + 2.5, c.y0 + 2.25))
    fp.pad('1', 'smd', 'rect', (0, 0), 1.6, net=f'N{c.id}a', mask_margin=0.15 if v < 0.3 else None)
    x = c.x0 + 2.5 + 0.8 + v + 0.1                           # трасса 0,2 мм: край в v от края площадки
    w.track([(x, c.y0 + 0.5), (x, c.y0 + 4.0)], 0.2, 'F.Cu', f'N{c.id}b')


def partial(c, w, v):                                        # A7 М4: окно сдвинуто с переходного
    x, y = c.x0 + 3, c.y0 + 2.25
    w.via((x, y), 0.6, 0.30, net=f'N{c.id}', remove_unused=True, keep_end=True)
    if v:
        w.graphic('circle', 'F.Mask', (x + v, y), (x + v + 0.2, y), width=0, fill=True)


def dams(c, w, v):                                           # A7 М1: перемычки между выводами
    fp = w.footprint(c.ref(), (c.x0 + 3, c.y0 + 2.25))
    for i in range(6):
        fp.pad(str(i + 1), 'smd', 'rect', ((i - 2.5) * v, 0), (0.25, 0.8), net=f'N{c.id}_{i}')


def open_via(c, w, v):                                       # A7 М5: открытое переходное у площадки
    n = f'N{c.id}'
    fp = w.footprint(c.ref(), (c.x0 + 2.5, c.y0 + 2.25))
    fp.pad('1', 'smd', 'rect', (0, 0), (0.5, 0.55), net=n)
    fp.pad('2', 'smd', 'rect', (-1.0, 0), (0.5, 0.55), net=f'{n}x')
    x = c.x0 + 2.5 + 0.25 + 0.1 + v + 0.1 + 0.3              # окно переходного Ø0,8 в v от окна площадки
    w.via((x, c.y0 + 2.25), 0.6, 0.30, net=n, remove_unused=True, keep_end=True, tenting=(v > 0.5, True))
    w.track([(c.x0 + 2.5, c.y0 + 2.25), (x, c.y0 + 2.25)], 0.2, 'F.Cu', n)


def edge(c, w, v):                                           # A7 М6: окно у края платы
    fp = w.footprint(c.ref(), (c.x0 + 3, v + 0.4))           # окно (0,6 + 2·0,1) нижним краем в v от края
    fp.pad('1', 'smd', 'rect', (0, 0), 0.6, net=f'N{c.id}')


def _cols(vals, ctrl):
    return [(v, True) for v in vals] + [(ctrl, False)]


# имя, функция, столбцы [(значение, дефект ли)], ожидаемые коды, допустимые побочные
KINDS = [
    ('narrow', narrow, _cols((0.04, 0.05, 0.06), 0.12), {('A1', 'Д1')}, ()),
    ('gap', gap, _cols((0.04, 0.06, 0.075), 0.15), {('A1', 'Д4')}, ()),
    ('ring', small_ring, _cols((0.36, 0.40, 0.42), 0.6), {('A3', 'К1'), ('A3', 'К2'), ('A3', 'К5')}, ()),
    ('offset', offset, _cols((0.15, 0.17, 0.19), 0.0), {('A3', 'К4')}, ()),
    ('inner', inner, _cols((0.40, 0.42, 0.45), 0.55), {('A3', 'К5')}, ()),
    ('antipad', antipad, _cols((0.125, 0.15, 0.175), 0.0), {('A5', 'З1'), ('A5', 'З2')}, ()),
    ('nfp', nfp, _cols((0.05, 0.07, 0.09), 0.15), {('A5', 'З3')}, {('A1', 'Д4')}),   # медь к меди < s_min A1 — тоже верно
    ('stale', stale, _cols((0.10, 0.15, 0.20), 0.0), {('A5', 'З8')}, ()),
    ('zero', zero_mask, _cols((0.0, 0.0, 0.0), None), {('A7', 'М2')}, ()),
    ('foreign', foreign, _cols((0.10, 0.12, 0.14), 0.4), {('A7', 'М3')}, ()),
    ('partial', partial, _cols((0.10, 0.15, 0.20), 0.0), {('A7', 'М4')}, {('A7', 'М6')}),
    ('dams', dams, _cols((0.45, 0.50, 0.52), 0.65), {('A7', 'М1')}, ()),
    ('openvia', open_via, _cols((0.0, 0.03, 0.06), 1.0), {('A7', 'М5')}, {('A7', 'М6')}),
    ('edge', edge, _cols((0.03, 0.05, 0.07), 0.3), {('A7', 'М6')}, ()),
]

# у порога: два значения чуть хуже порога (дефект) и два чуть лучше (не дефект). Пороги условного
# профиля: ширина наружного слоя 79,4 мкм, зазор 90 мкм, перемычка и край маски 100 мкм, зазор
# от площадки до слоя 100 мкм; вскрытие — по шкале вероятностей: от ствола 180 и 200 мкм дают
# 0,23 и 0,039 (ошибка, предупреждение), 225 и 245 мкм — 0,003 и 2·10⁻⁴ (информация, норма)
THRESHOLD = [
    ('narrow', narrow, [(0.074, True), (0.078, True), (0.081, False), (0.085, False)], {('A1', 'Д1')}, ()),
    ('gap', gap, [(0.084, True), (0.088, True), (0.092, False), (0.096, False)], {('A1', 'Д4')}, ()),
    ('dams', dams, [(0.54, True), (0.545, True), (0.60, False), (0.65, False)], {('A7', 'М1')}, ()),
    ('nfp', nfp, [(0.095, True), (0.098, True), (0.102, False), (0.105, False)], {('A5', 'З3')}, ()),
    ('antipad', antipad, [(0.205, True), (0.225, True), (0.25, False), (0.27, False)], {('A5', 'З2')}, ()),
    # край — последней строкой: его ячейки стоят у нижнего края платы
    ('edge', edge, [(0.09, True), (0.095, True), (0.105, False), (0.11, False)], {('A7', 'М6')}, ()),
]


def build(name, kinds):
    rows = len(kinds)
    W, H = 4 * CW + 1, rows * CH + 1
    w = KicadWriter(name, copper=4, thickness=1.655, clearance=0.2, track_width=0.2, mask_margin=0.1,
                    stackup=STACK, rules={'max_error': 0.001})
    w.outline_rect(0, 0, W, H)
    plane = [(0.5, 0.5), (W - 0.5, 0.5), (W - 0.5, H - 0.5), (0.5, H - 0.5)]
    w.zone('GND', 'In1.Cu', plane, clearance=0.2, min_thickness=0.1, connect='solid', island_removal=1)
    w.zone('+3V3', 'In2.Cu', plane, clearance=0.2, min_thickness=0.1, connect='solid', island_removal=1)
    w.dru.append("(rule \"вскрытие\" (condition \"A.Type == 'Via' || A.Type == 'Pad'\") "
                 "(constraint clearance (min 0.275mm)))")
    cells = []
    for r, (kind, fn, cols, expect, allow) in enumerate(kinds):
        y0 = 0.5 + (rows - 1 - r) * CH if kind != 'edge' else 0.0       # край — нижний ряд у края платы
        for col, (v, defect) in enumerate(cols):
            c = Cell(kind, col, 0.5 + col * CW, y0, defect, v, expect, allow)
            fn(c, w, None if (kind == 'zero' and not defect) else v)   # контроль «zero»: расширение платы
            cells.append(c)
    return w, cells


def move_after_refill(path, cells):
    t = path.read_text(encoding='utf-8')
    fmt = lambda v: f'{v:.6f}'.rstrip('0').rstrip('.')
    for c in cells:
        m = getattr(c, 'move', None)
        if not m:
            continue
        (x, y), (nx, ny) = m
        old = f'(at {fmt(x)} {fmt(-y)})'
        assert t.count(old) == 1, old
        t = t.replace(old, f'(at {fmt(nx)} {fmt(-ny)})')
    path.write_text(t, encoding='utf-8', newline='\n')


def locate(f, cells):
    pts = []
    if f.pos is not None:
        pts.append(f.pos)
    pts += [tuple(p) for p in f.measured.get('позиции', []) or []]
    refs = {o.get('ref') for o in f.objects if o.get('ref')}
    hit = {c.id for c in cells if any(c.contains(p) for p in pts) or (refs & c.refs)}
    return hit


def evaluate(name, kinds):
    out = HERE / 'inject_board'
    out.mkdir(exist_ok=True)
    w, cells = build(name, kinds)
    p = w.write(out / f'{name}.kicad_pcb')
    kicad_cli.refill_zones(p)
    p.with_suffix('.kicad_prl').unlink(missing_ok=True)
    move_after_refill(p, cells)
    b = load(p)
    fs = run(b, ctx=Context(b, load_profile()))
    by_id = {c.id: c for c in cells}
    found = defaultdict(set)                                  # ячейка → коды
    claims, info, summary = [], [], []
    for f in fs:
        if f.severity in (Severity.NOT_CHECKED,):
            summary.append(f)
            continue
        hit = locate(f, cells)
        if not hit:
            (summary if f.severity is not Severity.INFO else info).append(f)
            continue
        for cid in hit:
            found[cid].add((f.cls, f.code))
        (claims if f.severity in (Severity.WARNING, Severity.ERROR) else info).append((f, hit))
    tp = fp = 0
    fp_list = []
    for f, hit in claims:
        ok = any((f.cls, f.code) in by_id[c].expect | (by_id[c].allow if by_id[c].defect else set()) for c in hit)
        tp += ok
        fp += not ok
        if not ok:
            fp_list.append((f, hit))
    rows = []
    for kind, *_ in kinds:
        cs = [c for c in cells if c.kind == kind]
        det = [c for c in cs if c.defect and found[c.id] & c.expect]
        ctrl_bad = [c for c in cs if not c.defect and any(
            (f.cls, f.code) and f.severity in (Severity.WARNING, Severity.ERROR) for f, h in claims if c.id in h)]
        rows.append(dict(kind=kind, values=[c.value for c in cs if c.defect], control=[c.value for c in cs if not c.defect],
                         detected=f'{len(det)}/{sum(c.defect for c in cs)}',
                         codes={c.id: sorted(f'{a} {b}' for a, b in found[c.id]) for c in cs},
                         control_false=len(ctrl_bad)))
    n_def = sum(c.defect for c in cells)
    n_det = sum(1 for c in cells if c.defect and found[c.id] & c.expect)
    res = dict(cells=len(cells), defects=n_def, detected=n_det, recall=n_det / n_def,
               claims=len(claims), true=tp, false=fp, precision=tp / max(len(claims), 1), rows=rows,
               false_list=[dict(code=f'{f.cls} {f.code}', sev=f.severity.ru, title=f.title, cells=sorted(h)) for f, h in fp_list],
               info=[dict(code=f'{f.cls} {f.code}', title=f.title) for f in (x[0] if isinstance(x, tuple) else x for x in info)],
               summary=[dict(code=f'{f.cls} {f.code}', sev=f.severity.ru, title=f.title) for f in summary])
    (HERE / f'{name}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(f'{name}: дефектов {n_def}, найдено {n_det} — полнота {res["recall"]:.1%}; находок {len(claims)}, '
          f'ложных {fp} — точность {res["precision"]:.1%}')
    for r in rows:
        print(f'  {r["kind"]:8} {r["detected"]:5} ложных в контроле {r["control_false"]}  ', r['codes'])
    for x in res['false_list']:
        print('  ЛОЖНАЯ', x['code'], x['sev'], x['cells'], x['title'][:120])
    return res


def main():
    return evaluate('inject', KINDS), evaluate('inject_threshold', THRESHOLD)


if __name__ == '__main__':
    main()
