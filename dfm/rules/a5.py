"""Класс A5 — сплошные слои и зоны вскрытия (документ A5).

Для каждого сплошного слоя L (заливка покрывает ≥ plane_fill_fraction платы) и каждого
металлизированного отверстия другой цепи, проходящего через L:

    граница потенциала — оставленная площадка (нефункциональная) или ствол Ø d_св = d + 2t_мет + Δ;
    c = расстояние от неё до меди других цепей на L (измеряется по заливке, а не по настройкам);
    запас m = c − ℓ_w − c_тр(V), вероятность нарушения P = exp(−m²/2σ²) с σ слоя из registration.py.

  З1  c − ℓ_w < c_тр — нарушение на любой плате;
  З2  запаса на смещение нет — по вероятности; на плату — модель общих и местных составляющих;
  З3  у неподключённого отверстия оставлена площадка — зазор от её края;
  З4  не дефект платы, а проверки: зазор считается от сверла, а не от готового диаметра
      (KiCad строит вскрытие от ГОТОВОГО отверстия — на толщину металлизации меньше, чем кажется);
  З5  вскрытия слились в прорезь (несколько отверстий в одном вырезе слоя);
  З6  узкие перемычки слоя между вскрытиями; для полей — проводимость окна (уравнение Лапласа);
  З7  зазор по напряжению: c_тр = max(c_тр, D(V)) по таблице профиля (категория B1);
  З8  устаревшая заливка: зазор по сохранённой заливке меньше, чем после перезаливки.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import shapely

from ..board.model import Role
from ..board.shapes import Circle, Stroke
from ..geometry.copper import _subdivide, parts
from ..geometry.polygons import to_geometry
from .engine import MIN_HOLE_MM, rule
from .findings import Finding, Severity, probability_severity
from .profile import product_class
from .registration import hole_sigmas, layer_depths, p_board, p_breakout

from .voltage import max_voltage_difference, required_voltage_clearance, voltage_difference

__all__ = ['check', 'plane_layers', 'barrel_mm', 'measure', 'window_conductance', 'maxwell_garnett', 'antipad_min_mm']

FINE = 1e-5
EPS_UM = 1.0                                   # мкм: допуск номинальных проверок зазора


def web_tol(ctx) -> float:
    """Допуск ширины перемычки, мм: только погрешность собственного построения (точные формы → многоугольники
    с допуском 0,5 мкм, по 0,5 мкм с каждой стороны перемычки). Многоугольник вскрытия в файле (KiCad строит
    окружности снаружи с отклонением до max_error) — это то, что травит завод: сужение от него — реальное."""
    return 2 * ctx.tol


def _pct(p: float) -> str:
    v = p * 100
    return f'{v:.3g}' if v >= 0.01 else f'{v:.1e}'


def barrel_mm(d_mm: float, drilling: dict, via: bool = False) -> float:
    """Диаметр ствола (сверла): в KiCad записан готовый диаметр, сверло больше на 2t_мет + Δ
    (Δ — allowance_um, у переходных via_allowance_um, если задан)."""
    if drilling.get('drill_is_finished', True):
        allow = drilling.get('via_allowance_um', drilling.get('allowance_um', 0)) if via else drilling.get('allowance_um', 0)
        return d_mm + 2 * drilling.get('plating_um', 25) / 1000 + allow / 1000
    return d_mm


def wicking_um(planes: dict, product_class: int) -> tuple[float, float | None]:
    """(проектное затекание + подтрав диэлектрика, предельное по приёмке + подтрав или None), мкм.
    Проектное — wicking_design_um (ECSS-Q-ST-70-12C Rev.1, п. 13.8.2g: «+50 µm to allow for wicking»), без него —
    прежнее wicking_max_um; предельное — wicking_accept_max_um[класс] (ГОСТ Р 55693-2013, табл. 10: 125/100/80) —
    только предупреждение «при худшем допустимом затекании зазора не хватит»."""
    eb = planes.get('etchback_um', 0.0)
    design = planes.get('wicking_design_um', planes.get('wicking_max_um', 25.0)) + eb
    acc = planes.get('wicking_accept_max_um', {}).get(f'class{product_class}')
    return design, (acc + eb if acc is not None else None)


def antipad_min_mm(edge_mm: float, c_req_um: float, wick_um: float, sigma_um: float, p_star: float) -> float:
    """D_a,min = d_св (или D_нфп) + 2(c_тр + ℓ_w) + 2σ√(−2 ln P*) (документ A5, раздел 4.2)."""
    return edge_mm + 2 * (c_req_um + wick_um) / 1000 + 2 * sigma_um * math.sqrt(-2 * math.log(p_star)) / 1000


def maxwell_garnett(phi: float) -> float:
    """Проводимость слоя с решёткой круглых вырезов доли φ: (1 − φ)/(1 + φ) (точна до φ ≈ 0,4)."""
    return (1 - phi) / (1 + phi)


def plane_layers(ctx) -> dict[str, set]:
    """Сплошные слои → цепи их заливок. Слой сплошной, если заливка (всех цепей) покрывает не меньше
    plane_fill_fraction платы; цепь слоя — заливка цепи не меньше 5 % платы."""
    b = ctx.board
    outline = ctx.geo.outline_geom()
    area = outline.area if not outline.is_empty else None
    frac = ctx.profile['planes'].get('plane_fill_fraction', 0.5)
    out = {}
    for layer in b.copper[1:-1]:                 # сплошные слои — внутренние (наружную заливку проверяет A1)
        zones = [f for f in b.layers[layer].features if f.role is Role.ZONE and f.dark and f.net]
        if not zones or area is None:
            continue
        net_area = defaultdict(float)            # площадь по контурам (без построения геометрии слоя)
        for f in zones:
            net_area[f.net] += _area(f.shape)
        if sum(net_area.values()) >= frac * area:
            out[layer] = {n for n, a in net_area.items() if a >= 0.05 * area}
    return out


def _shoelace(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    return 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))


def _area(shape) -> float:
    """Площадь заливки по вершинам контуров (дуги — хордами): для порога «сплошной слой» хватает."""
    from ..board.shapes import Compound, Region
    if isinstance(shape, Region):
        return _shoelace(shape.outer.pts) - sum(_shoelace(h.pts) for h in shape.holes)
    if isinstance(shape, Compound):
        return sum(_area(x) for x in shape.parts)
    return to_geometry(shape).area


class _Foreign:
    """Медь слоя, нарезанная на плитки (те же, что у A1: LayerCopper.tiles), с цепью каждой плитки —
    для поиска ближайшей меди другой цепи. Плитка острова с несколькими цепями (замыкание) берётся
    по кускам цепей."""

    def __init__(self, lc):
        geoms, owner = lc.tiles()
        tiles, nets = [], []
        mixed = set()
        for g, o in zip(geoms, owner):
            ns = lc.island_nets[int(o)]
            if len(ns) == 1:
                tiles.append(g)
                nets.append(next(iter(ns)))
            else:
                mixed.add(int(o))
        for o in mixed:
            for net, piece in lc.island_pieces[o].items():
                for p in parts(piece):
                    for t in _subdivide(p, 256):
                        tiles.append(t)
                        nets.append(net)
        self.tiles = np.asarray(tiles, dtype=object)
        self.nets = np.asarray(nets, dtype=object)
        self.tree = shapely.STRtree(tiles) if tiles else None

    def nearest(self, geom, own_net, reach):
        """(расстояние мм, цепь, точка на меди по запросу) до ближайшей меди другой цепи."""
        if self.tree is None:
            return None
        idx = [int(i) for i in self.tree.query(geom, predicate='dwithin', distance=reach)
               if own_net is None or self.nets[int(i)] != own_net]
        if not idx:
            return None
        d = shapely.distance(self.tiles[idx], geom)
        k = int(np.argmin(d))
        tile = self.tiles[idx[k]]
        return float(d[k]), self.nets[idx[k]], lambda: tuple(shapely.shortest_line(geom, tile).coords[-1])

    def nearest_disks(self, centers, radii, own_nets, reach):
        """То же для многих кругов сразу: расстояние от круга до многоугольника = расстояние от
        центра − радиус. → список (расстояние мм, цепь, точка) или None для каждого круга."""
        n = len(centers)
        out = [None] * n
        if self.tree is None or not n:
            return out
        pts = shapely.points(np.asarray(centers, dtype=float))
        a, t = self.tree.query(pts, predicate='dwithin', distance=reach + float(np.max(radii)))
        if not len(a):
            return out
        own = np.asarray(own_nets, dtype=object)
        keep = self.nets[t] != own[a]
        a, t = a[keep], t[keep]
        d = shapely.distance(pts[a], self.tiles[t]) - np.asarray(radii)[a]
        best = np.full(n, np.inf)
        arg = np.full(n, -1)
        order = np.lexsort((d, a))                   # по кругу, внутри — по расстоянию
        first = np.ones(len(order), bool)
        first[1:] = a[order][1:] != a[order][:-1]
        for k in order[first]:
            best[a[k]], arg[a[k]] = d[k], t[k]
        for i in range(n):
            if arg[i] >= 0 and best[i] <= reach:
                tile, c, r = self.tiles[arg[i]], centers[i], radii[i]
                out[i] = (max(float(best[i]), 0.0), self.nets[arg[i]],
                          (lambda tile=tile, c=c, r=r: tuple(shapely.shortest_line(
                              shapely.Point(c).buffer(r, quad_segs=32), tile).coords[-1])))
        return out


def _hole_geom(h, diameter):
    if h.slot_end is not None:
        return to_geometry(Stroke(np.array([h.pos, h.slot_end]), diameter), FINE)
    return to_geometry(Circle(h.pos, diameter / 2), FINE)


def measure(ctx, planes: dict) -> list[dict]:
    """Зазоры «граница потенциала отверстия → медь другой цепи» на сплошных слоях."""
    b, prof = ctx.board, ctx.profile
    pl, dr = prof['planes'], prof['drilling']
    cu = b.copper
    depths = layer_depths(b)
    # дальше — запас > 8σ; не меньше зазора, которого требует напряжение (иначе пары молча пропускаются)
    w_design, w_acc = wicking_um(pl, product_class(prof))
    reach = (max(pl['clearance_req_um'] / 1000, voltage_reach_mm(ctx)) + max(w_design, w_acc or 0) / 1000) + 0.25
    jobs = defaultdict(list)                     # слой → [(отверстие, σ слоя, d_св)]
    for h in b.holes:
        if not h.plated or not h.net or h.diameter < MIN_HOLE_MM:   # без цепи — крепёжное
            continue
        i0, i1 = cu.index(h.span[0]), cu.index(h.span[1])
        layers = [L for L in cu[i0:i1 + 1] if L in planes and h.net not in planes[L]]
        if not layers:
            continue
        sig = hole_sigmas(b, prof, h, depths)
        d_b = barrel_mm(h.diameter, dr, via=h.role is Role.VIA)
        for L in layers:
            jobs[L].append((h, sig[L], d_b))
    out = []
    for L, items in jobs.items():
        fr = _Foreign(ctx.geo.layer(L))
        # стволы: круглые — одним запросом, пазы — по одному
        round_ = [k for k, (h, _, _) in enumerate(items) if h.slot_end is None]
        hits_b = [None] * len(items)
        res = fr.nearest_disks([items[k][0].pos for k in round_], [items[k][2] / 2 for k in round_],
                               [items[k][0].net for k in round_], reach)
        for k, r in zip(round_, res):
            hits_b[k] = r
        for k, (h, _, d_b) in enumerate(items):
            if h.slot_end is not None:
                hits_b[k] = fr.nearest(_hole_geom(h, d_b), h.net, reach)
        for k, (h, s, d_b) in enumerate(items):
            hit_b = hits_b[k]
            pad = h.padstack.get(L)
            hit_p, nfp = None, None
            if pad is not None:                  # нефункциональная площадка: её край — в рисунке слоя
                hit_p = fr.nearest(to_geometry(pad.shape, FINE), h.net, reach)
                nfp = 2 * pad.shape.radius if isinstance(pad.shape, Circle) else None
            hit = hit_p or hit_b
            if hit is None:
                continue
            dist, net, at = hit
            c_bar = hit_b[0] * 1000 if hit_b else float('inf')
            out.append(dict(hole=h, layer=L, c=dist * 1000, c_bar=c_bar, net=net, at=at,
                            kind='площадка' if hit_p else 'ствол', nfp=nfp, d_b=d_b,
                            c_finished=(c_bar + (d_b - h.diameter) / 2 * 1000) if hit_b else None,
                            s=s, key=(round(h.pos[0], 4), round(h.pos[1], 4), h.net, round(h.diameter, 4), L)))
    return out


def _judge(c, c_bar, kind, wick, c_req, s, verdict):
    """(номинально нарушено, запас ствола, вероятность, вердикт).

    Ствол смещается вместе с отверстием и «удлиняется» затеканием меди: c_min = c − ℓ_w − |ε|.
    Край оставленной площадки — часть рисунка слоя (травится вместе с медью слоя): смещение
    сверла и затекание (оно идёт от стенки отверстия на ℓ_w, а поясок площадки шире) этот
    зазор не меняют — он постоянный (документ A5, разделы 1.2 и 4.1)."""
    # допуск EPS_UM: вскрытия KiCad — многоугольники, «ровно 100 мкм» выходит как 99,96
    nominal = (c < c_req - EPS_UM) if kind == 'площадка' else (c - wick < c_req - EPS_UM)
    m = c_bar - wick - c_req
    p = 1.0 if nominal else p_breakout(m, s.total, s.bias)
    sev = Severity.ERROR if nominal else probability_severity(p, verdict)
    return nominal, m, p, sev


def _fresh(ctx, planes):
    """Зазоры после свежей перезаливки копии платы (общая для классов, Context.fresh) → {ключ: запись}
    или None и причина."""
    fctx, why = ctx.fresh()
    if fctx is None:
        return None, why
    return {r['key']: r for r in measure(fctx, planes)}, ''


def _eccentric(ctx, planes, recs) -> dict:
    """Сдвиг одиночного круглого вскрытия относительно своего отверстия, мм (ключ записи → сдвиг):
    признак отверстия, сдвинутого после заливки, — работает и без перезаливки, и для Gerber."""
    out = {}
    by_layer = defaultdict(list)
    for r in recs:
        if r['kind'] == 'ствол':
            by_layer[r['layer']].append(r)
    for L, rs in by_layer.items():
        lc = ctx.geo.layer(L)
        copper = shapely.union_all([lc.nets[n] for n in planes[L] if n in lc.nets])
        voids = [shapely.Polygon(ring) for poly in parts(copper) for ring in poly.interiors]
        if not voids:
            continue
        tree = shapely.STRtree(voids)
        pts = shapely.points(np.array([r['hole'].pos for r in rs]))
        src, dst = tree.query(pts, predicate='within')
        count = np.bincount(dst, minlength=len(voids))
        for a, v in zip(src, dst):
            if count[v] != 1:
                continue
            poly = voids[int(v)]
            c = poly.centroid
            rad = math.sqrt(poly.area / math.pi)
            ring = np.asarray(poly.exterior.coords)
            dev = np.abs(np.hypot(ring[:, 0] - c.x, ring[:, 1] - c.y) - rad).max()
            if dev > 0.01 * rad + 0.002:                 # не круг (например, вскрытие вокруг паза)
                continue
            off = math.dist((c.x, c.y), rs[int(a)]['hole'].pos)
            if off > 0.01:
                out[rs[int(a)]['key']] = off
    return out


def _voltage_req(ctx, a, b):
    """D(V) для пары цепей, мкм, и причина, если считать нельзя: (D, '') или (None, причина) — тогда З7
    «невыполнено» с этой причиной."""
    nets = ctx.profile.get('nets', {})
    volts = nets.get('voltages', {})
    table = (nets.get('voltage_table') or {}).get('B1')
    if a not in volts or b not in volts:
        return None, 'нет напряжения цепи'
    if not table:
        return None, 'нет таблицы зазоров по напряжению для внутренних слоёв (категория B1)'
    dv = voltage_difference(volts[a], volts[b])
    d = required_voltage_clearance(table, dv, nets.get('voltage_slope_mm_per_v', {}).get('B1'))
    if d is None:
        return None, f'{dv:.0f} В выше таблицы B1 (нет наклона)'
    return d * 1000, ''


def voltage_reach_mm(ctx) -> float:
    """Наибольший зазор по напряжению на внутреннем слое для пар цепей профиля, мм (0 — нет данных)."""
    nets = ctx.profile.get('nets', {})
    volts = nets.get('voltages', {})
    table = (nets.get('voltage_table') or {}).get('B1')
    if not volts or not table:
        return 0.0
    dv = max_voltage_difference(volts)
    d = required_voltage_clearance(table, dv, nets.get('voltage_slope_mm_per_v', {}).get('B1'))
    return d if d is not None else max(dd for _, dd in table)


@rule('A5')
def check(ctx) -> list[Finding]:
    prof = ctx.profile
    pl, verdict = prof['planes'], prof['verdict']
    planes = plane_layers(ctx)
    if not planes:
        return []
    quality = prof.quality('planes')
    recs = measure(ctx, planes)
    ecc = _eccentric(ctx, planes, recs)
    out: list[Finding] = []
    fresh, why = (None, 'выключено в профиле')
    if ctx.board.design is not None:
        fresh, why = _fresh(ctx, planes)
        if fresh is None and why not in ('выключено в профиле', 'не .kicad_pcb'):
            out.append(Finding('A5', 'З8', f'устаревшая заливка не проверена: {"нет KiCad для перезаливки" if why == "нет KiCad" else why}',
                               Severity.NOT_CHECKED, data_quality='нет данных',
                               fix='установить KiCad или перезалить зоны перед проверкой'))
    wick, wick_acc = wicking_um(pl, product_class(prof))
    c_req0, p_star = pl['clearance_req_um'], pl.get('breach_probability_max', 1e-3)
    volts = prof.get('nets', {}).get('voltages', {})
    groups = defaultdict(list)
    unknown_v, no_table = set(), set()
    board_rows = defaultdict(list)
    for r in recs:
        h, s = r['hole'], r['s']
        dv_req, why_v = _voltage_req(ctx, h.net, r['net']) if volts else (None, '')
        if volts and dv_req is None:
            if why_v == 'нет напряжения цепи':
                unknown_v.add(h.net if h.net not in volts else r['net'])
            else:
                no_table.add(why_v)
        c_req = max(c_req0, dv_req or 0.0)
        nominal, m, p, sev = _judge(r['c'], r['c_bar'], r['kind'], wick, c_req, s, verdict)
        board_rows[r['layer']].append((-1.0 if nominal else m, s.local, s.common))
        f = fresh.get(r['key']) if fresh else None
        if f is not None and (f['c'] > r['c'] + 5 or f['c_bar'] > r['c_bar'] + 5):
            sev_f = _judge(f['c'], f['c_bar'], f['kind'], wick, c_req, s, verdict)[3]
            if _rank(sev_f) < _rank(sev):
                groups[('З8', r['layer'], h.role, round(h.diameter, 4), None, '')].append(
                    (r, Severity.ERROR, p, c_req, f['c']))
                continue
        off = ecc.get(r['key']) if fresh is None else None
        if off is not None and sev is not Severity.OK:   # без перезаливки: вскрытие не по центру отверстия
            sev_f = _judge(r['c'] + off * 1000, r['c_bar'] + off * 1000, r['kind'], wick, c_req, s, verdict)[3]
            if _rank(sev_f) < _rank(sev):
                groups[('З8', r['layer'], h.role, round(h.diameter, 4), 'смещение', '')].append(
                    (r, Severity.WARNING, p, c_req, r['c'] + off * 1000))
                continue
        if sev is Severity.OK:
            # проект в порядке, но при предельном по приёмке затекании изоляции не хватит — предупреждение
            if wick_acc is not None and r['kind'] == 'ствол' and r['c'] - wick_acc < c_req - EPS_UM:
                groups[('З1', r['layer'], h.role, round(h.diameter, 4), r['nfp'], 'приёмка')].append(
                    (r, Severity.WARNING, p, c_req, None))
            continue
        code = 'З7' if dv_req and dv_req > c_req0 else 'З3' if r['kind'] == 'площадка' and nominal else \
            'З1' if nominal else 'З2'
        groups[(code, r['layer'], h.role, round(h.diameter, 4), r['nfp'], '')].append((r, sev, p, c_req, None))
    # внутри группы — кучки близких зазоров (разрыв > 3 мкм — новая находка): вскрытия KiCad строит
    # многоугольником, и одинаковые отверстия различаются на доли микрометра
    for (code, *rest), items in groups.items():
        items.sort(key=lambda t: t[0]['c'])
        accept = rest[-1] == 'приёмка'
        start = 0
        for k in range(1, len(items) + 1):
            if k == len(items) or items[k][0]['c'] - items[k - 1][0]['c'] > 3.0:
                out.append(_finding(code, items[start:k], wick_acc if accept else wick, p_star, quality, pl, accept))
                start = k
    if not volts:
        out.append(Finding('A5', 'З7', 'зазоры по напряжению до сплошных слоёв не проверены: в профиле нет напряжений '
                                       'цепей', Severity.NOT_CHECKED, data_quality='нет данных',
                           fix='задать [nets] voltages и voltage_table (категория B1: ГОСТ Р 53429-2009, табл. 7–9)'))
    else:
        if unknown_v:
            out.append(Finding('A5', 'З7', f'нет напряжений цепей {", ".join(sorted(map(str, unknown_v))[:8])} — их зазоры '
                                           f'по напряжению не проверены', Severity.NOT_CHECKED,
                               nets=tuple(sorted(map(str, unknown_v)))[:10], data_quality='нет данных',
                               fix='дополнить [nets] voltages'))
        for why_v in sorted(no_table):
            out.append(Finding('A5', 'З7', f'зазоры по напряжению не проверены: {why_v}', Severity.NOT_CHECKED,
                               data_quality='нет данных',
                               fix='задать [nets] voltage_table.B1 (ГОСТ Р 53429-2009, табл. 7–9) и наклон для высоких напряжений'))
    out += _fields(ctx, planes, quality)
    out += _board_level(board_rows, verdict, quality)
    return out


_RANK = {Severity.OK: 0, Severity.INFO: 1, Severity.WARNING: 2, Severity.ERROR: 3}


def _rank(s):
    return _RANK.get(s, 0)


def _objects(h):
    if h.role is Role.VIA:
        return [{'kind': 'via', 'uid': h.uid}]
    return [{'kind': 'pad', 'uid': h.uid, 'ref': h.component, 'pin': h.pin}]


def _finding(code, items, wick, p_star, quality, pl, accept=False) -> Finding:
    items.sort(key=lambda t: (t[0]['c'], -t[2]))
    r, sev, p, c_req, f_c = items[0]
    h, s, L = r['hole'], r['s'], r['layer']
    n = len(items)
    what = ('переходные' if h.role is Role.VIA else 'выводы') + f' Ø{h.diameter:.2f} мм' + (f' — {n} шт.' if n > 1 else '')
    edge = r['nfp'] if r['nfp'] else r['d_b']
    d_a_min = antipad_min_mm(edge, c_req, wick, s.total, p_star)
    d_a = edge + 2 * r['c'] / 1000
    cb = r['c'] if r['kind'] == 'ствол' else r['c_bar']          # зазор от ствола
    head = {'З1': f'зазор от ствола до меди {r["net"]} на {L} {cb:.0f} мкм: за вычетом '
                  f'{"предельного по приёмке " if accept else ""}затекания {wick:g} мкм {cb - wick:.0f} < {c_req:.0f} мкм',
            'З2': f'зазор от ствола до меди {r["net"]} на {L} {cb:.0f} мкм, запас {cb - wick - c_req:.0f} мкм '
                  f'при σ = {s.total:.1f} мкм — нарушение у {_pct(p)} % плат',
            'З3': f'на {L} оставлена нефункциональная площадка{f" Ø{r['nfp']:.2f}" if r["nfp"] else ""}: зазор от её '
                  f'края до {r["net"]} {r["c"]:.0f} мкм меньше требуемых {c_req:.0f} мкм (вместо {cb:.0f} мкм от ствола)',
            'З7': f'зазор по напряжению до {r["net"]} на {L}: {cb - wick:.0f} мкм за вычетом затекания при '
                  f'требуемых {c_req:.0f} мкм — нарушение у {_pct(p)} % плат',
            'З8': (f'устаревшая заливка {L}: зазор до {r["net"]} {r["c"]:.0f} мкм, после перезаливки '
                   f'{(f_c or 0):.0f} мкм') if sev is Severity.ERROR else
                  (f'вскрытие на {L} не по центру отверстия (отверстие сдвинуто после заливки?): зазор до '
                   f'{r["net"]} {r["c"]:.0f} мкм, после перезаливки ≈ {(f_c or 0):.0f} мкм')}[code]
    measured = {'отверстий': n, 'c_мкм': r['c'], 'граница': r['kind'], 'd_св_мм': r['d_b'], 'σ_мкм': s.total,
                'затекание_мкм': wick, 'D_a_мм': round(d_a, 4), 'цепь_слоя': r['net'],
                'позиции': [[round(v, 4) for v in t[0]['hole'].pos] for t in items[:20]]}
    if r['c_finished'] is not None:
        measured['c_от_готового_мкм'] = r['c_finished']         # З4: так завысила бы проверка «в лоб»
    if code == 'З7':
        l_eff = r['c'] - wick
        measured['L_эфф_мкм'] = l_eff
        measured['срок_отн'] = (max(l_eff, 0) / c_req) ** pl.get('caf_n', 2)
    if f_c is not None:
        measured['c_после_перезаливки_мкм'] = f_c
    fix = {'З8': 'перезалить зоны перед выпуском файлов (в KiCad: B — «Залить все зоны»)',
           'З3': f'удалить неподключённые площадки на сплошных слоях (remove_unused_layers) или вскрытие ≥ '
                 f'{d_a_min:.2f} мм', }.get(code, f'вскрытие не меньше {d_a_min:.2f} мм (D_a,min = d_св + 2(c_тр + ℓ_w) + '
                                          f'2σ√(−2 ln P*), P* = {p_star:g}) — зазор зоны или правило hole_clearance '
                                          f'≥ {(d_a_min - h.diameter) / 2:.3f} мм от готового отверстия')
    return Finding(
        'A5', code, f'{what}: {head}', sev, layer=L, pos=r['at'](),
        nets=tuple(dict.fromkeys([t[0]['hole'].net for t in items if t[0]['hole'].net] + [r['net']]))[:10],
        objects=[o for t in items[:20] for o in _objects(t[0]['hole'])], measured=measured,
        thresholds={'c_тр_мкм': c_req, 'D_a_min_мм': round(d_a_min, 3), 'P*': p_star},
        margin=(r['c'] - wick - c_req) / c_req if c_req else None,
        probability=1.0 if code == 'З8' else p, data_quality=quality,
        assumptions=['зазор измерен по заливке в файле; ствол — сверло d + 2t_мет (вскрытие KiCad строит от готового '
                     'отверстия, это на толщину металлизации меньше)', 'смещение — модель документа A3'],
        fix=fix)


def _board_level(board_rows, verdict, quality) -> list[Finding]:
    per = {}
    for L, rows in board_rows.items():
        a = np.asarray(rows, dtype=float)
        per[L] = p_board(a[:, 0], a[:, 1], float(a[:, 2].max()))
    if not per:
        return []
    total = 1 - float(np.prod([1 - p for p in per.values()]))
    sev = probability_severity(total, verdict)
    if sev is Severity.OK:
        return []
    worst = max(sorted(per), key=per.get)            # при равенстве — первый по имени (детерминированно)
    return [Finding('A5', 'З2', f'плата целиком: хотя бы один зазор до сплошного слоя будет нарушен у {_pct(total)} % '
                                f'плат (худший слой {worst} — {_pct(per[worst])} %)', sev, layer=worst,
                    measured={'p_плата': total, 'p_по_слоям': per}, probability=total, data_quality=quality,
                    assumptions=['общий сдвиг слоя + местные составляющие (документ A3, раздел 3.5)'],
                    fix='исправить отверстия из находок З1–З3 с наибольшей вероятностью')]


# ---------- поля: слившиеся вскрытия (З5), узкие перемычки (З6), проводимость ----------

def window_conductance(copper, window, h_mm=0.01, max_cells=250000):
    """Проводимость окна слоя по x и по y относительно сплошного окна + пиковая плотность тока
    (99-й процентиль / медиана). Уравнение Лапласа на сетке (документ A5, раздел 5.4): потенциалы на
    левой и правой стороне, верх и низ непроницаемы."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    x0, y0, x1, y1 = window.bounds
    h = max(h_mm, math.sqrt((x1 - x0) * (y1 - y0) / max_cells))
    xs = np.arange(x0, x1, h) + h / 2
    ys = np.arange(y0, y1, h) + h / 2
    X, Y = np.meshgrid(xs, ys)
    g = copper.intersection(window)
    shapely.prepare(g)
    cu0 = shapely.contains_xy(g, X.ravel(), Y.ravel()).reshape(X.shape)
    res = {}
    for direction in ('x', 'y'):
        cu = cu0 if direction == 'x' else cu0.T.copy()
        ny, nx = cu.shape
        idx = -np.ones(cu.shape, int)
        yy, xx = np.nonzero(cu)
        N = len(yy)
        if N == 0:
            res[direction] = (0.0, 0.0)
            continue
        idx[yy, xx] = np.arange(N)
        rows, cols, vals = [], [], []
        diag = np.full(N, -1e-9)                     # слабая связь с нулём: островки не делают систему вырожденной
        b = np.zeros(N)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            py, px = yy + dy, xx + dx
            inside = (py >= 0) & (py < ny) & (px >= 0) & (px < nx)
            pyc, pxc = np.clip(py, 0, ny - 1), np.clip(px, 0, nx - 1)
            ok = inside & cu[pyc, pxc]
            i = np.nonzero(ok)[0]
            rows.append(i)
            cols.append(idx[pyc[i], pxc[i]])
            vals.append(np.ones(len(i)))
            diag[i] -= 1
            if dx != 0:                              # левая сторона — потенциал 0, правая — 1 (через полклетки)
                e = np.nonzero(~inside)[0]
                diag[e] -= 2
                if dx == 1:
                    b[e] -= 2.0
        rows.append(np.arange(N))
        cols.append(np.arange(N))
        vals.append(diag)
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        v = spla.spsolve(A.tocsc(), b)
        V = np.full(cu.shape, np.nan)
        V[yy, xx] = v
        current = np.nansum(2 * V[:, 0])
        solid = ny / nx                              # сплошное окно: ток ny строк при сопротивлении nx клеток
        gy, gx = np.gradient(np.where(cu, V, np.nan))
        J = np.hypot(gx, gy)[cu]
        J = J[np.isfinite(J)]
        peak = float(np.percentile(J, 99) / np.median(J)) if len(J) and np.median(J) > 0 else float('nan')
        res[direction] = (float(current / solid), peak)
    phi = 1 - float(cu0.mean())
    return {'G_x': res['x'][0], 'G_y': res['y'][0], 'J_пик_x': res['x'][1], 'J_пик_y': res['y'][1], 'φ': phi,
            'шаг_сетки_мм': h}


def _fields(ctx, planes, quality) -> list[Finding]:
    pl = ctx.profile['planes']
    w_min, w_info = pl['web_min_mm'], pl.get('web_info_mm', pl['web_min_mm'])
    b = ctx.board
    cu = b.copper
    out = []
    all_pts = np.array([h.pos for h in b.holes]) if b.holes else np.zeros((0, 2))
    all_tree = shapely.STRtree(shapely.points(all_pts)) if len(all_pts) else None
    for L, nets in planes.items():
        lc = ctx.geo.layer(L)
        copper = shapely.union_all([lc.nets[n] for n in nets if n in lc.nets])
        voids = [shapely.Polygon(r) for p in parts(copper) for r in p.interiors]
        if not voids:
            continue
        vtree = shapely.STRtree(voids)
        holes = [h for h in b.holes if h.net not in nets
                 and cu.index(h.span[0]) <= cu.index(L) <= cu.index(h.span[1])]
        if len(holes) < 2:
            continue
        pts = shapely.points(np.array([h.pos for h in holes]))
        src, dst = vtree.query(pts, predicate='within')
        void_of = np.full(len(holes), -1)
        void_of[src] = dst
        members = defaultdict(list)
        for k, v in enumerate(void_of):
            if v >= 0:
                members[int(v)].append(k)
        merged = set()                               # слившиеся вскрытия: вырез покрыт кругами своих отверстий
        for v, ks in members.items():
            if len(ks) < 2:
                continue
            poly = voids[v]
            disks = [shapely.Point(holes[k].pos).buffer(1.05 * poly.exterior.distance(pts[k]) + 0.005, quad_segs=16)
                     for k in ks]
            if poly.difference(shapely.union_all(disks)).area <= 0.02 * poly.area:
                merged.add(v)
        # узкие перемычки между вскрытиями соседних отверстий
        a, c = shapely.STRtree(pts).query(pts, predicate='dwithin', distance=2.0)
        keep = (a < c) & (void_of[a] >= 0) & (void_of[c] >= 0) & (void_of[a] != void_of[c])
        webs = {}
        for i, j in zip(a[keep], c[keep]):
            vi, vj = int(void_of[i]), int(void_of[j])
            key = (min(vi, vj), max(vi, vj))
            if key not in webs:
                webs[key] = (voids[vi].distance(voids[vj]), int(i), int(j))
        tol = web_tol(ctx) + 1e-9                      # + запас на плавающую точку (граница не должна зависеть от пути)
        narrow = {k: v for k, v in webs.items() if v[0] < w_info - tol}
        # поля: связные группы отверстий из слившихся вырезов и узких перемычек
        parent = list(range(len(holes)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        involved = set()
        for v in merged:
            ks = members[v]
            involved.update(ks)
            for k in ks[1:]:
                parent[find(k)] = find(ks[0])
        for (vi, vj), (w, i, j) in narrow.items():
            involved.update((i, j))
            parent[find(i)] = find(j)
        comps = defaultdict(list)
        for k in involved:
            comps[find(k)].append(k)
        comps = sorted(comps.values(), key=len, reverse=True)
        n_field = pl.get('field_min_holes', 4)
        small = [ks for ks in comps if len(ks) < n_field]
        comps = [ks for ks in comps if len(ks) >= n_field]
        n_top = pl.get('fields_per_layer', 5)
        rest, comps = comps[n_top:], comps[:n_top]      # отдельно — крупнейшие поля слоя, прочие сводкой
        if rest:
            n_m = sum(1 for ks in rest if any(int(void_of[k]) in merged for k in ks))
            in_rest = {k for ks in rest for k in ks}
            w_rest = [w for w, i, _ in narrow.values() if i in in_rest]
            # как у отдельных полей: слияние или перемычка уже w_min — предупреждение, иначе информация
            sev_rest = Severity.WARNING if n_m or (w_rest and min(w_rest) < w_min - tol) else Severity.INFO
            out.append(Finding(
                'A5', 'З5' if n_m else 'З6', f'{L}: ещё {len(rest)} полей от {n_field} отверстий со слившимися '
                                             f'вскрытиями ({n_m}) или узкими перемычками', sev_rest,
                layer=L, pos=tuple(holes[rest[0][0]].pos), nets=tuple(sorted(nets)),
                objects=[o for ks in rest[:20] for o in _objects(holes[ks[0]])],
                measured={'полей': len(rest), 'со_слиянием': n_m, 'отверстий': sum(map(len, rest)),
                          'позиции': [[round(v, 4) for v in holes[ks[0]].pos] for ks in rest[:20]]},
                data_quality=quality, fix='см. крупнейшие поля этого слоя — исправления те же'))
        if small:                                    # 2–3 отверстия в общем вырезе (выводы разъёма) — сводкой
            n_m = sum(1 for ks in small if any(int(void_of[k]) in merged for k in ks))
            out.append(Finding(
                # слившиеся вскрытия — предупреждение, как и у больших полей (документ A5, раздел «Вердикт»);
                # только узкие перемычки в группе из 2–3 отверстий — информация
                'A5', 'З5' if n_m else 'З6', f'{L}: {len(small)} мелких групп по 2–{n_field - 1} отверстия '
                                             f'(слившиеся вскрытия или узкие перемычки)',
                Severity.WARNING if n_m else Severity.INFO, layer=L,
                pos=tuple(holes[small[0][0]].pos), nets=tuple(sorted(nets)),
                objects=[o for ks in small[:10] for k in ks for o in _objects(holes[k])][:20],
                measured={'групп': len(small), 'со_слиянием': n_m,
                          'позиции': [[round(v, 4) for v in holes[ks[0]].pos] for ks in small[:20]]},
                data_quality=quality, fix='проверить, что общий вырез слоя под этими выводами задуман'))
        n_cond = 0
        for rank, ks in enumerate(comps):
            vs = {int(void_of[k]) for k in ks}
            m_here = [v for v in vs if v in merged]
            w_here = [w for (vi, vj), (w, i, j) in narrow.items() if vi in vs or vj in vs]
            centers = np.array([holes[k].pos for k in ks])
            nn = [np.min(np.hypot(*(centers - c).T)[np.hypot(*(centers - c).T) > 1e-6]) for c in centers] \
                if len(centers) > 1 else [1.0]
            pitch = float(np.median(nn))
            # окно — поле вместе с соседними отверстиями любых цепей (ряды земли BGA)
            field = set(map(tuple, np.round(centers, 4)))
            frontier = list(centers)
            while frontier and all_tree is not None:
                c0 = frontier.pop()
                for idx in all_tree.query(shapely.points(*c0), predicate='dwithin', distance=1.25 * pitch):
                    p = tuple(np.round(all_pts[int(idx)], 4))
                    if p not in field:
                        field.add(p)
                        frontier.append(np.array(p))
            fp = np.array(sorted(field))
            margin = 0.75 * pitch
            window = shapely.box(fp[:, 0].min() - margin, fp[:, 1].min() - margin,
                                 fp[:, 0].max() + margin, fp[:, 1].max() + margin)
            cond = None
            if len(ks) >= pl.get('field_conductance_min_holes', 9) and n_cond < 2:   # два крупнейших поля слоя
                n_cond += 1
                cond = window_conductance(copper, window, pl.get('field_grid_um', 10) / 1000,
                                          pl.get('field_max_cells', 250000))
            if m_here:
                code, sev = 'З5', Severity.WARNING
                n_m = sum(len(members[v]) for v in m_here)
                title = f'{L}: вскрытия {n_m} отверстий слились в {len(m_here)} прорез{"ь" if len(m_here) == 1 else "и"}'
            else:
                code = 'З6'
                sev = Severity.WARNING if min(w_here) < w_min - tol else Severity.INFO
                title = (f'{L}: перемычки слоя между вскрытиями уже {w_info:.2f} мм — {len(w_here)} шт. '
                         f'(самая узкая {min(w_here):.3f} мм)')
            if cond and max(cond['G_x'], cond['G_y']) < 0.005:
                title += '; окно поля разрезано: ток через поле не проходит ни вдоль x, ни вдоль y'
            elif cond:
                title += f'; проводимость окна поля {cond["G_x"]:.2f} вдоль x и {cond["G_y"]:.2f} вдоль y от сплошного'
            c = window.centroid
            out.append(Finding(
                'A5', code, title, sev, layer=L, pos=(c.x, c.y), nets=tuple(sorted(nets)),
                objects=[o for k in ks[:20] for o in _objects(holes[k])],
                measured={'отверстий': len(ks), 'шаг_мм': pitch, 'перемычка_мин_мм': min(w_here) if w_here else None,
                          'окно_мм': [round(v, 3) for v in window.bounds], **(cond or {}),
                          **({'MG_оценка': maxwell_garnett(cond['φ'])} if cond else {})},
                thresholds={'w_min_мм': w_min, 'w_внимание_мм': w_info}, data_quality=quality,
                assumptions=['проводимость — двумерный расчёт окна поля, не всего слоя'],
                fix='шахматная расстановка переходных земли и сигналов, меньшие вскрытия в поле по окну '
                    'осуществимости D_a,min ≤ D_a ≤ p − w_min или явные перемычки между рядами (IPC-2221C §9.1.4.1)'))
    return out
