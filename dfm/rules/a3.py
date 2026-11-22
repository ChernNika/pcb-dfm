"""Класс A3 — пояски и совмещение отверстий (документ A3).

Для каждого металлизированного отверстия и каждого слоя, где у него есть площадка:
номинальный поясок r_n = dist(c, ∂P) − d/2 (P — медь площадки вместе с заливкой и каплей той же
цепи, c — центр или ось паза), погрешность σ слоя из общего бюджета (registration.py), запас
m = r_n − r_тр и вероятность выхода P = exp(−m²/2σ²) (Райс при систематическом сдвиге).

  К1  r_n < r_тр хоть на одном слое — ошибка на любой плате;
  К4  отверстие не в центре площадки (offset, паз): r_n меньше «размеры площадки − d» пополам;
  К5  худший слой — внутренний (решает межслойное совмещение);
  К2  запаса на смещение нет: вердикт по вероятности худшего слоя;
      на одно отверстие — одна из К1 > К4 > К5 > К2, одинаковые отверстия — одной находкой;
  К3  сужение соединения с трассой — моделирование: перемычки меди до углов стыка ≥ (1 − k)·w
      (Коробков 2025, ф. (37), (41)); вариант документа A3 (раздел 4.3) — выход в полосе трассы;
  К6  систематический сдвиг из-за положения на заготовке меняет вердикт (Райс);
  К7  микроотверстие не захватывает целевую площадку: exp(−(D_t − d_b)²/8σ²);
  К8  толстая плата: на нижнем слое заметен увод сверла.
Отдельно — вероятность на плату по модели общих и местных составляющих (раздел 3.5).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import shapely

from ..board.model import HoleKind, Role
from ..board.shapes import Circle, Stroke
from ..geometry.copper import _subdivide, parts
from ..geometry.polygons import to_geometry
from .engine import MIN_HOLE_MM, rule
from .findings import Finding, Severity, probability_severity
from .profile import product_class
from .registration import d_min_mm, hole_sigmas, layer_depths, p_board, p_breakout

__all__ = ['check', 'ring_um', 'nominal_rings', 'p_capture_miss', 'track_breakout_mc']

FINE = 1e-5                                   # мм: допуск многоугольников площадок
_ORDER = [Severity.OK, Severity.INFO, Severity.WARNING, Severity.ERROR]


def _bump(sev: Severity) -> Severity:
    """На ступень строже (выход у трассы опаснее выхода где угодно)."""
    return _ORDER[min(_ORDER.index(sev) + 1, 3)] if sev in _ORDER and sev is not Severity.OK else sev


def _pct(p: float) -> str:
    v = p * 100
    return '0' if v == 0 else f'{v:.3g}' if v >= 0.01 else f'{v:.1e}'


def ring_um(land, axis, half_width: float) -> float:
    """Наименьший поясок, мкм: расстояние от центра (оси паза) до границы куска меди, в котором он
    лежит, минус радиус отверстия. Центр вне меди — отрицательный поясок."""
    for poly in parts(land):
        if poly.covers(axis):
            return (poly.boundary.distance(axis) - half_width) * 1000
    return -(land.distance(axis) + half_width) * 1000 if not land.is_empty else -half_width * 1000


def nominal_rings(board) -> list:
    """Номинальный поясок площадок каждого металлизированного отверстия (без смещения и заливки),
    мкм: [(отверстие, {слой: поясок})] — то, что проверяет DRC KiCad («ширина пояска»)."""
    out = []
    for h in board.holes:
        if not h.plated or not h.padstack:
            continue
        axis = shapely.LineString([h.pos, h.slot_end]) if h.slot_end is not None else shapely.points(*h.pos)
        rings = {}
        for layer, f in h.padstack.items():
            if isinstance(f.shape, Circle) and h.slot_end is None:
                rings[layer] = (f.shape.radius - math.dist(f.shape.center, h.pos) - h.diameter / 2) * 1000
            else:
                rings[layer] = ring_um(to_geometry(f.shape, FINE), axis, h.diameter / 2)
        out.append((h, rings))
    return out


def p_capture_miss(target_mm: float, bottom_mm: float, sigma_um: float) -> float:
    """К7: дно микроотверстия выходит за целевую площадку, exp(−(D_t − d_b)²/8σ²)."""
    m = (target_mm - bottom_mm) * 1000
    return 1.0 if m <= 0 else float(math.exp(-m * m / (8 * sigma_um * sigma_um)))


def track_breakout_mc(land, center, d_mm, tracks, sigma_um, n=200000, seed=0, max_check=20000):
    """К3: доли выходов (где угодно, в секторе трассы) при случайном смещении отверстия.
    tracks — [(направление (ux, uy), ширина мм)]; сектор трассы — полоса шириной w + 60 мкм от
    центра в сторону трассы (документ A3, раздел 4.3)."""
    rng = np.random.default_rng(seed)
    shift = rng.normal(0, sigma_um / 1000, (n, 2))
    # сдвиг меньше наименьшего пояска выхода не даёт — проверяются только остальные (несколько %)
    r0 = max(ring_um(land, shapely.points(*center), d_mm / 2) / 1000, 0.0)
    cand = np.flatnonzero(np.hypot(shift[:, 0], shift[:, 1]) >= r0 - 1e-9)
    # точно проверяются не больше max_check кандидатов (случайная выборка), доли пересчитываются
    if len(cand) > max_check:
        cand = rng.choice(cand, max_check, replace=False)
        scale = np.count_nonzero(np.hypot(shift[:, 0], shift[:, 1]) >= r0 - 1e-9) / max_check
    else:
        scale = 1.0
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    ring = np.c_[np.cos(th), np.sin(th)] * d_mm / 2
    pts = center[None, None, :] + shift[cand, None, :] + ring[None, :, :]   # k × 180 × 2
    shapely.prepare(land)
    out = ~shapely.contains_xy(land, pts[..., 0], pts[..., 1])
    any_out = out.any(axis=1)
    at_track = np.zeros(len(cand), bool)
    rel = pts - center[None, None, :]
    for (ux, uy), w in tracks:
        u = rel[..., 0] * ux + rel[..., 1] * uy
        v = -rel[..., 0] * uy + rel[..., 1] * ux
        at_track |= (out & (u > 0) & (np.abs(v) <= (w + 0.06) / 2)).any(axis=1)
    return float(any_out.sum() * scale / n), float(at_track.sum() * scale / n)


def junction_corners(land, center, track):
    """Углы стыка трассы с медью площадки: точки, где края трассы (±w/2 от оси, из центра отверстия —
    допущение: ось трассы проходит через центр; так подводят трассы к площадкам почти всегда) выходят из
    меди площадки (с каплей — дальше по трассе)."""
    (ux, uy), w = track
    x0, y0, x1, y1 = land.bounds
    reach = math.hypot(x1 - x0, y1 - y0) + 0.1           # луч длиннее любой хорды меди площадки
    out = []
    for v in (w / 2, -w / 2):
        p0 = np.array([center[0] - uy * v, center[1] + ux * v])
        ray = shapely.LineString([p0, p0 + reach * np.array([ux, uy])])
        hit = land.boundary.intersection(ray)
        pts = [np.asarray(g.coords[0]) for g in shapely.get_parts(hit) if not g.is_empty]
        pts = [q for q in pts if np.hypot(*(q - p0)) > 1e-9]
        out.append(min(pts, key=lambda q: np.hypot(*(q - p0))) if pts and land.covers(shapely.points(*p0)) else p0)
    return out


def junction_margin(land, center, d_mm, tracks, k_min):
    """Запас критерия стыка без смещения, мм: min по трассам (перемычки − (1 − k)·w)/2. Сумма перемычек
    меняется не быстрее 2|ε|, поэтому при |ε| меньше запаса соединение годно наверняка."""
    r, c0 = d_mm / 2, np.asarray(center, float)
    out = math.inf
    for t in tracks:
        c1, c2 = junction_corners(land, center, t)
        bridge = max(np.hypot(*(c0 - c1)) - r, 0) + max(np.hypot(*(c0 - c2)) - r, 0)
        out = min(out, (bridge - (1 - k_min) * t[1]) / 2)
    return out


def junction_fail(land, center, d_mm, tracks, sigma_um, k_min, n=200000, seed=0):
    """К3 по Коробкову (диссертация 2025, ф. (37), (41), с. 84–94): соединение трассы с площадкой годно,
    если сумма перемычек меди от края отверстия до двух углов стыка не меньше (1 − k)·w — сужение в месте
    соединения не больше допуска приёмки (автор, с. 84: IPC-6012B, ГОСТ Р 55693, IPC-A-600G; k — допуск
    сужения проводника IPC-A-600G п. 2.10.1.1: 0,2 для классов 2, 3; 0,3 для класса 1). Граница — эллипс
    с фокусами в углах. Перемычка до угла, попавшего в отверстие, — 0 (меди нет), а не отрицательная, как
    в (37): физически верно и чуть мягче эллипса автора. Смещение — круговое нормальное σ без
    систематического сдвига (К6 — отдельно).
    Возвращает (доля смещений, при которых нарушена хотя бы одна трасса; наименьшая перемычка без
    смещения, мм)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(center, float)[None, :] + rng.normal(0, sigma_um / 1000, (n, 2))
    r = d_mm / 2
    fail = np.zeros(n, bool)
    nominal = math.inf
    for t in tracks:
        c1, c2 = junction_corners(land, center, t)
        bridge = (np.maximum(np.hypot(*(c - c1).T) - r, 0) + np.maximum(np.hypot(*(c - c2).T) - r, 0))
        fail |= bridge < (1 - k_min) * t[1]
        nominal = min(nominal, max(np.hypot(*(np.asarray(center) - c1)) - r, 0)
                      + max(np.hypot(*(np.asarray(center) - c2)) - r, 0))
    return float(fail.mean()), nominal


class _Lands:
    """Медь площадки на слое вместе с заливкой той же цепи рядом (капля, сплошное подключение).
    Заливки нарезаны на плитки: площадка в плоскости земли не пересекается со всей плоскостью."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._zones = {}

    def zones(self, layer):
        if layer not in self._zones:
            lc = self.ctx.geo.layer(layer)
            tiles, nets = [], []
            for f, g in zip(lc.features, lc.feature_geoms):
                if f.role is Role.ZONE and f.dark and f.net and not g.is_empty:
                    for p in parts(g):
                        for t in _subdivide(p, 256):
                            tiles.append(t)
                            nets.append(f.net)
            self._zones[layer] = (tiles, nets, shapely.STRtree(tiles) if tiles else None)
        return self._zones[layer]

    def land(self, layer, pad_geom, net, reach):
        tiles, nets, tree = self.zones(layer)
        if tree is None or not net:
            return pad_geom
        near = [tiles[i] for i in tree.query(pad_geom, predicate='intersects') if nets[i] == net]
        if not near:
            return pad_geom
        disk = pad_geom.buffer(reach, quad_segs=8)
        return shapely.union_all([pad_geom] + [t.intersection(disk) for t in near])


class _TrackEnds:
    """Концы трасс по слою и цепи: точки, направления от конца внутрь трассы и ширины — один раз на слой.
    Поиск трасс у площадки — выборка точек внутри неё (раньше — запрос к индексу объектов на каждое
    отверстие и слой: на Jetson 30 тыс. вызовов, ≈ 2 мин)."""

    def __init__(self, ctx):
        self.ctx, self._by_layer = ctx, {}

    def _layer(self, layer):
        if layer not in self._by_layer:
            ends = {}
            for f in self.ctx.board.layers[layer].features:
                if f.role is not Role.TRACK or not isinstance(f.shape, Stroke) or len(f.shape.pts) < 2 or not f.dark:
                    continue
                pts = np.asarray(f.shape.pts)
                for end, nxt in ((pts[0], pts[1]), (pts[-1], pts[-2])):
                    v = nxt - end
                    n = math.hypot(*v)
                    if n > 1e-9:
                        ends.setdefault(f.net, []).append((end[0], end[1], v[0] / n, v[1] / n, f.shape.width))
            self._by_layer[layer] = {k: np.array(v, float) for k, v in ends.items()}
        return self._by_layer[layer]

    def at(self, layer, pad_geom, net):
        """Трассы цепи net, начинающиеся в площадке: [(направление, ширина)]."""
        a = self._layer(layer).get(net)
        if a is None:
            return []
        x0, y0, x1, y1 = pad_geom.bounds
        a = a[(a[:, 0] >= x0) & (a[:, 0] <= x1) & (a[:, 1] >= y0) & (a[:, 1] <= y1)]
        if not len(a):
            return []
        inside = shapely.covers(pad_geom, shapely.points(a[:, 0], a[:, 1]))
        return [((float(r[2]), float(r[3])), float(r[4])) for r in a[inside]]


def _naive_ring_um(pad_geom, half_width):
    """Поясок «по размерам площадки»: половина меньшей стороны описанного прямоугольника − d/2."""
    r = shapely.minimum_rotated_rectangle(pad_geom)
    c = np.asarray(r.exterior.coords)[:4]
    sides = np.hypot(*(np.roll(c, -1, axis=0) - c).T)
    return (sides.min() / 2 - half_width) * 1000


def _size(r) -> str:
    if r['D'] is not None:
        return f"{r['D']:.2f}"
    x0, y0, x1, y1 = r['pad'].bounds
    return f'{x1 - x0:.2f}×{y1 - y0:.2f}'


def _sig(h, d, rec):
    """Признак группы: тип, отверстие, размеры площадок наружных и внутренних слоёв."""
    outer = tuple(sorted({_size(r) for r in rec.values() if r['s'].outer}))
    inner = tuple(sorted({_size(r) for r in rec.values() if not r['s'].outer}))
    return (h.role, round(d, 4), outer, inner)


def ring_bases(ctx, h) -> dict:
    """Диаметр, от которого считается поясок, мм, для наружных и внутренних слоёв.
    KiCad drill = готовое отверстие FHS; сверло d_св = FHS + 2t + Δ (Δ — припуск: allowance_um, у переходных
    via_allowance_um, по покрытию allowance_by_finish_um). Поясок наружного слоя — от внутренней поверхности
    металлизации (FHS + Δ), внутреннего — от стенки сверления (d_св): ГОСТ Р 55693-2013, пп. 5.7.3, 5.9.4.10;
    ECSS-Q-ST-70-60C Rev.1, п. 10.6.1. Ключи [annular_ring] ring_base_outer / ring_base_inner:
    "finished" | "plated" | "drill"; без них — прежний [drilling] ring_from для всех слоёв."""
    dr, ar = ctx.profile['drilling'], ctx.profile['annular_ring']
    fhs = h.diameter
    if not dr.get('drill_is_finished', True):          # в файле диаметр сверла
        t = dr.get('plating_um', 25) / 1000
        return {'outer': fhs - 2 * t, 'inner': fhs} if 'ring_base_inner' in ar else {'outer': fhs, 'inner': fhs}
    t = dr.get('plating_um', 25) / 1000
    allow = dr.get('allowance_um', 0)
    finish = getattr(ctx.board.stackup, 'finish', None) if ctx.board.stackup is not None else None
    by_finish = dr.get('allowance_by_finish_um', {})
    if finish:
        for k, v in by_finish.items():
            if k.lower() in str(finish).lower():
                allow = v
                break
    if h.role is Role.VIA and 'via_allowance_um' in dr:
        allow = dr['via_allowance_um']
    allow /= 1000
    val = {'finished': fhs, 'plated': fhs + allow, 'drill': fhs + 2 * t + allow}
    if 'ring_base_outer' in ar or 'ring_base_inner' in ar:
        return {'outer': val[ar.get('ring_base_outer', 'plated')], 'inner': val[ar.get('ring_base_inner', 'drill')]}
    old = 'drill' if dr.get('ring_from') == 'drill' else 'finished'
    return {'outer': val[old], 'inner': val[old]}


def breakout_offset_um(r_hole_um: float, r_pad_um: float, angle_deg: float) -> float:
    """Смещение отверстия (радиус r) от центра круглой площадки (радиус R), при котором оно выходит за
    площадку на дуге angle_deg: точки окружности отверстия под углом φ к направлению смещения e лежат вне
    площадки, когда e² + r² + 2er·cos φ > R²; дуга вне = 2·arccos((R² − e² − r²)/(2er)). Решение
    e² + 2r·cos(angle/2)·e + r² − R² = 0. angle = 0 — касание: e = R − r."""
    if angle_deg <= 0:
        return r_pad_um - r_hole_um
    c = math.cos(math.radians(angle_deg) / 2)
    disc = (r_hole_um * c) ** 2 - (r_hole_um ** 2 - r_pad_um ** 2)
    return -r_hole_um * c + math.sqrt(max(disc, 0.0))


def margin_for(r_n: float, r_req: float, d_mm: float, allowed_deg: float) -> float:
    """Запас по смещению до нарушения, мкм: без допуска разрыва — поясок минус требуемый (r_n − r_тр);
    с допуском разрыва (класс 2: 90°) — смещение до разрыва на этой дуге для круглой площадки
    радиуса r + r_n (для некруглой — по наименьшему пояску, консервативно)."""
    if not allowed_deg or r_n <= 0:
        return r_n - r_req
    r = d_mm * 500
    return max(breakout_offset_um(r, r + r_n, allowed_deg) - r_req, r_n - r_req)


def _describe_one(h, size):
    return f'{"переходные" if h.role is Role.VIA else "выводы"} {size}/{h.diameter:.2f} мм'


def _describe(h, rec):
    """«переходные 0.55/0.30 мм (внутренние площадки 0.45)»: площадка наружных слоёв / отверстие."""
    kind = 'переходные' if h.role is Role.VIA else 'выводы'
    outer = sorted({_size(r) for r in rec.values() if r['s'].outer})
    inner = sorted({_size(r) for r in rec.values() if not r['s'].outer})
    main = '/'.join(outer or inner)
    txt = f'{kind} {main}/{h.diameter:.2f} мм'
    if outer and inner and inner != outer:
        txt += f' (внутренние площадки {"/".join(inner)})'
    elif outer and not inner and len(h.padstack) < 3 and h.span[0] != h.span[1] and len(rec) == 2:
        txt += ' (внутренних площадок нет)'
    return txt


@rule('A3')
def check(ctx) -> list[Finding]:
    b, prof = ctx.board, ctx.profile
    dr, ar, verdict = prof['drilling'], prof['annular_ring'], prof['verdict']
    cls = product_class(ctx.profile)
    req = ar['required_um'][f'class{cls}']
    p_star = ar.get('probability_max', 1e-3)
    cu = b.copper
    depths = layer_depths(b)
    lands = _Lands(ctx)
    track_ends = _TrackEnds(ctx)
    quality = prof.quality('drilling')
    # класс 2: разрыв пояска до 90° допустим (ГОСТ Р 55693-2013, табл. 9; IPC-6012) — нарушение начинается
    # со смещения, при котором отверстие выходит за площадку на дуге больше breakout_allowed_deg
    b90 = ar.get('breakout_allowed_deg', {}).get(f'class{cls}', 0.0)
    out: list[Finding] = []
    groups = defaultdict(list)                      # ключ → [(отверстие, слои)]
    board_layers = defaultdict(list)                # слой → [(m, σ_мест, σ_общ)]
    k3_cache = {}
    k3 = defaultdict(list)                          # группа одинаковых стыков → [(p, …)]
    k3_mode = dr.get('k3_criterion', 'junction')    # junction — по Коробкову; strip — полоса w + 60 мкм
    # допуск сужения в месте соединения — допуск сужения проводника (IPC-A-600G п. 2.10.1.1, [yield].k_width)
    k_junction = prof.get('yield', {}).get('k_width', {}).get(f'class{cls}', 0.3 if cls == 1 else 0.2)
    micro_max = dr.get('microvia_max_mm', 0.15)

    for h in b.holes:
        if not h.plated or not h.padstack or h.diameter < MIN_HOLE_MM:
            continue
        i0, i1 = cu.index(h.span[0]), cu.index(h.span[1])
        # микроотверстие: по типу переходного KiCad; без уровня D (Gerber) — несквозное между соседними
        # слоями и не больше microvia_max (сквозное 0,15 мм на 2-слойной плате — механическое)
        micro = h.microvia if h.microvia is not None else (
            i1 - i0 == 1 and h.diameter <= micro_max and h.kind(cu) is not HoleKind.THROUGH)
        if micro:
            from_bottom = i1 == len(cu) - 1 and i0 > 0     # прожигается со стороны B.Cu: цель — внутренний слой
            entry, target = (cu[i1], cu[i0]) if from_bottom else (cu[i0], cu[i1])
            out += _microvia(ctx, h, entry, target, quality)
            continue
        d_by = ring_bases(ctx, h)                   # {'outer': мм, 'inner': мм} — от чего считается поясок
        d = d_by['outer']
        axis = shapely.LineString([h.pos, h.slot_end]) if h.slot_end is not None else shapely.points(*h.pos)
        sig = hole_sigmas(b, prof, h, depths)
        rec = {}
        for layer, feat in h.padstack.items():
            if layer not in sig:
                continue
            # поясок чувствителен к долям микрометра (exp(−m²/2σ²)): многоугольник с отклонением
            # 0,01 мкм, а круглая площадка без заливки рядом — точно
            pad = to_geometry(feat.shape, FINE)
            s = sig[layer]
            r_req = req['outer'] if s.outer else req['inner']
            dl = d_by['outer'] if s.outer else d_by['inner']
            if isinstance(feat.shape, Circle) and h.slot_end is None:
                r_pad = (feat.shape.radius - math.dist(feat.shape.center, h.pos) - dl / 2) * 1000
            else:
                r_pad = ring_um(pad, axis, dl / 2)
            # заливка той же цепи вокруг только увеличивает поясок: собирается, лишь если по одной
            # площадке выход заметен (на Jetson 3269 переходных × 10 слоёв — иначе минута впустую)
            land, r_n = pad, r_pad
            if p_breakout(margin_for(r_pad, r_req, dl, b90), s.total, s.bias) >= verdict['probability_info'] * 1e-3:
                land = lands.land(layer, pad, h.net, 0.5)
                if land is not pad:
                    r_n = ring_um(land, axis, dl / 2)
            naive = (feat.shape.radius - dl / 2) * 1000 if isinstance(feat.shape, Circle) else _naive_ring_um(pad, dl / 2)
            m = margin_for(r_n, r_req, dl, b90)
            p = p_breakout(m, s.total, s.bias)
            p0 = p_breakout(m, s.total) if s.bias else p
            D = 2 * feat.shape.radius if isinstance(feat.shape, Circle) else None
            # нецентрированность — по всей меди вокруг (площадка, подключённая к заливке, не страдает)
            rec[layer] = dict(r_n=r_n, r_req=r_req, p=p, p0=p0, s=s, D=D, off=naive - r_n > 5,
                              naive=naive, pad=pad, land=land, r_pad=r_pad, d=dl, m=m)
            board_layers[layer].append((m, s.local, s.common))
        if not rec:
            continue
        # металлизированное крепёжное отверстие: без цепи, площадка не больше отверстия — пояска нет
        # по замыслу (вывод «» у BAT1, USB1, DC1 на One-Air-Max)
        if not h.net and all(r['r_pad'] <= 1.0 for r in rec.values()):
            for r in rec.values():
                board_layers[r['s'].layer].pop()
            continue
        # К3: трассы у отверстий. strip — где заметна вероятность выхода; junction — где запас критерия стыка
        # может быть исчерпан: стык нарушается раньше, чем отверстие выходит за площадку (автор, с. 92)
        for layer, r in rec.items():
            if k3_mode == 'strip' and p_breakout(r['r_n'], r['s'].total, r['s'].bias) < verdict['probability_info']:
                continue
            # трасса не уже отверстия сама окружает его медью — сужать нечего (документ A3, раздел 4.1:
            # опасен выход сбоку от тонкой трассы)
            tracks = [t for t in track_ends.at(layer, r['pad'], h.net) if t[1] < r['d']]
            if not tracks:
                continue
            if k3_mode == 'junction':
                # P(нарушение стыка) ≤ P(|ε| ≥ запас) = exp(−m²/2σ²); ниже probability_info находки не будет —
                # отбор точный, а не приближённый
                gate = lambda land: p_breakout(junction_margin(land, h.pos, r['d'], tracks, k_junction) * 1000,
                                               r['s'].total) < verdict['probability_info']
                if gate(r['land']):
                    continue
                if r['land'] is r['pad']:                    # капля или заливка той же цепи увеличивают перемычки
                    r['land'] = lands.land(layer, r['pad'], h.net, 0.5)
                    if r['land'] is not r['pad'] and gate(r['land']):
                        continue
            shp = h.padstack[layer].shape
            # круглая площадка без заливки, центр — на оси отверстия: важны только углы между трассами
            # (смещённая площадка — по абсолютным направлениям и месту, иначе кэш смешал бы разные стыки)
            circ = isinstance(shp, Circle) and r['land'] is r['pad'] and math.dist(shp.center, h.pos) < 1e-4
            if circ:
                a0 = math.atan2(tracks[0][0][1], tracks[0][0][0])
                rel = tuple(sorted((round((math.atan2(u[1], u[0]) - a0) % (2 * math.pi), 3), round(w, 4))
                                   for u, w in tracks))
            else:
                rel = tuple((round(u[0], 3), round(u[1], 3), round(w, 4)) for u, w in tracks)
            key = (circ, round(r['r_n'], 1), round(r['d'], 4), rel, round(r['land'].area - r['pad'].area, 4),
                   round(r['s'].total, 1),
                   None if circ else (round(h.pos[0], 4), round(h.pos[1], 4), layer, r['land'].wkb_hex))
            if key not in k3_cache:
                nmc = dr.get('teardrop_mc', 200000)
                k3_cache[key] = (*track_breakout_mc(r['land'], np.asarray(h.pos, float), r['d'], tracks,
                                                    r['s'].total, n=nmc),
                                 *junction_fail(r['land'], np.asarray(h.pos, float), r['d'], tracks, r['s'].total,
                                                k_junction, n=nmc))
            p_any, p_strip, p_tr, bridge0 = k3_cache[key]
            if k3_mode == 'strip':                   # прежний заменитель: любой выход в полосе w + 60 мкм
                p_tr = p_strip
            sev = probability_severity(p_tr, verdict)
            if k3_mode == 'strip':
                sev = _bump(sev)
            if sev is Severity.OK:
                continue
            teardrop = r['land'].area - r['pad'].area > 1e-4
            k3[(round(r['d'], 4), _size(r), teardrop)].append(
                (p_tr, p_any, h, layer, r['s'].total, sev, min(w for _, w in tracks), p_strip, bridge0))
        worst = max(rec, key=lambda L: rec[L]['p'])
        w = rec[worst]
        if any(r['r_n'] < r['r_req'] for r in rec.values()):
            code = 'К1'
        elif any(r['off'] and r['p'] >= verdict['probability_info'] for r in rec.values()):
            code = 'К4'                              # нецентрированность, которая заметно влияет
        elif not w['s'].outer:
            code = 'К5'
        else:
            code = 'К2'
        sev = Severity.ERROR if code == 'К1' else probability_severity(w['p'], verdict)
        if sev is not Severity.OK:
            groups[(code,) + _sig(h, d, rec)].append((h, rec, worst, sev))
        # К6: систематический сдвиг меняет вердикт
        if w['s'].bias > 0 and probability_severity(w['p0'], verdict) != probability_severity(w['p'], verdict):
            groups[('К6',) + _sig(h, d, rec)].append(
                (h, rec, worst, probability_severity(w['p'], verdict)))
        # К8: увод сверла на нижнем слое — только если нижний слой и есть худший (документ A3, каталог, К8)
        bottom = cu[-1]
        if bottom in rec and h.span == (cu[0], cu[-1]) and worst == bottom:
            s = rec[bottom]['s']
            share = s.parts['увод'] / s.total ** 2
            sb = probability_severity(rec[bottom]['p'], verdict)
            if share >= dr.get('wander_share_min', 0.25) and sb is not Severity.OK:
                groups[('К8',) + _sig(h, d, rec)].append(
                    (h, rec, bottom, sb))

    for (code, *_), items in groups.items():
        out.append(_group_finding(ctx, code, items, req, p_star, quality))
    for (d, size, teardrop), items in k3.items():
        items.sort(key=lambda t: -t[0])
        p_tr, p_any, h, layer, sigma, sev, w, p_strip, bridge0 = items[0]
        n = len(items)
        what = ('сужение соединения с трассой' if k3_mode == 'junction' else 'выход отверстия у трассы')
        how = (f'соединение годно, если перемычки меди от отверстия до углов стыка в сумме ≥ (1 − {k_junction:g})·w '
               '(Коробков 2025, ф. (37), (41); допуск IPC-A-600G п. 2.10.1.1)' if k3_mode == 'junction' else
               'сектор трассы — полоса w + 60 мкм; выход у трассы опаснее выхода где угодно — вердикт на ступень строже')
        out.append(Finding(
            'A3', 'К3', f'{_describe_one(h, size)}{f" — {n} шт." if n > 1 else ""}: {what} '
                        f'{w:.2f} мм {"с каплей" if teardrop else "без каплевидного перехода"} на {layer} — до '
                        f'{_pct(p_tr)} % плат (выход где угодно {_pct(p_any)} %)', sev,
            layer=layer, pos=h.pos, nets=tuple(dict.fromkeys(x[2].net for x in items if x[2].net))[:10],
            objects=[o for x in items[:20] for o in _objects(x[2])],
            measured={'отверстий': n, 'p_у_трассы': p_tr, 'p_выход': p_any, 'p_выход_в_полосе': p_strip,
                      'перемычки_без_смещения_мм': bridge0, 'σ_мкм': sigma, 'ширина_трассы_мм': w,
                      'капля': teardrop, 'позиции': [[round(v, 4) for v in x[2].pos] for x in items[:20]]},
            probability=p_tr, data_quality=quality,
            assumptions=[f'моделирование {dr.get("teardrop_mc", 200000)} смещений; {how}'] +
                        ([f'без смещения перемычки {bridge0 * 1000:.0f} мкм < (1 − k)·w: площадка мала для трассы '
                          f'(нужно D_пл ≥ (1 − k)·w + d, Коробков 2025, с. 93)'] if bridge0 < (1 - k_junction) * w else []),
            fix='каплевидный переход (KiCad: Правка → Каплевидные переходы) и перезаливка; или больше площадка'))
    out += _board_level(ctx, board_layers, quality)
    return out


def _objects(h):
    if h.role is Role.VIA:
        return [{'kind': 'via', 'uid': h.uid}]
    return [{'kind': 'pad', 'uid': h.uid, 'ref': h.component, 'pin': h.pin}]


def _group_finding(ctx, code, items, req, p_star, quality) -> Finding:
    items.sort(key=lambda t: -t[1][t[2]]['p'])
    h, rec, worst, sev = items[0]
    w = rec[worst]
    n = len(items)
    what = _describe(h, rec)
    refs = sorted({f'{x.component}.{x.pin}' for x, *_ in items if x.component})
    head = f'{what}' + (f' ({", ".join(refs[:4])}{"…" if len(refs) > 4 else ""})' if refs else '') + \
        (f' — {n} шт.' if n > 1 else '')
    per = {L: r['p'] for L, r in rec.items()}
    outer = [r for r in rec.values() if r['s'].outer]
    inner = [r for r in rec.values() if not r['s'].outer]
    fix_d = {}
    for name, rs, rq in (('наружные', outer, req['outer']), ('внутренние', inner, req['inner'])):
        if rs:
            s = max(r['s'].total for r in rs)
            fix_d[name] = round(d_min_mm(rs[0]['d'], rq, s, p_star), 3)
    fix_txt = 'площадка не меньше ' + ', '.join(f'{k} {v:.2f} мм' for k, v in fix_d.items()) + \
        f' (D_min = d + 2r_тр + 2σ√(−2 ln P*), P* = {p_star:g})'
    if code == 'К1':
        bad = [L for L, r in rec.items() if r['r_n'] < r['r_req']]
        title = (f'{head}: номинальный поясок {min(rec[L]["r_n"] for L in bad):.0f} мкм меньше требуемого '
                 f'{max(rec[L]["r_req"] for L in bad):.0f} мкм ({", ".join(bad)})')
        prob = 1.0
    elif code == 'К4':
        off = [L for L, r in rec.items() if r['off']]
        r0 = rec[off[0]]
        title = (f'{head}: отверстие не в центре площадки — наименьший поясок {r0["r_n"]:.0f} мкм вместо '
                 f'{r0["naive"]:.0f}; выход на {worst} у {_pct(w["p"])} % плат')
        prob = w['p']
        fix_txt = 'убрать offset или увеличить площадку в сторону узкого пояска; ' + fix_txt
    elif code == 'К8':
        s = w['s']
        title = (f'{head}: на нижнем слое {worst} увод сверла — {s.parts["увод"] / s.total ** 2 * 100:.0f} % σ² '
                 f'(σ = {s.total:.1f} мкм, глубина {s.z_mm:.2f} мм): выход у {_pct(w["p"])} % плат')
        prob = w['p']
        fix_txt = 'увеличить площадки нижнего слоя или диаметр отверстия; сверление с двух сторон; ' + fix_txt
    elif code == 'К6':
        title = (f'{head}: систематический сдвиг {w["s"].bias:.0f} мкм (положение на заготовке) — выход на {worst} '
                 f'у {_pct(w["p"])} % плат вместо {_pct(w["p0"])} %')
        prob = w['p']
        fix_txt = 'запас пояска на наихудшее положение или размещение в центре заготовки; ' + fix_txt
    else:
        where = 'внутренний слой' if code == 'К5' else 'слой'
        b90 = ctx.profile['annular_ring'].get('breakout_allowed_deg', {}).get(
            f'class{product_class(ctx.profile)}', 0.0)
        what_out = f'разрыв пояска больше {b90:.0f}°' if b90 else 'выход'
        title = (f'{head}: запас пояска на {where} {worst} {w["r_n"] - w["r_req"]:.0f} мкм при σ = '
                 f'{w["s"].total:.1f} мкм — {what_out} у {_pct(w["p"])} % плат')
        prob = w['p']
    return Finding(
        'A3', code, title, sev, layer=worst, pos=h.pos,
        nets=tuple(dict.fromkeys(x.net for x, *_ in items if x.net))[:10],
        objects=[o for x, *_ in items[:20] for o in _objects(x)],
        measured={'отверстий': n, 'd_мм': h.diameter,
                  'D_мм': {L: r['D'] for L, r in rec.items() if r['D'] is not None},
                  'поясок_мкм': {L: round(r['r_n'], 2) for L, r in rec.items()},
                  'σ_мкм': {L: round(r['s'].total, 2) for L, r in rec.items()},
                  'p_выход': per, 'худший_слой': worst,
                  'позиции': [list(map(lambda v: round(v, 4), x.pos)) for x, *_ in items[:20]]},
        thresholds={'r_тр_мкм': {L: r['r_req'] for L, r in rec.items()}, 'P*': p_star, 'D_min_мм': fix_d},
        probability=prob, data_quality=quality,
        assumptions=['нормальные независимые составляющие смещения; для некруглой площадки — по наименьшему пояску '
                     '(консервативно)'],
        fix=fix_txt)


def _microvia(ctx, h, top, target, quality) -> list[Finding]:
    dr = ctx.profile['drilling']
    feat = h.padstack.get(target)
    if feat is None:
        return [Finding('A3', 'К7', f'микроотверстие {top}→{target} без целевой площадки', Severity.ERROR,
                        layer=target, pos=h.pos, nets=(h.net,) if h.net else (), objects=_objects(h),
                        data_quality=quality, fix='добавить целевую площадку')]
    pad = to_geometry(feat.shape, FINE)
    d_b = h.diameter * dr.get('microvia_bottom_ratio', 0.8)
    margin = ring_um(pad, shapely.points(*h.pos), d_b / 2)                # (D_t − d_b)/2 для круга
    s = dr['sigma_microvia_um']
    p = 1.0 if margin <= 0 else float(math.exp(-margin * margin / (2 * s * s)))
    sev = probability_severity(p, ctx.profile['verdict'])
    if sev is Severity.OK:
        return []
    return [Finding(
        'A3', 'К7', f'микроотверстие {top}→{target}: дно Ø{d_b:.2f} мм, запас до края целевой площадки '
                    f'{margin:.0f} мкм — промах у {_pct(p)} % отверстий', sev, layer=target, pos=h.pos,
        nets=(h.net,) if h.net else (), objects=_objects(h),
        measured={'запас_мкм': margin, 'дно_мм': d_b, 'σ_мв_мкм': s}, probability=p, data_quality=quality,
        assumptions=[f'дно = {dr.get("microvia_bottom_ratio", 0.8)} верхнего диаметра (профиль)'],
        fix=f'целевая площадка не меньше {d_b + 2 * s * math.sqrt(-2 * math.log(ctx.profile["annular_ring"].get("probability_max", 1e-3))) / 1000:.2f} мм '
            f'или более точное совмещение лазера')]


def _board_level(ctx, board_layers, quality) -> list[Finding]:
    """Вероятность выхода хотя бы одного отверстия на плате: по слоям — общие + местные составляющие,
    слои между собой — как независимые (общие сдвиги слоёв независимы)."""
    verdict = ctx.profile['verdict']
    per = {}
    for layer, rows in board_layers.items():
        a = np.asarray(rows, dtype=float)
        common = float(a[:, 2].max()) if len(a) else 0.0
        per[layer] = p_board(a[:, 0], a[:, 1], common)
    if not per:
        return []
    total = 1 - float(np.prod([1 - p for p in per.values()]))
    sev = probability_severity(total, verdict)
    if sev is Severity.OK:
        return []
    worst = max(sorted(per), key=per.get)            # при равенстве — первый по имени (детерминированно)
    n = sum(len(v) for v in board_layers.values())
    return [Finding(
        'A3', 'К2', f'плата целиком: хотя бы одно отверстие выйдет за площадку у {_pct(total)} % плат '
                    f'(худший слой {worst} — {_pct(per[worst])} %)', sev, layer=worst,
        measured={'p_плата': total, 'p_по_слоям': per, 'площадок': n}, probability=total, data_quality=quality,
        assumptions=['общий сдвиг слоя (межслойное совмещение, масштаб) один для всех отверстий, положение сверла '
                     'и увод — свои (документ A3, раздел 3.5); модель независимых отверстий завысила бы риск'],
        fix='исправить отверстия из находок К1–К5 с наибольшей вероятностью; при равномерно малом запасе — '
            'завод с лучшим межслойным совмещением')]
