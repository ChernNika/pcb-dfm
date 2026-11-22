"""Класс A7 — паяльная маска (документ A7).

Окна маски берутся из уровня P (слои F.Mask / B.Mask), медь — с наружного слоя той же стороны.
Расширение окна e измеряется по геометрии (окно против меди площадки), а не по настройкам KiCad,
поэтому класс одинаково работает для .kicad_pcb и Gerber.

Совмещение маски (документ A7, раздел 2): σ = T/3 (T — допуск ±3σ завода для фотошаблона или
LDI), с масштабной ошибкой σ_полн = √(σ² + (σ_k·r)²), r — половина диагонали заготовки.

  М1  перемычка между окнами выводов одного компонента уже минимума для цвета маски;
      вероятность потерять хотя бы одну в ряду P = 1 − Π(1 − p_отсл(w)); окно осуществимости
      3σ_полн ≤ e ≤ (s − w_d,min)/2;
  М2  наползание маски на площадку: P_прям = 1 − (1 − 2Q(e/σ))², P_кругл = exp(−e²/2σ²) —
      вероятность НА ПЛАТУ (маска сдвигается целиком, решает площадка с наименьшим e);
  М3  чужая медь в окне (A_чуж > 0) или ближе 3σ к его краю (откроется с вероятностью Q(d/σ));
  М4  частично закрытое переходное: доли пояска φ_L и отверстия φ_H в окнах;
  М5  открытое переходное у площадки той же цепи без перемычки: V_отв = πd²h/4 против припоя
      на площадке k_мет·k_перен·A·t_трафарета;
  М6  узкие полоски маски вне выводов одного компонента и у края платы, p_отсл(w);
  М7  смесь SMD (e < −0,5 мкм) и NSMD (e > 0,5 мкм) в матрице круглых площадок компонента (медь ±30 % от медианы);
  М8  тонкая маска над толстой медью: t_верх = t̄ − β(1 − f)t_Cu.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import shapely
from scipy.stats import norm

from ..board.model import Role
from ..board.shapes import Circle
from ..geometry.copper import parts
from .engine import MIN_HOLE_MM, rule
from .findings import Finding, Severity, probability_severity
from .registration import DEFAULT_BOARD_MM

__all__ = ['check', 'sigma_um', 'p_encroach_rect', 'p_encroach_circ', 'p_peel', 'mask_window_quality']

Q = norm.sf


# ---------- формулы (документ A7, сводка формул) ----------

def sigma_um(mask: dict) -> tuple[float, float]:
    """(σ, σ_полн) совмещения маски, мкм."""
    s = mask['registration_3sigma_um'][mask['exposure']] / 3
    scale = mask['scale_sigma_ppm'] * 1e-6 * mask['panel_half_diagonal_mm'] * 1000
    return s, math.hypot(s, scale)


def p_encroach_rect(e: float, s: float) -> float:
    return 1.0 if e <= 0 else float(1 - (1 - 2 * Q(e / s)) ** 2)


def p_encroach_circ(e: float, s: float) -> float:
    return 1.0 if e <= 0 else float(math.exp(-e * e / (2 * s * s)))


def p_peel(w_um: float, mask: dict) -> float:
    """Доля отслоившихся перемычек шириной w (условная логистическая кривая по купону)."""
    c = mask['color']
    return float(1 / (1 + math.exp((w_um - mask['peel_w50_um'][c]) / mask['peel_slope_um'][c])))


def _circle_overlap(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    b = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    c = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a + b - c


def solder_area_loss(a: float, e: float, shift: float) -> float:
    """η — доля паяемой площади круглой NSMD-площадки радиуса a, ушедшая под маску при сдвиге окна."""
    return 1 - _circle_overlap(a, a + e, shift) / (math.pi * a * a)


def _pct(p: float) -> str:
    """Вероятность в процентах для текста: 100, 15, 0,67, 3·10⁻⁵ …"""
    v = p * 100
    return f'{v:.3g}' if v >= 0.01 else f'{v:.1e}'


def mask_window_quality(ctx) -> str:
    return ctx.profile.quality('mask')


# ---------- окна и площадки стороны ----------

class _Side:
    """Окна маски одной стороны и площадки наружного слоя под ними."""

    def __init__(self, ctx, side: str):
        b = ctx.board
        self.ctx = ctx
        self.mask_name = 'F.Mask' if side == 'top' else 'B.Mask'
        self.cu_name = b.copper[0] if side == 'top' else b.copper[-1]
        self.ok = self.mask_name in b.layers and bool(b.layers[self.mask_name].features)
        if not self.ok:
            return
        self.mask = ctx.geo.layer(self.mask_name)
        self.cu = ctx.geo.layer(self.cu_name)
        self.parts = self.mask.islands                      # слившиеся окна
        # окно каждого объекта маски и его кусок
        self.win_feats = [(f, g) for f, g in zip(self.mask.features, self.mask.feature_geoms)
                          if f.dark and not g.is_empty]
        pts = shapely.point_on_surface([g for _, g in self.win_feats]) if self.win_feats else []
        self.win_part = np.full(len(self.win_feats), -1)
        if len(pts):
            src, dst = self.mask.locate(pts)
            self.win_part[src] = dst
        # площадки меди и их окна: по компоненту и выводу, иначе — по месту
        by_pin = defaultdict(list)
        index = {id(f): k for k, (f, _) in enumerate(self.win_feats)}
        for k, (f, g) in enumerate(self.win_feats):
            if f.role is Role.PAD and f.component is not None:
                by_pin[(f.component, f.pin)].append(k)
        self.pads = []                                     # (Feature меди, геометрия меди, № окна или None)
        for f, g in zip(self.cu.features, self.cu.feature_geoms):
            if f.role is not Role.PAD or not f.dark or g.is_empty:
                continue
            win = None
            for k in by_pin.get((f.component, f.pin), []):
                if self.win_feats[k][1].intersects(g):
                    win = k
                    break
            if win is None:                          # нет компонента и вывода (Gerber без .P): по месту
                cand = [index[id(m)] for m in self.mask.features_near(g) if id(m) in index]
                if cand:
                    win = max(cand, key=lambda k: self.win_feats[k][1].intersection(g).area)
            self.pads.append((f, g, win))
        self._geom_of = {id(f): g for f, g in zip(self.cu.features, self.cu.feature_geoms)}
        self.sigma, self.sigma_full = sigma_um(ctx.profile['mask'])
        self.pads_of_window = defaultdict(list)            # № окна → геометрии меди его площадок
        self.pads_of_part = defaultdict(list)              # (компонент, кусок окон) → геометрии меди
        for f, g, w in self.pads:
            if w is not None:
                self.pads_of_window[w].append(g)
                self.pads_of_part[(f.component, int(self.win_part[w]))].append(g)
        # куски меди по цепям для поиска чужой меди, нарезанные на плитки: окно рядом с плоскостью
        # земли проверяется против маленьких плиток, а не против всей плоскости
        from ..geometry.copper import _subdivide
        tiles, self.tile_piece, self.piece_net = [], [], []
        for net, g in self.cu.nets.items():
            for p in parts(g):
                k = len(self.piece_net)
                self.piece_net.append(net)
                for t in _subdivide(p, 256):
                    tiles.append(t)
                    self.tile_piece.append(k)
        self.tiles = np.asarray(tiles, dtype=object)
        self.tile_piece = np.asarray(self.tile_piece, dtype=np.int64)
        self.tile_tree = shapely.STRtree(tiles) if tiles else None

    def local_windows(self, g, distance: float):
        """Окна маски не дальше distance от g (объединение слившихся кусков рядом)."""
        idx = self.mask.island_tree.query(g, predicate='dwithin', distance=distance)
        if not len(idx):
            return shapely.Polygon()
        return self.parts[int(idx[0])] if len(idx) == 1 else shapely.MultiPolygon([self.parts[int(i)] for i in idx])

    def expansion_um(self, pad, window_feature) -> float:
        """Расширение окна над площадкой, мкм: > 0 — окно больше меди (NSMD), < 0 — меньше (SMD),
        0 — окно сдвинуто и край меди уже под маской. Круг в круге — точно; иначе по ломаным
        с отклонением 0,01 мкм (вероятность наползания чувствительна к e)."""
        from ..geometry.polygons import to_geometry
        a, b = pad.shape, window_feature.shape
        if isinstance(a, Circle) and isinstance(b, Circle):
            return (b.radius - a.radius - math.dist(a.center, b.center)) * 1000
        copper, window = to_geometry(a, 1e-5), to_geometry(b, 1e-5)
        if window.contains(copper):
            return min(shapely.distance(q.exterior, copper) for q in parts(window)) * 1000
        if copper.contains(window):
            return -min(shapely.distance(q.exterior, window) for q in parts(copper)) * 1000
        return 0.0

    def own_copper(self, k: int):
        """Медь, которой принадлежит окно k: площадка (по компоненту и выводу), переходное (по uuid)
        или, если связи нет (Gerber без атрибутов), объект меди площадки/переходного с наибольшим
        перекрытием окна."""
        f, win = self.win_feats[k]
        own = self.pads_of_window.get(k)
        if own:
            return shapely.union_all(own)
        geom_of = self._geom_of
        cands = [(cf, geom_of[id(cf)]) for cf in self.cu.features_near(win) if cf.role in (Role.PAD, Role.VIA)]
        same = [cg for cf, cg in cands if f.uid and cf.uid == f.uid]
        if same:
            return same[0]
        if cands:
            return max(cands, key=lambda c: c[1].intersection(win).area)[1]
        return None

    def foreign_near(self, k: int, distance: float):
        """Чужая медь не дальше distance от окна k → [(кусок, цепь)]. Чужая — кусок другой цепи (или
        без цепи), НЕ касающийся меди самого окна: то, что касается её, уже электрически соединено
        с площадкой (например, медная графика теплоотвода в посадочном месте)."""
        f, win = self.win_feats[k]
        if self.tile_tree is None:
            return []
        own = self.own_copper(k)
        if own is None:                  # в окне нет своей меди (крепёжное отверстие) — паять нечего
            return []
        idx = self.tile_tree.query(win, predicate='dwithin', distance=distance)
        idx = [int(i) for i in idx if f.net is None or self.piece_net[self.tile_piece[i]] != f.net]
        if not idx:
            return []
        # куски, касающиеся своей меди, — по плиткам: кусок касается, если касается любая его плитка
        touching = {int(self.tile_piece[i]) for i in self.tile_tree.query(own, predicate='intersects')}
        return [(self.tiles[i], self.piece_net[self.tile_piece[i]]) for i in idx
                if int(self.tile_piece[i]) not in touching]


def _q(ctx) -> str:
    return mask_window_quality(ctx)


# ---------- М2, М7 ----------

def _inherited_zero(ctx, s: _Side):
    """Площадки, у которых нулевое расширение окна не задано, а унаследовано от платы (в KiCad 6+
    расширение маски по умолчанию 0 — тогда его выбирает завод): {(компонент, вывод)} или None,
    если уровня D нет (Gerber) — тогда так считается вся плата, где ≥ 90 % окон с e = 0."""
    d = ctx.board.design
    if d is None:
        return None
    if abs(d.rules.pad_to_mask_clearance) > 1e-9:
        return set()
    return {(fp.ref, p.number) for fp in d.footprints for p in fp.pads
            if p.solder_mask_margin is None and fp.solder_mask_margin is None}


def m2_m7(ctx, s: _Side) -> list[Finding]:
    out = []
    mask = ctx.profile['mask']
    inherited = _inherited_zero(ctx, s)
    fab_zero = []                                    # площадки с e = 0 «по умолчанию платы»
    groups = defaultdict(list)                       # (вид, e мкм) → площадки
    bga = defaultdict(list)                          # компонент → [(e, площадка, паяемый диаметр)]
    # матрица — круглые площадки компонента без отверстия с медью близкого диаметра (±30 % от медианы):
    # смесь SMD и NSMD в матрице опасна (пример документа A7, раздел 5.7: медь SMD Ø0,50 с окном Ø0,35 при
    # NSMD Ø0,40 — 0,50/0,40 = 1,25 входит), а крупные тепловые и крепёжные площадки в матрицу не входят
    drilled = {id(x) for h in ctx.board.holes for x in (h.padstack or {}).values()}
    for f, g, win in s.pads:
        if win is None:
            continue
        e = s.expansion_um(f, s.win_feats[win][0])
        circ = isinstance(f.shape, Circle)
        if circ and f.component:
            # паяемый диаметр: у NSMD задаёт медь, у SMD — окно (документ A7, раздел 5.7)
            d_def = f.shape.radius * 2 + (2 * e / 1000 if e < 0 else 0.0)
            if id(f) not in drilled:
                bga[f.component].append((e, f, d_def))
        if e < 0:
            continue                                 # SMD — маска по краю намеренно (М7)
        if abs(e) < 0.5 and inherited is not None and (f.component, f.pin) in inherited:
            fab_zero.append((f, g))
            continue
        groups[('кругл' if circ else 'прям', round(e, 1))].append((f, g))
    if inherited is None:                            # Gerber: нули по всей плате — договорённость с заводом
        n_all = sum(len(v) for v in groups.values())
        zeros = [k for k in groups if abs(k[1]) < 0.5]
        if zeros and sum(len(groups[k]) for k in zeros) >= 0.9 * n_all:
            for k in zeros:
                fab_zero += groups.pop(k)
    if fab_zero:
        f, g = fab_zero[0]
        c = g.representative_point()
        out.append(Finding(
            'A7', 'М2', f'расширение окон маски не задано (0 по умолчанию платы) у {len(fab_zero)} площадок — его '
                        f'выберет завод; без расширения маска наползёт на каждой плате', Severity.INFO,
            layer=s.mask_name, pos=(c.x, c.y),
            objects=[{'kind': 'pad', 'uid': f.uid, 'ref': f.component, 'pin': f.pin} for f, _ in fab_zero[:20]],
            measured={'площадок': len(fab_zero), 'σ_полн_мкм': s.sigma_full},
            thresholds={'e_min_мкм': 3 * s.sigma_full}, data_quality=_q(ctx),
            assumptions=['в KiCad 6+ расширение по умолчанию 0: обычно завод добавляет своё (документ A7, раздел 3.1)'],
            fix=f'уточнить у завода, какое расширение он добавит, или задать на плате ≥ 3σ_полн = '
                f'{3 * s.sigma_full:.0f} мкм с проверкой перемычек М1'))
    for (kind, e), pads in sorted(groups.items(), key=lambda kv: kv[0][1]):
        p = p_encroach_circ(e, s.sigma_full) if kind == 'кругл' else p_encroach_rect(e, s.sigma_full)
        sev = probability_severity(p, ctx.profile['verdict'])
        if sev is Severity.OK:
            continue
        f, g = pads[0]
        c = g.representative_point()
        extra = {}
        if kind == 'кругл' and e > 0:
            a = f.shape.radius * 1000
            extra = {'η_при_3σ': solder_area_loss(a, e, 3 * s.sigma_full)}
        out.append(Finding(
            'A7', 'М2', f'расширение окна {e:.0f} мкм у {len(pads)} {"круглых" if kind == "кругл" else "некруглых"} '
                        f'площадок: наползание маски у {_pct(p)} % плат', sev,
            layer=s.mask_name, pos=(c.x, c.y), nets=tuple(sorted({f.net for f, _ in pads if f.net}))[:10],
            objects=[{'kind': 'pad', 'uid': f.uid, 'ref': f.component, 'pin': f.pin} for f, _ in pads[:20]],
            measured={'e_мкм': e, 'площадок': len(pads), 'σ_полн_мкм': s.sigma_full, **extra},
            thresholds={'e_min_мкм': 3 * s.sigma_full}, probability=p, data_quality=_q(ctx),
            assumptions=[f'смещение маски общее для платы, нормальное, σ = {s.sigma:.1f} мкм '
                         f'({mask["exposure"]}), с масштабом {s.sigma_full:.1f} мкм'],
            fix=f'расширение не меньше 3σ_полн = {3 * s.sigma_full:.0f} мкм (с проверкой перемычек М1) '
                f'или более точное экспонирование'))
    n_min = mask.get('bga_min_pads', 16)
    for ref, items in bga.items():
        med = float(np.median([f.shape.radius for _, f, _ in items]))
        items = [t for t in items if abs(t[1].shape.radius - med) <= 0.3 * med]
        if len(items) < n_min:
            continue
        smd = [f for e, f, _ in items if e < -0.5]      # |e| ≤ 0,5 мкм — окно по меди: ни то, ни другое
        nsmd = [f for e, f, _ in items if e > 0.5]
        if smd and nsmd:
            minority = smd if len(smd) < len(nsmd) else nsmd
            g = minority[0].shape
            d_lo, d_hi = min(d for *_, d in items), max(d for *_, d in items)
            dia = f'Ø{d_lo:.2f} мм' if d_hi - d_lo < 0.005 else f'Ø{d_lo:.2f}–{d_hi:.2f} мм'
            out.append(Finding(
                'A7', 'М7', f'{ref}: в матрице {len(items)} площадок ({dia} паяемый) смешаны SMD ({len(smd)}) и NSMD '
                            f'({len(nsmd)})', Severity.WARNING, layer=s.mask_name, pos=g.center,
                objects=[{'kind': 'pad', 'uid': f.uid, 'ref': ref, 'pin': f.pin} for f in minority[:20]],
                measured={'smd': len(smd), 'nsmd': len(nsmd), 'выводы_меньшинства': [f.pin for f in minority[:20]]},
                data_quality=_q(ctx),
                assumptions=['разная высота паяных соединений (модель усечённой сферы, документ A7, раздел 5.7)'],
                fix='единое определение площадок матрицы по рекомендации производителя корпуса'))
    return out


# ---------- М3 ----------

def m3_foreign_copper(ctx, s: _Side) -> list[Finding]:
    out = []
    far = 3 * s.sigma_full / 1000
    for k, (f, win) in enumerate(s.win_feats):
        if f.role not in (Role.PAD, Role.VIA):
            continue
        near = s.foreign_near(k, far)
        if not near:
            continue
        inside = [(p, n) for p, n in near if p.intersects(win)]
        area = sum(win.intersection(p).area for p, _ in inside)
        if inside and area >= 1e-6:                 # меньше 1 мкм² — касание: ниже по расстоянию 0, P = Q(0)
            p0 = win.intersection(inside[0][0]).representative_point()
            out.append(Finding(
                'A7', 'М3', f'в окне {f.component or "переходного"}{"." + f.pin if f.pin else ""} открыта чужая медь '
                            f'({area:.3f} мм²) — припой может замкнуть', Severity.ERROR, layer=s.mask_name,
                pos=(p0.x, p0.y), nets=tuple(x for x in dict.fromkeys([f.net] + [n for _, n in inside]))[:6],
                objects=[{'kind': f.role.value, 'uid': f.uid, 'ref': f.component, 'pin': f.pin}],
                measured={'площадь_мм2': area}, data_quality=_q(ctx),
                fix=f'уменьшить расширение окна (e ≤ d_Cu − 3σ = d_Cu − {3 * s.sigma_full:.0f} мкм) или отодвинуть '
                    f'чужую медь'))
            continue
        d = min(win.distance(p) for p, _ in near)
        prob = float(Q(d * 1000 / s.sigma_full))
        sev = probability_severity(prob, ctx.profile['verdict'])
        if sev is Severity.OK:
            continue
        p0 = shapely.shortest_line(win, min(near, key=lambda pn: win.distance(pn[0]))[0]).interpolate(0.5,
                                                                                                    normalized=True)
        out.append(Finding(
            'A7', 'М3', f'чужая медь в {d * 1000:.0f} мкм от окна {f.component or "переходного"}'
                        f'{"." + f.pin if f.pin else ""}: откроется смещением маски у {_pct(prob)} % плат', sev,
            layer=s.mask_name, pos=(p0.x, p0.y), nets=tuple(dict.fromkeys([f.net] + [n for _, n in near]))[:6],
            objects=[{'kind': f.role.value, 'uid': f.uid, 'ref': f.component, 'pin': f.pin}],
            measured={'расстояние_мкм': d * 1000}, probability=prob, data_quality=_q(ctx),
            fix=f'зазор от окна до чужой меди не меньше 3σ = {3 * s.sigma_full:.0f} мкм'))
    return out


# ---------- М4, М5 ----------

def _hole_length_mm(b, h) -> float:
    """Длина канала отверстия, мм: между его крайними медными слоями по стеку (глухое и скрытое — не вся
    плата); сквозное или без стека — толщина платы."""
    from .registration import layer_depths
    cu = b.copper
    if (h.span[0], h.span[1]) == (cu[0], cu[-1]):
        return b.thickness or DEFAULT_BOARD_MM
    z = layer_depths(b)
    return abs(z[h.span[1]] - z[h.span[0]]) or (b.thickness or DEFAULT_BOARD_MM)

def _via_state(s: _Side, land, hole_g):
    """Состояние переходного: (φ_L, φ_H, класс) — closed / open / partial / edge.

    Ловушку растворов создаёт частично закрытое ОТВЕРСТИЕ, поэтому запас 3σ проверяется по
    отверстию (документ A7, раздел 3.7): закрыто надёжно —
    окна не ближе 3σ к отверстию; открыто надёжно — отверстие с запасом 3σ целиком в окне;
    частично — 1 % < φ_H < 99 %; остальное — «на грани»."""
    m = 3 * s.sigma_full / 1000
    lg = land if land is not None else hole_g
    w = s.local_windows(lg, m + 0.001)
    fl = lg.intersection(w).area / lg.area if lg.area else 0.0
    fh = hole_g.intersection(w).area / hole_g.area if hole_g.area else 0.0
    if w.is_empty:
        return fl, fh, 'closed'
    if fh <= 0.01 and hole_g.distance(w) >= m:
        return fl, fh, 'closed'
    if hole_g.buffer(m).difference(w).area <= 0.01 * hole_g.area:
        return fl, fh, 'open'
    if 0.01 < fh < 0.99:
        return fl, fh, 'partial'
    return fl, fh, 'edge'


def m4_m5(ctx, s: _Side) -> list[Finding]:
    from ..geometry.polygons import to_geometry
    out = []
    b = ctx.board
    paste = ctx.profile.get('paste', {})
    pw_list = [(f, g, s.win_feats[win][1]) for f, g, win in s.pads if win is not None and f.net]
    pw_tree = shapely.STRtree([w for _, _, w in pw_list]) if pw_list else None
    m5 = defaultdict(list)                           # компонент → открытые переходные у его площадок
    dam_min = ctx.profile['mask']['dam_min_um'][ctx.profile['mask']['color']] / 1000
    for h, hg in zip(b.holes, ctx.geo.hole_geoms()):
        if h.role is not Role.VIA or not h.plated or s.cu_name not in b.copper or h.diameter < MIN_HOLE_MM:
            continue
        cu = b.copper
        if not (cu.index(h.span[0]) <= cu.index(s.cu_name) <= cu.index(h.span[1])):
            continue
        land_f = h.padstack.get(s.cu_name)
        land = to_geometry(land_f.shape) if land_f is not None else None
        fl, fh, state = _via_state(s, land, hg)
        if state in ('partial', 'edge'):
            out.append(Finding(
                'A7', 'М4', ('переходное частично закрыто маской' if state == 'partial' else
                             'окно переходного без запаса 3σ: закрытие зависит от смещения маски')
                + f' (поясок в окне {fl * 100:.0f} %, отверстие {fh * 100:.0f} %)',
                Severity.ERROR if state == 'partial' else Severity.WARNING, layer=s.mask_name, pos=h.pos,
                nets=(h.net,) if h.net else (), objects=[{'kind': 'via', 'uid': h.uid}] if h.uid else [],
                measured={'φ_L': fl, 'φ_H': fh}, data_quality=_q(ctx),
                assumptions=['в частично закрытом отверстии задерживаются растворы — коррозия, пузыри'],
                fix=f'закрыть целиком (окна не ближе 3σ = {3 * s.sigma_full:.0f} мкм) или открыть с запасом 3σ; '
                    f'одно правило закрытия для платы (IPC-4761)'))
        if fh <= 0.01 or not h.net:
            continue
        # М5: открытое переходное рядом с площадкой той же цепи без перемычки
        if pw_tree is None:
            continue
        vw = hg if land is None else land
        m = 3 * s.sigma_full / 1000
        via_win = s.local_windows(vw, m).intersection(vw.buffer(m))
        if via_win.is_empty:
            continue
        near = sorted(int(i) for i in pw_tree.query(via_win, predicate='dwithin', distance=dam_min))
        for pf, pg, pw in (pw_list[i] for i in near):
            if pf.net != h.net:
                continue
            gap = pw.distance(via_win)
            if gap >= dam_min:
                continue
            v_hole = math.pi * h.diameter ** 2 * _hole_length_mm(b, h) / 4
            v_solder = paste.get('metal_fraction', 0.5) * paste.get('transfer', 0.9) * pg.area *                 paste.get('stencil_um', 100) / 1000
            in_pad = pg.contains(hg)
            m5[pf.component].append((v_hole / v_solder, h, pf, v_hole, v_solder, gap, in_pad))
            break
    # М5 — одна находка на компонент: у разъёма с переходными в каждой площадке их сотни
    for ref, items in sorted(m5.items(), key=lambda kv: str(kv[0])):
        items.sort(key=lambda t: -t[0])
        ratio, h, pf, v_hole, v_solder, gap, in_pad = items[0]
        bad = [t for t in items if t[0] > 1]
        n_in = sum(t[6] for t in items)
        where = ('в площадках' if n_in == len(items) else 'у площадок' if n_in == 0 else
                 f'в площадках ({n_in}) и рядом без перемычки')
        out.append(Finding(
            'A7', 'М5', f'{ref or "?"}: открытые переходные {where} — {len(items)} шт.; худшее — {pf.component}.{pf.pin}: '
                        f'канал {v_hole:.3f} мм³ против ≈ {v_solder:.3f} мм³ припоя'
                        + (f'; канал больше припоя — {len(bad)} шт.' if bad else ''),
            Severity.WARNING if bad else Severity.INFO, layer=s.mask_name, pos=h.pos,
            nets=tuple(dict.fromkeys(t[1].net for t in items))[:10],
            objects=[{'kind': 'footprint', 'ref': ref}] + [
                {'kind': 'via', 'uid': t[1].uid, 'pin': t[2].pin} for t in items[:20]],
            measured={'переходных': len(items), 'в_площадке': n_in, 'канал_больше_припоя': len(bad),
                      'V_отв_мм3': v_hole, 'V_припоя_мм3': v_solder, 'перемычка_мкм': gap * 1000},
            data_quality=_q(ctx),
            assumptions=['припой ≈ доля металла × перенос × площадь × трафарет (профиль [paste]); '
                         'весь канал заполняется, если переходное не заполнено и не закрыто крышкой',
                         *([f'толщины платы нет в файле — принято {DEFAULT_BOARD_MM} мм'] if not b.thickness else [])],
            fix='закрыть переходные, отодвинуть до перемычки ≥ минимума или заполнить с крышкой '
                '(IPC-4761, тип VII)'))
    return out


# ---------- М1, М6 ----------

def m1_m6(ctx, s: _Side) -> list[Finding]:
    out = []
    mask = ctx.profile['mask']
    color = mask['color']
    dam_min = mask['dam_min_um'][color] / 1000
    edge_min = mask.get('edge_strip_min_um', 100) / 1000
    eps = 0.0005                                     # мм: перемычка ровно на минимуме — не узкая
    # перемычки выводов одного компонента смотрим до двух минимумов: даже на минимуме ряд из
    # нескольких перемычек теряет хотя бы одну заметно часто (документ A7: U2, 7 по 100 мкм — 4,6 %)
    look = 2 * dam_min
    d = ctx.board.design
    bridges_ok = bool(d and d.rules.allow_soldermask_bridges_in_footprints)
    n = len(s.parts)
    # кто в каждом куске окон
    comps = [set() for _ in range(n)]
    nets = [set() for _ in range(n)]
    has_via = [False] * n
    for k, (f, _) in enumerate(s.win_feats):
        p = s.win_part[k]
        if p < 0:
            continue
        if f.role is Role.PAD and f.component:
            comps[p].add(f.component)
        if f.net:
            nets[p].add(f.net)
        if f.role is Role.VIA:
            has_via[p] = True
    # общее окно на выводах разных цепей одного компонента
    part_refs = defaultdict(lambda: defaultdict(set))   # кусок окон → компонент → цепи его выводов
    for k, (f, _) in enumerate(s.win_feats):
        if s.win_part[k] >= 0 and f.role is Role.PAD:
            part_refs[int(s.win_part[k])][f.component].add(f.net)
    for p in sorted(part_refs):
        for ref, nn in part_refs[p].items():
            if ref and len({x for x in nn if x}) > 1:
                c = s.parts[p].representative_point()
                out.append(Finding(
                    'A7', 'М1', f'{ref}: общее окно маски на выводах разных цепей ({len(nn)} цепи) — перемычек нет',
                    Severity.INFO if bridges_ok else Severity.WARNING, layer=s.mask_name, pos=(c.x, c.y),
                    nets=tuple(sorted(x for x in nn if x))[:8], data_quality=_q(ctx),
                    assumptions=['общие окна внутри компонента разрешены настройкой платы'] if bridges_ok else [],
                    fix='если это намеренно — меньше пасты; иначе перемычки маски'))
    # пары соседних кусков окон
    tree = s.mask.island_tree
    arr = np.asarray(s.parts, dtype=object)
    by_fp = defaultdict(list)                        # компонент → [(ширина, точка, пара)]
    if n > 1:
        a, bb = tree.query(s.parts, predicate='dwithin', distance=look)
        keep = a < bb
        for i, j in zip(a[keep], bb[keep]):
            w = float(arr[i].distance(arr[j]))
            if w <= 0:
                continue
            (ax, ay), (bx, by) = shapely.shortest_line(arr[i], arr[j]).coords
            pos = ((ax + bx) / 2, (ay + by) / 2)
            common = comps[i] & comps[j]
            same_net_via = (has_via[i] != has_via[j]) and bool(nets[i] & nets[j])
            if common and not has_via[i] and not has_via[j]:
                by_fp[sorted(common)[0]].append((w, pos, (int(i), int(j))))
            elif same_net_via or w >= dam_min - eps:
                continue                              # площадка и переходное той же цепи — М5
            else:
                pp = p_peel(w * 1000, mask)
                sev = max(probability_severity(pp, ctx.profile['verdict']), Severity.WARNING)
                out.append(Finding(
                    'A7', 'М6', f'узкая полоска маски {w * 1000:.0f} мкм (минимум {dam_min * 1000:.0f} мкм): '
                                f'отслоится с вероятностью {_pct(pp)} %', sev, layer=s.mask_name, pos=pos,
                    nets=tuple(sorted((nets[i] | nets[j]) - {None}))[:6],
                    measured={'ширина_мкм': w * 1000, 'p_отсл': pp}, thresholds={'цел': dam_min},
                    probability=pp, data_quality=_q(ctx),
                    assumptions=[f'кривая отслоения для цвета «{color}» условная (купон завода)'],
                    fix='закрыть переходное, отодвинуть окна или намеренно объединить их — узкая полоска хуже, '
                        'чем никакой'))
    for ref, dams in by_fp.items():
        dams.sort()
        ps = [p_peel(w * 1000, mask) for w, _, _ in dams]
        p_row = 1 - float(np.prod([1 - p for p in ps]))
        narrow = [w for w, _, _ in dams if w < dam_min - eps]
        if not narrow and probability_severity(p_row, ctx.profile['verdict']) is Severity.OK:
            continue
        w0, pos, (i, j) = dams[0]
        s_cu = _copper_gap(s, ref, i, j)
        e_min = 3 * s.sigma_full / 1000
        e_max = (s_cu - dam_min) / 2 if s_cu is not None else None
        feasible = e_max is not None and e_max >= e_min
        sev = probability_severity(p_row, ctx.profile['verdict'])
        if narrow:
            sev = max(sev, Severity.WARNING)
        if bridges_ok:
            sev = Severity.INFO
        window_note = (f'окно осуществимости {e_min * 1000:.0f}…{e_max * 1000:.0f} мкм' if feasible else
                       f'окна осуществимости нет: промежуток {s_cu * 1000:.0f} мкм < w_d,min + 2·3σ = '
                       f'{(dam_min + 2 * e_min) * 1000:.0f} мкм' if s_cu is not None else '')
        out.append(Finding(
            'A7', 'М1', f'{ref}: {len(dams)} перемычек маски, самая узкая {w0 * 1000:.0f} мкм (минимум '
                        f'{dam_min * 1000:.0f} мкм), хотя бы одна отслоится у {_pct(p_row)} % плат', sev,
            layer=s.mask_name, pos=pos,
            objects=[{'kind': 'footprint', 'ref': ref}],
            measured={'ширина_мкм': w0 * 1000, 'перемычек': len(dams), 'P_ряд': p_row,
                      **({'промежуток_меди_мкм': s_cu * 1000} if s_cu is not None else {})},
            thresholds={'цел': dam_min}, probability=p_row, data_quality=_q(ctx),
            assumptions=[f'кривая отслоения для цвета «{color}» условная; узкую перемычку завод может удалить '
                         f'(общее окно)'] + ([window_note] if window_note else []),
            fix=('уменьшить расширение до ' + (f'{e_max * 1000:.0f} мкм' if feasible else 'предела совмещения')
                 + '; иначе намеренное общее окно с меньшей пастой, SMD-площадки или LDI')))
    # полоски у края платы
    outline = ctx.geo.outline_geom()
    if not outline.is_empty and n:
        edge = outline.boundary
        for i in tree.query(edge, predicate='dwithin', distance=edge_min):
            w = float(arr[i].distance(edge))
            if w <= 0 or w >= edge_min - eps:
                continue
            pp = p_peel(w * 1000, mask)
            ln = shapely.shortest_line(arr[i], edge)
            c = ln.interpolate(0.5, normalized=True)
            out.append(Finding(
                'A7', 'М6', f'полоска маски у края платы {w * 1000:.0f} мкм (минимум {edge_min * 1000:.0f} мкм)',
                max(probability_severity(pp, ctx.profile['verdict']), Severity.WARNING), layer=s.mask_name,
                pos=(c.x, c.y), nets=tuple(sorted(nets[i] - {None}))[:4],
                measured={'ширина_мкм': w * 1000, 'p_отсл': pp}, thresholds={'цел': edge_min}, probability=pp,
                data_quality=_q(ctx), fix='отодвинуть окно от края или открыть край целиком'))
    return out


def _copper_gap(s: _Side, ref, i, j) -> float | None:
    """Промежуток меди между площадками компонента ref, чьи окна в кусках i и j, мм."""
    gi = s.pads_of_part.get((ref, i), [])
    gj = s.pads_of_part.get((ref, j), [])
    if not gi or not gj:
        return None
    return float(min(a.distance(b) for a in gi for b in gj))


# ---------- М8 ----------

def m8_mask_thickness(ctx, s: _Side) -> list[Finding]:
    from .thresholds import copper_thickness_um
    mask = ctx.profile['mask']
    outline = ctx.geo.outline_geom()
    if outline.is_empty:
        return []
    f = min(s.cu.area / outline.area, 1.0)
    t_cu, _, _ = copper_thickness_um(ctx.board, s.cu_name, ctx.profile)
    t_top = mask['thickness_um'] - mask['levelling_beta'] * (1 - f) * t_cu
    need = mask['over_conductor_min_um']
    if t_top >= need:
        return []
    need_bar = need + mask['levelling_beta'] * (1 - f) * t_cu
    return [Finding(
        'A7', 'М8', f'маска {mask["thickness_um"]} мкм над медью {t_cu:.0f} мкм: над проводником ≈ {t_top:.1f} мкм '
                    f'(нужно ≥ {need} мкм)' + (' — проводник выступает' if t_top < 0 else ''),
        # вывод целиком определяется β; пока β условный (не от завода), это информация, а не предупреждение
        Severity.WARNING if ctx.profile.quality('mask') == 'профиль завода' else Severity.INFO,
        layer=s.mask_name, measured={'t_верх_мкм': t_top, 'доля_меди': f, 't_Cu_мкм': t_cu},
        thresholds={'цел': need}, data_quality=_q(ctx),
        assumptions=[f'выравнивание β = {mask["levelling_beta"]} (по микрошлифу), доля меди по слою'],
        fix=f'в документацию: средняя толщина маски ≥ {need_bar:.0f} мкм (двойное нанесение)')]


@rule('A7')
def check(ctx) -> list[Finding]:
    out = []
    for side in ('top', 'bottom'):
        s = _Side(ctx, side)
        if not s.ok:
            continue
        out += m1_m6(ctx, s) + m2_m7(ctx, s) + m3_foreign_copper(ctx, s) + m4_m5(ctx, s) + m8_mask_thickness(ctx, s)
    return out

