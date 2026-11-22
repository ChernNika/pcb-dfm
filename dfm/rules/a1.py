"""Класс A1 — ширина проводников и зазоры (документ A1, глава 6).

  Д1  трасса уже минимума слоя — ширина линии меди с ролью «проводник» против w_min слоя;
  Д2  тонкий текст и графика на меди — толщина линий против w_min, высота символа против
      2(s_min + ℓ) (иначе промежутки в «8», «В» не протравятся);
  Д3  перешеек в заливке — узкое место, соединяющее две широкие части заливки (открытие
      кругом диаметра w_min), и настройка зоны min_thickness меньше w_min;
  Д4  зазор между КУСКАМИ меди меньше технологического минимума слоя — по всей меди слоя, а
      не по цепям: щель между двумя кусками одной цепи, соединёнными на другом слое, DRC не
      видит, а травление — видит. Пустота внутри одного куска (карман) — не Д4, а B1;
  Д5  зазор мал для напряжения — только если в профиле есть напряжения цепей и таблица
      «напряжение → зазор» по категориям (ГОСТ Р 53429-2009, табл. 7–9, или ECSS-Q-ST-70-12C,
      Table 133; числа в профиль не переписаны), иначе находка «невыполнено»;
  Д7  трасса тонка для тока — сечение по формуле IPC-2221 (документ A1, раздел 6.7), полный
      ток в каждом куске цепи (консервативно); только если в профиле есть токи цепей;
  Д8  лишний риск от случайных дефектов — ранжирование по L/s длинных участков с зазором
      меньше двух минимумов (критическая площадь x₀²L/s, документ A1, раздел 6.8); рекомендация.
  Д9  вероятность выйти за допуск приёмки по ширине и зазору (Коробков 2025, ф. (24)): ширина после
      фотолитографии и травления W ~ N(w + δ, σ_л² + 2σ_т²), зазор G ~ N(g − δ, …); брак — сужение
      больше k·w (k·g) по IPC-A-600G п. 2.10.1.1/2.10.1.2. Слой — как у автора (с. 162–163, 244–245): самая
      узкая трасса и самый узкий зазор, P = 1 − (1 − P_w)(1 − P_g); yield.combine = "product" — произведение
      по всем группам трасс (цепь, ширина) и парам островов (наше расширение, допущение независимости).
      h — стравливаемая медь. Только если в профиле [yield] enabled = true (нужны σ процесса).
Д6 (путь утечки по IEC 60664-1) — опциональный и здесь не проверяется.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import Point

from ..board.layers import LayerKind, kind_of
from ..board.model import Role
from ..board.shapes import Stroke
from ..geometry.copper import parts
from .engine import guarded, rule
from .findings import Finding, Severity, geometric_severity, margin, probability_severity
from .voltage import (
    MAINS_TOLERANCE, max_voltage_difference, peak_voltage as _peak_voltage,
    required_voltage_clearance, voltage_difference,
)

__all__ = ['check', 'current_width_mm', 'required_voltage_clearance']

NECK_AREA_C = 0.3                 # фрагменты открытия меньше c·r² — срезанные уголки (раздел 4.9)
SLIT_CLOSE_MM = 0.005             # Д3: щели уже 5 мкм между кусками меди одного острова (стык двух заливок) не
                                  #   протравятся (разрешение резиста ≥ 25 мкм) — при замере перешейка медь сплошная
MIL2_MM2 = 0.0254 ** 2            # 1 мил² в мм²
IPC2221_K = {True: 0.048, False: 0.024}   # наружный / внутренний слой (IPC-2221; документ A1, разделы 3.1, 6.7)


def _objects(features) -> list:
    """Объекты находки для плагина: роль, uuid KiCad, компонент и вывод (у Gerber uuid нет)."""
    out = []
    for f in features:
        o = {'kind': f.role.value}
        if f.uid:
            o['uid'] = f.uid
        if f.component:
            o['ref'] = f.component
        if f.pin:
            o['pin'] = f.pin
        out.append(o)
    return out


def _quality(ctx, thr) -> str:
    return f'{ctx.profile.quality("etch")}; порог — {thr.source}'


def _um(v: float) -> str:
    return f'{v * 1000:.0f}'


# ---------- Д1 ----------

def d1_track_width(ctx) -> list[Finding]:
    """Отрезки одной цепи одинаковой ширины на слое — одна находка."""
    out = []
    for layer in ctx.board.copper:
        thr = ctx.thresholds[layer].width
        groups = defaultdict(list)
        for f in ctx.board.layers[layer].features:
            if f.role is Role.TRACK and isinstance(f.shape, Stroke) and f.dark:
                groups[(f.net, round(f.shape.width, 6))].append(f)
        for (net, w), feats in sorted(groups.items(), key=lambda kv: kv[0][1]):
            sev = geometric_severity(w, thr.target, thr.level_a, thr.level_c)
            if sev is Severity.OK:
                continue
            p = feats[0].shape.pts[0]
            out.append(Finding(
                'A1', 'Д1', f'трасса {_um(w)} мкм уже минимума слоя {thr.target * 1000:.1f} мкм', sev,
                layer=layer, pos=(float(p[0]), float(p[1])), nets=(net,) if net else (),
                objects=_objects(feats), measured={'ширина_мм': w, 'отрезков': len(feats)},
                thresholds=thr.as_dict(), margin=margin(w, thr.target), data_quality=_quality(ctx, thr),
                assumptions=[x for x in (thr.note,) if x],
                fix=f'расширить до {thr.target * 1000:.0f} мкм и более с учётом соседей '
                    f'(w_max = w + (d_L − s_min) + (d_R − s_min)) или перенести на другой слой'))
    return out


# ---------- Д2 ----------

def d2_thin_text(ctx) -> list[Finding]:
    out = []
    d = ctx.board.design
    if d is not None:
        for t in d.texts:
            if kind_of(t.layer) is not LayerKind.COPPER or t.layer not in ctx.thresholds:
                continue
            th = ctx.thresholds[t.layer]
            if t.thickness is not None:
                sev = geometric_severity(t.thickness, th.width.target, th.width.level_a, th.width.level_c)
                if sev is not Severity.OK:
                    out.append(Finding(
                        'A1', 'Д2', f'линии текста «{t.text}» {_um(t.thickness)} мкм уже минимума '
                                    f'{th.width.target * 1000:.1f} мкм', sev, layer=t.layer, pos=t.pos,
                        objects=[{'kind': 'text', 'uid': t.uid}] if t.uid else [],
                        measured={'толщина_мм': t.thickness}, thresholds=th.width.as_dict(),
                        margin=margin(t.thickness, th.width.target), data_quality=_quality(ctx, th.width),
                        fix='утолщить линии текста или перенести надпись на маркировку'))
            if t.size is not None and t.thickness is not None:
                need = 2 * (th.clearance.target + t.thickness)
                if t.size[0] < need:
                    out.append(Finding(
                        'A1', 'Д2', f'текст «{t.text}» высотой {t.size[0]:.2f} мм: промежутки в буквах не '
                                    f'протравятся, нужно h ≥ 2(s_min + ℓ) = {need:.2f} мм', Severity.WARNING,
                        layer=t.layer, pos=t.pos, objects=[{'kind': 'text', 'uid': t.uid}] if t.uid else [],
                        measured={'высота_мм': t.size[0], 'толщина_мм': t.thickness},
                        thresholds={'цел': need}, margin=margin(t.size[0], need),
                        data_quality=_quality(ctx, th.clearance),
                        assumptions=['промежуток символа оценён по «8» как два кольца (документ A1, раздел 6.2)'],
                        fix=f'увеличить высоту до {need:.2f} мм, утоньшить линии или перенести на маркировку'))
    # графика на меди (и текст, пришедший из Gerber готовыми линиями)
    for layer in ctx.board.copper:
        thr = ctx.thresholds[layer].width
        groups = defaultdict(list)
        for f in ctx.board.layers[layer].features:
            if f.role is Role.GRAPHIC and isinstance(f.shape, Stroke) and f.dark:
                groups[round(f.shape.width, 6)].append(f)
        for w, feats in sorted(groups.items()):
            sev = geometric_severity(w, thr.target, thr.level_a, thr.level_c)
            if sev is Severity.OK:
                continue
            p = feats[0].shape.pts[0]
            out.append(Finding(
                'A1', 'Д2', f'графика на меди: линии {_um(w)} мкм уже минимума {thr.target * 1000:.1f} мкм', sev,
                layer=layer, pos=(float(p[0]), float(p[1])), objects=_objects(feats[:20]),
                measured={'ширина_мм': w, 'линий': len(feats)}, thresholds=thr.as_dict(),
                margin=margin(w, thr.target), data_quality=_quality(ctx, thr),
                fix='утолщить линии или перенести графику на маркировку'))
    return out


# ---------- Д3 ----------

def d3_zone_necks(ctx) -> list[Finding]:
    """Перешейки по сохранённой заливке (её получит завод, если не перезалить) и, если можно, по
    свежей перезаливке (KiCad перезаливает зоны при выводе Gerber): перешейки, которые есть только
    в одной из заливок, помечаются."""
    saved = _d3_necks(ctx)
    # перезаливка копии (десятки секунд в KiCad) нужна, только если на плате есть зоны
    has_zones = any(f.role is Role.ZONE for L in ctx.board.copper for f in ctx.board.layers[L].features)
    fctx, _ = ctx.fresh() if has_zones else (None, 'зон нет')
    if fctx is not None:
        fresh = _d3_necks(fctx)
        pts_s = [(f.layer, p) for f in saved for p in f.measured['точки']]
        pts_f = [(f.layer, p) for f in fresh for p in f.measured['точки']]

        def near(f, pts):
            return any(L == f.layer and math.dist(p, q) < 0.1 for p in f.measured['точки'] for L, q in pts)
        for f in saved:
            if not near(f, pts_f):
                f.assumptions.append('после перезаливки зон этого перешейка нет — заливка в файле устарела')
        for f in fresh:
            if not near(f, pts_s):
                f.title = 'после перезаливки: ' + f.title
                f.assumptions.append('в сохранённой заливке перешейка нет — он появится, когда KiCad перезальёт '
                                     'зоны (например, при выводе Gerber)')
                saved.append(f)
    return saved + _d3_settings(ctx)


def _neck_width(n, wide_near, r, tol=1e-4, depth=0):
    """Ширина перешейка, мм — самое узкое место куска n между широкими частями заливки.

    Проход: 2·s*, s* — наибольший радиус, при котором центры кругов (эрозия области n ∪ широкие куски на s)
    ещё связывают две широкие части (деление пополам до tol). Связность открытия не годится: круги из двух
    частей перекрываются через щель, в которую сами не проходят. Область — только n и куски широких частей в
    пределах 2r: обходы через соседние перешейки (поле отверстий) не учитываются. Если в n несколько
    параллельных путей (при большом пороге узкие места сливаются в один кусок), проход даёт ширину самого
    широкого пути — поэтому кусок разбирается ещё раз с радиусом s*: более узкий путь выделяется своим куском
    (до трёх уровней). Вписанная окружность n не годится: с ростом порога в n входят «плечи» у отверстий."""
    h = SLIT_CLOSE_MM / 2
    local = shapely.union_all([n, *wide_near]).buffer(h, quad_segs=8).buffer(-h, quad_segs=8)
    lo, hi = 0.0, 2 * r                       # без настоящего сужения проход шире 2r — перешейка нет
    while hi - lo > tol:
        s = (lo + hi) / 2
        centers = local.buffer(-s, quad_segs=8)
        linked = any(sum(g.intersects(w) for w in wide_near) >= 2 for g in parts(centers))
        lo, hi = (s, hi) if linked else (lo, s)
    width = 2 * lo
    r2 = lo - 2 * tol
    if depth < 3 and 2 * tol < r2 < r:
        opened = local.buffer(-r2, quad_segs=8).buffer(r2, quad_segs=8)
        wide2 = parts(opened)
        for n2 in parts(local.difference(opened)):
            if n2.area < NECK_AREA_C * r2 * r2:
                continue
            near2 = [w.intersection(n2.buffer(2 * r2, quad_segs=8)) for w in wide2 if w.intersects(n2.buffer(tol))]
            near2 = [w for w in near2 if not w.is_empty]
            if len(near2) >= 2:
                width = min(width, _neck_width(n2, near2, r2, tol, depth + 1))
    return width


def _d3_necks(ctx) -> list[Finding]:
    out = []
    for layer in ctx.board.copper:
        thr = ctx.thresholds[layer].width
        lc = ctx.geo.layer(layer)
        zones = [g for f, g in zip(lc.features, lc.feature_geoms) if f.role is Role.ZONE and f.dark]
        if not zones:
            continue
        zone_geom = shapely.union_all(zones)
        r = thr.target / 2
        floor = 2 * ctx.tol                       # уже — это погрешность ломаных, а не перешеек
        for k, island in enumerate(lc.islands):
            if Role.ZONE not in lc.island_roles[k]:
                continue
            opened = island.buffer(-r, quad_segs=8).buffer(r, quad_segs=8)
            narrow = island.difference(opened)
            if narrow.is_empty:
                continue
            wide = parts(opened)
            tree = shapely.STRtree(wide) if wide else None
            necks = []                            # (ширина, точка, кусок)
            for n in parts(narrow):
                if n.area < NECK_AREA_C * r * r or not n.intersects(zone_geom) or tree is None:
                    continue
                touched = tree.query(n.buffer(floor), predicate='intersects')
                if len(touched) < 2:              # тупиковая полоска или уголок — классы B2 и B1
                    continue
                # широкие части у перешейка: их куски в пределах 2r — там помещается круг радиусом до r у устья
                wide_near = [wide[i].intersection(n.buffer(2 * r, quad_segs=8)) for i in touched]
                width = _neck_width(n, [w for w in wide_near if not w.is_empty], r)
                if width >= floor and geometric_severity(width, thr.target, thr.level_a, thr.level_c) \
                        is not Severity.OK:
                    p = n.representative_point()
                    necks.append((width, (p.x, p.y), n))
            if not necks:
                continue
            # перешейки одного куска заливки (обычно ряд отверстий) — одна находка со всеми точками
            necks.sort(key=lambda x: x[0])
            width, pos, n = necks[0]
            sev = geometric_severity(width, thr.target, thr.level_a, thr.level_c)
            nets = tuple(sorted(x for x in lc.island_nets[k] if x))
            out.append(Finding(
                'A1', 'Д3', f'перешейки в заливке: {len(necks)} шт., самый узкий {_um(width)} мкм при минимуме '
                            f'{thr.target * 1000:.1f} мкм',
                sev, layer=layer, pos=pos, nets=nets, objects=_objects(lc.features_near(n)[:10]),
                measured={'ширина_мм': width, 'перешейков': len(necks),
                          'точки': [(round(p[0], 4), round(p[1], 4)) for _, p, _ in necks],
                          'ширины_мм': [round(w_, 5) for w_, _, _ in necks]},
                thresholds=thr.as_dict(), margin=margin(width, thr.target), data_quality=_quality(ctx, thr),
                assumptions=['ширина — диаметр наибольшего круга, проходящего через узкое место между широкими '
                             'частями заливки (не зависит от порога)'],
                fix='шаг отверстий p ≥ D + 2c + w_min, меньше зазор зоны c или площадка D, шахматный '
                    'порядок; опасность перешейка оценивает класс B2'))
    return out


def _d3_settings(ctx) -> list[Finding]:
    out = []
    d = ctx.board.design
    if d is not None:                            # настройка зоны KiCad
        for z in d.all_zones():
            # капли (teardrop) KiCad строит сам с min_thickness 25 мкм: это настройка генератора
            # капель, а не зоны разработчика; их медь — площадка с трассой, а не заливка с перешейками
            if z.is_rule_area or z.min_thickness is None or z.teardrop:
                continue
            for layer in z.layers:
                if layer not in ctx.thresholds:
                    continue
                thr = ctx.thresholds[layer].width
                if z.min_thickness < thr.target:
                    pt = z.outline[0].pts[0] if z.outline else None
                    out.append(Finding(
                        'A1', 'Д3', f'зона {z.name or z.net or ""}: min_thickness {_um(z.min_thickness)} мкм меньше '
                                    f'минимума ширины {thr.target * 1000:.1f} мкм — заливка может дать узкие перешейки',
                        Severity.WARNING, layer=layer, pos=(float(pt[0]), float(pt[1])) if pt is not None else None,
                        nets=(z.net,) if z.net else (), objects=[{'kind': 'zone', 'uid': z.uid}] if z.uid else [],
                        measured={'min_thickness_мм': z.min_thickness}, thresholds=thr.as_dict(),
                        data_quality=_quality(ctx, thr),
                        fix=f'min_thickness зоны не меньше {thr.target * 1000:.0f} мкм, перезалить'))
    return out


# ---------- Д4 ----------

def d4_clearance(ctx) -> list[Finding]:
    """Пары кусков меди ближе порога слоя (поиск по плиткам островов, см. LayerCopper.tiles).

    Щель между кусками ОДНОЙ цепи — предупреждение, а не ошибка: электрически она безопасна
    (медь сольётся), но это нарушение технологии — недопроявленная полоска резиста может
    оторваться и замкнуть другое место; DRC такие щели не видит (документ A1, раздел 6.4).
    Промежутки между кусками графики без цепи (буквы текста из Gerber) — это Д2."""
    out = []
    for layer in ctx.board.copper:
        lc = ctx.geo.layer(layer)
        thr = ctx.thresholds[layer].clearance
        wmin = ctx.thresholds[layer].width.target
        lim = max(x for x in (thr.target, thr.level_a) if x is not None)
        for (i, j), (dd, (ax, ay), (bx, by)) in sorted(lc.island_pairs_within(lim).items()):
            sev = geometric_severity(dd, thr.target, thr.level_a, thr.level_c)
            if sev is Severity.OK:
                continue
            na = tuple(sorted(x for x in lc.island_nets[i] if x))
            nb = tuple(sorted(x for x in lc.island_nets[j] if x))
            same = bool(set(na) & set(nb))
            delta = thr.target - dd
            graphic = not na and not nb and lc.island_roles[i] <= {Role.GRAPHIC} and lc.island_roles[j] <= {Role.GRAPHIC}
            if graphic:
                code, title = 'Д2', (f'промежуток {_um(dd)} мкм в тексте или графике на меди меньше минимума '
                                     f'{thr.target * 1000:.1f} мкм')
            else:
                code = 'Д4'
                title = (f'зазор {_um(dd)} мкм между кусками меди меньше технологического минимума '
                         f'{thr.target * 1000:.1f} мкм' + (' (одна цепь — DRC такой зазор не проверяет)' if same else ''))
                if same and sev is Severity.ERROR:
                    sev = Severity.WARNING
            feats = lc.features_near(Point(ax, ay), 1e-4) + lc.features_near(Point(bx, by), 1e-4)
            out.append(Finding(
                'A1', code, title, sev, layer=layer, pos=((ax + bx) / 2, (ay + by) / 2),
                nets=tuple(dict.fromkeys(na + nb)), objects=_objects(feats),
                measured={'зазор_мм': dd, 'точка_a': (ax, ay), 'точка_b': (bx, by), 'одна_цепь': same},
                thresholds=thr.as_dict(), margin=margin(dd, thr.target), data_quality=_quality(ctx, thr),
                assumptions=[x for x in (thr.note,) if x] + (['одна цепь: электрически безопасно, риск — полоска '
                                                             'резиста'] if same else []),
                fix=f'раздвинуть на {delta * 1000:.0f} мкм или сузить трассу на {2 * delta * 1000:.0f} мкм, если '
                    f'останется не меньше {wmin * 1000:.0f} мкм; трасса проходит в промежуток G только при '
                    f'G ≥ w_min + 2·s_min'))
    return _group_same_net(out)


def _group_same_net(findings: list[Finding]) -> list[Finding]:
    """Щели внутри одной цепи на слое — одной находкой (на больших платах их сотни: переходные
    одной цепи вплотную), с числом, самой узкой щелью и списком мест."""
    keep, groups = [], defaultdict(list)
    for f in findings:
        if f.code == 'Д4' and f.measured.get('одна_цепь'):
            groups[(f.layer, f.nets)].append(f)
        else:
            keep.append(f)
    for (layer, nets), fs in groups.items():
        fs.sort(key=lambda f: f.measured['зазор_мм'])
        f = fs[0]
        if len(fs) > 1:
            f.title = (f'щели между кусками меди одной цепи {", ".join(nets)} — {len(fs)} шт., самая узкая '
                       f'{_um(f.measured["зазор_мм"])} мкм при минимуме {f.thresholds["цел"] * 1000:.1f} мкм '
                       f'(DRC такие зазоры не проверяет)')
            f.measured['щелей'] = len(fs)
            f.measured['позиции'] = [[round(v, 4) for v in g.pos] for g in fs[:20]]
            f.measured['зазоры_мм'] = [round(g.measured['зазор_мм'], 5) for g in fs[:20]]
        keep.append(f)
    return keep


# ---------- Д5 ----------

def d5_voltage(ctx) -> list[Finding]:
    nets = ctx.profile.get('nets', {})
    volts = dict(nets.get('voltages', {}))           # исходные записи профиля: число или {rms, …}
    table = nets.get('voltage_table') or {}
    if not volts or not table:
        missing = ' и '.join(x for x, ok in (('напряжений цепей', volts), ('таблицы зазоров по напряжению', table))
                             if not ok)
        return [Finding('A1', 'Д5', f'зазоры по напряжению не проверены: в профиле нет {missing}',
                        Severity.NOT_CHECKED, data_quality='нет данных',
                        fix='задать [nets] voltages и voltage_table (ГОСТ Р 53429-2009, табл. 7–9) в профиле')]
    slopes = nets.get('voltage_slope_mm_per_v', {})
    open_cat = 'B3' if nets.get('altitude_m', 0) > 3050 else 'B2'
    # радиус поиска пар: не меньше требуемого зазора при наибольшей разности напряжений (с продолжением
    # таблицы по наклону), и не меньше удвоенной последней строки таблицы
    dv_all = max_voltage_difference(volts)
    need = [required_voltage_clearance(t, dv_all, slopes.get(c)) for c, t in table.items()]
    dmax = max([2 * max(max(d for _, d in t) for t in table.values())] + [1.2 * d for d in need if d is not None])
    out, unknown = [], set()
    cu = ctx.board.copper
    for layer in cu:
        outer = layer in (cu[0], cu[-1])
        lc = ctx.geo.layer(layer)
        s_min = ctx.thresholds[layer].clearance.target
        windows, no_mask = None, False
        if outer:
            side = ctx.board.layers[layer].side           # маска — по стороне слоя (одна медь B.Cu — B.Mask)
            m = 'F.Mask' if side == 'top' else 'B.Mask'
            no_mask = m not in ctx.board.layers
            windows = None if no_mask else ctx.geo.layer(m).geom
        hv = sorted(n for n in lc.nets if n in volts)
        for i, a in enumerate(hv):
            ga = lc.nets[a]
            for other, gb in lc.nets.items():          # соседи без напряжения — проверить нельзя
                if other is not None and other not in volts and ga.distance(gb) < dmax:
                    unknown.add(other)
            for b in hv[i + 1:]:
                dv = voltage_difference(volts[a], volts[b])
                gb = lc.nets[b]
                if dv == 0 or ga.distance(gb) >= dmax:
                    continue
                # Категория по месту (документ A1, раздел 6.5): внутренний слой — B1; на наружном
                # открытая медь (в окнах маски) — B2 (выше 3050 м — B3), закрытая маской — B4.
                # Открытая и закрытая медь проверяются отдельно: у них разные требования.
                if not outer:
                    checks = [('B1', ga, gb)]
                elif no_mask:                            # слоя маски нет — вся медь открыта (консервативно)
                    checks = [(open_cat, ga, gb)]
                else:
                    oa, ob = ga.intersection(windows), gb.intersection(windows)
                    ca, cb = ga.difference(windows), gb.difference(windows)
                    checks = [(open_cat, oa, gb), (open_cat, ga, ob), ('B4', ca, cb)]
                worst = {}
                for cat, x, y in checks:
                    if x.is_empty or y.is_empty:
                        continue
                    d = x.distance(y)
                    if cat not in worst or d < worst[cat][0]:
                        worst[cat] = (d, x, y)
                for cat, (d, x, y) in sorted(worst.items()):
                    pa, pb = shapely.shortest_line(x, y).coords
                    pos = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
                    if cat not in table:
                        out.append(Finding('A1', 'Д5', f'нет строки таблицы для категории {cat}', Severity.NOT_CHECKED,
                                           layer=layer, pos=pos, nets=(a, b), data_quality='нет данных'))
                        continue
                    dreq = required_voltage_clearance(table[cat], dv, slopes.get(cat))
                    if dreq is None:
                        out.append(Finding('A1', 'Д5', f'{dv:.0f} В выше таблицы категории {cat}',
                                           Severity.NOT_CHECKED, layer=layer, pos=pos, nets=(a, b),
                                           data_quality='нет данных'))
                        continue
                    req = max(dreq, s_min)
                    if d >= req:
                        continue
                    where = {'B1': 'внутренний слой', 'B2': 'открытая медь', 'B3': 'открытая медь выше 3050 м',
                             'B4': 'под маской'}.get(cat, cat)
                    out.append(Finding(
                        'A1', 'Д5', f'зазор {d:.3f} мм мал для {dv:.0f} В ({where}, категория {cat}: нужно {req:.3f} мм)',
                        Severity.ERROR, layer=layer, pos=pos, nets=(a, b),
                        objects=_objects(lc.features_near(Point(pa), 1e-4) + lc.features_near(Point(pb), 1e-4)),
                        measured={'зазор_мм': d, 'разность_В': dv, 'категория': cat},
                        thresholds={'цел': req}, margin=margin(d, req), data_quality=ctx.profile.quality('nets'),
                        assumptions=['напряжения — из профиля (сеть — пик √2·U·(1 + κ), с переменным — худшая '
                                     'полярность); таблица — с округлением вверх до строки']
                        + (['слоя маски нет — медь считается открытой'] if no_mask else []),
                        fix=f'раздвинуть до {req:.3f} мм, закрыть маской, увести на внутренний слой или покрыть лаком'))
    if unknown:
        out.append(Finding('A1', 'Д5', f'для {len(unknown)} цепей рядом с цепями под напряжением напряжение не задано '
                                       f'— их зазоры не проверены', Severity.NOT_CHECKED,
                           nets=tuple(sorted(unknown))[:20], data_quality='нет данных'))
    return out


# ---------- Д7 ----------

def current_width_mm(current_a: float, rise_c: float, thickness_um: float, outer: bool) -> float:
    """Ширина по току (IPC-2221, документ A1, формула 21): A = (I/(k·ΔT^0,44))^(1/0,725) мил²."""
    area_mil2 = (current_a / (IPC2221_K[outer] * rise_c ** 0.44)) ** (1 / 0.725)
    return area_mil2 * MIL2_MM2 / (thickness_um / 1000)


def d7_current(ctx) -> list[Finding]:
    nets = ctx.profile.get('nets', {})
    currents = nets.get('currents', {})
    if not currents:
        return [Finding('A1', 'Д7', 'трассы по току не проверены: в профиле нет токов цепей', Severity.NOT_CHECKED,
                        data_quality='нет данных', fix='задать [nets] currents в профиле')]
    rise = nets.get('temperature_rise_c', 10.0)
    out = []
    for layer in ctx.board.copper:
        th = ctx.thresholds[layer]
        groups = defaultdict(list)
        for f in ctx.board.layers[layer].features:
            if f.role is Role.TRACK and isinstance(f.shape, Stroke) and f.net in currents:
                groups[(f.net, round(f.shape.width, 6))].append(f)
        for (net, w), feats in sorted(groups.items()):
            need = current_width_mm(currents[net], rise, th.copper_um, th.outer)
            if w >= need:
                continue
            p = feats[0].shape.pts[0]
            density = currents[net] / (w * th.copper_um / 1000)
            out.append(Finding(
                'A1', 'Д7', f'трасса {w:.3f} мм тонка для {currents[net]:g} А (нужно {need:.3f} мм, '
                            f'{density:.0f} А/мм²)', Severity.ERROR, layer=layer, pos=(float(p[0]), float(p[1])),
                nets=(net,), objects=_objects(feats), measured={'ширина_мм': w, 'ток_А': currents[net],
                                                                'плотность_А_мм2': density},
                thresholds={'цел': need}, margin=margin(w, need), data_quality=ctx.profile.quality('nets'),
                assumptions=[f'IPC-2221 (графики заменены IPC-2152 — консервативная иллюстрация); перегрев {rise:g} °C; '
                             f'медь {th.copper_um:g} мкм; полный ток цепи в каждом куске'],
                fix='расширить, продублировать на другом слое (A₁ + A₂) или залить'))
    return out


# ---------- Д8 ----------

def _boundary_samples(islands, step: float):
    """Точки на границах островов не реже step: (координаты n×2, остров, длина участка у точки)."""
    rings, owner = [], []
    for k, isl in enumerate(islands):
        for r in shapely.get_rings(isl):
            rings.append(r)
            owner.append(k)
    if not rings:
        return np.zeros((0, 2)), np.zeros(0, np.int64), np.zeros(0)
    rings = shapely.segmentize(np.asarray(rings, dtype=object), step)
    pts, own, ln = [], [], []
    for r, k in zip(rings, owner):
        c = shapely.get_coordinates(r)            # кольцо замкнуто: последняя точка = первой
        seg = np.hypot(*np.diff(c, axis=0).T)
        pts.append(c[:-1])
        own.append(np.full(len(seg), k))
        ln.append(seg)
    return np.vstack(pts), np.concatenate(own), np.concatenate(ln)


def d8_defect_risk(ctx, top: int = 10) -> list[Finding]:
    """Длинные участки с зазором от минимума до двух минимумов: риск замыкания пылинкой ∝ L/s.

    s — точное расстояние между кусками меди (по плиткам); L — длина края куска, идущего не
    дальше 1,5·s от соседа (по точкам границы с шагом s_min/2 и KD-дереву)."""
    from scipy.spatial import cKDTree
    rows = []
    for layer in ctx.board.copper:
        lc = ctx.geo.layer(layer)
        s_min = ctx.thresholds[layer].clearance.target
        pairs = {k: v for k, v in lc.island_pairs_within(2 * s_min).items() if v[0] >= s_min}
        if not pairs:
            continue
        involved = sorted({i for p in pairs for i in p})
        pts, own, seg = _boundary_samples([lc.islands[i] for i in involved], s_min / 2)
        own = np.asarray(involved)[own]
        tree = cKDTree(pts)
        dist, idx = tree.query(pts, k=12, distance_upper_bound=3 * s_min)
        n = len(pts)
        found = np.isfinite(dist)
        nb_own = np.where(found, own[np.minimum(idx, n - 1)], -1)
        foreign = found & (nb_own != own[:, None]) & (nb_own >= 0)
        has = foreign.any(axis=1)
        first = np.argmax(foreign, axis=1)                 # ближайшая точка чужого куска
        rows_ = np.nonzero(has)[0]
        other = nb_own[rows_, first[rows_]]
        dfor = dist[rows_, first[rows_]]
        me = own[rows_]
        lo, hi = np.minimum(me, other), np.maximum(me, other)
        run = defaultdict(float)
        for r, a_, b_, dd in zip(rows_, lo, hi, dfor):
            key = (int(a_), int(b_))
            if key in pairs and dd <= 1.5 * pairs[key][0] + s_min / 2:
                run[key] += seg[r] / 2                    # обе стороны щели — делим пополам
        for key, length in run.items():
            s, (ax, ay), (bx, by) = pairs[key]
            if length < 10 * s:
                continue
            nets = tuple(sorted(x for x in lc.island_nets[key[0]] | lc.island_nets[key[1]] if x))
            rows.append((length / s, length, s, layer, ((ax + bx) / 2, (ay + by) / 2), nets, s_min))
    rows.sort(key=lambda r: -r[0])
    out = []
    for ratio, length, s, layer, pos, nets, s_min in rows[:top]:
        out.append(Finding(
            'A1', 'Д8', f'участок {length:.1f} мм с зазором {_um(s)} мкм (меньше двух минимумов): L/s = {ratio:.0f}',
            Severity.INFO, layer=layer, pos=pos, nets=nets,
            measured={'длина_мм': length, 'зазор_мм': s, 'L_s': ratio},
            thresholds={'цел': s_min}, data_quality='модель критической площади',
            assumptions=['риск ∝ x₀²L/s; x₀ и плотность дефектов неизвестны — только ранжирование; L — длина края, '
                         'идущего не дальше 1,5·s от соседа'],
            fix='где есть место — раздвинуть: зазор ×1,5 снижает риск в 1,5 раза'))
    return out


# ---------- Д9 ----------

def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def yield_sigma_um(y: dict, t_um: float, ef: float) -> float:
    """σ ширины после фотолитографии и травления (Коробков 2025, ф. (22)): σ² = σ_л² + 2σ_т², σ_т — край
    при травлении; "h/F" — допущение автора σ_т = h/F (весь подтрав, с. 72; h — стравливаемая медь)."""
    se = y.get('etch_edge_sigma_um', 'h/F')
    se = t_um / ef if se == 'h/F' else float(se)
    return math.sqrt(float(y['litho_sigma_um']) ** 2 + 2 * se ** 2)


def p_out_of_tolerance(size_um: float, k: float, shift_um: float, sigma_um: float) -> float:
    """P(размер сузится больше чем на k·size): размер ~ N(size + shift, σ²)."""
    return _phi((-k * size_um - shift_um) / sigma_um)


def _pct(p: float) -> str:
    return ('> 99,9' if p > 0.999 else f'{p * 100:.1f}' if p >= 1e-3 else f'{p * 100:.1g}').replace('.', ',')


def d9_yield(ctx) -> list[Finding]:
    from .profile import product_class
    from .thresholds import etched_copper_um
    prof = ctx.profile
    y = prof.get('yield', {})
    if not y.get('enabled', False):
        return []
    combine = y.get('combine', 'worst')
    if combine not in ('worst', 'product'):
        raise ValueError(f'yield.combine = {combine!r}: ожидается "worst" или "product"')
    cls = product_class(prof)
    kw, kg = y['k_width'][f'class{cls}'], y['k_gap'][f'class{cls}']
    bias = float(y.get('bias_um', 0.0))
    verdict = prof['verdict']
    out = []
    for layer in ctx.board.copper:
        t_um = etched_copper_um(ctx.board, layer, prof)
        sig = yield_sigma_um(y, t_um, prof['etch']['etch_factor'])
        widths = defaultdict(list)
        for f in ctx.board.layers[layer].features:
            if f.role is Role.TRACK and isinstance(f.shape, Stroke) and f.dark:
                widths[(f.net, round(f.shape.width, 6))].append(f)
        pw = [(p_out_of_tolerance(w * 1000, kw, bias, sig), w, feats) for (_, w), feats in widths.items()]
        lc = ctx.geo.layer(layer)
        # зазоры, у которых P хотя бы probability_info / 100 (остальные на результат не влияют): зазор
        # G ~ N(g − δ, σ²), P(G < (1 − k)g) = Φ((δ − k·g)/σ) ≥ p  ⇔  g ≤ (σ·z + δ)/k
        z = -statistics_ppf(verdict['probability_info'] / 100)
        lim = (sig * z + bias) / kg / 1000
        pg = []
        for (i, j), (dd, a, b) in lc.island_pairs_within(max(lim, 0.0)).items():
            na, nb = set(lc.island_nets[i]) - {None}, set(lc.island_nets[j]) - {None}
            if na & nb or (not na and not nb):            # одна цепь или графика — не зазор между проводниками
                continue
            pg.append((p_out_of_tolerance(dd * 1000, kg, -bias, sig), dd, a, b))
        if combine == 'worst':                            # метод автора: худшая трасса и худший зазор слоя
            terms = [max(pw, key=lambda r: r[0], default=None), max(pg, key=lambda r: r[0], default=None)]
            terms = [r for r in terms if r]
        else:
            terms = pw + pg
        log_ok = sum(math.log1p(-min(p, 1 - 1e-12)) for p, *_ in terms)
        p_bad = -math.expm1(log_ok)
        sev = probability_severity(p_bad, verdict)
        if sev is Severity.OK:
            continue
        ww = max(pw, key=lambda r: r[0], default=None)
        wg = max(pg, key=lambda r: r[0], default=None)
        parts = []
        if ww and ww[0] > 0:
            parts.append(f'трасса {_um(ww[1])} мкм — {_pct(ww[0])} %')
        if wg and wg[0] > 0:
            parts.append(f'зазор {_um(wg[1])} мкм — {_pct(wg[0])} %')
        worst = max([r for r in (ww, wg) if r], key=lambda r: r[0])
        pos = ((worst[2][0] + worst[3][0]) / 2, (worst[2][1] + worst[3][1]) / 2) if worst is wg else \
            tuple(float(v) for v in worst[2][0].shape.pts[0])
        out.append(Finding(
            'A1', 'Д9', f'вероятность выйти за допуск приёмки по ширине и зазору на слое — {_pct(p_bad)} %'
                        f' ({"; ".join(parts)})', sev, layer=layer, pos=pos,
            objects=_objects(worst[2]) if worst is ww else [], probability=p_bad,
            measured={'σ_ширины_мкм': sig, 'h_мкм': t_um, 'трасс_групп': len(pw), 'зазоров': len(pg),
                      'P_трасса_макс': ww[0] if ww else 0.0, 'P_зазор_макс': wg[0] if wg else 0.0,
                      'ширина_худшая_мм': ww[1] if ww else None, 'зазор_худший_мм': wg[1] if wg else None},
            thresholds={'k_ширина': kw, 'k_зазор': kg, 'δ_мкм': bias, 'класс': cls},
            data_quality=prof.quality('yield'),
            assumptions=[f'модель Коробкова 2025 (ф. (22)–(24)): σ = √(σ_л² + 2σ_т²) = {sig:.1f} мкм, σ_л = '
                         f'{y["litho_sigma_um"]} мкм, σ_т = {y.get("etch_edge_sigma_um", "h/F")}, h = {t_um:g} мкм '
                         '(стравливаемая медь); допуск IPC-A-600G п. 2.10.1.1/2.10.1.2',
                         'слой — по самой узкой трассе и самому узкому зазору (метод автора, с. 162–163)'
                         if combine == 'worst' else 'произведение по всем трассам и зазорам слоя — допущение '
                         'независимости (при общей составляющей разброса брак завышен; у автора — худшие w и g)'],
            fix='расширить самые узкие трассы и зазоры: при σ процесса нужен запас z·σ/k (P = 0,997 → 2,75σ/k)'))
    return out


def statistics_ppf(p: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


@rule('A1')
def check(ctx) -> list[Finding]:
    out = []
    for code, fn in (('Д1', d1_track_width), ('Д2', d2_thin_text), ('Д3', d3_zone_necks), ('Д4', d4_clearance),
                     ('Д5', d5_voltage), ('Д7', d7_current), ('Д8', d8_defect_risk), ('Д9', d9_yield)):
        out += guarded('A1', code, fn, ctx)
    return out
