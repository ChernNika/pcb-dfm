"""Класс B1 — кислотные ловушки и шипы (документ B1): гибрид «формула клина по эффективному углу +
поправка нейросети» (шаги 4.4–4.5) или одна формула клина.

Для каждой цепи на каждом медном слое — объединение её меди (LayerCopper.nets), контуры с медью
слева. Вершины контуров:
  * вогнутая, угол пустоты θ = 360° − α < θ_max и на масштабе ℓ тоже → кандидат в ловушки;
  * выпуклая, угол меди φ < φ_max и на масштабе тоже → кандидат в шипы.

Ловушка (Л1–Л6). Карман — часть пустоты у вершины, куда не помещается круг радиуса δ
(N_δ = Ū ∖ (Ū ∘ B_δ), пустота — окно минус медь своей цепи). По площади кармана — эффективный угол
θ_эфф из A_ид(θ) = δ²[ctg(θ/2) − (π − θ)/2]. Карманы ищутся и θ_эфф считается при δ = 50 мкм
(POCKET_DELTA_UM) при любом профиле: так размечены данные гибрида и таблица клина; δ профиля входит
в формулу клина и в профиль процесса гибрида (для идеального клина θ_эфф от δ не зависит). Сужение y = Δ / w (w — меньшая ширина меди у стенок кармана):
  * b1.method = "nn" (по умолчанию) — гибрид dfm/ml/infer.py: формула клина + поправка сети на
    дескрипторах, квантили y и P(y > τ) (numpy, без PyTorch);
  * "formula" — формула клина: таблица численной модели на сетке θ × Da × δ × u (dfm/ml/tables), т. е.
    с толщиной меди слоя и углами от 8°; "doc_table" — таблица документа (С/О/З, медь 35 мкм, от 15°).
Вердикт: P ≥ 0,5 (у формулы y ≥ τ) — ошибка, но только при b1.calibrated = true (профиль подтверждён
купоном с клиньями, раздел 3.7), иначе предупреждение; P ≥ p_warning (у формулы y ≥ y_warning·τ) —
предупреждение; иначе информация, если перетрав ≥ report_min; недотрав не сообщается. Пороги
предупреждения выбраны по требованию раздела 9.2 «полнота ≥ 0,95» на проверочных частях
(research/b1_nn/warning_threshold.md). Ожидаемый ущерб = P × тяжесть
(сигнал 10, цепь питания из nets.power_nets 30).
Ситуация определяется по объектам цепи у вершины: заливка — Л4, площадка — Л1 (круглая — Л6),
три и больше трассы из одной точки — Л5, ответвление с коротким продолжением — Л3, иначе Л2.

Шип (Л7). Хрупкий кончик резиста: x_f = (b₅₀ − 2u)/(2 tg(φ/2)), A_f = 2u·x_f + tg(φ/2)·x_f²,
сравнение с прямым углом A_f(90°). Шип, направленный на чужую цепь, при известных напряжениях —
Л8: напряжение начала короны V_кор = E_c·ρ·ln(4d/ρ)/2 (Мейсон), E_c = 3,0(1 + 0,301/√ρ_см) кВ/мм
(Пик), ρ — радиус острия после травления ≈ u.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry.polygon import orient

from ..board.model import Role
from ..board.shapes import Circle, Stroke
from ..geometry.copper import parts
from .engine import rule
from .findings import Finding, Severity
from .voltage import voltage_difference

# поиск карманов и θ_эфф — при тех же δ и ℓ, что в данных обучения (единственное место: dataset.B1_GEOM,
# признаки для обучения и анализа платы используют это значение); δ и ℓ профиля на ловушки не влияют
POCKET_GEOM = dict(delta_um=50.0, ell_um=50.0)
POCKET_DELTA_UM = POCKET_GEOM['delta_um']


class B1ProfileError(ValueError):
    """Не хватает данных профиля процесса B1 — карманы не оцениваются, остальное (шипы) проверяется."""

__all__ = ['check', 'trap_candidates', 'corner_area', 'theta_from_area', 'overetch', 'resist_tip_area', 'corona_onset_kv']


# ---------- формулы ----------

def corner_area(theta_rad: float, r: float) -> float:
    """Площадь кармана идеального клина угла θ: часть угла, куда не входит круг радиуса r."""
    return r * r * (1 / math.tan(theta_rad / 2) - (math.pi - theta_rad) / 2)


def theta_from_area(area: float, r: float) -> float:
    """Эффективный угол, °: угол идеального клина с той же площадью кармана (деление пополам)."""
    lo, hi = math.radians(0.5), math.radians(179.5)
    if area >= corner_area(lo, r):
        return 0.5
    if area <= 0:
        return 179.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if corner_area(mid, r) > area:
            lo = mid
        else:
            hi = mid
    return math.degrees((lo + hi) / 2)


def overetch(theta_deg: float, b1: dict) -> float:
    """Перетрав стенки кармана, мкм, для профиля процесса — по таблице численной модели; острее
    15° — как при 15° (таблица дальше не считалась), тупее 120° — как при 120°."""
    a = np.asarray(b1['angles_deg'], dtype=float)
    v = np.asarray(b1['overetch_um'][b1['process']], dtype=float)
    return float(np.interp(theta_deg, a, v))


def resist_tip_area(phi_deg: float, u_um: float, b50_um: float) -> float:
    """Площадь хрупкого кончика резиста над шипом, мкм² (документ B1, раздел 2.9)."""
    t = math.tan(math.radians(phi_deg) / 2)
    xf = (b50_um - 2 * u_um) / (2 * t)
    return 0.0 if xf <= 0 else 2 * u_um * xf + t * xf * xf


def corona_onset_kv(rho_mm: float, d_mm: float) -> float:
    """Напряжение начала короны у острия радиуса ρ на расстоянии d (Мейсон + Пик), кВ. Формула Мейсона
    верна при d ≫ ρ; при d ≤ ρ (острие почти касается чужой меди) — 0: корона при любом напряжении,
    а зазор и так нарушен (A1)."""
    if d_mm <= rho_mm:
        return 0.0
    ec = 3.0 * (1 + 0.301 / math.sqrt(rho_mm / 10))
    return ec * rho_mm * math.log(4 * d_mm / rho_mm) / 2


# ---------- геометрия контуров ----------

def _rings(geom):
    for p in parts(geom):
        p = orient(p, 1.0)                           # медь слева: внешний против часовой, дырки по часовой
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)[:-1]
            if len(c) >= 3:
                keep = np.r_[True, np.hypot(*np.diff(c, axis=0).T) > 1e-4]
                c = c[keep]
                if len(c) >= 3 and math.dist(c[0], c[-1]) <= 1e-4:
                    c = c[:-1]
                if len(c) >= 3:
                    yield c


def _angles(P):
    """Угол меди в каждой вершине, ° (медь слева от обхода)."""
    a = np.roll(P, 1, axis=0) - P
    b = np.roll(P, -1, axis=0) - P
    return (np.degrees(np.arctan2(a[:, 1], a[:, 0]) - np.arctan2(b[:, 1], b[:, 0]))) % 360


def _scale_angles(P, idx, ell):
    """Угол меди на масштабе ℓ: по хордам к точкам контура на расстоянии ℓ в обе стороны."""
    seg = np.hypot(*(np.roll(P, -1, axis=0) - P).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = cum[-1]

    def at(s):
        t = s % total
        j = np.clip(np.searchsorted(cum, t, side='right') - 1, 0, len(P) - 1)
        f = np.where(seg[j] > 0, (t - cum[j]) / np.where(seg[j] > 0, seg[j], 1), 0)
        return P[j] + f[:, None] * (np.roll(P, -1, axis=0)[j] - P[j])
    a = at(cum[idx] - ell) - P[idx]
    b = at(cum[idx] + ell) - P[idx]
    return (np.degrees(np.arctan2(a[:, 1], a[:, 0]) - np.arctan2(b[:, 1], b[:, 0]))) % 360, a, b


# ---------- класс ----------

NAMES = {'Л1': 'трасса подходит к площадке под острым углом', 'Л2': 'острый угол между трассами',
         'Л3': '«хвост» за точкой стыка', 'Л4': 'вход трассы в заливку под острым углом',
         'Л5': 'веер трасс из одной точки', 'Л6': 'трасса к круглой площадке', 'Л7': 'шип', 'Л8': 'шип на чужую цепь'}


def _situation(lc, geom_of, v, net):
    """Код ситуации Л1–Л6 по объектам цепи у вершины и ширина проводника у кармана, мм."""
    pt = shapely.points(*v)
    near = [f for f in lc.features_near(pt.buffer(0.02)) if f.net == net and f.dark]
    tracks = [f for f in near if f.role is Role.TRACK and isinstance(f.shape, Stroke)]
    widths = [f.shape.width for f in tracks]
    w = min(widths) if widths else None
    roles = {f.role for f in near}
    if Role.ZONE in roles and tracks:
        return 'Л4', w
    pads = [f for f in near if f.role in (Role.PAD, Role.VIA)]
    if pads and tracks:
        return ('Л6' if any(isinstance(f.shape, Circle) for f in pads) else 'Л1'), w
    if len(tracks) >= 2:
        ends = [np.asarray(f.shape.pts)[[0, -1]] for f in tracks]
        allends = np.vstack(ends)
        e = allends[np.argmin(np.hypot(*(allends - v).T))]     # точка стыка — ближайший конец трассы
        tol = max(w or 0.1, 0.05)
        n_at = sum(np.hypot(*(x - e).T).min() < tol / 2 for x in ends)
        at_pad = any(geom_of[id(f)].covers(shapely.points(*e)) for f in lc.features_near(shapely.points(*e))
                     if f.net == net and f.role in (Role.PAD, Role.VIA))
        if n_at >= 3 or (n_at >= 2 and at_pad):        # веер: трассы расходятся из одной точки (площадки)
            return 'Л5', w
        for f, x in zip(tracks, ends):                          # трасса, через которую проходит стык
            if np.hypot(*(x - e).T).min() >= tol / 2 and geom_of[id(f)].distance(shapely.points(*e)) < 1e-6:
                return ('Л3' if np.hypot(*(x - e).T).min() < 4 * tol else 'Л2'), w
        return 'Л2', w
    if w is None:
        sizes = [min(f.shape.radius * 2, 9) if isinstance(f.shape, Circle) else 0.3 for f in near]
        w = min(sizes) if sizes else 0.3
    return 'Л2', w


def _pocket(net_geom, v, delta, theta_deg, R=0.6):
    """Карман вершины → (площадь кармана у вершины мм², глубина мм, щель: ширина мм или None) или None.

    Карман — компонента N_δ у вершины. Если узкая пустота тянется дальше, чем карман клина угла θ
    (щель между двумя кусками меди цепи уже 2δ, например трасса вдоль края площадки), формула клина
    к ней неприменима: площадь щели дала бы ложно острый эффективный угол. Щель — если компонента
    больше чем в 1,5 раза превосходит свою часть в круге 1,5·H_ид(θ) + δ около вершины (H_ид =
    δ/tg(θ/2) — глубина кармана клина); тогда клин не оценивается, щель сообщается отдельно."""
    win = shapely.box(v[0] - R, v[1] - R, v[0] + R, v[1] + R)
    gap = win.difference(net_geom)
    opened = gap.buffer(-delta, quad_segs=24).buffer(delta, quad_segs=24)
    # открытие по многоугольникам оставляет вдоль скруглённых краёв полоски тоньше микрометра; они
    # цепляются к карману и раздувают площадь (так в документе вышло 13 337 мкм² у Л1 при идеальных
    # 12 250) — срезаем всё тоньше 2 мкм
    narrow = gap.difference(opened).buffer(-1e-3).buffer(1e-3)
    pt = shapely.points(*v)
    comps = [g for g in parts(narrow) if g.distance(pt) < 0.02]
    if not comps:
        return None
    pk = max(comps, key=lambda g: g.area)
    h_id = delta / math.tan(math.radians(max(theta_deg, 1.0)) / 2)
    near = pk.intersection(pt.buffer(1.5 * h_id + delta, quad_segs=24))
    near = max(parts(near), key=lambda g: g.area) if not near.is_empty else pk
    c = np.asarray(near.exterior.coords)
    slot = None
    # щель: пустота тянется дальше кармана клина или у вершины её больше, чем у клина того же угла
    # (у клина с прямыми или выпуклыми стенками площадь кармана не больше идеальной)
    if pk.area > 1.5 * near.area or near.area > 1.5 * corner_area(math.radians(min(theta_deg, 179)), delta):
        slot = 2 * shapely.maximum_inscribed_circle(pk, tolerance=1e-4).length
    return near.area, float(np.hypot(*(c - v).T).max()), slot


def trap_candidates(net_geom, b1):
    """Кандидаты в ловушки одной цепи на слое: вогнутые вершины с углом пустоты < θ_max (и на масштабе ℓ)
    и их карманы. Общий перебор для проверки и для генератора данных нейросети (dfm/ml/dataset.py).
    Возвращает словари: v (мм), theta, theta_l (°), area (мм²), depth (мм), slot (мм или None),
    beta (°) — направление биссектрисы кармана в пустоту по хордам масштаба ℓ."""
    delta, ell = b1['delta_um'] / 1000, b1['ell_um'] / 1000
    for P in _rings(net_geom):
        ang = _angles(P)
        idx = np.flatnonzero((ang > 180) & (360 - ang < b1['theta_max_deg'] - 0.5))
        if not len(idx):
            continue
        sc, va, vb = _scale_angles(P, idx, ell)
        for k, i in enumerate(idx):
            th_v, th_l = 360 - ang[i], 360 - sc[k]
            if th_l >= b1['theta_max_deg'] - 0.5:
                continue
            pk = _pocket(net_geom, P[i], delta, min(th_v, th_l))   # клин сравнения — по более острому углу
            if pk is None:
                continue
            area, depth, slot = pk
            if slot is None and area < 0.02 * corner_area(math.radians(min(th_l, 179)), delta):
                continue
            bis = va[k] / (np.hypot(*va[k]) or 1) + vb[k] / (np.hypot(*vb[k]) or 1)
            yield dict(v=P[i].copy(), theta=float(th_v), theta_l=float(th_l), area=float(area), depth=float(depth),
                       slot=slot, beta=float(np.degrees(np.arctan2(bis[1], bis[0]))))


@rule('B1')
def check(ctx) -> list[Finding]:
    from .thresholds import etched_copper_um
    b, prof = ctx.board, ctx.profile
    b1 = prof['b1']
    geom = dict(b1, **POCKET_GEOM)
    ell = b1['ell_um'] / 1000
    quality = prof.quality('b1')
    volts = prof.get('nets', {}).get('voltages', {})
    out = []
    for layer in b.copper:
        lc = ctx.geo.layer(layer)
        geom_of = {id(f): g for f, g in zip(lc.features, lc.feature_geoms)}
        t_cu = etched_copper_um(b, layer, prof)                    # стравливаемая медь — как в A1
        u = t_cu / prof['etch']['etch_factor']                     # подтрав плоской стенки, мкм
        a90 = resist_tip_area(90, u, b1['resist_b50_um'])
        traps = []
        for net, g in lc.nets.items():
            if net is None:                                        # графика и текст без цепи — не проводники
                continue
            for c in trap_candidates(g, geom):
                v, th_v = c['v'], c['theta']
                if c['slot'] is not None:                         # щель, а не клин: формула клина не годится
                    out.append(_slot_finding(lc, layer, v, net, th_v, c['slot'], geom, quality))
                    continue
                traps.append((net, c))                            # оценка — партией по слою, ниже
            for P in _rings(g):
                ang = _angles(P)
                idx = np.flatnonzero(ang < b1['phi_max_deg'] - 0.5)
                if not len(idx):
                    continue
                sc, va, vb = _scale_angles(P, idx, ell)
                for k, i in enumerate(idx):
                    v = P[i]
                    phi, phi_l = ang[i], sc[k]
                    if phi_l >= b1['phi_max_deg'] - 0.5:
                        continue
                    af = resist_tip_area(phi, u, b1['resist_b50_um'])
                    ratio = af / a90 if a90 else 0.0
                    f8 = _spike_hv(ctx, lc, layer, v, va[k], vb[k], net, volts, u)
                    if f8 is not None:
                        out.append(f8)
                        continue
                    if ratio < b1['spike_ratio_info']:
                        continue
                    sev = Severity.WARNING if ratio >= b1['spike_ratio_warning'] else Severity.INFO
                    out.append(Finding(
                        'B1', 'Л7', f'шип {phi:.0f}°: хрупкий кончик резиста {af:.0f} мкм² — в {ratio:.1f} раза '
                                    f'больше, чем у прямого угла; острие отступит на ≈ '
                                    f'{u / math.sin(math.radians(phi) / 2):.0f} мкм', sev,
                        layer=layer, pos=(float(v[0]), float(v[1])), nets=(net,), objects=_objs(lc, v, net),
                        measured={'φ_°': phi, 'φ_ℓ_°': phi_l, 'A_f_мкм2': af, 'A_f_отн': ratio, 'u_мкм': u},
                        thresholds={'отн_информация': b1['spike_ratio_info'],
                                    'отн_предупреждение': b1['spike_ratio_warning']},
                        data_quality=quality,
                        assumptions=['пороги для шипа условные: абсолютная вероятность отрыва кончика резиста '
                                     'требует купона с шипами (документ B1, раздел 2.9)'],
                        fix=f'скруглить острие радиусом r ≥ b₅₀/2 − u = {max(b1["resist_b50_um"] / 2 - u, 0):.0f} мкм '
                            f'(документ B1, Л7; с запасом — 0,12 мм) или срезать фаской'))
        if traps:
            try:
                out += _trap_findings(lc, geom_of, layer, traps, prof,
                                      dict(t=t_cu, ef=prof['etch']['etch_factor'], u=u), quality)
            except B1ProfileError as e:                       # нет данных профиля процесса — шипы проверены
                out.append(Finding('B1', 'Л1', f'карманы слоя не оценены: {e}', Severity.NOT_CHECKED, layer=layer,
                                   data_quality=quality, fix='дополнить [b1] профиля'))
    return out


TRAP_FIX = ('заполнить угол (каплевидный переход, скругление стыка), подвести трассу под углом '
            'не острее 45° или изменить порядок стыка (документ B1, раздел 10.4)')


METHODS = ('nn', 'formula', 'doc_table')
# nn — гибрид: формула клина по θ_эфф + поправка сети (шаг 4.4), квантили сужения и P(y > τ);
# formula — формула клина по θ_эфф: таблица численной модели на сетке θ 8–90° × Da × δ × u (dfm/ml/tables),
#   т. е. с толщиной меди слоя; doc_table — таблица документа (профили С/О/З, u = 12 мкм, 15–120°).


def process_params(b1):
    """Параметры профиля процесса: Da и u_r (мкм) — из b1.process_params[b1.process]."""
    proc = b1['process']
    pp = b1.get('process_params', {}).get(proc)
    if pp is None:
        raise B1ProfileError(f'нет параметров профиля процесса {proc!r} в [b1.process_params]')
    return float(pp['Da']), float(pp['u_r_um'])


def _method(b1):
    m = b1.get('method', 'nn')
    if m not in METHODS:
        raise B1ProfileError(f'b1.method = {m!r}: ожидается одно из {METHODS}')
    notes = []
    if m == 'doc_table' and b1['process'] not in b1.get('overetch_um', {}):
        m = 'formula'
        notes.append(f'таблица документа есть только для профилей {", ".join(b1.get("overetch_um", {}))} — '
                     'формула клина')
    if m == 'nn':
        from ..ml import infer
        if not infer.available():
            m = 'formula'
            notes.append('гибрид недоступен (нет весов dfm/ml/weights/b1_hybrid.npz или они обучены с другой '
                         'таблицей клина) — формула клина')
    if m == 'formula':
        from ..ml.baselines import TABLE_NPZ
        if not TABLE_NPZ.exists():
            if b1['process'] not in b1.get('overetch_um', {}):
                raise B1ProfileError(f'нет таблицы клина dfm/ml/tables/b1_wedge.npz, а таблица документа для профиля '
                                     f'{b1["process"]!r} не задана')
            m = 'doc_table'
            notes.append('нет таблицы клина dfm/ml/tables/b1_wedge.npz — таблица документа')
    return m, notes


def _severity(b1, y, p=None):
    """Вердикт по вероятности превысить допуск, если она есть (гибрид), иначе по сужению (формула).
    Модель B1 — гипотеза до купона с клиньями (раздел 3.7), поэтому без b1.calibrated = true вердикт не
    выше предупреждения. Пороги предупреждения p_warning и y_warning·τ — наименьшая строгость, при которой
    на проверочных частях полнота ≥ 0,95 (раздел 9.2) и при слабой, и при сильной промывке
    (research/b1_nn/warning_threshold.md)."""
    top = Severity.ERROR if b1.get('calibrated', False) else Severity.WARNING
    if p is not None:
        return top if p >= 0.5 else Severity.WARNING if p >= b1.get('p_warning', 0.05) else Severity.INFO
    tau = b1['tau']
    return top if y >= tau else Severity.WARNING if y >= b1.get('y_warning', 0.75) * tau else Severity.INFO


def tau_of(prof) -> float:
    """Допуск сужения τ: число из b1.tau или "class" — допуск сужения проводника IPC-A-600G п. 2.10.1.1
    для класса изделия ([yield].k_width: 0,2 для классов 2–3, 0,3 для класса 1)."""
    from .profile import product_class
    t = prof['b1'].get('tau', 'class')
    if t == 'class':
        cls = product_class(prof)
        return float(prof.get('yield', {}).get('k_width', {}).get(f'class{cls}', 0.3 if cls == 1 else 0.2))
    try:
        v = float(str(t).replace(',', '.'))
    except ValueError:
        v = math.nan
    if not 0 < v < 1:
        raise B1ProfileError(f'b1.tau = {t!r}: ожидается "class" или доля от 0 до 1')
    return v


def _trap_findings(lc, geom_of, layer, traps, prof, etch, quality):
    """Ловушки слоя: дескрипторы кармана (ширина меди у стенок — как в данных обучения), сужение
    y = Δ / w_мин гибридом или формулой клина, вердикт и ожидаемый ущерб (раздел 6.2)."""
    from ..ml import infer
    b1 = dict(prof['b1'], tau=tau_of(prof))
    proc, tau, delta_um = b1['process'], b1['tau'], b1['delta_um']
    Da, u_r = process_params(b1)
    method, notes = _method(b1)
    params = {'Da': Da, 'δ_мкм': delta_um, 't_мкм': etch['t'], 'EF': etch['ef'], 'u_r_мкм': u_r}
    power = set(prof.get('nets', {}).get('power_nets', []))
    w_conseq = b1.get('consequence', {'сигнал': 10, 'питание': 30})
    rows = []
    for net, c in traps:
        own, foreign = infer.window(lc, net, c['v'])
        d, w_min = infer.descriptors(own, foreign, c, delta_um)
        if d is not None:
            rows.append((net, c, d, w_min))
    if not rows:
        return []
    D = np.array([r[2] for r in rows], float)
    P = np.tile([math.log(Da), delta_um, etch['t'], etch['ef'], etch['u']], (len(rows), 1))
    if method == 'nn':
        q, p, f = infer.model().predict(D, P, u_r, tau)
        src = 'гибрид'
    else:
        if method == 'formula':
            f = infer.formula_y(D, P, np.full(len(D), u_r))
            src = 'формула клина'
        else:
            f = np.array([overetch(r[2][3] * 90, b1) for r in rows]) / (D[:, 6] * 1000)
            src = 'таблица документа'
        q, p = np.c_[f, f, f], [None] * len(rows)
    extra = infer.out_of_range(params) if method != 'doc_table' else []
    out = []
    for (net, c, d, w_min), qk, pk, fk in zip(rows, q, p, f):
        y10, y50, y90 = (float(x) for x in qk)
        dlt = y50 * w_min * 1000           # перетрав, мкм: y·w_мин — нижняя оценка, если максимум y у более широкой стенки
        if pk is None and dlt < b1['report_min_um']:
            continue
        if pk is not None and pk < 0.01 and dlt < b1['report_min_um']:
            continue
        sev = _severity(b1, y50, pk)
        code, w_sit = _situation(lc, geom_of, c['v'], net)
        th_v, th_e = c['theta'], d[3] * 90
        weight = w_conseq['питание' if net in power else 'сигнал']
        v = c['v']
        if pk is not None:
            title = (f'{NAMES[code]}: карман {th_v:.0f}° (эффективный {th_e:.0f}°) — сужение проводника '
                     f'{y50 * 100:.0f} % (80 %: {max(y10, 0) * 100:.0f}–{y90 * 100:.0f} %), '
                     f'вероятность превысить допуск {tau * 100:.0f} % — {pk * 100:.0f} %')
        else:
            title = (f'{NAMES[code]}: карман {th_v:.0f}° (эффективный {th_e:.0f}°) — перетрав ≈ {dlt:.0f} мкм, '
                     f'сужение проводника {y50 * 100:.0f} % (допуск {tau * 100:.0f} %)')
        meas = {'θ_°': th_v, 'θ_ℓ_°': c['theta_l'], 'θ_эфф_°': th_e, 'площадь_кармана_мкм2': c['area'] * 1e6,
                'глубина_мкм': c['depth'] * 1000, 'перетрав_мкм': dlt, 'ширина_мм': w_min, 'сужение': y50,
                'сужение_формула': float(fk), 'профиль': proc, 'источник': src}
        if pk is not None:
            pk = float(f'{pk:.3g}') if pk >= 1e-4 else 0.0               # P < 10⁻⁴ — за пределом точности модели
            meas.update({'сужение_q10': y10, 'сужение_q90': y90, 'P_сверх_допуска': pk,
                         'тяжесть': weight, 'ожидаемый_ущерб': pk * weight})
        if w_sit:
            meas['ширина_трассы_мм'] = w_sit
        how = {'гибрид': 'гибрид: формула клина по θ_эфф + поправка сети (5 моделей перекрёстной проверки, '
                         'суррогат численной модели травления); интервал 80 % калиброван (CQR)',
               'формула клина': 'формула клина по эффективному углу (таблица численной модели для Da, δ, u, u_r)',
               'таблица документа': f'таблица документа B1 (раздел 3.6, профиль {proc}, рассчитана для меди 35 мкм)'}[src]
        assumptions = [f'{how}; профиль {proc}: Da = {Da:g}, u_r = {u_r:g} мкм, медь {etch["t"]:g} мкм, '
                       f'EF {etch["ef"]:g}', *notes, *extra]
        if th_e < 8 and method != 'doc_table':
            assumptions.append(f'θ_эфф {th_e:.0f}° острее сетки таблицы (8°) — оценка занижена')
        if not b1.get('calibrated', False):
            assumptions.append('модель B1 не откалибрована купоном с клиньями (документ B1, раздел 3.7) — '
                               'вердикт не выше предупреждения')
        out.append(Finding(
            'B1', code, title, sev, layer=layer, pos=(float(v[0]), float(v[1])), nets=(net,),
            objects=_objs(lc, v, net), probability=pk, measured=meas,
            thresholds={'τ': tau, 'δ_мкм': delta_um, 'Da': Da, 'u_r_мкм': u_r}, data_quality=quality,
            assumptions=assumptions, fix=TRAP_FIX))
    return out


def _slot_finding(lc, layer, v, net, th_v, slot, b1, quality):
    """Узкая щель внутри цепи у вершины: формула клина её не оценивает — информация."""
    return Finding(
        'B1', 'Л2', f'у вершины {th_v:.0f}° пустота продолжается узкой щелью ≈ {slot * 1000:.0f} мкм внутри цепи (уже '
                    f'2δ = {2 * b1["delta_um"]:.0f} мкм): раствор в ней обновляется хуже, формула клина её не оценивает',
        Severity.INFO, layer=layer, pos=(float(v[0]), float(v[1])), nets=(net,), objects=_objs(lc, v, net),
        measured={'θ_°': th_v, 'щель_мкм': slot * 1000}, thresholds={'2δ_мкм': 2 * b1['delta_um']}, data_quality=quality,
        assumptions=['узкий канал (документ B1, раздел 2.5): количественно — численной моделью или нейросетью (глава 8)'],
        fix='раздвинуть медь шире 2δ или заполнить щель медью той же цепи')


def _objs(lc, v, net):
    out = []
    for f in lc.features_near(shapely.points(*v).buffer(0.02)):
        if f.net == net and f.uid:
            out.append({'kind': f.role.value, 'uid': f.uid, 'ref': f.component, 'pin': f.pin})
    return out[:6]


def _spike_hv(ctx, lc, layer, v, va, vb, net, volts, u_um):
    """Л8: шип, направленный на медь другой цепи, при известной разности напряжений."""
    if not volts or net not in volts:
        return None
    ua, ub = va / (np.hypot(*va) or 1), vb / (np.hypot(*vb) or 1)
    out_dir = -(ua + ub)
    n = np.hypot(*out_dir)
    if n < 1e-9:
        return None
    out_dir = out_dir / n
    best = None
    for other, g in lc.nets.items():
        if other in (net, None) or other not in volts:
            continue
        d = g.distance(shapely.points(*v))
        if d > 3.0:
            continue
        q = np.asarray(shapely.shortest_line(shapely.points(*v), g).coords[-1]) - v
        if np.hypot(*q) > 0 and np.dot(q / np.hypot(*q), out_dir) < 0.7:      # не в сторону острия
            continue
        dv = voltage_difference(volts[net], volts[other])         # как в A1 Д5: фазы и полярность пика
        if best is None or dv / max(d, 1e-3) > best[0]:
            best = (dv / max(d, 1e-3), other, d, dv)
    if best is None:
        return None
    _, other, d, dv = best
    rho = max(u_um / 1000, 0.005)                                  # радиус острия после травления ≈ u
    vk = corona_onset_kv(rho, d) * 1000
    ks = ctx.profile['b1'].get('corona_margin', 1.5)            # запас k_s (документ B1, Л8)
    if vk >= ks * dv:
        return None
    sev = Severity.ERROR if vk < dv else Severity.WARNING          # корона уже при рабочем напряжении / мал запас
    r_need = _corona_radius(ks * dv / 1000, d)
    return Finding(
        'B1', 'Л8', f'шип направлен на {other} в {d:.2f} мм: разность {dv:.0f} В при напряжении начала короны '
                    f'≈ {vk:.0f} В', sev, layer=layer, pos=(float(v[0]), float(v[1])), nets=(net, other),
        objects=_objs(lc, v, net), measured={'d_мм': d, 'ΔV_В': dv, 'V_кор_В': vk, 'ρ_мм': rho},
        thresholds={'k_s': ks, 'V_кор_нужно_В': ks * dv},
        data_quality=ctx.profile.quality('b1'),
        assumptions=['формулы Мейсона и Пика — оценки для воздуха; под маской пороги другие; зазор по напряжению '
                     'проверяет A1 Д5'],
        fix=(f'скруглить острие радиусом не меньше {r_need:.2f} мм (V_кор ≥ k_s·ΔV, документ B1, Л8)' if r_need
             else 'скругления не хватит — отодвинуть острие от цепи или закрыть покрытием') +
            '; требование к самому зазору — A1 Д5')


def _corona_radius(v_kv, d_mm):
    """Наименьший радиус острия, мм, при котором V_кор(ρ, d) ≥ v_kv (V_кор растёт с ρ до ρ ≈ d) — деление
    пополам (документ B1, Л8); None, если не хватает и ρ = d/2."""
    lo, hi = 1e-3, d_mm / 2
    if corona_onset_kv(hi, d_mm) < v_kv:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if corona_onset_kv(mid, d_mm) < v_kv else (lo, mid)
    return hi
