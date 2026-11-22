"""Численная модель травления у острых углов (документ B1, глава 3) — учитель для нейросети B1.

Перенесено из research/b1/physics.py и research/b1/demo_sim.py без изменения физики. Единицы — мкм.
Плёночная модель: дальше δ от меди раствор перемешан (c = 1), в плёнке — диффузия (уравнение
Лапласа), на меди — реакция первого порядка (Робен), Da = kδ/D. Фронт меди сдвигается со
скоростью k·c (метод уровня, 48 шагов до подтрава плоской стенки u). Промывка — время пребывания
T (уравнение Пуассона в плёнке, нормировано на плоскую стенку). Смещение стенки кармана:

    Δ(x) = (s_тр(x) − u) + u_r·(T(x) − 1)

Оно линейно по u_r, поэтому один расчёт (геометрия, Da, δ, u) даёт метки для любого u_r:
`simulate` возвращает вдоль обеих стенок кармана a(s) = s_тр − u, b(s) = T − 1 и местную ширину
меди w(s); метка y(u_r) = max_s (a + u_r·b)/w (документ B1, разделы 7.1 и 7.3).
Для обучения T на стенках ограничено пределом клина того же угла (`t_cap`).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import shapely
from scipy import ndimage
from scipy.ndimage import map_coordinates
from shapely.geometry.polygon import orient

from ..geometry.copper import parts

__all__ = ['simulate', 'labels_for', 'flat_speed', 't_cap', 'T_CAP_K']

NSTEPS = 48


def _lap_system(film, bulk, copper, coef_cu, rhs_const=0.0, dirichlet_bulk=1.0):
    idx = -np.ones(film.shape, dtype=np.int64)
    ys, xs = np.nonzero(film)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    H, W = film.shape
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    b = np.full(n, rhs_const, dtype=float)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nyc, nxc = np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)
        is_film = inside & film[nyc, nxc]
        is_bulk = inside & bulk[nyc, nxc]
        is_cu = inside & copper[nyc, nxc]
        i_f = np.nonzero(is_film)[0]
        rows.append(i_f)
        cols.append(idx[nyc[i_f], nxc[i_f]])
        vals.append(np.ones(len(i_f)))
        diag[i_f] -= 1
        i_b = np.nonzero(is_bulk)[0]
        diag[i_b] -= 1
        b[i_b] -= dirichlet_bulk
        i_c = np.nonzero(is_cu)[0]
        diag[i_c] -= coef_cu
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A, b, ys, xs


def _bulk(copper, h, delta, frame_bulk):
    gap = ~copper
    dist = ndimage.distance_transform_edt(gap) * h
    bulk = gap & (dist > delta)
    if frame_bulk:                               # края окна — объём раствора
        f = np.zeros(gap.shape, bool)
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = True
        bulk |= gap & f
    lab, n = ndimage.label(gap)                  # замкнутые полости без связи с объёмом — объём
    if n:
        has = ndimage.maximum(bulk, lab, index=np.arange(1, n + 1))
        iso = np.nonzero(has == 0)[0] + 1
        if len(iso):
            bulk = bulk | np.isin(lab, iso)
    return gap, bulk, gap & ~bulk


def concentration(copper, h, delta, Da, frame_bulk=True):
    _, bulk, film = _bulk(copper, h, delta, frame_bulk)
    A, b, ys, xs = _lap_system(film, bulk, copper, Da / delta * h, 0.0, 1.0)
    c = np.zeros(copper.shape)
    c[bulk] = 1.0
    if len(ys):
        c[ys, xs] = spla.spsolve(A.tocsc(), b)
    return c


def residence(copper, h, delta, frame_bulk=True):
    _, bulk, film = _bulk(copper, h, delta, frame_bulk)
    A, b, ys, xs = _lap_system(film, bulk, copper, 0.0, -h * h, 0.0)
    T = np.zeros(copper.shape)
    if len(ys):
        T[ys, xs] = spla.spsolve(A.tocsc(), b)
    return T * 2 / delta ** 2


def _run_etch(phi0, h, delta, Da, dt, nsteps=NSTEPS, frame_bulk=True):
    k = Da / delta
    phi = phi0.copy()
    disp = np.zeros_like(phi0)
    for _ in range(nsteps):
        copper = phi > 0
        c = concentration(copper, h, delta, Da, frame_bulk)
        cu_nb = ndimage.binary_dilation(copper) & ~copper
        v_b = np.where(cu_nb, k * c, 0.0)
        _, inds = ndimage.distance_transform_edt(~cu_nb, return_indices=True)
        v_ext = v_b[inds[0], inds[1]]
        phi = phi - v_ext * dt
        disp = disp + v_ext * dt
    return phi, disp


_FLAT = {}


def flat_speed(Da, delta, u, h):
    """Скорость отступа плоской стенки на сетке (калибровка: за NSTEPS шагов стенка уходит на u)."""
    key = (round(Da, 6), round(delta, 3), round(u, 3), h)
    if key not in _FLAT:
        xs = np.arange(-160, 160 + h / 2, h)
        Xf, Yf = np.meshgrid(xs, xs)
        v0 = (Da / delta) / (1 + Da)
        _, dsp = _run_etch(-Yf.astype(float), h, delta, Da, (u / NSTEPS) / v0, frame_bulk=False)
        meas = _sample(dsp, Xf, Yf, 0.0, -6.0, h)
        _FLAT[key] = v0 * meas / u
    return _FLAT[key]


def _sample(field, X, Y, x, y, h):
    return float(map_coordinates(field, [[(y - Y[0, 0]) / h], [(x - X[0, 0]) / h]], order=1)[0])


# ---------- окно и стенки кармана ----------

def _window(allcu, v, Rw, h):
    xs = np.arange(-Rw, Rw + h / 2, h)
    X, Y = np.meshgrid(xs, xs)
    pts = shapely.points((v[0] * 1000 + X).ravel() / 1000, (v[1] * 1000 + Y).ravel() / 1000)
    shapely.prepare(allcu)
    inside = shapely.contains(allcu, pts).reshape(X.shape)
    dist = shapely.distance(allcu.boundary, pts).reshape(X.shape) * 1000
    return X, Y, np.where(inside, dist, -dist)


def _ring_through(own, v):
    best = None
    for p in parts(own):
        p = orient(p, 1.0)
        for ring in [p.exterior, *p.interiors]:
            P = np.asarray(ring.coords)[:-1]
            d = np.hypot(*(P - v).T)
            i = int(d.argmin())
            if best is None or d[i] < best[0]:
                best = (d[i], P, i)
    return best[1], best[2]


def _arc_point(P, cum, total, i, s):
    t = (cum[i] + s) % total
    j = min(int(np.searchsorted(cum, t, side='right') - 1), len(P) - 1)
    L = cum[j + 1] - cum[j]
    f = (t - cum[j]) / L if L > 0 else 0
    return P[j] + f * (P[(j + 1) % len(P)] - P[j])


def _walls(own, v, Lw):
    """Точки обеих стенок кармана от вершины вдоль контура до длины Lw (мм) с нормалью в пустоту."""
    P, i = _ring_through(own, v)
    seg = np.hypot(*(np.roll(P, -1, axis=0) - P).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = cum[-1]
    out = []
    for side in (-1, 1):
        for s in np.arange(0.004, Lw, 0.002):
            p = _arc_point(P, cum, total, i, side * s)
            t = _arc_point(P, cum, total, i, side * s + 0.001) - _arc_point(P, cum, total, i, side * s - 0.001)
            t = t / (np.hypot(*t) or 1)
            out.append((side, s, p, np.array([t[1], -t[0]])))      # медь слева → пустота справа
    return out


def _local_width(allcu, p, n_gap):
    ray = shapely.LineString([p - n_gap * 1e-4, p - n_gap * 1.5])
    inter = allcu.intersection(ray)
    if inter.is_empty:
        return np.nan
    segs = [g for g in shapely.get_parts(inter) if g.geom_type == 'LineString']
    if not segs:
        return np.nan
    first = min(segs, key=lambda s: shapely.points(*p).distance(s))
    return first.length + 1e-4


def wall_length_mm(theta_deg, delta_um, lw_max=340.0):
    """Длина стенок кармана от вершины, на которой ищется наибольший перетрав, мм: 1,3 глубины кармана клина
    δ/sin(θ/2), но не больше lw_max мкм (как в research/b1/demo_sim.py; дальше стенка травится как плоская).
    Общая для учителя (simulate) и проверки (infer.wall_widths) — иначе дескрипторы разойдутся с обучением."""
    th = math.radians(max(theta_deg, 5.0))
    return min(1.3 * delta_um / math.sin(th / 2), lw_max) / 1000


def simulate(own, allcu, v, theta_deg, Da, delta, u, h=2.0, rw_max=2000.0, lw_max=340.0):
    """Расчёт одного кармана: вершина v (мм), медь своей цепи own и вся медь окна allcu (мм),
    угол пустоты θ, профиль (Da, δ мкм, u мкм). Возвращает стенки: s (мм, со знаком стороны),
    a = s_тр − u, b = T − 1 (мкм и доли), w (мм) — и время расчёта."""
    import time
    t0 = time.perf_counter()
    th = math.radians(max(theta_deg, 5.0))
    Rw = min(max(360.0, 1.25 * delta / math.sin(th / 2) + 100), rw_max)
    X, Y, phi0 = _window(allcu, v, Rw, h)
    Lw = wall_length_mm(theta_deg, delta, lw_max)
    walls = _walls(own, np.asarray(v, float), Lw)
    T = residence(phi0 > 0, h, delta)
    _, disp = _run_etch(phi0, h, delta, Da, (u / NSTEPS) / flat_speed(Da, delta, u, h))
    out_s, out_a, out_b, out_w = [], [], [], []
    for side, s, p, n in walls:
        pc, pg = p - n * 0.003, p + n * 0.0015
        dx, dy = (pc - v) * 1000
        gx, gy = (pg - v) * 1000
        if max(abs(dx), abs(dy), abs(gx), abs(gy)) > Rw - 2 * h:
            continue
        out_s.append(side * s)
        out_a.append(_sample(disp, X, Y, dx, dy, h) - u)
        out_b.append(_sample(T, X, Y, gx, gy, h) - 1)
        out_w.append(_local_width(allcu, p, n))
    return dict(s=np.array(out_s), a=np.array(out_a), b=np.array(out_b), w=np.array(out_w),
                rw_um=Rw, h_um=h, sec=time.perf_counter() - t0)


T_CAP_K = 1.25


def t_cap(theta_deg, k=T_CAP_K):
    """Предел времени пребывания на стенках кармана с углом θ: k·(1 + A(θ)), A(θ) = 1/(2 sin²(θ/2)) —
    время пребывания у вершины клина (документ B1). У обычных клиньев расчётное T_max ≈ 1,1·(1 + A(θ))
    (реальные платы: ≤ 1,17 в 99 % примеров), а в 2,6–19 раз больше — только там, где раствор держит
    узкая щель своей цепи вдоль стенки (зазоры термобарьеров и т. п.). Член промывки u_r·(T − 1) не
    учитывает расход задержанного раствора и там неограниченно растёт, поэтому T обрезается пределом
    клина с тем же углом (отступление от модели документа, см. dfm/ml/dataset.py)."""
    th = math.radians(min(max(theta_deg, 5.0), 180.0))
    return k * (1 + 1 / (2 * math.sin(th / 2) ** 2))


def labels_for(sim, u_r, tau=0.2, t_max=None):
    """Метки при промывке u_r (мкм): наибольшее смещение стенки Δ_max, наименьшее (недотрав) и
    наибольшее относительное сужение y = Δ/w; опасно — y > τ. t_max — предел времени пребывания
    T (`t_cap`), None — без предела, как в документе."""
    b = sim['b'] if t_max is None else np.minimum(sim['b'], t_max - 1)
    d = sim['a'] + u_r * b
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = d / (sim['w'] * 1000)
    ok = np.isfinite(rel)
    if not ok.any():
        return dict(notch=float('nan'), fillet=float('nan'), y=float('nan'), risky=False)
    j = int(np.nanargmax(np.where(ok, rel, -np.inf)))
    return dict(notch=float(np.nanmax(d)), fillet=float(np.nanmin(d)), y=float(rel[j]), w_at=float(sim['w'][j]),
                risky=bool(rel[j] > tau))
