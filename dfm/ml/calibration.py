"""Калибровка интервалов сужения и вероятность превышения допуска B1."""
import math

import numpy as np

TAU = 0.2
UR_GRID = (0.0, 0.5, 1.0, 2.0)
CQR_BINS = (0.0, 15.0, 20.0, 30.0, 45.0, 181.0)   # группы углов θ для калибровки интервалов
CQR_MIN = 30                         # меньше примеров в группе — общая калибровка


def cqr_offsets(y, q, level=0.8):
    """Конформная калибровка квантилей (Romano и др., 2019): насколько расширить [ŷ₀,₁; ŷ₀,₉], чтобы на
    калибровочной выборке покрытие было не меньше level. Отрицательное значение — интервал сужается."""
    s = np.sort(np.maximum(q[:, 0] - y, y - q[:, 2]))
    k = min(int(math.ceil((len(s) + 1) * level)), len(s)) - 1
    return float(s[k])


def _angle_bins(theta):
    return np.clip(np.searchsorted(CQR_BINS, theta, side='right') - 1, 0, len(CQR_BINS) - 2)


def fit_cqr(pv, theta, grid=UR_GRID):
    """Калибровка интервалов по группам угла (условная конформная, «мондриановская»): у острых карманов
    разброс ошибки больше, и общая поправка оставила бы их интервалы узкими. pv[u] = (y, q, logit)."""
    bins = _angle_bins(theta)
    out = {'all': {str(u): cqr_offsets(pv[u][0], pv[u][1]) for u in grid}}
    for group in range(len(CQR_BINS) - 1):
        members = bins == group
        if members.sum() >= CQR_MIN:
            out[str(group)] = {str(u): cqr_offsets(pv[u][0][members], pv[u][1][members]) for u in grid}
    return out


def prob_from_quantiles(q, tau=TAU):
    """P(y > τ) из калиброванных квантилей: двухчастное нормальное распределение с медианой ŷ₀,₅ и
    половинами, проходящими через ŷ₀,₁ и ŷ₀,₉. Отдельная выходная ветка вероятности (фокальная потеря)
    ранжирует опасность хуже, чем сами квантили (шаг 4.4), поэтому основная вероятность — эта."""
    from scipy.stats import norm
    z = norm.ppf(0.9)
    lo = np.maximum((q[:, 1] - q[:, 0]) / z, 1e-4)
    hi = np.maximum((q[:, 2] - q[:, 1]) / z, 1e-4)
    x = tau - q[:, 1]
    return np.where(x >= 0, norm.sf(x / hi), 1 - norm.cdf(x / lo) + 0.0)


def apply_cqr(q, u, cqr, theta=None):
    """Сдвиг границ интервала по калибровке (линейная интерполяция по u_r между точками сетки).
    cqr — {u: e} или {группа: {u: e}} из fit_cqr (тогда нужен theta)."""
    def shift(table):
        grid = [float(key) for key in table]
        return float(np.interp(u, grid, list(table.values())))

    # Ранние модели хранили одну таблицу поправок, современные — ещё и таблицы групп угла.
    if 'all' in cqr and theta is not None:
        bins = _angle_bins(np.asarray(theta))
        shifts = [shift(cqr.get(str(group), cqr['all'])) for group in range(len(CQR_BINS) - 1)]
        correction = np.asarray(shifts)[bins]
    else:
        correction = shift(cqr.get('all', cqr))
    result = q.copy()
    result[:, 0] = np.minimum(q[:, 0] - correction, q[:, 1])
    result[:, 2] = np.maximum(q[:, 2] + correction, q[:, 1])
    return result
