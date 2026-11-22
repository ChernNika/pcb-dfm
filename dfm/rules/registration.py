"""Бюджет погрешностей совмещения отверстий по слоям (документ A3, глава 2) — общий для A3 и A5.

Смещение отверстия относительно меди слоя k — сумма независимых нормальных составляющих:

    положение сверла      σ_св             местная (своя у каждого отверстия, общая для его слоёв)
    увод сверла           c_w · z          местная (направление одно на все слои отверстия)
    наружный рисунок      σ_нар            местная (рисунок совмещается с просверленными отверстиями)
    межслойное совмещение σ_сл             ОБЩАЯ для всех отверстий слоя заготовки (внутренние слои)
    остаток масштаба      σ_k · r          ОБЩАЯ (внутренние слои), r — расстояние от центра заготовки

    σ²_нар(z) = σ_св² + (c_w z)² + σ_нар²,   σ²_вн(z) = σ_св² + (c_w z)² + σ_сл² + (σ_k r)².

Систематический сдвиг μ = k_сист · r (некомпенсированный масштаб) превращает распределение
Рэлея в распределение Райса (документ A3, раздел 3.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rice

from ..board.model import StackKind

__all__ = ['LayerSigma', 'layer_depths', 'hole_sigmas', 'p_breakout', 'p_board', 'd_min_mm', 'DEFAULT_BOARD_MM']

DEFAULT_BOARD_MM = 1.6        # толщина платы, если её нет в файле: значение KiCad по умолчанию (Board Setup →
                              # стек) — допущение; пишется в допущения находки


@dataclass(slots=True)
class LayerSigma:
    """Погрешность положения отверстия на одном слое, мкм."""
    layer: str
    z_mm: float                    # глубина слоя от стороны входа сверла
    outer: bool
    local: float                   # местная часть (своя у отверстия)
    common: float                  # общая часть (одна на слой заготовки)
    bias: float = 0.0              # систематический сдвиг μ
    parts: dict = field(default_factory=dict)   # составляющие σ², мкм²

    @property
    def total(self) -> float:
        return math.hypot(self.local, self.common)


def layer_depths(board) -> dict[str, float]:
    """Глубина медных слоёв от верхней поверхности, мм: сумма меди и диэлектрика над слоем.
    Если толщины слоёв неизвестны (job-файл без них), слои ставятся поровну по толщине платы."""
    cu = board.copper
    out = {}
    known = board.stackup is not None and all(
        it.thickness for it in board.stackup.items if it.kind in (StackKind.COPPER, StackKind.DIELECTRIC))
    if known and board.stackup.copper:
        z = 0.0
        for it in board.stackup.items:
            if it.kind is StackKind.COPPER and it.name in cu:
                out[it.name] = z
            if it.kind in (StackKind.COPPER, StackKind.DIELECTRIC):
                z += it.thickness or 0.0
    if len(out) != len(cu):                 # стека или толщин нет — слои поровну по толщине платы
        h = board.thickness or DEFAULT_BOARD_MM
        out = {name: h * i / max(len(cu) - 1, 1) for i, name in enumerate(cu)}
    return out


def registration_sigmas(dr: dict) -> tuple[float, float]:
    """σ наружного рисунка и межслойного совмещения, мкм. Если в профиле задан допуск завода
    registration_tol_um = {outer, inner} (±T «отверстие → медь» на готовой плате — суммарная ошибка),
    σ составляющей выводится вычитанием сверла: σ = √((T/k)² − σ_св²), k = registration_k
    (4 ≈ Cpk 1,33; 3 — консервативно; 5 — Cpk 1,67; правила перевода в отрасли нет — допущение).
    Иначе — sigma_outer_pattern_um и sigma_layer_um как есть."""
    tol = dr.get('registration_tol_um')
    if not tol:
        return dr['sigma_outer_pattern_um'], dr['sigma_layer_um']
    k = dr.get('registration_k', 4.0)
    sd = dr['sigma_drill_um']
    part = lambda T: math.sqrt(max((T / k) ** 2 - sd * sd, 0.0))
    return part(tol['outer']), part(tol['inner'])


def hole_sigmas(board, profile, hole, depths: dict | None = None) -> dict[str, LayerSigma]:
    """σ отверстия на каждом медном слое его пролёта."""
    dr = profile['drilling']
    cu = board.copper
    depths = depths or layer_depths(board)
    i0, i1 = cu.index(hole.span[0]), cu.index(hole.span[1])
    from_bottom = i1 == len(cu) - 1 and i0 > 0          # глухое снизу — сверлится с нижней стороны
    z_entry = depths[cu[i1]] if from_bottom else depths[cu[i0]]
    r_mm = dr['scale_radius_mm']
    scale = dr['scale_sigma_ppm'] * 1e-6 * r_mm * 1000
    bias = dr.get('systematic_scale_ppm', 0.0) * 1e-6 * r_mm * 1000
    s_out, s_lay = registration_sigmas(dr)
    out = {}
    for name in cu[i0:i1 + 1]:
        z = abs(depths[name] - z_entry)
        outer = name in (cu[0], cu[-1])
        parts = {'сверло': dr['sigma_drill_um'] ** 2, 'увод': (dr['wander_um_per_mm'] * z) ** 2}
        if outer:
            parts['наружный рисунок'] = s_out ** 2
            common = 0.0
        else:
            parts['межслойное'] = s_lay ** 2
            parts['масштаб'] = scale ** 2
            common = math.sqrt(parts['межслойное'] + parts['масштаб'])
        local = math.sqrt(sum(v for k, v in parts.items() if k in ('сверло', 'увод', 'наружный рисунок')))
        out[name] = LayerSigma(name, z, outer, local, common, 0.0 if outer else bias, parts)
    return out


def p_breakout(margin_um, sigma_um, bias_um=0.0):
    """P(|ε| > m): Рэлей exp(−m²/2σ²), с систематическим сдвигом — Райс Q₁(μ/σ, m/σ).
    m ≤ 0 → 1 (номинальный поясок уже меньше требуемого). Принимает числа и массивы."""
    if np.isscalar(margin_um) and np.isscalar(sigma_um) and np.isscalar(bias_um):   # быстрый путь
        if margin_um <= 0:
            return 1.0
        if bias_um <= 0:
            return math.exp(-margin_um * margin_um / (2 * sigma_um * sigma_um))
        return float(rice.sf(margin_um / sigma_um, bias_um / sigma_um))
    m = np.asarray(margin_um, dtype=float)
    s = np.asarray(sigma_um, dtype=float)
    b = np.asarray(bias_um, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(b > 0, rice.sf(np.maximum(m, 0) / s, b / s), np.exp(-np.maximum(m, 0) ** 2 / (2 * s * s)))
    p = np.where(m <= 0, 1.0, p)
    return float(p) if p.ndim == 0 else p


def p_board(margins_um, local_um, common_um, n_common: int = 4096) -> float:
    """Вероятность, что на слое хотя бы одно из отверстий выйдет за площадку (документ A3,
    раздел 3.5): общий сдвиг слоя g ~ N(0, σ_общ²) по двум осям; при известном g отверстия
    независимы, у каждого — Райс с систематическим сдвигом |g| и случайной частью σ_мест;
    результат усредняется по g. margins_um, local_um — массивы по отверстиям.
    Усреднение — не случайными испытаниями (в документе 20 000), а по n_common равновероятным узлам
    распределения Рэлея |g| (середины квантилей): без случайного шума. Последний узел — |g| ≈ 4,1σ_общ,
    хвост за ним отрезан, поэтому очень редкие события занижаются: −1 % при P ≈ 10⁻³, −14 % при 6·10⁻⁵
    (сверка с адаптивным интегралом) — на вердикты (пороги 10⁻³…10⁻¹) это почти не влияет."""
    m = np.asarray(margins_um, dtype=float)
    loc = np.asarray(local_um, dtype=float)
    if not len(m):
        return 0.0
    if np.any(m <= 0):
        return 1.0
    # одинаковые отверстия — одной строкой с кратностью
    key, inv, cnt = np.unique(np.round(np.c_[m, loc], 3), axis=0, return_inverse=True, return_counts=True)
    if common_um <= 0:
        p = p_breakout(key[:, 0], key[:, 1])
        return float(1 - np.exp(np.sum(cnt * np.log1p(-np.minimum(p, 1 - 1e-16)))))
    uq = (np.arange(n_common) + 0.5) / n_common
    g = common_um * np.sqrt(-2 * np.log1p(-uq))           # квантили Рэлея
    p = rice.sf(key[None, :, 0] / key[None, :, 1], g[:, None] / key[None, :, 1])
    log_ok = np.sum(cnt[None, :] * np.log1p(-np.minimum(p, 1 - 1e-16)), axis=1)
    return float(np.mean(1 - np.exp(log_ok)))


def d_min_mm(d_mm: float, r_req_um: float, sigma_um: float, p_star: float) -> float:
    """Нужный диаметр площадки: D_min = d + 2r_тр + 2σ√(−2 ln P*) (документ A3, раздел 3.6)."""
    return d_mm + 2 * r_req_um / 1000 + 2 * sigma_um * math.sqrt(-2 * math.log(p_star)) / 1000
