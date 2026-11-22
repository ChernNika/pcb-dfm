"""Напряжения цепей и выбор зазора по таблицам профиля для A1, A5 и B1."""
import math
from itertools import chain, combinations

MAINS_TOLERANCE = 0.10       # κ: отклонение напряжения сети ±10 % (ГОСТ 32144-2013, п. 4.2.2; документ A1, раздел 6.5.2)


def _voltage(v) -> tuple[float, bool, float | None]:
    """Напряжение цепи из профиля → (пик, переменное ли, фаза °). Число — постоянное или ПИКОВОЕ
    (таблицы зазоров заданы в пиковых вольтах); {rms = …, tolerance = …, phase_deg = …} — переменное:
    пик √2·U·(1 + κ), κ по умолчанию 10 % (доля, не проценты)."""
    if isinstance(v, dict):
        if 'rms' not in v:
            raise ValueError(f'напряжение {v!r}: для переменного нужно rms = …')
        tol = v.get('tolerance', MAINS_TOLERANCE)
        if not 0 <= tol < 1:
            raise ValueError(f'напряжение {v!r}: tolerance — доля (0,1 = 10 %), а не проценты')
        return math.sqrt(2) * v['rms'] * (1 + tol), True, v.get('phase_deg')
    return float(v), False, None


def peak_voltage(v) -> float:
    return _voltage(v)[0]


def voltage_difference(va, vb) -> float:
    """Наибольшая разность потенциалов пары цепей, В (пик): постоянные — |Va − Vb|; если хотя бы одна
    переменная — худшая полярность |Va| + |Vb| (пик бывает обоих знаков); две переменные с заданными
    фазами — модуль разности векторов (фазы 0° и 120° по 325 В → 563 В; одна фаза → 0)."""
    peak_a, ac_a, phase_a = _voltage(va)
    peak_b, ac_b, phase_b = _voltage(vb)
    if ac_a and ac_b and phase_a is not None and phase_b is not None:
        phi = math.radians(phase_a - phase_b)
        return math.sqrt(max(peak_a * peak_a + peak_b * peak_b - 2 * peak_a * peak_b * math.cos(phi), 0.0))
    if ac_a or ac_b:
        return abs(peak_a) + abs(peak_b)
    return abs(peak_a - peak_b)


def required_voltage_clearance(table: list, dv: float, slope: float | None = None) -> float | None:
    """Зазор по таблице «напряжение → зазор» с округлением ВВЕРХ до строки (документ A1, раздел
    6.5.3); выше последней строки — по наклону slope (мм/В), если он задан, иначе None."""
    rows = sorted(table)
    for v, d in rows:
        if dv <= v:
            return float(d)
    if slope is None:
        return None
    v, d = rows[-1]
    return float(d + slope * (dv - v))


def max_voltage_difference(voltages: dict) -> float:
    """Пиковая разность для выбора радиуса поиска, включая зазор до цепи с потенциалом 0 В."""
    peaks_to_ground = (abs(peak_voltage(v)) for v in voltages.values())
    pairs = combinations(voltages.values(), 2)
    between_nets = (voltage_difference(first, second) for first, second in pairs)
    return max(chain(between_nets, peaks_to_ground), default=0.0)
