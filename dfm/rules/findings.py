"""Находки правил и вердикты (документ A1, разделы 5.7, 5.8 и 7.3; документ A7, раздел 4.3).

Находка — одна проблема в одном месте платы: класс и код дефекта, слой, координаты, объекты
KiCad (uuid), измеренное значение, пороги, вердикт, запас, откуда взят порог, допущения и
предложение по исправлению. Она же уходит в отчёт JSON и (пункт 3) в плагин KiCad.

Вердикт — четыре ступени и «невыполнено»:
  * геометрический (величина против порогов уровней A, целевого и C):
        d ≥ d_A → норма;  d_цел ≤ d < d_A → информация;  d_C ≤ d < d_цел → предупреждение;
        d < d_C → ошибка.
    Если уровня A в профиле нет, «информации» не бывает; если нет уровня C, всё ниже целевого —
    ошибка (консервативно: нет данных — не «пройдено»).
  * вероятностный (вероятность на плату против шкалы профиля): < info — норма, < warning —
    информация, < error — предупреждение, иначе ошибка.
  * «невыполнено» — проверку нельзя провести без данных (напряжения, токи, таблица стандарта).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum

__all__ = ['Severity', 'Finding', 'geometric_severity', 'probability_severity', 'margin', 'to_json']


class Severity(IntEnum):
    OK = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    NOT_CHECKED = -1

    @property
    def ru(self) -> str:
        return {0: 'норма', 1: 'информация', 2: 'предупреждение', 3: 'ошибка', -1: 'невыполнено'}[int(self)]


def geometric_severity(value: float, target: float, level_a: float | None = None,
                       level_c: float | None = None) -> Severity:
    """Вердикт для величины, которой должно быть НЕ МЕНЬШЕ порога (ширина, зазор, поясок)."""
    if level_a is not None and value >= level_a:
        return Severity.OK
    if value >= target:
        return Severity.INFO if level_a is not None else Severity.OK
    if level_c is not None and value >= level_c:
        return Severity.WARNING
    return Severity.ERROR


def probability_severity(p: float, scale: dict) -> Severity:
    if p < scale['probability_info']:
        return Severity.OK
    if p < scale['probability_warning']:
        return Severity.INFO
    if p < scale['probability_error']:
        return Severity.WARNING
    return Severity.ERROR


def margin(value: float, threshold: float) -> float:
    """Относительный запас m = (d − d_порог)/d_порог (отрицательный — не хватает)."""
    return (value - threshold) / threshold if threshold else float('inf')


@dataclass
class Finding:
    cls: str                          # класс: A1, A3, A5, A7
    code: str                         # дефект: Д4, М1, К2, З5 …
    title: str                        # коротко, что не так
    severity: Severity
    layer: str | None = None
    pos: tuple | None = None          # точка в координатах модели (мм, ось Y вверх)
    nets: tuple = ()
    objects: list = field(default_factory=list)   # [{'kind': 'pad', 'uid': …, 'ref': 'U1', 'pin': '3'}]
    measured: dict = field(default_factory=dict)   # {'зазор_мм': 0.07, …}
    thresholds: dict = field(default_factory=dict)  # {'цел': 0.073, 'A': …, 'C': …}
    margin: float | None = None
    probability: float | None = None
    data_quality: str = ''            # профиль завода / условный профиль / модель / таблица / умолчание
    assumptions: list = field(default_factory=list)
    fix: str = ''

    def to_dict(self) -> dict:
        d = asdict(self)
        d['severity'] = self.severity.ru
        if self.pos is not None:
            d['pos_kicad'] = (round(self.pos[0], 4), round(-self.pos[1] + 0.0, 4))
        return d


def to_json(findings: list[Finding], **meta) -> str:
    return json.dumps({**meta, 'findings': [f.to_dict() for f in findings]}, ensure_ascii=False, indent=1,
                      default=float)
