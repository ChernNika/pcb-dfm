"""Запуск правил: контекст проверки и реестр классов.

    findings = run(board, profile, classes=('A1', 'A7'))

Контекст общий для всех классов: плата, профиль, геометрия (BoardGeometry, слои строятся по
требованию и один раз), связность и пороги A1 — тоже по требованию. Класс регистрируется
декоратором @rule('A1') над функцией check(ctx) → list[Finding].

Если правило упало, проверка остальных продолжается, а в отчёт идёт находка «невыполнено» с
текстом ошибки (переменная окружения DFM_STRICT=1 — пробросить исключение, для отладки).
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field

from .findings import Finding, Severity
from .profile import Profile, load_profile

__all__ = ['Context', 'rule', 'RULES', 'available_classes', 'run']

RULES: dict[str, object] = {}


def rule(cls: str):
    def deco(fn):
        RULES[cls] = fn
        return fn
    return deco


@dataclass
class Context:
    board: object
    profile: Profile
    tol: float = 0.0005
    timings: dict = field(default_factory=dict)
    _geo: object = None
    _conn: object = None
    _thr: dict | None = None
    _fresh: object = None

    def fresh(self):
        """(контекст платы после свежей перезаливки зон, причина) — копия .kicad_pcb перезаливается
        в KiCad один раз на проверку (её используют A1 Д3 и A5 З8). None — нельзя: не .kicad_pcb,
        нет KiCad или выключено в профиле ([planes] check_fresh_fill)."""
        if self._fresh is None:
            self._fresh = self._make_fresh()
        return self._fresh

    def _make_fresh(self):
        import shutil
        import tempfile
        from pathlib import Path
        from .. import kicad_cli
        from ..board import load
        if not self.profile.get('planes', {}).get('check_fresh_fill', True):
            return None, 'выключено в профиле'
        src = getattr(self.board, 'path', None)
        if src is None or Path(src).suffix != '.kicad_pcb' or self.board.design is None:
            return None, 'не .kicad_pcb'
        try:
            kicad_cli.find()
        except FileNotFoundError:
            return None, 'нет KiCad'
        src = Path(src)
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / src.name
            for ext in ('.kicad_pcb', '.kicad_pro', '.kicad_dru'):
                if src.with_suffix(ext).exists():
                    shutil.copy(src.with_suffix(ext), dst.with_suffix(ext))
            try:
                kicad_cli.refill_zones(dst)
                fb = load(dst)
            except (RuntimeError, OSError, ValueError) as e:     # сбой kicad-cli или чтения — запоминается
                return None, f'перезаливка не удалась: {type(e).__name__}: {e}'
        return Context(fb, self.profile, self.tol), ''

    @property
    def geo(self):
        if self._geo is None:
            from ..geometry.copper import BoardGeometry
            self._geo = BoardGeometry(self.board, self.tol)
        return self._geo

    @property
    def connectivity(self):
        if self._conn is None:
            from ..geometry.connectivity import connectivity
            self._conn = connectivity(self.geo)
        return self._conn

    @property
    def thresholds(self):
        if self._thr is None:
            from .thresholds import layer_thresholds
            self._thr = layer_thresholds(self.board, self.profile)
        return self._thr


# отверстие меньше 10 мкм — не отверстие: такие «переходные» бывают в файлах KiCad (RoyalBlue54L —
# сверло 0,00001 мм, в Excellon KiCad пишет C0.000), сверлить там нечего; правила их пропускают
MIN_HOLE_MM = 0.01

CLASS_MODULES = ('a1', 'a3', 'a5', 'a7', 'b1')        # модули классов: при импорте регистрируют свои правила


def _load_rules():
    import importlib
    for module in CLASS_MODULES:
        importlib.import_module(f'{__package__}.{module}')


def available_classes() -> list[str]:
    """Коды зарегистрированных проверок в порядке вывода в интерфейсе."""
    _load_rules()
    return sorted(RULES)


def guarded(cls: str, code: str, fn, ctx) -> list[Finding]:
    """Подпроверка класса отдельно: её сбой — одна находка «невыполнено» с кодом подпроверки,
    остальные подпроверки класса выполняются (раньше сбой одной гасил весь класс)."""
    try:
        return fn(ctx)
    except Exception as e:
        if os.environ.get('DFM_STRICT'):
            raise
        return [Finding(cls, code, f'проверка {code} не выполнена: {type(e).__name__}: {e}', Severity.NOT_CHECKED,
                        data_quality='сбой', assumptions=[traceback.format_exc(limit=3)])]


def run(board, profile: Profile | None = None, classes=None, ctx: Context | None = None) -> list[Finding]:
    _load_rules()
    ctx = ctx or Context(board, profile or load_profile())
    out = []
    for cls in (classes or sorted(RULES)):
        if cls not in RULES:
            raise KeyError(f'класса {cls} нет (есть: {", ".join(sorted(RULES))})')
        t = time.perf_counter()
        try:
            out += RULES[cls](ctx)
        except Exception as e:
            if os.environ.get('DFM_STRICT'):
                raise
            out.append(Finding(cls, 'сбой', f'правило не выполнено: {type(e).__name__}: {e}', Severity.NOT_CHECKED,
                               assumptions=[traceback.format_exc(limit=3)]))
        ctx.timings[cls] = time.perf_counter() - t
    return out
