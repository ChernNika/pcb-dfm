"""Шаг 2.6: внесённые дефекты (research/validation/inject.py) — все находятся, ложных нет."""
import sys
from pathlib import Path

import pytest

from dfm import kicad_cli

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'validation'))


@pytest.fixture(scope='module')
def inject():
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    import inject as mod
    return mod


@pytest.mark.kicad
def test_injected_defects_found(inject):
    """14 видов дефектов по три размера и контроль: полнота и точность 100 %."""
    res = inject.evaluate('inject', inject.KINDS)
    assert res['recall'] == 1.0 and res['false'] == 0
    assert all(r['control_false'] == 0 for r in res['rows'])


@pytest.mark.kicad
def test_thresholds_exact(inject):
    """Значения вплотную к порогу: чуть хуже — находка, чуть лучше — нет."""
    res = inject.evaluate('inject_threshold', inject.THRESHOLD)
    assert res['recall'] == 1.0 and res['false'] == 0
