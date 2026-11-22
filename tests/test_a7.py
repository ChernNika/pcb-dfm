"""Класс A7 (шаг 2.3): формулы против таблиц документа и демо-плата A7 из документа."""
from collections import Counter
from pathlib import Path

import pytest

from dfm import kicad_cli
from dfm.board import load
from dfm.rules.a7 import p_encroach_circ, p_encroach_rect, p_peel, sigma_um, solder_area_loss
from dfm.rules.engine import run
from dfm.rules.findings import Severity
from dfm.rules.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
A7 = ROOT / 'boards' / 'demo' / 'a7_demo.kicad_pcb'
MASK = load_profile()['mask']


@pytest.mark.parametrize('s, e, rect, circ', [
    (25, 25, 0.534, 0.607), (25, 50, 0.0889, 0.135), (25, 75, 0.0054, 0.0111), (10, 25, 0.0247, 0.0439)])
def test_encroachment_table(s, e, rect, circ):
    """Документ A7, раздел 2.3: формулы наползания (совпали с моделированием на 2 млн смещений)."""
    assert p_encroach_rect(e, s) == pytest.approx(rect, rel=0.01)
    assert p_encroach_circ(e, s) == pytest.approx(circ, rel=0.01)
    assert p_encroach_rect(0, s) == p_encroach_circ(0, s) == 1.0


def test_sigma_with_scale():
    """LDI, 25 ppm на 200 мм — 11,2 мкм; фотошаблон — 25,5 мкм (документ A7, раздел 2.4)."""
    for exposure, expected in (('ldi', 11.2), ('photo', 25.5)):
        m = dict(MASK, exposure=exposure, panel_half_diagonal_mm=200)
        assert sigma_um(m)[1] == pytest.approx(expected, abs=0.05)


def test_peel_and_row():
    """Зелёная маска: 125 / 100 / 70 мкм — 0,03 % / 0,67 % / 22 %; ряд из 7 по 100 мкм — 4,6 %,
    у чёрной — 59 % (документ A7, раздел 2.5)."""
    assert [p_peel(w, MASK) for w in (125, 100, 70)] == pytest.approx([0.0003, 0.0067, 0.22], rel=0.02)
    assert 1 - (1 - p_peel(100, MASK)) ** 7 == pytest.approx(0.046, abs=0.001)
    black = dict(MASK, color='black')
    assert 1 - (1 - p_peel(100, black)) ** 7 == pytest.approx(0.59, abs=0.01)


def test_solder_area_loss():
    """Площадка Ø0,40, окно Ø0,50: до 50 мкм потерь нет, 75 мкм — 4,9 %, 100 мкм — 12 %."""
    assert solder_area_loss(200, 50, 50) == pytest.approx(0.0, abs=1e-9)
    assert solder_area_loss(200, 50, 75) == pytest.approx(0.049, abs=0.002)
    assert solder_area_loss(200, 50, 100) == pytest.approx(0.12, abs=0.005)


@pytest.fixture(scope='module')
def demo():
    return run(load(A7), classes=['A7'])


def test_a7_demo_board_findings(demo):
    """Демо-плата документа: находки М1–М7 в тех местах, которые описывает документ A7, глава 6."""
    by = lambda code: [f for f in demo if f.code == code]
    m1 = {f.title.split(':')[0]: f for f in by('М1')}
    assert set(m1) == {'U2', 'U3'}
    assert m1['U3'].measured['ширина_мкм'] == pytest.approx(70, abs=1) and m1['U3'].probability == pytest.approx(0.83, abs=0.02)
    assert m1['U2'].measured['ширина_мкм'] == pytest.approx(100, abs=1) and m1['U2'].probability == pytest.approx(0.046, abs=0.003)
    m2 = sorted((f.measured['e_мкм'], f.probability) for f in by('М2'))
    assert m2[0] == (pytest.approx(0, abs=1e-6), 1.0)                     # R1 без расширения
    m3 = by('М3')
    assert len(m3) == 1 and set(m3[0].nets) == {'CONN', 'SIG'} and m3[0].measured['площадь_мм2'] == pytest.approx(0.09, abs=0.01)
    m4 = by('М4')
    assert len(m4) == 1 and m4[0].nets == ('B22',) and m4[0].severity is Severity.ERROR
    assert m4[0].measured['φ_L'] == pytest.approx(0.48, abs=0.02) and m4[0].measured['φ_H'] == pytest.approx(0.79, abs=0.02)
    m5 = {f.objects[0]['ref']: f for f in by('М5')}
    assert 'C1' in m5 and m5['C1'].severity is Severity.WARNING
    widths = Counter(round(f.measured['ширина_мкм']) for f in by('М6'))
    assert widths == {41: 3, 64: 2, 50: 2}                                 # B21, B22, N5–N6 и TP1 у края
    m7 = by('М7')
    assert len(m7) == 1 and m7[0].measured['smd'] == 1 and m7[0].measured['nsmd'] == 15


def test_correct_places_give_no_findings(demo):
    """Правильные места документа (ряд U1 с шагом 0,65 мм, C2 с закрытым переходным, закрытые
    переходные BGA) находок не дают."""
    assert not any('U1' in f.title for f in demo if f.code == 'М1')
    assert not any(any(o.get('ref') == 'C2' for o in f.objects) for f in demo)
    assert len([f for f in demo if f.code == 'М4']) == 1


@pytest.mark.kicad
def test_gerber_path_same_mask_findings(tmp_path, demo):
    """Окна маски из Gerber дают те же находки: правило работает на уровне P."""
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    kicad_cli.export_gerbers(A7, tmp_path / 'g')
    kicad_cli.export_drill(A7, tmp_path / 'g')
    fs = run(load(tmp_path / 'g'), classes=['A7'])
    key = lambda fs: sorted((f.code, f.severity.ru) for f in fs)
    assert key(fs) == key(demo)
