"""Класс A5 (шаг 2.5): формулы против таблиц документа A5, демо-плата A5, путь через Gerber."""
from collections import Counter
from pathlib import Path

import pytest
import shapely

from dfm import kicad_cli
from dfm.board import load
from dfm.board.kicad_writer import KicadWriter
from dfm.rules.a5 import antipad_min_mm, barrel_mm, maxwell_garnett, window_conductance
from dfm.rules.engine import Context, run
from dfm.rules.findings import Severity
from dfm.rules.profile import load_profile
from dfm.rules.registration import p_breakout

ROOT = Path(__file__).resolve().parents[1]
A5 = ROOT / 'boards' / 'demo' / 'a5_demo.kicad_pcb'


def _kicad():
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')


def test_barrel_and_rayleigh_table():
    """Документ A5, раздел 4.1 (ℓ_w = 25, c_тр = 100 мкм): V1 — 1,2·10⁻⁴ / 1,9·10⁻⁴, V2 — 0,70 / 0,71;
    З4: от готового диаметра запас 50 мкм и 0,24 — риск занижен втрое."""
    assert barrel_mm(0.30, load_profile()['drilling']) == pytest.approx(0.35)
    assert p_breakout(250 - 25 - 100, 29.41) == pytest.approx(1.2e-4, rel=0.1)
    assert p_breakout(250 - 25 - 100, 30.22) == pytest.approx(1.9e-4, rel=0.1)
    assert p_breakout(150 - 25 - 100, 29.41) == pytest.approx(0.70, abs=0.01)
    assert p_breakout(150 - 25 - 100, 30.22) == pytest.approx(0.71, abs=0.01)
    assert p_breakout(175 - 25 - 100, 29.41) == pytest.approx(0.24, abs=0.01)


def test_antipad_min():
    """Раздел 4.2: сверло 0,35, σ = 29,4 — 0,78 / 0,82 / 0,85 мм при P* = 10⁻² / 10⁻³ / 10⁻⁴."""
    got = [antipad_min_mm(0.35, 100, 25, 29.41, p) for p in (1e-2, 1e-3, 1e-4)]
    assert got == pytest.approx([0.778, 0.82, 0.852], abs=0.003)


@pytest.mark.parametrize('phi, numeric', [(0.1, 0.816), (0.3, 0.535), (0.5, 0.322)])
def test_cell_conductance(phi, numeric):
    """Раздел 5.3: ячейка решётки с круглым вырезом — численный расчёт документа; сплошная — 1."""
    cell = shapely.box(0, 0, 1, 1)
    hole = shapely.Point(0.5, 0.5).buffer((phi / 3.141592653589793) ** 0.5, quad_segs=64)
    g = window_conductance(cell.difference(hole), cell, h_mm=1 / 240)
    assert g['G_x'] == pytest.approx(numeric, abs=0.01) and g['G_y'] == pytest.approx(numeric, abs=0.01)
    assert window_conductance(cell, cell, h_mm=1 / 100)['G_x'] == pytest.approx(1.0, abs=1e-5)   # утечка 1e-9 на землю
    assert maxwell_garnett(phi) == pytest.approx((1 - phi) / (1 + phi))


def _no_refill():
    pr = load_profile()
    pr['planes']['check_fresh_fill'] = False
    return pr


@pytest.fixture(scope='module')
def demo():
    b = load(A5)
    return run(b, ctx=Context(b, _no_refill()), classes=['A5'])


def _titles(fs, code):
    return [f for f in fs if f.code == code and not f.title.startswith('плата')]


def test_a5_demo_findings(demo):
    """Демо-плата документа: V3 — З1 (75 < 100 мкм), V2 — З2 (≈ 0,70), поле L3 (вскрытия 0,70) —
    З2 25 %, V6 — смещённое вскрытие (З8), L2 — вскрытия поля слились (проводимость 0,47 / 0,15),
    L3 — перемычки 0,10 мм (0,59 / 0,43); V1, V4 (площадка 0,60 при вскрытии 0,85) и V5 — норма."""
    z1 = _titles(demo, 'З1')
    assert {f.layer for f in z1} == {'In1.Cu', 'In2.Cu'} and all(set(f.nets) == {'S3', f.measured['цепь_слоя']} for f in z1)
    z2 = {(f.layer, f.measured['отверстий']): f for f in _titles(demo, 'З2')}
    assert z2[('In1.Cu', 1)].nets[0] == 'S2' and z2[('In1.Cu', 1)].probability == pytest.approx(0.69, abs=0.02)
    assert z2[('In2.Cu', 15)].probability == pytest.approx(0.25, abs=0.01)
    assert z2[('In1.Cu', 1)].measured['c_от_готового_мкм'] == pytest.approx(175, abs=1.5)     # З4
    z8 = [f for f in demo if f.code == 'З8']
    assert len(z8) == 2 and all(f.nets[0] == 'S6' and f.severity is Severity.WARNING for f in z8)
    assert all(f.measured['c_мкм'] == pytest.approx(100, abs=1.5) for f in z8)
    z5, = [f for f in demo if f.code == 'З5']
    assert z5.layer == 'In1.Cu' and z5.measured['отверстий'] == 15
    assert z5.measured['G_x'] == pytest.approx(0.47, abs=0.02) and z5.measured['G_y'] == pytest.approx(0.15, abs=0.02)
    z6, = [f for f in demo if f.code == 'З6']
    assert z6.layer == 'In2.Cu' and z6.severity is Severity.INFO
    assert z6.measured['перемычка_мин_мм'] == pytest.approx(0.10, abs=0.002)
    assert z6.measured['G_x'] == pytest.approx(0.59, abs=0.02) and z6.measured['G_y'] == pytest.approx(0.43, abs=0.02)
    nets = {n for f in demo if f.severity is not Severity.NOT_CHECKED and not f.title.startswith('плата')
            for n in f.nets}
    assert not nets & {'S1', 'S4', 'HV'}
    assert any(f.code == 'З7' and f.severity is Severity.NOT_CHECKED for f in demo)


def test_voltage_clearance_z7(tmp_path):
    """З7: HV 300 В против земли по ТЕСТОВОЙ таблице (0,4 мм) — 225 мкм мало; срок по модели нити."""
    prof = tmp_path / 'v.toml'
    prof.write_text('[nets]\nvoltages = { HV = 300.0, GND = 0.0, "+3V3" = 3.3 }\n'
                    'voltage_table = { B1 = [[15, 0.05], [400, 0.4]] }\n[planes]\ncheck_fresh_fill = false\n',
                    encoding='utf-8')
    b = load(A5)
    fs = run(b, ctx=Context(b, load_profile(prof)), classes=['A5'])
    z7 = [f for f in fs if f.code == 'З7' and f.severity is Severity.ERROR]
    assert {f.layer for f in z7} == {'In1.Cu', 'In2.Cu'} and all(f.nets[0] == 'HV' for f in z7)
    assert z7[0].measured['срок_отн'] == pytest.approx((225 / 400) ** 2, abs=0.01)


@pytest.mark.kicad
def test_stale_fill_with_refill():
    """З8 по перезаливке копии: 100 мкм по сохранённой заливке против 250 после — ошибка."""
    _kicad()
    fs = run(load(A5), classes=['A5'])
    z8 = [f for f in fs if f.code == 'З8']
    assert len(z8) == 2 and all(f.severity is Severity.ERROR for f in z8)
    assert all(f.measured['c_после_перезаливки_мкм'] == pytest.approx(250, abs=2) for f in z8)


@pytest.mark.kicad
def test_nonfunctional_pad_model(tmp_path):
    """Край оставленной площадки — в рисунке слоя: смещение сверла его не меняет. Площадка 0,60 при
    вскрытии 0,85 (125 мкм) — норма; при вскрытии 0,75 (75 мкм < 100) — З3."""
    _kicad()

    def make(clear):
        w = KicadWriter(f'nfp{int(clear * 1000)}', copper=4, clearance=0.1, rules={'max_error': 0.001})
        w.outline_rect(0, 0, 3, 3)
        w.zone('GND', 'In1.Cu', [(0.2, 0.2), (2.8, 0.2), (2.8, 2.8), (0.2, 2.8)], clearance=0.1,
               min_thickness=0.05, island_removal=1)
        w.via((1.5, 1.5), 0.60, 0.30, net='S', remove_unused=False)
        w.dru.append(f"(rule \"s\" (condition \"A.NetName == 'S'\") (constraint clearance (min {clear}mm)))")
        p = w.write(tmp_path / f'{w.name}.kicad_pcb')
        kicad_cli.refill_zones(p)
        b = load(p)
        return [f for f in run(b, ctx=Context(b, _no_refill()), classes=['A5']) if f.severity is not Severity.NOT_CHECKED]
    assert make(0.125) == []
    z3 = make(0.075)
    assert [f.code for f in z3 if not f.title.startswith('плата')] == ['З3']


@pytest.mark.kicad
def test_gerber_path_same_a5(tmp_path, demo):
    """Через Gerber находки те же (устаревшая заливка — по смещению вскрытия)."""
    _kicad()
    kicad_cli.export_gerbers(A5, tmp_path / 'g')
    kicad_cli.export_drill(A5, tmp_path / 'g')
    gb = run(load(tmp_path / 'g'), classes=['A5'])
    key = lambda fs: Counter((f.code, f.severity.ru, round(f.probability or 0, 3), f.layer) for f in fs)
    assert key(gb) == key(demo)
