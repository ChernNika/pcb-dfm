"""Класс A1 (шаг 2.2): формулы документа и плата t09_a1 с дефектами известного размера."""
import os
import re
from pathlib import Path

import pytest

from dfm import kicad_cli
from dfm.board import load
from dfm.rules.engine import Context, run
from dfm.rules.a1 import current_width_mm, required_voltage_clearance
from dfm.rules.findings import Severity
from dfm.rules.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
T09 = ROOT / 'boards' / 'test' / 't09_a1.kicad_pcb'
PROFILE = ROOT / 'boards' / 'test' / 't09_a1_profile.toml'
BOARDS = ROOT / 'boards'
DEMOS = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'KiCad' / '10.0' / 'share' / 'kicad' / 'demos'


def test_current_width_document_example():
    """Документ A1, раздел 6.7: 2 А, перегрев 10 °C, наружный слой, медь 35 мкм → ≈ 0,78 мм."""
    assert current_width_mm(2.0, 10.0, 35, outer=True) == pytest.approx(0.78, abs=0.005)
    assert current_width_mm(2.0, 10.0, 35, outer=False) > 1.5        # внутри k вдвое меньше


def test_voltage_table_rounds_up():
    table = [[50, 0.1], [150, 0.4], [300, 0.8]]
    assert required_voltage_clearance(table, 50) == 0.1
    assert required_voltage_clearance(table, 51) == 0.4                # округление вверх до строки
    assert required_voltage_clearance(table, 400) is None              # выше таблицы без наклона — нельзя
    assert required_voltage_clearance(table, 400, slope=0.005) == pytest.approx(1.3)


@pytest.fixture(scope='module')
def t09():
    b = load(T09)
    fs = run(b, ctx=Context(b, load_profile(PROFILE)), classes=['A1'])
    return b, fs


def _by(fs, code):
    return [f for f in fs if f.code == code]


def test_t09_d4_exact(t09):
    """Ровно заложенные зазоры: 60 и 85 мкм снаружи, 70 мкм внутри, 50 мкм между кусками одной цепи;
    85 мкм внутри (C0–C1) — норма (порог 73,3 мкм)."""
    _, fs = t09
    got = sorted((f.layer, f.nets, round(f.measured['зазор_мм'] * 1000), f.measured['одна_цепь']) for f in _by(fs, 'Д4'))
    assert got == [('F.Cu', ('A0', 'A1'), 60, False), ('F.Cu', ('B0', 'B1'), 85, False), ('F.Cu', ('S',), 50, True),
                   ('In1.Cu', ('D0', 'D1'), 70, False)]
    assert all(f.severity is (Severity.WARNING if f.measured['одна_цепь'] else Severity.ERROR) and f.objects
               for f in _by(fs, 'Д4'))


def test_t09_other_defects(t09):
    _, fs = t09
    assert [(f.layer, f.nets) for f in _by(fs, 'Д1')] == [('F.Cu', ('W',))]
    d3 = sorted(_by(fs, 'Д3'), key=lambda f: f.severity)
    assert d3[0].severity is Severity.WARNING and 'min_thickness' in d3[0].title
    assert d3[1].measured['перешейков'] == 15 and d3[1].measured['ширина_мм'] == pytest.approx(0.060, abs=0.002)
    assert sorted(f.title.split(':')[0] for f in _by(fs, 'Д2')) == \
        ['графика на меди', 'линии текста «TEST» 50 мкм уже минимума 79.4 мкм', 'текст «88» высотой 0.35 мм']
    d5 = sorted((f.layer, f.measured['категория']) for f in _by(fs, 'Д5') if f.severity is Severity.ERROR)
    assert d5 == [('B.Cu', 'B2'), ('F.Cu', 'B2'), ('F.Cu', 'B4')]
    d7 = _by(fs, 'Д7')
    assert len(d7) == 1 and d7[0].nets == ('PWR',) and d7[0].thresholds['цел'] == pytest.approx(0.456, abs=0.002)
    assert sorted(f.nets for f in _by(fs, 'Д8')) == [('C0', 'C1'), ('E0', 'E1')]


def test_electrical_checks_not_done_without_data():
    """Нет напряжений и токов — Д5 и Д7 «невыполнены», а не «пройдены»."""
    fs = run(load(T09), classes=['A1'])
    nc = {f.code for f in fs if f.severity is Severity.NOT_CHECKED}
    assert nc == {'Д5', 'Д7'} and not any(f.code in ('Д5', 'Д7') and f.severity is Severity.ERROR for f in fs)


def test_gerber_path_gives_same_clearances(tmp_path):
    """Д4 по модели из Gerber — те же зазоры, что по .kicad_pcb (правила работают на уровне P)."""
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    kicad_cli.export_gerbers(T09, tmp_path / 'g')
    kicad_cli.export_drill(T09, tmp_path / 'g')
    gb = load(tmp_path / 'g')
    fs = run(gb, classes=['A1'])
    # текст в Gerber пришёл линиями: промежутки между буквами — Д2, а не Д4
    assert any(f.code == 'Д2' and 'промежуток' in f.title for f in fs)
    got = sorted((f.layer, f.nets, round(f.measured['зазор_мм'] * 1000)) for f in _by(fs, 'Д4'))
    assert got == [('F.Cu', ('A0', 'A1'), 60), ('F.Cu', ('B0', 'B1'), 85), ('F.Cu', ('S',), 50),
                   ('In1.Cu', ('D0', 'D1'), 70)]


test_gerber_path_gives_same_clearances = pytest.mark.kicad(test_gerber_path_gives_same_clearances)


@pytest.mark.kicad
def test_d4_against_kicad_drc(tmp_path):
    """Норма класса Default на t09 — 0,09 мм = наш порог для наружных слоёв. На F.Cu пары разных цепей
    совпадают с DRC KiCad; два куска одной цепи (S) видит только Д4 — DRC зазоры внутри цепи не проверяет."""
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    report = kicad_cli.drc(T09, tmp_path / 'drc.json', all_track_errors=True)
    kicad = set()
    for v in report['violations']:
        if v['type'] != 'clearance':
            continue
        items = [i['description'] for i in v['items']]
        if all(' на F.Cu' in d for d in items):
            kicad.add(tuple(sorted(re.search(r'\[([^\]]*)\]', d).group(1) for d in items)))
    _, fs = load(T09), run(load(T09), classes=['A1'])
    ours = {f.nets for f in _by(fs, 'Д4') if f.layer == 'F.Cu' and not f.measured['одна_цепь']}
    assert ours == kicad == {('A0', 'A1'), ('B0', 'B1')}


@pytest.mark.skipif(not DEMOS.exists(), reason='нет демо-плат KiCad 10')
@pytest.mark.parametrize('name', ['pic_programmer', 'StickHub'])
def test_clean_demo_boards(name):
    """Разведённые демо-платы KiCad при условном заводе не дают технологических ошибок Д1, Д3, Д4."""
    fs = run(load(next(DEMOS.rglob(f'{name}.kicad_pcb'))), classes=['A1'])
    bad = [f for f in fs if f.code in ('Д1', 'Д3', 'Д4') and f.severity is Severity.ERROR]
    assert not bad, [(f.code, f.layer, f.title) for f in bad[:5]]


@pytest.mark.skipif(not DEMOS.exists(), reason='нет демо-плат KiCad 10')
def test_one_air_max_real_findings():
    """На One-Air-Max Д4 находит только щели между переходными ОДНОЙ цепи, поставленными вплотную
    (площадки на внутренних слоях в 50 и 3 мкм — DRC их не видит). Две заливки PGND стыкуются по линии со
    щелью 1,6 мкм (касаются только у края): щель уже 5 мкм не протравится, медь сплошная — перешейка Д3 нет
    (раньше вписанная окружность давала ложные 20 мкм). Чужие цепи технологический минимум не нарушают."""
    fs = run(load(next(DEMOS.rglob('One-Air-Max.kicad_pcb'))), classes=['A1'])
    d4 = [f for f in fs if f.code == 'Д4']
    assert d4 and all(f.measured['одна_цепь'] for f in d4)
    assert {f.nets for f in d4} == {('/Solar_module/SW1',), ('/Solar_module/SW2',)}
    assert min(f.measured['зазор_мм'] for f in d4) == pytest.approx(0.003, abs=0.001)
    assert not [f for f in fs if f.code == 'Д3' and 'ширина_мм' in f.measured and f.severity is Severity.ERROR]


def test_mains_peak_voltage_default_tolerance():
    """Документ A1, раздел 6.5.2: 230 В действующих → 1,414·230·1,1 ≈ 358 В пиковых (κ = 10 %, ГОСТ 32144-2013)."""
    from dfm.rules.a1 import _peak_voltage
    assert _peak_voltage({'rms': 230}) == pytest.approx(357.8, abs=0.1)
    assert _peak_voltage({'rms': 230, 'tolerance': 0.0}) == pytest.approx(325.3, abs=0.1)
    assert _peak_voltage(48) == 48.0                       # число — постоянное или пиковое


def test_yield_model_korobkov():
    """Д9: модель Коробкова (2025, рис. 31): медь 35 мкм, класс 3, P = 0,97 — «производственные» ξ = 4,
    F = 5 → около 124 мкм, «лабораторные» ξ = 14, F = 2 → около 344 мкм (две стороны допуска ±20 %)."""
    import math
    from dfm.rules.a1 import p_out_of_tolerance, yield_sigma_um
    for xi, ef, w in ((4, 5, 124), (14, 2, 344)):
        s = yield_sigma_um({'litho_sigma_um': math.sqrt(2) * xi}, 35, ef)
        assert 2 * p_out_of_tolerance(w, 0.2, 0, s) == pytest.approx(0.03, abs=0.002)


def test_mai_lab_profile():
    """Профиль лаборатории МАИ: основа — открытые данные, свои числа — из диссертации; Д9 включён только в нём."""
    from dfm.rules.profile import OPEN_DATA_PROFILE, load_profile
    p = load_profile(OPEN_DATA_PROFILE.with_name('mai_lab.toml'))
    assert p['etch']['etch_factor'] == 2.3 and p['etch']['slot_aspect_max'] == 0.91   # своё и из основы
    assert p.quality('etch') == 'лабораторная линия, открытые данные' and p.quality('planes') == 'открытые данные, условный профиль'
    assert p.quality('yield') == 'лабораторная линия, условный профиль'     # k_width, k_gap, combine — из условного
    assert p['yield']['enabled'] and not load_profile()['yield']['enabled']
    # зазор — по модели линии (g_res 150 + 2t/EF), а не по таблицам заводов из основы
    from dfm.board import load
    from dfm.rules.thresholds import layer_thresholds
    th = layer_thresholds(load(BOARDS / 'demo' / 'a3_demo.kicad_pcb'), p)
    assert th['In1.Cu'].clearance.source.startswith('модель')
    assert th['In1.Cu'].clearance.target * 1000 == pytest.approx(150 + 2 * 35 / 2.3, abs=0.5)
    assert th['In1.Cu'].width.target * 1000 == pytest.approx(190)


def test_d9_finding_korobkov_method():
    """Д9 на a3_demo с профилем МАИ: слой — по самой узкой трассе и самому узкому зазору (метод автора,
    с. 162–163), h — стравливаемая медь; P = 1 − (1 − P_w)(1 − P_g)."""
    from dfm.board import load
    from dfm.rules.engine import Context, run
    from dfm.rules.profile import OPEN_DATA_PROFILE, load_profile
    b = load(BOARDS / 'demo' / 'a3_demo.kicad_pcb')
    p = load_profile(OPEN_DATA_PROFILE.with_name('mai_lab.toml'))
    d9 = {f.layer: f for f in run(b, ctx=Context(b, p), classes=['A1']) if f.code == 'Д9'}
    f = d9['F.Cu']
    m = f.measured
    assert m['h_мкм'] == 50                                              # 35 + стравливаемая гальваника 15
    assert f.probability == pytest.approx(1 - (1 - m['P_трасса_макс']) * (1 - m['P_зазор_макс']), rel=1e-9)
    p['yield']['combine'] = 'product'
    d9p = {f.layer: f for f in run(b, ctx=Context(b, p), classes=['A1']) if f.code == 'Д9'}
    assert d9p['F.Cu'].probability >= f.probability
