"""Класс B1 без нейросети (шаг 4.1): формулы против примеров документа B1 и демо-плата B1."""
import math
from pathlib import Path

import pytest

from dfm.board import load
from dfm.board.kicad_writer import KicadWriter
from dfm.rules.b1 import corner_area, corona_onset_kv, overetch, resist_tip_area, theta_from_area
from dfm.rules.engine import Context, run
from dfm.rules.findings import Severity
from dfm.rules.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
B1 = ROOT / 'boards' / 'demo' / 'b1_demo.kicad_pcb'
PROF = load_profile()['b1']


def test_corner_area_and_effective_angle():
    """Раздел 4.6: δ = 50 мкм, θ = 30° — 6057 мкм²; обратная задача возвращает угол."""
    assert corner_area(math.radians(30), 0.05) * 1e6 == pytest.approx(6057, abs=1)
    for th in (15, 30, 45, 66.7):
        assert theta_from_area(corner_area(math.radians(th), 0.05), 0.05) == pytest.approx(th, abs=0.01)


def test_wedge_formula_profile_z():
    """Раздел 3.6, профиль З: Л4 (25°) — 15,1 мкм, Л1 по углу 18° — 35,8 мкм, Л5 (15°) — 52,2 мкм."""
    assert overetch(25, PROF) == pytest.approx(15.1, abs=0.05)
    assert overetch(18, PROF) == pytest.approx(35.8, abs=0.1)
    assert overetch(15, PROF) == pytest.approx(52.2, abs=0.1)
    assert overetch(90, PROF) < 0                                  # тупые карманы недотравливаются


def test_spike_formulas():
    """Раздел 2.9: шип 28°, u = 11,7 мкм — кончик ≈ 3060 мкм², прямой угол ≈ 760 мкм²; раздел 2.10:
    острие 15 мкм в 0,45 мм — корона с 0,94 кВ, скруглённое 0,25 мм — с 2,15 кВ."""
    assert resist_tip_area(28, 11.7, 60) == pytest.approx(3060, rel=0.01)
    assert resist_tip_area(90, 11.7, 60) == pytest.approx(760, rel=0.01)
    assert corona_onset_kv(0.015, 0.45) == pytest.approx(0.94, abs=0.01)
    assert corona_onset_kv(0.25, 0.45) == pytest.approx(2.15, abs=0.02)


def _run(proc, volts=None, method='doc_table', **b1):
    b = load(B1)
    pr = load_profile()
    pr['b1']['process'] = proc
    pr['b1']['method'] = method
    pr['b1'].update(b1)
    if volts:
        pr['nets']['voltages'] = volts
    return run(b, ctx=Context(b, pr), classes=['B1'])


def test_demo_board_profiles():
    """Раздел 6.4: предупреждений по карманам при профилях С / О / З — 0 / 0 / 2 (Л1 и Л5) у таблицы документа
    и формулы клина. Гибрид при З предупреждает только о Л5: наружная медь демо-платы 60 мкм (35 + 25), u = 20
    мкм, и у Л1 сужение 0,17 при узком интервале — P(y > 0,2) ≈ 0,02 < p_warning (пример документа — медь 35)."""
    for method in ('doc_table', 'formula', 'nn'):
        for proc, n in (('С', 0), ('О', 0), ('З', 2)):
            fs = [f for f in _run(proc, method=method) if f.code not in ('Л7', 'Л8') and f.severity >= Severity.WARNING]
            want = (['Л5'] if method == 'nn' else ['Л1', 'Л5']) if n else []
            assert sorted(f.code for f in fs) == want, (method, proc)
            assert all(f.severity is Severity.WARNING for f in fs)     # без купона — не выше предупреждения
    fs = {f.code: f for f in _run('З') if f.code not in ('Л7',)}
    assert fs['Л1'].measured['θ_эфф_°'] == pytest.approx(18.0, abs=0.2)
    assert fs['Л1'].measured['перетрав_мкм'] == pytest.approx(35.8, abs=0.3)
    assert fs['Л5'].measured['перетрав_мкм'] == pytest.approx(52.2, abs=0.3)
    assert fs['Л4'].measured['перетрав_мкм'] == pytest.approx(15.1, abs=0.2) and fs['Л4'].severity is Severity.INFO
    assert fs['Л2'].measured['θ_эфф_°'] == pytest.approx(30, abs=0.3)


def test_demo_spikes():
    """Л7 и Л8 — шипы 28° и 25,6°: кончик резиста вчетверо больше прямого угла; при 1 кВ на HV1 — Л8."""
    sp = sorted((round(f.measured['φ_°'], 1), round(f.measured['A_f_отн'], 1)) for f in _run('З') if f.code == 'Л7')
    assert sp == [(25.6, 4.4), (28.0, 4.0)]
    l8 = [f for f in _run('З', {'HV1': 1000.0, 'HV2': 0.0}) if f.code == 'Л8']
    assert len(l8) == 1 and l8[0].measured['d_мм'] == pytest.approx(0.45, abs=0.01)


def test_slot_is_not_a_wedge(tmp_path):
    """Трасса вдоль края площадки той же цепи в 60 мкм: узкая щель — информация, а не перетрав клина."""
    w = KicadWriter('slot')
    w.outline_rect(0, 0, 4, 3)
    fp = w.footprint('U1', (2, 1.5))
    fp.pad('1', 'smd', 'rect', (0, 0), (1.0, 1.0), net='A')
    x = 2 + 0.5 + 0.06 + 0.075                     # ось трассы: край в 60 мкм от края площадки
    w.track([(2.3, 1.8), (x, 1.8 - (x - 2.3)), (x, 0.2)], 0.15, 'F.Cu', 'A')   # отход под 45°, затем вдоль края
    b = load(w.write(tmp_path / 'slot.kicad_pcb'))
    fs = run(b, classes=['B1'])
    assert all(f.severity is Severity.INFO for f in fs)
    assert any('щель' in f.title for f in fs)


def test_hybrid_on_demo():
    """Гибрид (numpy): квантили упорядочены, P — по ним; Л5 почти наверняка сверх допуска, при откалиброванном
    профиле — ошибка; гибрид и формула клина близки; источник и ожидаемый ущерб записаны."""
    fs = {f.code: f for f in _run('З', method='nn') if f.code in ('Л1', 'Л5')}
    for f in fs.values():
        m = f.measured
        assert m['источник'] == 'гибрид' and m['сужение_q10'] <= m['сужение'] <= m['сужение_q90']
        assert m['ожидаемый_ущерб'] == pytest.approx(10 * f.probability)
        assert m['сужение'] == pytest.approx(m['сужение_формула'], abs=0.05)
    assert fs['Л5'].probability > 0.9 and fs['Л5'].measured['сужение'] > 0.2
    assert {f.code: f.severity for f in _run('З', method='nn', calibrated=True)}['Л5'] is Severity.ERROR


def test_hybrid_matches_formula_without_correction():
    """Гибрид с нулевой головой сети — ровно формула клина (так он начинает обучение)."""
    import numpy as np
    from dfm.ml import infer
    h = infer.Hybrid()
    m = h.models[0]
    m = dict(m, W={**m['W'], 'head.2': (np.zeros_like(m['W']['head.2'][0]), np.zeros_like(m['W']['head.2'][1]))})
    d = np.array([[0.2, 0.2, 0.2, 0.2, 1.0, 4.0, 0.15, 0.7, 1.0, 0.0]])
    prof = np.array([[np.log(3.0), 50.0, 35.0, 3.0, 35 / 3]])
    f = h.formula_y(d, prof, np.array([2.0]))
    q = h._forward(m, d, prof, np.array([2.0]), f)
    assert q[0, 1] == pytest.approx(f[0]) and f[0] > 0.2           # 18° при профиле З — сверх допуска


def test_out_of_range_profile_is_flagged():
    """Профиль вне диапазона обучающих данных — пометка об экстраполяции в находке."""
    fs = [f for f in _run('З', method='nn', process_params={'З': {'Da': 100.0, 'u_r_um': 2.0}}) if f.code == 'Л5']
    assert fs and any('экстраполяция' in a for a in fs[0].assumptions)


def test_pocket_delta_independent_of_profile():
    """Карманы ищутся при δ = 50 мкм, как в данных обучения: δ профиля меняет только физику (формулу и z
    гибрида), а не эффективный угол — иначе при δ = 80 θ_эфф Л1 падал с 18° до 8°."""
    base = {f.code: f.measured['θ_эфф_°'] for f in _run('З', method='formula') if f.code in ('Л1', 'Л5')}
    for d in (30, 80):
        fs = {f.code: f.measured['θ_эфф_°'] for f in _run('З', method='formula', delta_um=d) if f.code in ('Л1', 'Л5')}
        assert fs == pytest.approx(base, abs=1e-6), d


def test_tau_and_custom_process():
    """P(y > τ) — при τ профиля; свой процесс без таблицы документа — формула клина с пометкой, а не сбой."""
    p2 = {f.code: f.probability for f in _run('З', method='nn') if f.code == 'Л5'}
    p3 = {f.code: f.probability for f in _run('З', method='nn', tau=0.3) if f.code == 'Л5'}
    assert p3['Л5'] < p2['Л5']
    fs = _run('МАИ', method='doc_table', process_params={'МАИ': {'Da': 3.0, 'u_r_um': 1.0}})
    traps = [f for f in fs if f.code in ('Л1', 'Л5')]
    assert traps and all(f.measured['источник'] == 'формула клина' for f in traps)
    assert any('таблица документа есть только' in a for a in traps[0].assumptions)
    fs = _run('МАИ', method='formula')                       # нет process_params — карманы «невыполнено», шипы есть
    assert any(f.severity is Severity.NOT_CHECKED for f in fs) and any(f.code == 'Л7' for f in fs)


def test_corona_voltage_difference():
    """Л8: разность напряжений — как в A1 Д5 (две фазы — не 0 В); нужный радиус — из V_кор ≥ k_s·ΔV
    (документ B1, Л8: d = 0,45 мм, 1,2 кВ, k_s = 1,5 → между 0,10 и 0,15 мм)."""
    from dfm.rules.b1 import _corona_radius
    assert 0.10 < _corona_radius(1.8, 0.45) < 0.15
    assert corona_onset_kv(0.01, 0.005) == 0.0                       # острие ближе своего радиуса — без log(0)
    l8 = [f for f in _run('З', {'HV1': {'rms': 700, 'phase_deg': 0}, 'HV2': {'rms': 700, 'phase_deg': 180}}) if f.code == 'Л8']
    assert len(l8) == 1 and l8[0].measured['ΔV_В'] > 1900
