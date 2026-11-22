"""Регрессионные тесты на ошибки правил, найденные чек-апом 20.11.2026 (research_notes/Чек-ап/rules_review.md)."""
import pytest

from dfm.rules.engine import Context, run
from dfm.rules.findings import Severity
from dfm.rules.profile import load_profile

from model_boards import board, via


def _run(b, classes, prof=None):
    prof = prof or load_profile()
    return run(b, classes=classes, ctx=Context(b, prof))


def test_two_layer_small_through_via_is_not_microvia():
    """Сквозное 0,15 мм на 2-слойной плате — механическое: поясок проверяется так же, как у 0,16 мм."""
    out = {}
    for d in (0.15, 0.16):
        b = board()
        via(b, (10, 10), d, ('F.Cu', 'B.Cu'), {'F.Cu': d + 0.01, 'B.Cu': 0.45}, 'N1')
        fs = _run(b, ['A3'])
        out[d] = sorted({f.code for f in fs if f.severity is not Severity.OK})
        assert 'К7' not in out[d]
    assert out[0.15] == out[0.16] and 'К2' in out[0.15]


def test_bottom_microvia_checks_inner_target_pad():
    """Микроотверстие In2.Cu→B.Cu прожигается со стороны B.Cu: целевая площадка — In2.Cu."""
    res = {}
    for span, pads in ((('F.Cu', 'In1.Cu'), {'F.Cu': 0.30, 'In1.Cu': 0.12}),
                       (('In2.Cu', 'B.Cu'), {'In2.Cu': 0.12, 'B.Cu': 0.30})):
        b = board(copper=('F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'))
        via(b, (10, 10), 0.10, span, pads, 'N1', microvia=True)
        k7 = [f for f in _run(b, ['A3']) if f.code == 'К7']
        res[span] = (len(k7), k7[0].layer if k7 else None)
    assert res[('F.Cu', 'In1.Cu')] == (1, 'In1.Cu')
    assert res[('In2.Cu', 'B.Cu')] == (1, 'In2.Cu')


# ---------- напряжения: A1 Д5 и A5 З7 ----------

def _hv_profile(**nets):
    prof = load_profile()
    prof['nets'].update(nets)
    return prof


def test_voltage_difference_worst_polarity_and_phase():
    from dfm.rules.a1 import voltage_difference
    assert voltage_difference(15, 5) == 10                                    # постоянные — разность
    mains = {'rms': 230}
    assert voltage_difference(mains, 15) == pytest.approx(357.8 + 15, abs=0.2)   # худшая полярность
    assert voltage_difference(mains, mains) == pytest.approx(2 * 357.8, abs=0.2)  # фазы неизвестны
    l1, l2 = {'rms': 230, 'phase_deg': 0}, {'rms': 230, 'phase_deg': 120}
    assert voltage_difference(l1, l2) == pytest.approx(357.8 * 3 ** 0.5, abs=0.3)
    assert voltage_difference(l1, dict(l1)) == pytest.approx(0, abs=1e-9)       # одна фаза
    with pytest.raises(ValueError):
        voltage_difference({'peak': 400}, 0)
    with pytest.raises(ValueError):
        voltage_difference({'rms': 230, 'tolerance': 10}, 0)                   # проценты вместо доли


def test_d5_without_mask_treats_copper_as_open():
    """Нет слоя маски — наружная медь открыта (категория B2), а не «под маской» (B4)."""
    from model_boards import track
    tables = {'B2': [[50, 0.6], [400, 1.25]], 'B4': [[50, 0.13], [400, 0.25]], 'B1': [[50, 0.1], [400, 0.25]]}
    res = {}
    for mask in (False, True):
        b = board(mask=mask)
        track(b, 'F.Cu', (2, 5), (18, 5), 0.5, 'MAINS')
        track(b, 'F.Cu', (2, 5.8), (18, 5.8), 0.5, 'GND')                     # зазор 0,3 мм
        prof = _hv_profile(voltages={'MAINS': {'rms': 230}, 'GND': 0}, voltage_table=tables)
        d5 = [f for f in _run(b, ['A1'], prof) if f.code == 'Д5' and f.severity is Severity.ERROR]
        res[mask] = sorted(f.measured['категория'] for f in d5)
    assert res[False] == ['B2']           # без маски: открытая медь
    assert res[True] == []                # маска без окон: закрытая (B4: нужно 0,25 мм — зазор 0,3 проходит)


def test_a5_voltage_reach_and_one_failing_subcheck():
    """Высокое напряжение: вскрытие дальше 0,375 мм от ствола всё равно проверяется (раньше молча
    пропускалось); A1: сбой одной подпроверки не гасит остальные."""
    from model_boards import region
    from dfm.rules import a5
    res = {}
    for R in (0.40, 1.00):
        b = board(copper=('F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'))
        for L in ('In1.Cu', 'In2.Cu'):
            region(b, L, [(0, 0), (20, 0), (20, 20), (0, 20)], 'GND')
        # вскрытие вокруг отверстия: заливка с круглым вырезом радиуса R
        from dfm.board.model import Feature, Role
        from dfm.board.shapes import Circle
        for L in ('In1.Cu', 'In2.Cu'):
            b.layers[L].features.append(Feature(Circle((10, 10), R), None, Role.ZONE, dark=False))
        via(b, (10, 10), 0.3, ('F.Cu', 'B.Cu'), {'F.Cu': 0.6, 'B.Cu': 0.6}, 'HV')
        prof = _hv_profile(voltages={'HV': 1000, 'GND': 0}, voltage_table={'B1': [[15, 0.05], [500, 0.25]]},
                           voltage_slope_mm_per_v={'B1': 0.0025})
        z7 = [f for f in _run(b, ['A5'], prof) if f.code == 'З7' and f.severity is not Severity.NOT_CHECKED]
        res[R] = len(z7)
        assert a5.voltage_reach_mm(Context(b, prof)) == pytest.approx(0.25 + 0.0025 * 500, abs=1e-9)
    assert res[0.40] >= 1 and res[1.00] >= 1


# ---------- A7 М7 ----------

def _matrix(b, ref, n, cu_d, win_d, x0=3.0, pitch=0.8):
    from model_boards import mask_window, pad
    k = 0
    for i in range(n):
        c = (x0 + (i % 5) * pitch, 3.0 + (i // 5) * pitch)
        pad(b, 'F.Cu', c, cu_d, f'N{ref}{i}', component=ref, pin=str(i + 1))
        mask_window(b, 'F.Mask', c, win_d)
        k += 1
    return k


def test_m7_finds_document_example_and_ignores_thermal_pads():
    from model_boards import mask_window, pad
    # пример документа A7, раздел 5.7: NSMD Ø0,40 (окно 0,50) и одна SMD-площадка Ø0,50 с окном Ø0,35
    b = board()
    _matrix(b, 'U1', 15, 0.40, 0.50)
    pad(b, 'F.Cu', (9.0, 9.0), 0.50, 'NU1x', component='U1', pin='16')
    mask_window(b, 'F.Mask', (9.0, 9.0), 0.35)
    m7 = [f for f in _run(b, ['A7']) if f.code == 'М7']
    assert len(m7) == 1 and m7[0].measured['smd'] == 1 and m7[0].measured['nsmd'] == 15
    # LGA: 16 NSMD Ø0,40 и 4 крупные круглые тепловые SMD Ø1,5 — не матрица со смесью
    b = board()
    _matrix(b, 'U2', 16, 0.40, 0.50)
    for j in range(4):
        c = (12.0 + 2.0 * j, 15.0)
        pad(b, 'F.Cu', c, 1.5, 'GND', component='U2', pin=f'T{j}')
        mask_window(b, 'F.Mask', c, 1.3)
    assert not [f for f in _run(b, ['A7']) if f.code == 'М7']


# ---------- устойчивость: сбой перезаливки и сбой одной подпроверки ----------

def test_refill_failure_is_cached_and_reported(monkeypatch):
    from pathlib import Path
    from dfm import kicad_cli
    from dfm.board import load
    calls = []

    def boom(path, *a, **k):
        calls.append(path)
        raise RuntimeError('kicad-cli упал')
    monkeypatch.setattr(kicad_cli, 'refill_zones', boom)
    monkeypatch.setattr(kicad_cli, 'find', lambda: 'kicad-cli')
    b = load(Path(__file__).resolve().parents[1] / 'boards' / 'demo' / 'a5_demo.kicad_pcb')
    ctx = Context(b, load_profile())
    f1, why = ctx.fresh()
    f2, _ = ctx.fresh()
    assert f1 is None and f2 is None and why.startswith('перезаливка не удалась') and len(calls) == 1


def test_one_failing_a1_subcheck_keeps_others(monkeypatch):
    from model_boards import track
    from dfm.rules import a1
    b = board()
    track(b, 'F.Cu', (2, 5), (18, 5), 0.05, 'N1')                  # трасса уже минимума — Д1
    monkeypatch.setattr(a1, 'd8_defect_risk', lambda ctx: 1 / 0)
    fs = _run(b, ['A1'])
    assert any(f.code == 'Д1' and f.severity is Severity.ERROR for f in fs)
    assert any(f.code == 'Д8' and f.severity is Severity.NOT_CHECKED for f in fs)
    assert not any(f.code == 'сбой' for f in fs)


# ---------- профиль по открытым данным и модели, которых он требует ----------

def _open():
    from dfm.rules.profile import OPEN_DATA_PROFILE
    return load_profile(OPEN_DATA_PROFILE)


def test_open_data_profile_loads_and_is_labelled():
    p = _open()
    assert p.quality('etch') == 'открытые данные' and p.quality('b1') == 'условный профиль'
    assert p['planes']['wicking_design_um'] == 50 and p['b1']['process'] == load_profile()['b1']['process']


def test_ring_base_by_layer():
    """Открытые данные: наружный поясок — от FHS + Δ, внутренний — от сверла FHS + 2t + Δ; условный — от FHS."""
    from dfm.rules.a3 import ring_bases
    b = board(copper=('F.Cu', 'In1.Cu', 'B.Cu'))
    h = via(b, (10, 10), 0.30, ('F.Cu', 'B.Cu'), {'F.Cu': 0.6, 'In1.Cu': 0.6, 'B.Cu': 0.6}, 'N1')
    assert ring_bases(Context(b, _open()), h) == pytest.approx({'outer': 0.35, 'inner': 0.40})
    assert ring_bases(Context(b, load_profile()), h) == pytest.approx({'outer': 0.30, 'inner': 0.30})


def test_class2_breakout_90_lowers_probability():
    from dfm.rules.a3 import margin_for
    m0 = margin_for(100.0, 0.0, 0.40, 0.0)        # без допуска разрыва: запас = поясок
    m90 = margin_for(100.0, 0.0, 0.40, 90.0)      # разрыв до 90°: смещение до разрыва 90° больше пояска
    assert m0 == 100.0 and m90 == pytest.approx(123.15, abs=0.05)


def test_sigma_from_fab_tolerance():
    from dfm.rules.registration import registration_sigmas
    dr = {'sigma_drill_um': 15, 'sigma_outer_pattern_um': 8, 'sigma_layer_um': 25}
    assert registration_sigmas(dr) == (8, 25)
    dr.update(registration_tol_um={'outer': 100, 'inner': 150}, registration_k=4.0)
    so, sl = registration_sigmas(dr)
    assert so == pytest.approx(20.0) and sl == pytest.approx((37.5 ** 2 - 15 ** 2) ** 0.5)


def test_refined_clearance_model_and_tables():
    from dfm.rules.thresholds import _model_clearance, layer_thresholds
    e = _open()['etch']
    assert [round(_model_clearance(e, f)) for f in (18, 35, 70, 105)] == [95, 125, 187, 249]
    b = board(copper=('F.Cu', 'In1.Cu', 'B.Cu'))
    th = layer_thresholds(b, _open())                           # стека нет — фольга 35 мкм
    assert th['In1.Cu'].clearance.target == pytest.approx(0.125) and th['F.Cu'].clearance.target == pytest.approx(0.150)
    assert th['In1.Cu'].clearance.source == 'таблица'


def test_a5_acceptance_wicking_warning():
    """Зазор 170 мкм от ствола: с проектным затеканием 50 мкм хватает (≥ 104), с предельным 100 — нет → предупреждение."""
    from dfm.board.model import Feature, Role
    from dfm.board.shapes import Circle
    from model_boards import region
    b = board(copper=('F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'))
    for L in ('In1.Cu', 'In2.Cu'):
        region(b, L, [(0, 0), (20, 0), (20, 20), (0, 20)], 'GND')
        b.layers[L].features.append(Feature(Circle((10, 10), 0.20 + 0.170), None, Role.ZONE, dark=False))
    via(b, (10, 10), 0.30, ('F.Cu', 'B.Cu'), {'F.Cu': 0.6, 'B.Cu': 0.6}, 'SIG')   # ствол 0,40 мм (r = 0,20)
    prof = _open()
    prof['planes']['check_fresh_fill'] = False
    # почти точное совмещение: проверяется ветка «проект проходит, худшая приёмка — нет» (при σ ≈ 34 мкм
    # вероятностная проверка сама требует запаса больше 50 мкм, и до этой ветки дело не доходит)
    prof['drilling'].update(sigma_drill_um=1, sigma_outer_pattern_um=1, sigma_layer_um=1, scale_sigma_ppm=0,
                            wander_um_per_mm=0)
    z1 = [f for f in _run(b, ['A5'], prof) if f.code == 'З1']
    assert z1 and all(f.severity is Severity.WARNING and 'предельного по приёмке' in f.title for f in z1)


def test_product_class_single_source():
    """Класс изделия — etch.product_class; старый ключ annular_ring.product_class своего профиля учитывается,
    если etch.product_class не задан своим профилем; два разных значения — ошибка."""
    from dfm.rules.profile import Profile, product_class
    base = load_profile()
    assert product_class(base) == 2
    old = Profile({**base, 'annular_ring': {**base['annular_ring'], 'product_class': 3}}, 'свой', {'annular_ring': {'product_class': 'fab'}})
    assert product_class(old) == 3
    both = Profile({**base, 'etch': {**base['etch'], 'product_class': 2}, 'annular_ring': {**base['annular_ring'], 'product_class': 3}},
                   'свой', {'etch': {'product_class': 'fab'}, 'annular_ring': {'product_class': 'fab'}})
    with pytest.raises(ValueError):
        product_class(both)
