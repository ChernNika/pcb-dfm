"""Основа правил (шаг 2.1): профиль, пороги A1 по слоям, вердикты, движок, отчёт, A1 Д1."""
import json

import pytest

from dfm.board import load
from dfm.board.kicad_writer import KicadWriter
from dfm.rules import engine
from dfm.rules.engine import Context, run
from dfm.rules.findings import Finding, Severity, geometric_severity, margin, probability_severity
from dfm.rules.profile import load_profile
from dfm.rules.thresholds import _table_clearance, layer_thresholds


def _board(tmp_path, tracks, copper=4, name='b'):
    w = KicadWriter(name, copper=copper)
    w.outline_rect(0, 0, 30, 20)
    fp = w.footprint('J1', (2, 10))
    for i, (width, layer) in enumerate(tracks):
        fp.pad(str(i + 1), 'thru_hole', 'circle', (0, -6 + 3 * i), 1.5, drill=0.8, net=f'N{i}')
        y = 4 + 3 * i
        w.track([(2, y), (20, y)], width, layer, f'N{i}')
    return load(w.write(tmp_path / f'{name}.kicad_pcb'))


# ---------- профиль и пороги ----------

def test_profile_defaults_and_merge(tmp_path):
    p = load_profile()
    assert p.illustrative and p['etch']['etch_factor'] == 3.0 and p.quality('etch') == 'условный профиль'
    user = tmp_path / 'fab.toml'
    user.write_text('[etch]\netch_factor = 2.5\n', encoding='utf-8')
    q = load_profile(user)
    assert q['etch']['etch_factor'] == 2.5 and q['etch']['resist_resolution_um'] == 50      # остальное — условное
    # в [etch] задан только EF — остальные числа раздела из условного, и метка это говорит
    assert not q.illustrative and q.quality('etch') == 'профиль завода, условный профиль'
    assert q.quality('mask') == 'условный профиль'


def test_thresholds_match_document_a1(tmp_path):
    """Документ A1, раздел 5: условный завод (EF 3 ± 0,3, резист 50 мкм, гальваника 25 мкм) —
    зазор внутри ≈ 73,3 мкм, снаружи (35 + 25 мкм) 90 мкм; ширина ≈ 77,6 и ≈ 79,4 мкм."""
    th = layer_thresholds(_board(tmp_path, []), load_profile())
    assert th['In1.Cu'].clearance.target * 1000 == pytest.approx(73.33, abs=0.01)
    assert th['F.Cu'].clearance.target * 1000 == pytest.approx(90.0)
    assert th['In2.Cu'].width.target * 1000 == pytest.approx(77.6, abs=0.05)
    assert th['B.Cu'].width.target * 1000 == pytest.approx(79.4, abs=0.05)
    assert th['F.Cu'].copper_um == 60 and th['F.Cu'].outer and not th['In1.Cu'].outer


def test_clearance_table_interpolation_and_extrapolation():
    """Таблица (18; 75), (35; 90), (70; 125) мкм: внутри — интерполяция, снаружи — прямая МНК
    (документ A1, раздел 5.5: k ≈ 0,967, a ≈ 57,0 мкм)."""
    table = [[18, 75], [35, 90], [70, 125]]
    assert _table_clearance(table, 35) == (90.0, 'таблица')
    s, src = _table_clearance(table, 105)
    assert src == 'экстраполяция' and s == pytest.approx(57.0 + 0.967 * 105, abs=0.6)


# ---------- вердикты ----------

def test_geometric_severity_ladder():
    assert geometric_severity(0.12, target=0.09, level_a=0.11, level_c=0.07) is Severity.OK
    assert geometric_severity(0.10, 0.09, 0.11, 0.07) is Severity.INFO
    assert geometric_severity(0.08, 0.09, 0.11, 0.07) is Severity.WARNING
    assert geometric_severity(0.06, 0.09, 0.11, 0.07) is Severity.ERROR
    # без уровней A и C: всё, что не меньше целевого, — норма; ниже — ошибка (консервативно)
    assert geometric_severity(0.09, 0.09) is Severity.OK and geometric_severity(0.0899, 0.09) is Severity.ERROR
    assert margin(0.080, 0.073) == pytest.approx(0.0959, abs=1e-3)      # документ A1: +9,6 %


def test_probability_severity():
    scale = load_profile()['verdict']
    assert [probability_severity(p, scale) for p in (1e-4, 5e-3, 0.05, 0.3)] == \
        [Severity.OK, Severity.INFO, Severity.WARNING, Severity.ERROR]


# ---------- движок и A1 Д1 ----------

def test_d1_narrow_tracks(tmp_path):
    """0,07 мм снаружи и 0,077 мм внутри — ошибки (минимумы 79,4 и 77,6 мкм); 0,2 мм — норма."""
    b = _board(tmp_path, [(0.07, 'F.Cu'), (0.077, 'In1.Cu'), (0.2, 'B.Cu'), (0.078, 'In2.Cu')])
    fs = [f for f in run(b, classes=['A1']) if f.code == 'Д1']
    got = sorted((f.layer, f.code, f.severity, f.nets) for f in fs)
    assert got == [('F.Cu', 'Д1', Severity.ERROR, ('N0',)), ('In1.Cu', 'Д1', Severity.ERROR, ('N1',))]
    f = next(f for f in fs if f.layer == 'F.Cu')
    assert f.objects and all(o['uid'] for o in f.objects) and f.margin < 0
    assert f.data_quality.startswith('условный профиль')


def test_d1_info_with_level_a(tmp_path):
    """Если завод даёт уровень A, ширина между целевым и A — «информация»."""
    prof = tmp_path / 'fab.toml'
    prof.write_text('[etch.level_A]\nwidth_um = 120\n', encoding='utf-8')
    b = _board(tmp_path, [(0.1, 'In1.Cu'), (0.13, 'In2.Cu')])
    fs = [f for f in run(b, ctx=Context(b, load_profile(prof)), classes=['A1']) if f.code == 'Д1']
    assert [(f.layer, f.severity) for f in fs] == [('In1.Cu', Severity.INFO)]


def test_engine_reports_crashed_rule(tmp_path, monkeypatch):
    b = _board(tmp_path, [])
    engine._load_rules()

    def boom(ctx):
        raise RuntimeError('нет данных')
    monkeypatch.setitem(engine.RULES, 'X9', boom)
    fs = run(b, classes=['X9'])
    assert len(fs) == 1 and fs[0].severity is Severity.NOT_CHECKED and 'нет данных' in fs[0].title


def test_check_command_writes_report(tmp_path):
    from dfm.__main__ import main
    b = _board(tmp_path, [(0.07, 'F.Cu')], copper=2, name='cli')
    code = main(['check', str(tmp_path / 'cli.kicad_pcb'), '-o', str(tmp_path / 'out')])
    assert code == 1                                   # есть ошибка — код возврата 1 (для CI)
    data = json.loads((tmp_path / 'out' / 'findings.json').read_text(encoding='utf-8'))
    d1 = [f for f in data['findings'] if f['code'] == 'Д1']
    assert data['illustrative'] and len(d1) == 1
    assert d1[0]['severity'] == 'ошибка' and 'pos_kicad' in d1[0]
    assert (tmp_path / 'out' / 'report.md').exists() and (tmp_path / 'out' / 'layers' / 'F_Cu.png').exists()
    assert b.name == 'cli'


def test_finding_json_roundtrip():
    f = Finding('A1', 'Д4', 'зазор', Severity.WARNING, layer='F.Cu', pos=(1.0, -2.0), nets=('A', 'B'))
    d = f.to_dict()
    assert d['severity'] == 'предупреждение' and d['pos_kicad'] == (1.0, 2.0)
