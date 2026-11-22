"""Командная строка: --set раздел.ключ=значение — ключ из профиля, значение по типу прежнего, ошибки без трассировки."""
from pathlib import Path

import pytest

from dfm.__main__ import SetError, main, set_value
from dfm.rules.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
A7 = ROOT / 'boards' / 'demo' / 'a7_demo.kicad_pcb'


def test_set_value_types():
    p = load_profile()
    set_value(p, 'etch.etch_factor=2,5')                      # десятичная запятая
    assert p['etch']['etch_factor'] == 2.5
    set_value(p, 'etch.etch_factor=4')                        # было float — осталось float
    assert p['etch']['etch_factor'] == 4.0 and isinstance(p['etch']['etch_factor'], float)
    set_value(p, 'etch.resist_resolution_um = 40 ')
    assert p['etch']['resist_resolution_um'] == 40
    set_value(p, 'etch.outer_includes_plating=TRUE')
    assert p['etch']['outer_includes_plating'] is True
    set_value(p, 'b1.process=О')                              # кириллица — как передаёт плагин
    set_value(p, 'b1.method=nn')
    assert p['b1']['process'] == 'О' and p['b1']['method'] == 'nn'
    set_value(p, 'mask.registration_3sigma_um.ldi=25')        # ключ во вложенной таблице
    assert p['mask']['registration_3sigma_um']['ldi'] == 25


@pytest.mark.parametrize('kv, msg', [
    ('etch.etch_fator=2', "нет ключа 'etch_fator'; есть: etch_factor"),   # опечатка — со списком ключей
    ('etc.etch_factor=2', "в профиле нет ключа 'etc'"),
    ('etch.etch_factor.x=1', 'значение, а не раздел'),
    ('etch.etch_factor=nan', 'конечное число'),
    ('etch.etch_factor=inf', 'конечное число'),
    ('etch.etch_factor=', 'пустое значение'),
    ('etch.etch_factor=2,3,4', 'ожидается число'),
    ('etch.etch_factor=abc', 'ожидается число'),
    ('etch.outer_includes_plating=1', 'true или false'),
    ('etch=1', 'ожидается раздел.ключ=значение'),
    ('etch.etch_factor', 'ожидается раздел.ключ=значение'),
    ('mask.registration_3sigma_um=1', 'это раздел'),
])
def test_set_value_errors(kv, msg):
    p = load_profile()
    with pytest.raises(SetError, match=msg.replace('(', r'\(')):
        set_value(p, kv)
    assert p == load_profile()                                # профиль не изменён


def test_set_value_inherited_profile(tmp_path):
    """Ключ ищется в действующем профиле: свой профиль + meta.base + условный."""
    f = tmp_path / 'завод.toml'
    f.write_text('[meta]\nname = "завод"\nbase = "open_data.toml"\n[b1]\nprocess = "С"\n', encoding='utf-8')
    p = load_profile(f)
    set_value(p, 'etch.etch_factor=3,2')                      # раздела etch в файле нет — взят из основы
    set_value(p, 'b1.process=З')
    assert p['etch']['etch_factor'] == 3.2 and p['b1']['process'] == 'З'


def test_check_set_error_exit_code(capsys):
    """Ошибка --set — сообщение argparse и код 2 (до чтения платы), без трассировки."""
    with pytest.raises(SystemExit) as e:
        main(['check', str(A7), '--set', 'etch.etch_fator=2'])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert 'error: --set etch.etch_fator' in err and 'Traceback' not in err
    with pytest.raises(SystemExit) as e:
        main(['check', str(A7), '--profile', str(ROOT / 'нет_такого.toml')])
    assert e.value.code == 2 and 'профиль:' in capsys.readouterr().err


def test_set_value_checked_keys():
    """b1.tau — "class" или число в (0; 1); b1.method, b1.process — из допустимых; кавычки снимаются;
    целое по смыслу остаётся целым; величина с единицей может стать дробной."""
    p = load_profile()
    set_value(p, 'b1.tau=0,3')
    assert p['b1']['tau'] == 0.3 and isinstance(p['b1']['tau'], float)
    set_value(p, 'b1.tau="class"')
    assert p['b1']['tau'] == 'class'
    set_value(p, 'b1.process="О"')
    set_value(p, "b1.method='formula'")
    assert p['b1']['process'] == 'О' and p['b1']['method'] == 'formula'
    set_value(p, 'etch.product_class=3')
    assert p['etch']['product_class'] == 3 and type(p['etch']['product_class']) is int
    set_value(p, 'etch.product_class=2,0')
    assert type(p['etch']['product_class']) is int
    set_value(p, 'etch.plating_um=17,5')
    assert p['etch']['plating_um'] == 17.5


@pytest.mark.parametrize('kv, msg', [
    ('b1.tau=0,3,1', 'ожидается число'),
    ('b1.tau=1,5', r'"class" или число в \(0; 1\)'),
    ('b1.tau=0', r'"class" или число в \(0; 1\)'),
    ('b1.method=xx', 'ожидается одно из nn, formula, doc_table'),
    ('b1.process=Ж', 'нет процесса'),
    ('b1.process=""', 'пустое значение'),
    ('etch.product_class=2.5', 'целое число'),
    ('mask.bga_min_pads=1,5', 'целое число'),
    ('meta.name=x', 'раздел meta'),
    ('meta.illustrative=false', 'раздел meta'),
])
def test_set_value_checked_errors(kv, msg):
    p = load_profile()
    with pytest.raises(SetError, match=msg):
        set_value(p, kv)
    assert p == load_profile()


def _bad_profiles(d):
    d.mkdir()
    (d / 'цикл_а.toml').write_text('[meta]\nbase = "цикл_б.toml"\n', encoding='utf-8')
    (d / 'цикл_б.toml').write_text('[meta]\nbase = "цикл_а.toml"\n', encoding='utf-8')
    (d / 'не_таблица.toml').write_text('b1 = 1\n', encoding='utf-8')
    (d / 'мета.toml').write_text('meta = 1\n', encoding='utf-8')
    (d / 'битый.toml').write_text('x = \n', encoding='utf-8')
    (d / 'хороший.toml').write_text('[meta]\nname = "хороший"\n', encoding='utf-8')
    return d


def test_profiles_bad_files(tmp_path, capsys):
    """Испорченный профиль в папке — запись с ошибкой, остальные профили на месте, код 0."""
    import json
    d = _bad_profiles(tmp_path / 'профили')
    assert main(['profiles', '--dir', str(d)]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    by = {Path(p['path']).stem: p for p in out['profiles'] if p['path']}
    for stem in ('цикл_а', 'цикл_б', 'не_таблица', 'мета', 'битый'):
        assert by[stem]['error'] and by[stem]['name'] == stem, stem
    assert 'цикл meta.base' in by['цикл_а']['error'] and 'b1 = 1' in by['не_таблица']['error']
    assert 'error' not in by['хороший'] and by['хороший']['name'] == 'хороший'
    assert out['profiles'][0]['path'] == '' and 'error' not in out['profiles'][0]


@pytest.mark.parametrize('argv, msg', [
    (['--profile', 'цикл_а.toml'], 'профиль: цикл meta.base'),
    (['--profile', 'не_таблица.toml'], 'ожидается таблица [b1]'),
    (['--profile', 'мета.toml'], 'профиль:'),
    (['--classes', 'Z9'], "нет классов 'Z9'"),
    (['--classes', 'A1,'], "нет классов ''"),
    (['--set', 'b1.tau=0,3,1'], '--set b1.tau'),
])
def test_check_bad_input_exit_code(tmp_path, capsys, argv, msg):
    """Плохой профиль, неизвестный класс — сообщение argparse и код 2 (до чтения платы), без трассировки."""
    d = _bad_profiles(tmp_path / 'профили')
    argv = [str(d / a) if a.endswith('.toml') else a for a in argv]
    with pytest.raises(SystemExit) as e:
        main(['check', str(A7), '-o', str(tmp_path / 'out'), *argv])
    err = capsys.readouterr().err
    assert e.value.code == 2 and msg in err and 'Traceback' not in err
    assert not (tmp_path / 'out').exists()
