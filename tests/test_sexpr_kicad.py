"""Проверки читателя S-выражений на настоящих платах из поставки KiCad 10."""
import os
import shutil
from pathlib import Path

import pytest

from dfm import kicad_cli
from dfm.board.sexpr import load, dump, parse, dumps

DEMOS = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'KiCad' / '10.0' / 'share' / 'kicad' / 'demos'
SMALL = sorted((p for p in DEMOS.rglob('*.kicad_pcb') if p.stat().st_size < 7e6), key=lambda p: p.stat().st_size) \
    if DEMOS.exists() else []

pytestmark = pytest.mark.skipif(not SMALL, reason='нет демо-плат KiCad 10')


@pytest.mark.parametrize('board', SMALL, ids=lambda p: p.stem)
def test_roundtrip_demo_board(board):
    """Разбор → запись → разбор даёт то же дерево."""
    tree = load(board)
    assert tree.tag == 'kicad_pcb'
    assert parse(dumps(tree)) == tree


def test_kicad9_missing_paren_repaired():
    """В демо-плате RoyalBlue54L-Feather (KiCad 9) 349 пропущенных «(» в параметрах капель."""
    w = []
    tree = load(DEMOS / 'royalblue54L_feather' / 'RoyalBlue54L-Feather.kicad_pcb', w)
    assert len(w) == 1 and '349' in w[0]
    td = next(p.get('teardrops') for fp in tree.all('footprint') for p in fp.all('pad') if p.get('teardrops'))
    assert td.val('curved_edges') == 'no' and td.val('filter_ratio') == 0.9 and td.val('enabled') == 'yes'


def _gerber_lines(path):
    """Строки файла без отметок времени создания (в Gerber и в шапке Excellon)."""
    return [l for l in Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
            if not l.startswith(('G04 Created by', '; DRILL file')) and 'CreationDate' not in l]


@pytest.mark.kicad
def test_kicad_reads_our_file(tmp_path):
    """KiCad читает файл, записанный нашим модулем, и выгружает те же Gerber и сверловку, что из оригинала.
    Зоны не перезаливаются: перезаливка в KiCad недетерминирована (см. dfm/kicad_cli.py)."""
    try:
        kicad_cli.find()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    src = DEMOS / 'pic_programmer' / 'pic_programmer.kicad_pcb'
    a, b = tmp_path / 'orig', tmp_path / 'ours'
    a.mkdir(); b.mkdir()
    shutil.copy(src, a / src.name)
    shutil.copy(src.with_suffix('.kicad_pro'), a)
    shutil.copy(src.with_suffix('.kicad_pro'), b)
    dump(load(src), b / src.name)
    fa = kicad_cli.export_gerbers(a / src.name, a / 'g') + kicad_cli.export_drill(a / src.name, a / 'g')
    fb = kicad_cli.export_gerbers(b / src.name, b / 'g') + kicad_cli.export_drill(b / src.name, b / 'g')
    na = {p.name: p for p in fa if p.is_file()}
    nb = {p.name: p for p in fb if p.is_file()}
    assert na.keys() == nb.keys() and len(na) > 10
    for name in na:
        assert _gerber_lines(na[name]) == _gerber_lines(nb[name]), name
