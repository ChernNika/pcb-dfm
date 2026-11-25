"""HTML-отчёт (шаг 3.1): по карточке на находку, картинки мест, без внешних ресурсов."""
import re
from html import unescape
from pathlib import Path

from dfm.board import load
from dfm.report.check import write_report
from dfm.rules.engine import Context, run
from dfm.rules.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]


def test_html_report(tmp_path):
    b = load(ROOT / 'boards' / 'demo' / 'a7_demo.kicad_pcb')
    ctx = Context(b, load_profile())
    fs = run(b, ctx=ctx, classes=['A7'])
    out = write_report(b, ctx, fs, tmp_path, pictures=True)
    page = (out / 'report.html').read_text(encoding='utf-8')
    assert page.count('<article class="card') == len(fs)
    assert page.count('<svg class="crop"') == sum(f.pos is not None for f in fs)
    # После сортировки карточек каждая метка должна вести к исходной находке.
    markers = re.findall(r'<a href="#f(\d+)"[^>]*><circle[^>]*><title>(.*?)</title>', page)
    assert len(markers) == sum(f.pos is not None for f in fs)
    cards = dict(re.findall(r'<article [^>]*id="f(\d+)"[^>]*>.*?<h3>(.*?)</h3>', page, re.S))
    for index, title in markers:
        finding = fs[int(index)]
        assert unescape(cards[index]) == finding.title
        assert unescape(title) == f'{finding.cls} {finding.code}: {finding.title}'
    assert not re.search(r'(src|href)="https?:', page)                            # всё внутри файла
    assert 'J1.1' in page and 'Профиль условный' in page
    for f in fs:                                                                  # заголовки экранированы
        assert f.title.replace('<', '&lt;').replace('>', '&gt;') in page.replace('&#x27;', "'").replace('&quot;', '"')


def test_filter_hides_cards():
    """Фильтры скрывают карточки атрибутом hidden; у .card свой display:flex, он перебивает правило браузера."""
    from dfm.report.html import CSS
    assert re.search(r'\[hidden\]\s*\{\s*display:\s*none\s*!important;?\s*\}', CSS)


def test_yield_panel_uses_supplied_gerbers(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from dfm.report import yield_panel

    folder = tmp_path / 'gerber'
    folder.mkdir()
    copper = '%TF.FileFunction,Copper,L1,Top*%\\n%FSLAX46Y46*%\\n%MOMM*%\\nM02*'
    (folder / 'top.gbr').write_text(copper, encoding='utf-8')
    (folder / 'board.drl').write_text('M48\\nMETRIC\\nT1C0.30\\n%\\nT1\\nX1.0Y2.0\\nM30', encoding='utf-8')
    profile = load_profile()
    profile['drilling']['sigma_drill_um'] = 17.25
    board = SimpleNamespace(path=folder)

    def missing_cli(*args, **kwargs):
        raise FileNotFoundError('нет KiCad')

    monkeypatch.setattr(yield_panel.kicad_cli, 'export_gerbers', missing_cli)
    page = yield_panel.report_panel(board, profile)
    data = json.loads(re.search(r'id="yield-data">(.*?)</script>', page, re.S)[1])
    assert data['profile']['fields']['drilling.sigma_drill_um']['value'] == 17.25
    assert {f['name'] for f in data['files']} == {'board.drl', 'top.gbr'}
    assert 'id="pad-model"' in page
    board.path = tmp_path / 'board.kicad_pcb'
    fallback = yield_panel.report_panel(board, profile)
    assert 'нет KiCad' in fallback and 'id="yield-data"' not in fallback
