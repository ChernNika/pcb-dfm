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
