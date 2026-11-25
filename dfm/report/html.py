"""HTML-отчёт проверки: один самодостаточный файл без внешних библиотек (его открывает браузер
и плагин KiCad).

    write_html(board, ctx, findings, путь)

Сводка «класс × дефект × вердикт» (клик — фильтр), обзорная схема платы с метками, фильтры
и поиск, карточки находок. У каждой находки с местом — картинка места в SVG: медь слоя (для
маски — медь стороны и окна), отверстия и отметка; геометрия берётся из плиток меди
(LayerCopper.tiles), поэтому сотни картинок строятся за секунды. Координаты в тексте — как
в KiCad (мм, ось Y вниз).
"""
from __future__ import annotations

import math
from collections import Counter
from html import escape
from pathlib import Path

from ..rules.findings import Severity
from .yield_panel import report_panel
from .svg import SEV_CLASS, BoardScene, crop_svg, overview_svg

__all__ = ['write_html', 'CODE_NAMES']

CODE_NAMES = {
    'Д1': 'узкая трасса', 'Д2': 'тонкий текст и графика', 'Д3': 'перешеек в заливке', 'Д4': 'зазор меньше минимума',
    'Д5': 'зазор по напряжению', 'Д6': 'путь утечки', 'Д7': 'ширина по току', 'Д8': 'риск от пылинок', 'Д9': 'выход за допуск приёмки',
    'К1': 'поясок меньше требуемого', 'К2': 'нет запаса на смещение', 'К3': 'сужение соединения с трассой',
    'К4': 'отверстие не в центре', 'К5': 'внутренние слои без запаса', 'К6': 'положение на заготовке',
    'К7': 'промах микроотверстия', 'К8': 'увод на толстой плате',
    'З1': 'зазор до слоя мал', 'З2': 'нет запаса на смещение', 'З3': 'оставленная площадка', 'З4': 'зазор от готового диаметра',
    'З5': 'слившиеся вскрытия', 'З6': 'узкие перемычки слоя', 'З7': 'зазор по напряжению', 'З8': 'устаревшая заливка',
    'М1': 'перемычки между выводами', 'М2': 'наползание маски', 'М3': 'чужая медь в окне', 'М4': 'частично закрытое переходное',
    'М5': 'открытое переходное у площадки', 'М6': 'узкие полоски маски', 'М7': 'SMD и NSMD в BGA', 'М8': 'тонкая маска',
    'Л1': 'трасса к площадке под острым углом', 'Л2': 'острый угол между трассами, щель', 'Л3': 'хвост за стыком',
    'Л4': 'вход в заливку под острым углом', 'Л5': 'веер из одной точки', 'Л6': 'трасса к круглой площадке',
    'Л7': 'шип', 'Л8': 'шип на высоковольтную цепь',
}
CLASS_NAMES = {'A1': 'ширина и зазоры', 'A3': 'пояски и совмещение', 'A5': 'сплошные слои', 'A7': 'паяльная маска',
               'B1': 'кислотные ловушки и шипы'}
SEV_ORDER = [Severity.ERROR, Severity.WARNING, Severity.INFO, Severity.NOT_CHECKED]


def _fmt(v):
    if isinstance(v, bool):
        return 'да' if v else 'нет'
    if isinstance(v, float):
        if v == 0 or math.isinf(v) or math.isnan(v):
            return str(v)
        a = abs(v)
        return f'{v:.3g}' if a < 1e-3 or a >= 1e5 else f'{v:.4g}'.replace('.', ',')
    if isinstance(v, dict):
        return '; '.join(f'{k}: {_fmt(x)}' for k, x in v.items())
    if isinstance(v, (list, tuple)):
        if len(v) > 6:
            return f'{len(v)} знач.: ' + ', '.join(_fmt(x) for x in v[:6]) + ' …'
        return '(' + '; '.join(_fmt(x) for x in v) + ')'
    return str(v)


def _kv(d: dict, skip=()) -> str:
    rows = [f'<tr><th>{escape(str(k))}</th><td>{escape(_fmt(v))}</td></tr>' for k, v in d.items() if k not in skip]
    return f'<table class="kv">{"".join(rows)}</table>' if rows else ''


def _objects(f) -> str:
    out = []
    for o in f.objects[:20]:
        if o.get('ref'):
            out.append(o['ref'] + (f'.{o["pin"]}' if o.get('pin') else ''))
        elif o.get('kind'):
            out.append({'via': 'переходное', 'zone': 'зона', 'track': 'трасса', 'footprint': 'компонент'}.get(o['kind'], o['kind']))
    c = Counter(out)
    return ', '.join(f'{k} ×{n}' if n > 1 else k for k, n in c.items())


def _card(i, f, scene) -> str:
    sev = SEV_CLASS[f.severity]
    at = f'({f.pos[0]:.3f}; {-f.pos[1] + 0.0:.3f})'.replace('.', ',') if f.pos is not None else ''
    uids = ','.join(dict.fromkeys(o['uid'] for o in f.objects if o.get('uid')))
    refs = ','.join(dict.fromkeys(o['ref'] for o in f.objects if o.get('ref') and not o.get('uid')))
    crop = crop_svg(scene, f) if scene is not None else None
    text = ' '.join([f.cls, f.code, f.title, f.layer or '', ' '.join(n for n in f.nets if n), _objects(f)]).lower()
    meta = []
    if f.layer:
        meta.append(f'<span>{escape(f.layer)}</span>')
    if at:
        meta.append(f'<span title="координаты KiCad, мм">{at}</span>')
    if f.probability is not None and f.severity is not Severity.NOT_CHECKED:
        meta.append(f'<span title="вероятность на плату">P = {escape(_fmt(float(f.probability)))}</span>')
    if f.data_quality:
        meta.append(f'<span class="dq">{escape(f.data_quality)}</span>')
    nets = ', '.join(n for n in f.nets if n)
    objs = _objects(f)
    details = ''.join([
        f'<h4>Измерено</h4>{_kv(f.measured)}' if f.measured else '',
        f'<h4>Пороги</h4>{_kv(f.thresholds)}' if f.thresholds else '',
        ('<h4>Допущения</h4><ul>' + ''.join(f'<li>{escape(a)}</li>' for a in f.assumptions) + '</ul>') if f.assumptions else '',
        f'<p class="uids">uuid: {escape(uids)}</p>' if uids else '',
    ])
    return (
        f'<article class="card {sev}" id="f{i}" data-cls="{f.cls}" data-code="{f.code}" data-sev="{sev}" '
        f'data-rank="{int(f.severity)}" data-prob="{f.probability if f.probability is not None else -1}" '
        f'data-layer="{escape(f.layer or "")}" data-text="{escape(text)}" data-uids="{escape(uids)}">'
        f'<div class="body"><div class="head"><span class="badge {sev}">{escape(f.severity.ru)}</span>'
        f'<span class="code">{f.cls} {f.code}</span><span class="cname">{escape(CODE_NAMES.get(f.code, ""))}</span></div>'
        f'<h3>{escape(f.title)}</h3><div class="meta">{"".join(meta)}</div>'
        + (f'<p class="nets"><b>Цепи:</b> {escape(nets)}</p>' if nets else '')
        + (f'<p class="objs"><b>Объекты:</b> {escape(objs)}</p>' if objs else '')
        + (f'<p class="fix"><b>Исправление:</b> {escape(f.fix)}</p>' if f.fix else '')
        + (f'<p><button class="kicad" data-uids="{escape(uids)}" data-refs="{escape(refs)}">Показать в KiCad</button>'
           f'<span class="kres"></span></p>' if uids or refs else '')
        + (f'<details><summary>Подробности</summary>{details}</details>' if details else '')
        + '</div>' + (f'<div class="pic">{crop}</div>' if crop else '') + '</article>')


def _summary(findings) -> str:
    c = Counter((f.cls, f.code, f.severity) for f in findings)
    head = ''.join(f'<th class="{SEV_CLASS[s]}">{s.ru}</th>' for s in SEV_ORDER)
    rows = []
    for cls, code in sorted({(k[0], k[1]) for k in c}):
        cells = []
        for s in SEV_ORDER:
            n = c.get((cls, code, s), 0)
            cells.append(f'<td><button class="cnt {SEV_CLASS[s]}" data-code="{code}" data-sev="{SEV_CLASS[s]}">{n}</button></td>'
                         if n else '<td></td>')
        rows.append(f'<tr><td>{cls}</td><td><button class="lnk" data-code="{code}">{code}</button> '
                    f'{escape(CODE_NAMES.get(code, ""))}</td>{"".join(cells)}</tr>')
    return f'<table class="summary"><thead><tr><th>Класс</th><th>Дефект</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


ASSETS = Path(__file__).with_name('assets')
CSS = (ASSETS / 'report.css').read_text(encoding='utf-8')
JS = (ASSETS / 'report.js').read_text(encoding='utf-8')


def _finding_order(finding):
    """Сначала вердикт, внутри него — ожидаемый ущерб. Невыполненные проверки идут в конце."""
    loss = float(finding.measured.get('ожидаемый_ущерб', 0) or 0)
    unchecked = finding.severity is Severity.NOT_CHECKED
    return unchecked, -int(finding.severity), -loss, finding.cls, finding.code


def write_html(board, ctx, findings, path, pictures: bool = True) -> Path:
    path = Path(path)
    profile = ctx.profile
    # Идентификатор находки берём до сортировки. Тот же индекс использует метка на схеме.
    ordered = sorted(enumerate(findings), key=lambda item: _finding_order(item[1]))
    scene = BoardScene(ctx) if pictures and findings else None
    counts = Counter(f.severity for f in findings)
    totals = ''.join(f'<div class="tot {SEV_CLASS[s]}"><b>{counts[s]}</b> {s.ru}</div>' for s in SEV_ORDER)
    classes = sorted({f.cls for f in findings})
    copper_order = {layer: index for index, layer in enumerate(board.copper)}
    layers = sorted({f.layer for f in findings if f.layer},
                    key=lambda layer: (layer not in copper_order, copper_order.get(layer, 0), layer))
    prof_name = profile.get('meta', {}).get('name', profile.source)
    timing = ', '.join(f'{k} {v:.1f} с' for k, v in ctx.timings.items())
    src = getattr(board, 'path', None)
    chips_sev = ''.join(f'<button class="chip on" data-sev="{SEV_CLASS[s]}">{s.ru}</button>' for s in SEV_ORDER)
    chips_cls = ''.join(f'<button class="chip" data-cls="{k}" title="{escape(CLASS_NAMES.get(k, ""))}">{k}</button>' for k in classes)
    opts = ''.join(f'<option value="{escape(L)}">{escape(L)}</option>' for L in layers)
    cards = ''.join(_card(index, finding, scene) for index, finding in ordered)
    doc = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Проверка {escape(board.name)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<button class="chip theme" id="theme" title="тема: авто / светлая / тёмная">◐ тема</button>
<h1>Проверка технологичности: {escape(board.name)}</h1>
<div class="sub">{escape(str(src) if src else board.name)} · слоёв меди {len(board.copper)} · отверстий {len(board.holes)}
{(' · время: ' + escape(timing)) if timing else ''}</div>
<div class="sub">Профиль: {escape(str(prof_name))}</div>
{'<div class="warnline">Профиль условный: числа документов, а не данные завода. Вероятности и пороги — иллюстративные.</div>' if profile.illustrative else ''}
<div class="totals">{totals}</div>
{report_panel(board, profile) if src else ''}
<div class="grid"><section><h2>Сводка</h2>{_summary(findings) if findings else '<p class="empty">Находок нет.</p>'}
<p class="sub">Клик по коду — только этот дефект; по числу — дефект и вердикт.</p></section>
<section><h2>Плата</h2>{overview_svg(scene, findings) if scene else ''}<p class="sub">Метки — места находок (клик — к карточке).</p></section></div>
<h2>Находки</h2>
<div class="controls">{chips_sev}<span>·</span>{chips_cls}
<select id="layer" aria-label="слой"><option value="">все слои</option>{opts}</select>
<input id="q" type="search" placeholder="поиск: цепь, вывод, текст">
<select id="sort" aria-label="сортировка"><option value="sev">по вердикту</option><option value="prob">по вероятности</option>
<option value="code">по коду</option></select><button class="chip" id="reset">сбросить</button><span id="shown"></span></div>
<div id="list">{cards}</div><p id="nothing" class="empty" hidden>Под фильтр ничего не попало.</p>
<footer>Координаты — как в KiCad (мм, ось Y вниз). Картинка места: медь слоя (для маски — медь стороны и окна маски,
зелёные), отверстия, кружок — место находки; серым — за пределами платы.</footer>
</div><script>{JS}</script></body></html>"""
    path.write_text(doc, encoding='utf-8')
    return path
