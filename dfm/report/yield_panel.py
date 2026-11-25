"""Данные одной проверки для автономного раздела выхода годных в HTML-отчёте."""
import json
import tempfile
from html import escape
from pathlib import Path

from .. import kicad_cli
from ..board.gerber import classify_files
from ..plugin_install import plugin_source
from ..process_parameters import process_parameters


def _files(path):
    _, _, layers, drills, _ = classify_files(path)
    sources = []
    used = set()
    for file, kind, side, number, from_header in layers:
        if kind != 'Copper':
            continue
        if side in ('Top', 'Bot'):
            role = 'Top' if side == 'Top' else 'Bottom'
        else:
            role = f'In{number - int(from_header)}' if number is not None else '?'
        if role != '?' and role in used:
            continue
        used.add(role)
        sources.append((file, 'copper:' + role))
    files = [dict(name=file.name, text=file.read_text(encoding='utf-8', errors='replace'), role=role)
             for file, role in sources]
    for file in drills:
        files.append(dict(name=file.name, text=file.read_text(encoding='utf-8', errors='replace')))
    return files


def report_panel(board, profile):
    source = Path(board.path)
    try:
        if source.is_dir() or source.suffix.lower() == '.gbrjob':
            files = _files(source)
        else:
            with tempfile.TemporaryDirectory(prefix='dfm_yield_') as tmp:
                kicad_cli.export_gerbers(source, tmp, check_zones=True)
                kicad_cli.export_drill(source, tmp)
                files = _files(tmp)
        if not files:
            raise ValueError('нет файлов меди и сверловки')
    except (OSError, ValueError, RuntimeError) as error:
        return ('<section><h2>Выход годных плат</h2><p class="sub">Расчёт недоступен. '
                + escape(str(error)) + '</p></section>')

    ui = plugin_source() / 'ui'

    def asset(name):
        return (ui / name).read_text(encoding='utf-8')

    # Разметка и скрипты общие с окном ручной загрузки. Отчёт можно перенести одним файлом.
    _, marker, remainder = asset('gerber.html').partition('<!-- board-analysis -->')
    panel, end, _ = remainder.partition('<!-- /board-analysis -->')
    if not marker or not end:
        raise ValueError('нет шаблона раздела выхода годных')
    payload = json.dumps(dict(files=files, profile=process_parameters(profile)), ensure_ascii=False,
                         separators=(',', ':')).replace('<', '\\u003c')
    scripts = ''.join('<script>' + asset(name) + '</script>'
                      for name in ('via-view.js', 'via.js', 'gerber-engine.js', 'gerber.js'))
    return (f'<section id="yield-report"><h2>Выход годных плат</h2><div id="yield-panel"></div>'
            f'<template id="yield-template"><style>{asset("analysis-theme.css")}{asset("gerber.css")}</style>{panel}</template>'
            f'<template id="yield-via-style"><style>{asset("via.css")}</style></template>'
            f'<script type="application/json" id="yield-data">{payload}</script>'
            f'{scripts}</section>')
