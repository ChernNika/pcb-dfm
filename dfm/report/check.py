"""Отчёт проверки: findings.json (для плагина и сверки), report.md (для человека), картинки слоёв
с отмеченными находками.

    write_report(board, ctx, findings, папка)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..rules.findings import Severity, to_json

__all__ = ['write_report', 'summary_lines']

_COLORS = {Severity.ERROR: '#C0392B', Severity.WARNING: '#E67E22', Severity.INFO: '#2E86C1',
           Severity.NOT_CHECKED: '#7F8C8D', Severity.OK: '#27AE60'}


def summary_lines(findings) -> list[str]:
    c = Counter((f.cls, f.code, f.severity) for f in findings)
    lines = ['| Класс | Дефект | Ошибка | Предупреждение | Информация | Невыполнено |', '|---|---|---|---|---|---|']
    for cls, code in sorted({(k[0], k[1]) for k in c}):
        n = lambda s: c.get((cls, code, s), 0) or ''
        lines.append(f'| {cls} | {code} | {n(Severity.ERROR)} | {n(Severity.WARNING)} | {n(Severity.INFO)} | '
                     f'{n(Severity.NOT_CHECKED)} |')
    return lines


def _md(board, ctx, findings) -> str:
    p = ctx.profile
    lines = [f'# Проверка технологичности: {board.name}', '',
             f'Профиль: {p.get("meta", {}).get("name", p.source)}'
             + (' — **условные числа документов, не данные завода**' if p.illustrative else ''), '',
             f'Находок: {len(findings)}. Время по классам: '
             + ', '.join(f'{k} {v:.1f} с' for k, v in ctx.timings.items()) + '.', '']
    if findings:
        lines += summary_lines(findings) + ['']
    for f in sorted(findings, key=lambda f: (-int(f.severity), -float(f.measured.get('ожидаемый_ущерб', 0) or 0),
                                             f.cls, f.code)):
        where = f' — {f.layer}' if f.layer else ''
        at = f' ({f.pos[0]:.3f}; {-f.pos[1] + 0.0:.3f})' if f.pos else ''
        lines.append(f'- **{f.cls} {f.code} · {f.severity.ru}**{where}{at}: {f.title}'
                     + (f'. Цепи: {", ".join(n for n in f.nets if n)}' if any(f.nets) else '')
                     + (f'. Исправление: {f.fix}' if f.fix else ''))
    lines += ['', 'Координаты — как в KiCad (мм, ось Y вниз).']
    return '\n'.join(lines) + '\n'


def write_report(board, ctx, findings, out_dir, pictures: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'findings.json').write_text(
        to_json(findings, board=board.name, profile=ctx.profile.source, illustrative=ctx.profile.illustrative,
                timings=ctx.timings), encoding='utf-8')
    (out / 'report.md').write_text(_md(board, ctx, findings), encoding='utf-8')
    from .html import write_html
    write_html(board, ctx, findings, out / 'report.html', pictures=pictures)
    if pictures and findings:
        _pictures(board, ctx, findings, out / 'layers')
    return out


def _pictures(board, ctx, findings, out_dir):
    from ..draw import COPPER, MASK, _frame, _save, geom_patch
    by_layer = {}
    for f in findings:
        if f.pos is not None:
            by_layer.setdefault(f.layer or board.copper[0], []).append(f)
    for layer, fs in by_layer.items():
        if layer not in board.layers:
            continue
        lc = ctx.geo.layer(layer)
        fig, ax = _frame(ctx.geo, f'{board.name}: {layer} — находок {len(fs)}')
        patch = geom_patch(lc.geom, facecolor=COPPER if layer in board.copper else MASK, edgecolor='none', alpha=0.55)
        if patch is not None:
            ax.add_patch(patch)
        for sev in sorted({f.severity for f in fs}):
            pts = [f.pos for f in fs if f.severity is sev]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=60, facecolors='none', edgecolors=_COLORS[sev],
                       lw=1.3, label=f'{sev.ru}: {len(pts)}')
        for f in fs:
            ax.annotate(f'{f.cls} {f.code}', f.pos, xytext=(4, 4), textcoords='offset points', fontsize=6,
                        color=_COLORS[f.severity])
        ax.legend(loc='upper right', fontsize=7)
        _save(fig, out_dir / f'{layer.replace(".", "_")}.png')
