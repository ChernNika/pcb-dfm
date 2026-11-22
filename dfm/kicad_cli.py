"""Обёртка над kicad-cli: поиск программы, выгрузка Gerber/сверловки, DRC.

Нужна для сверки ядра (пункт 1) и для пакетной обработки плат.
Путь можно задать переменной окружения KICAD_CLI; иначе ищется в PATH
и в стандартных папках установки KiCad 10 под Windows.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

_CANDIDATES = [
    Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'KiCad' / '10.0' / 'bin' / 'kicad-cli.exe',
    Path(r'C:\Program Files\KiCad\10.0\bin\kicad-cli.exe'),
]


def find() -> Path:
    env = os.environ.get('KICAD_CLI')
    if env and Path(env).exists():
        return Path(env)
    exe = shutil.which('kicad-cli')
    if exe:
        return Path(exe)
    for p in _CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError('kicad-cli не найден: установите KiCad 10 или задайте переменную KICAD_CLI')


def _run(*args) -> str:
    r = subprocess.run([str(find()), *map(str, args)], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if r.returncode != 0:
        raise RuntimeError(f'kicad-cli {" ".join(map(str, args))}\n{r.stdout}\n{r.stderr}')
    return r.stdout


def version() -> str:
    return _run('version').strip()


# ВАЖНО: перезаливка зон в KiCad 10 недетерминирована — один и тот же файл, перезалитый дважды,
# даёт заливку, отличающуюся на единицы–десятки мкм (проверено на демо-плате RoyalBlue54L-Feather).
# Поэтому для сверки зоны перезаливаются ОДИН раз (refill_zones), плата сохраняется,
# и дальше все выгрузки и наш разбор работают с этим сохранённым файлом.

def refill_zones(board) -> None:
    """Перезаливает все зоны и сохраняет плату на месте (через DRC с --save-board)."""
    board = Path(board)
    report = board.with_suffix('.refill_drc.json')
    _run('pcb', 'drc', '--refill-zones', '--save-board', '--format', 'json', '-o', report, board)
    report.unlink(missing_ok=True)


def export_gerbers(board, out_dir, check_zones=False) -> list[Path]:
    """Gerber X2 с атрибутами цепей и job-файлом.

    check_zones=True перезаливает устаревшие зоны перед выгрузкой — для сверки не использовать
    (см. примечание выше), вместо этого вызвать refill_zones() заранее."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ['pcb', 'export', 'gerbers', '--precision', '6', '-o', f'{out_dir}{os.sep}']
    if check_zones:
        args.insert(3, '--check-zones')
    _run(*args, board)
    return sorted(out_dir.glob('*'))


def export_drill(board, out_dir) -> list[Path]:
    """Excellon в мм, металлизированные и неметаллизированные отверстия раздельно."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run('pcb', 'export', 'drill', '--format', 'excellon', '--excellon-units', 'mm',
         '--excellon-separate-th', '-o', f'{out_dir}{os.sep}', board)
    return sorted(out_dir.glob('*.drl'))


def drc(board, out_json, refill=False, all_track_errors=False) -> dict:
    """DRC KiCad с отчётом в JSON (для сверки найденных нарушений).
    По умолчанию работает с заливкой, сохранённой в файле (см. примечание о перезаливке).
    all_track_errors: без этого ключа KiCad сообщает только ПЕРВОЕ нарушение каждой дорожки
    (на multichannel_mixer из-за этого пропадали нарушения в одной из четырёх копий канала)."""
    args = ['pcb', 'drc', '--format', 'json', '--severity-all', '--units', 'mm', '-o', out_json]
    if refill:
        args.append('--refill-zones')
    if all_track_errors:
        args.append('--all-track-errors')
    _run(*args, board)
    return json.loads(Path(out_json).read_text(encoding='utf-8'))
