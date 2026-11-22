"""Входные данные: адаптеры форматов (.kicad_pcb, Gerber X2 + Excellon) и единая модель платы.

Уровень P (производственный, обязательный): медь по слоям, отверстия, цепи, стек, маска, контур.
Уровень D (проектный, необязательный): объекты KiCad и правила проекта — только из .kicad_pcb.
"""
from pathlib import Path

from .model import Board, Feature, Hole, Layer, Role, Stackup, StackItem
from .layers import LayerKind, Side

__all__ = ['load', 'Board', 'Feature', 'Hole', 'Layer', 'Role', 'Stackup', 'StackItem', 'LayerKind', 'Side']


def load(path) -> Board:
    """Читает плату в единую модель. Формат определяется по пути:
    файл .kicad_pcb — адаптер KiCad; папка с Gerber (или её job-файл .gbrjob) — адаптер Gerber."""
    path = Path(path)
    if path.suffix == '.kicad_pcb':
        from .kicad import load_kicad
        board = load_kicad(path)
    elif path.is_dir() or path.suffix.lower() == '.gbrjob':
        from .gerber import load_gerber
        board = load_gerber(path)
    else:
        raise ValueError(f'неизвестный формат входных данных: {path}')
    board.path = path
    return board
