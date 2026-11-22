"""Имена и порядок слоёв.

В модели слои называются так же, как в KiCad (F.Cu, In1.Cu … B.Cu, F.Mask, Edge.Cuts):
эти имена однозначны и понятны. Адаптер Gerber переводит в них функции файлов X2
(Copper,L1,Top → F.Cu; Copper,L2,Inr → In1.Cu; Soldermask,Top → F.Mask и т. д.).

Порядок медных слоёв определяется по имени, а не по номеру из файла: номера слоёв
в KiCad 9 поменялись (F.Cu = 0, B.Cu = 2, In1.Cu = 4 …; до KiCad 9 было B.Cu = 31).
"""
from __future__ import annotations

import re
from functools import lru_cache
from enum import StrEnum

__all__ = ['LayerKind', 'Side', 'kind_of', 'side_of', 'copper_order', 'copper_index',
           'expand', 'copper_span']


class LayerKind(StrEnum):
    COPPER = 'copper'
    MASK = 'mask'          # паяльная маска (в модели — окна маски)
    PASTE = 'paste'        # паста (трафарет)
    SILK = 'silk'          # шелкография
    OUTLINE = 'outline'    # контур платы (Edge.Cuts)
    OTHER = 'other'        # чертёжные, курт-ярды, слои пользователя и т. п.


class Side(StrEnum):
    TOP = 'top'
    BOTTOM = 'bottom'
    INNER = 'inner'


_INNER = re.compile(r'In(\d+)\.Cu\Z')
_SUFFIX_KIND = {'Cu': LayerKind.COPPER, 'Mask': LayerKind.MASK, 'Paste': LayerKind.PASTE,
                'SilkS': LayerKind.SILK, 'Cuts': LayerKind.OUTLINE}


@lru_cache(maxsize=None)
def kind_of(name: str) -> LayerKind:
    suffix = name.rsplit('.', 1)[-1]
    return _SUFFIX_KIND.get(suffix, LayerKind.OTHER)


def side_of(name: str) -> Side | None:
    if name.startswith('F.'):
        return Side.TOP
    if name.startswith('B.'):
        return Side.BOTTOM
    if _INNER.match(name):
        return Side.INNER
    return None


def _copper_key(name: str) -> int:
    if name == 'F.Cu':
        return 0
    if name == 'B.Cu':
        return 10 ** 6
    m = _INNER.match(name)
    if not m:
        raise ValueError(f'не медный слой: {name}')
    return int(m.group(1))


def copper_order(names) -> list[str]:
    """Медные слои из списка имён — сверху вниз."""
    return sorted((n for n in names if kind_of(n) is LayerKind.COPPER), key=_copper_key)


def copper_index(name: str, copper: list[str]) -> int:
    return copper.index(name)


def expand(names, board_layers) -> list[str]:
    """Раскрывает обозначения KiCad «*.Cu», «F&B.Cu», «*.Mask» в имена слоёв платы.

    board_layers — все слои платы (порядок сохраняется). «*.Cu» даёт все медные сверху вниз,
    «F&B.Cu» — только наружные. Имена, которых нет на плате, сохраняются как есть:
    решать, ошибка ли это, должна проверка модели (Board.validate)."""
    board_layers = list(board_layers)
    copper = copper_order(board_layers)
    out = []
    for n in names:
        if n == '*.Cu':
            out += copper
        elif n.startswith('*.'):
            suf = n[1:]
            out += [l for l in board_layers if l.endswith(suf)]
        elif n.startswith('F&B.'):
            suf = n[3:]
            out += ['F' + suf, 'B' + suf]
        else:
            out.append(n)
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def copper_span(first: str, last: str, copper: list[str]) -> list[str]:
    """Медные слои от first до last включительно (в любом порядке аргументов), сверху вниз."""
    i, j = copper.index(first), copper.index(last)
    if i > j:
        i, j = j, i
    return copper[i:j + 1]
