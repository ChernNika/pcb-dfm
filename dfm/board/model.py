"""Модель платы, уровень P (производственный).

Уровень P — то, что получает завод и что можно восстановить из ЛЮБОГО формата:
.kicad_pcb, Gerber X2 + Excellon + .gbrjob, IPC-2581. Понятия взяты из IPC-2581
(стек слоёв, медь слоя с цепями, отверстия с падстеками, маска, контур).

    Board
      ├─ stackup   Stackup        слои меди и диэлектрика: толщины, материалы, εr
      ├─ layers    {имя: Layer}   медь, маска, паста, … — список Feature на каждом слое
      ├─ holes     [Hole]         отверстия: место, диаметр, металлизация, слои, падстек
      ├─ outline   [Region]       контур платы с вырезами
      ├─ nets      {имя цепи}
      └─ design    Design | None  уровень D (только из KiCad), см. design.py

Правила, которым подчиняются все адаптеры:
  * координаты — мм, ось Y вверх; углы — градусы против часовой стрелки;
  * диаметр отверстия — готовый (после металлизации), как в KiCad и в Excellon от KiCad;
  * Feature на слое маски — это ОКНО в маске (так же, как тёмное изображение в Gerber маски);
  * net = None — у объекта нет цепи (или формат её не сообщает).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .layers import LayerKind, Side, copper_order, kind_of, side_of
from .shapes import Pt, Shape

__all__ = ['Role', 'Feature', 'Layer', 'StackItem', 'Stackup', 'Hole', 'HoleKind', 'Board']


class Role(StrEnum):
    """Роль объекта на слое. Совпадает по смыслу с атрибутом .AperFunction Gerber X2."""
    PAD = 'pad'            # площадка компонента (SMD или выводная)
    VIA = 'via'            # площадка переходного отверстия
    TRACK = 'track'        # проводник (дорожка, дуга)
    ZONE = 'zone'          # заливка зоны
    GRAPHIC = 'graphic'    # графика, нарисованная прямо на слое
    TEXT = 'text'          # текст на слое
    OTHER = 'other'


@dataclass(slots=True, eq=False)
class Feature:
    """Один объект на слое: точная форма + цепь + роль.

    dark=False — «светлая» полярность Gerber (вычитает уже нарисованное); объекты слоя
    применяются по порядку. KiCad почти всегда даёт только dark=True.
    uid — ссылка на объект уровня D (uuid KiCad), если он есть.
    component, pin — позиционное обозначение и номер вывода (для площадок)."""
    shape: Shape
    net: str | None = None
    role: Role = Role.OTHER
    dark: bool = True
    uid: str | None = None
    component: str | None = None
    pin: str | None = None


@dataclass(slots=True, eq=False)
class Layer:
    name: str
    kind: LayerKind
    side: Side | None
    features: list[Feature] = field(default_factory=list)

    @classmethod
    def named(cls, name: str) -> Layer:
        return cls(name, kind_of(name), side_of(name))


class StackKind(StrEnum):
    COPPER = 'copper'
    DIELECTRIC = 'dielectric'
    MASK = 'mask'
    PASTE = 'paste'
    SILK = 'silk'


@dataclass(slots=True)
class StackItem:
    """Слой физического стека (сверху вниз)."""
    name: str                       # F.Cu, dielectric 1, F.Mask …
    kind: StackKind
    thickness: float | None = None  # мм
    material: str | None = None     # FR4, …
    epsilon_r: float | None = None
    loss_tangent: float | None = None
    dielectric_type: str | None = None   # core / prepreg (для диэлектрика)
    color: str | None = None


@dataclass(slots=True)
class Stackup:
    items: list[StackItem]
    finish: str | None = None       # покрытие меди (ENIG, HAL …)
    synthetic: bool = False         # True — стека в файле не было, построен по умолчанию

    @property
    def copper(self) -> list[StackItem]:
        return [i for i in self.items if i.kind is StackKind.COPPER]

    @property
    def thickness(self) -> float:
        """Сумма известных толщин (медь + диэлектрик + маска)."""
        return sum(i.thickness or 0 for i in self.items
                   if i.kind in (StackKind.COPPER, StackKind.DIELECTRIC, StackKind.MASK))


class HoleKind(StrEnum):
    THROUGH = 'through'      # сквозное
    BLIND = 'blind'          # глухое: один конец на наружном слое
    BURIED = 'buried'        # скрытое: оба конца на внутренних слоях


@dataclass(slots=True, eq=False)
class Hole:
    """Отверстие.

    span — первый и последний медный слои (сверху вниз), между которыми проходит отверстие;
    у неметаллизированных — вся плата.
    slot_end — второй конец для овального отверстия (паза); тогда diameter — его ширина.
    padstack — площадка на каждом медном слое внутри span (те же объекты Feature, что лежат
    на слоях). Отсутствие слоя в словаре = площадки на этом слое нет (например, удалена
    как неиспользуемая) — это важно для пояска (A3) и антипадов (A5)."""
    pos: Pt
    diameter: float
    plated: bool
    span: tuple[str, str]
    net: str | None = None
    role: Role = Role.VIA            # VIA или PAD (отверстие вывода/крепёжное)
    slot_end: Pt | None = None
    padstack: dict[str, Feature] = field(default_factory=dict)
    uid: str | None = None
    component: str | None = None
    pin: str | None = None
    microvia: bool | None = None     # лазерное микроотверстие (KiCad: тип переходного micro); None — неизвестно (Gerber)

    def kind(self, copper: list[str]) -> HoleKind:
        top = self.span[0] == copper[0]
        bottom = self.span[1] == copper[-1]
        if top and bottom:
            return HoleKind.THROUGH
        return HoleKind.BLIND if (top or bottom) else HoleKind.BURIED


@dataclass(slots=True, eq=False)
class Board:
    name: str
    source: str                                   # 'kicad', 'gerber', …
    layers: dict[str, Layer] = field(default_factory=dict)
    stackup: Stackup | None = None
    holes: list[Hole] = field(default_factory=list)
    outline: list = field(default_factory=list)   # [Region]
    nets: set[str] = field(default_factory=set)
    thickness: float | None = None                # толщина платы, мм
    design: object | None = None                  # Design (уровень D) или None
    path: object | None = None                    # путь к исходному файлу или папке Gerber
    warnings: list[str] = field(default_factory=list)

    @property
    def copper(self) -> list[str]:
        """Медные слои сверху вниз."""
        return copper_order(self.layers)

    def layer(self, name: str) -> Layer:
        return self.layers[name]

    def add_layer(self, name: str) -> Layer:
        if name not in self.layers:
            self.layers[name] = Layer.named(name)
        return self.layers[name]

    def validate(self) -> list[str]:
        """Проверка внутренней согласованности модели. Возвращает список проблем (пустой — всё в порядке).
        Ничего не исправляет: это работа адаптеров, здесь — только диагностика."""
        from .validate import validate_board
        return validate_board(self)

    def summary(self) -> dict:
        from .validate import summarize
        return summarize(self)
