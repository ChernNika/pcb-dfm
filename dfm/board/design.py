"""Модель платы, уровень D (проектный) — объекты KiCad.

Этого уровня нет в Gerber: там дорожка — уже просто полоса меди, а площадка — вспышка
апертуры. Уровень D нужен, чтобы
  * знать замысел проектировщика: ширину дорожки, класс цепи, настройки зоны, правила;
  * показывать находки в KiCad на конкретном объекте (по uuid);
  * строить уровень P из .kicad_pcb (kicad_build.py).

Координаты — в системе модели (мм, ось Y вверх, абсолютные координаты платы),
то есть уже не такие, как в файле KiCad (там Y вниз, а площадки — относительно компонента).
Исключение — формы площадок (PadShape и примитивы произвольных площадок): они описаны
в собственной системе площадки (центр площадки — начало координат, без поворота), а
поворот и сдвиг применяются при построении меди. Перевод обратно в координаты KiCad
(для плагина) — kicad_xy().
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

from .shapes import Contour, Pt

__all__ = ['Track', 'ArcTrack', 'Via', 'ViaType', 'PadType', 'PadShape', 'Drill', 'Pad',
           'ZoneConnection', 'Graphic', 'Text', 'Footprint', 'Zone', 'NetClass', 'DesignRules',
           'Design', 'kicad_xy']


def kicad_xy(p: Pt) -> Pt:
    """Точка модели → координаты KiCad (ось Y вниз)."""
    return (p[0], -p[1] + 0.0)


@dataclass(slots=True, eq=False)
class Track:
    """Прямой отрезок дорожки с круглыми концами."""
    start: Pt
    end: Pt
    width: float
    layer: str
    net: str | None
    uid: str | None = None
    locked: bool = False


@dataclass(slots=True, eq=False)
class ArcTrack:
    """Дорожка-дуга по трём точкам."""
    start: Pt
    mid: Pt
    end: Pt
    width: float
    layer: str
    net: str | None
    uid: str | None = None
    locked: bool = False


class ViaType(StrEnum):
    THROUGH = 'through'
    BLIND = 'blind'          # глухое или скрытое: в KiCad 9 оба записываются как «blind»
    MICRO = 'micro'          # микроотверстие (лазерное, между соседними слоями)


@dataclass(slots=True, eq=False)
class Via:
    pos: Pt
    size: float                 # диаметр площадки (по умолчанию на всех слоях)
    drill: float                # готовый диаметр отверстия
    layers: tuple[str, str]     # первый и последний медный слои (сверху вниз)
    via_type: ViaType
    net: str | None
    uid: str | None = None
    # Падстек KiCad 9+: mode 'normal' | 'front_inner_back' | 'custom';
    # sizes — диаметр площадки по слою ('F.Cu', 'Inner', 'B.Cu' или конкретный In1.Cu …)
    padstack_mode: str = 'normal'
    sizes: dict[str, float] = field(default_factory=dict)
    remove_unused_layers: bool = False    # убрать площадки на слоях, где нет подключений
    keep_end_layers: bool = True          # …но оставить на крайних слоях
    zone_layer_connections: tuple[str, ...] = ()   # слои, где площадка сохранена (её подключает зона)
    tenting: dict[str, bool | None] = field(default_factory=dict)    # {'front': True, …}; None — как в плате
    covering: dict[str, bool | None] = field(default_factory=dict)
    plugging: dict[str, bool | None] = field(default_factory=dict)
    filling: bool | None = None
    capping: bool | None = None
    free: bool = False
    locked: bool = False

    def size_on(self, layer: str, copper: list[str]) -> float:
        """Диаметр площадки на медном слое с учётом падстека. В режиме front_inner_back незаданный
        B.Cu берёт размер Inner (если он задан), незаданный Inner — основной размер; в режиме custom
        незаданные слои — основной размер. Так делает KiCad 10.0 (проверено на t04_padstacks)."""
        if layer in self.sizes:
            return self.sizes[layer]
        if self.padstack_mode == 'front_inner_back' and layer != copper[0]:
            return self.sizes.get('Inner', self.size)
        return self.size


class PadType(StrEnum):
    THRU_HOLE = 'thru_hole'      # металлизированное отверстие
    SMD = 'smd'
    CONNECT = 'connect'          # краевой разъём (без маски и пасты)
    NPTH = 'np_thru_hole'        # неметаллизированное отверстие


class ZoneConnection(IntEnum):
    """Как зона подключается к площадке (значения — как в файле KiCad)."""
    INHERITED = -1
    NONE = 0
    THERMAL = 1      # через термобарьер
    FULL = 2         # сплошное
    THT_THERMAL = 3  # термобарьер только для выводных, SMD — сплошное


@dataclass(slots=True, eq=False)
class PadShape:
    """Форма площадки на одном слое — в системе самой площадки (без поворота).

    shape: circle | rect | oval | roundrect | trapezoid | custom; прямоугольник со
    срезанными углами в KiCad — это roundrect/rect с chamfer_ratio > 0.
    offset — сдвиг формы относительно отверстия (в KiCad записан внутри (drill … (offset)));
    центр формы на плате = позиция площадки + поворот(offset)."""
    shape: str
    size: tuple[float, float]
    offset: Pt = (0.0, 0.0)
    roundrect_rratio: float = 0.0
    chamfer_ratio: float = 0.0
    chamfer: frozenset[str] = frozenset()       # top_left, top_right, bottom_left, bottom_right
    rect_delta: tuple[float, float] = (0.0, 0.0)   # трапеция
    anchor: str = 'circle'                      # опорная форма произвольной площадки
    primitives: list[Graphic] = field(default_factory=list)   # произвольная площадка
    thermal_bridge_width: float | None = None
    thermal_gap: float | None = None
    thermal_bridge_angle: float | None = None
    zone_connect: ZoneConnection = ZoneConnection.INHERITED


@dataclass(slots=True, eq=False)
class Drill:
    """Отверстие площадки. Овальное отверстие (паз) — size по двум осям, иначе круг size[0]."""
    size: tuple[float, float]
    oval: bool = False


@dataclass(slots=True, eq=False)
class Pad:
    number: str
    type: PadType
    pos: Pt                       # центр отверстия / опорная точка, координаты платы
    angle: float                  # полный угол поворота на плате (уже включает поворот компонента)
    shape: PadShape               # форма на наружном (или единственном) слое
    layers: tuple[str, ...]       # слои площадки (уже раскрытые: *.Cu → F.Cu, In1.Cu …)
    net: str | None
    footprint: str = ''           # позиционное обозначение компонента
    drill: Drill | None = None
    local_pos: Pt = (0.0, 0.0)    # позиция относительно компонента (как в файле, Y вниз)
    padstack_mode: str = 'normal'
    layer_shapes: dict[str, PadShape] = field(default_factory=dict)   # падстек: 'Inner', 'B.Cu', 'In2.Cu' …
    solder_mask_margin: float | None = None
    solder_paste_margin: float | None = None
    solder_paste_margin_ratio: float | None = None
    clearance: float | None = None
    remove_unused_layers: bool = False
    keep_end_layers: bool = True
    zone_layer_connections: tuple[str, ...] = ()
    tenting: dict[str, bool | None] = field(default_factory=dict)
    pintype: str | None = None
    pinfunction: str | None = None
    uid: str | None = None

    def shape_on(self, layer: str, copper: list[str]) -> PadShape:
        """Форма площадки на медном слое с учётом падстека (правила незаданных слоёв — как у
        Via.size_on: в front_inner_back незаданный B.Cu берёт форму Inner)."""
        if layer in self.layer_shapes:
            return self.layer_shapes[layer]
        if self.padstack_mode == 'front_inner_back' and copper and layer != copper[0]:
            return self.layer_shapes.get('Inner', self.shape)
        return self.shape

    @property
    def plated(self) -> bool:
        return self.type is PadType.THRU_HOLE


@dataclass(slots=True, eq=False)
class Graphic:
    """Графический объект: line, arc, circle, rect, poly, curve (кривая Безье).

    points: line/rect — (начало, конец); arc — (начало, середина, конец); circle — (центр, точка
    на окружности); curve — 4 точки Безье. poly — контур в contour (может содержать дуги).
    width — толщина линии (0 — без обводки); filled — заливка (для circle, rect, poly)."""
    kind: str
    layer: str
    width: float
    points: tuple[Pt, ...] = ()
    contour: Contour | None = None
    filled: bool = False
    net: str | None = None
    footprint: str | None = None
    uid: str | None = None


@dataclass(slots=True, eq=False)
class Text:
    """Текст на медном слое или слое маски (сам текст в медь пока не переводится — нужны шрифты KiCad)."""
    text: str
    layer: str
    pos: Pt
    footprint: str | None = None
    knockout: bool = False
    uid: str | None = None
    size: tuple[float, float] | None = None     # высота и ширина символа, мм (font size h w)
    thickness: float | None = None              # толщина линии шрифта, мм (None — не задана в файле)
    bold: bool = False


@dataclass(slots=True, eq=False)
class Footprint:
    ref: str
    value: str
    lib_id: str
    pos: Pt
    angle: float
    side: str                      # 'top' или 'bottom'
    pads: list[Pad] = field(default_factory=list)
    graphics: list[Graphic] = field(default_factory=list)   # в координатах платы
    zones: list[Zone] = field(default_factory=list)
    attrs: frozenset[str] = frozenset()     # smd, through_hole, dnp, board_only, exclude_from_bom …
    solder_mask_margin: float | None = None
    solder_paste_margin: float | None = None
    clearance: float | None = None
    zone_connect: ZoneConnection = ZoneConnection.INHERITED
    uid: str | None = None
    locked: bool = False


@dataclass(slots=True, eq=False)
class Zone:
    """Зона (заливка медью) или область правил (keepout).

    outline — контур, нарисованный проектировщиком (может содержать дуги).
    filled — залитая медь по слоям, как её рассчитал и сохранил KiCad: {слой: [контур]}.
    KiCad хранит заливку «разрезанной»: дырки соединены с внешним краем разрезами нулевой
    ширины, поэтому каждый контур — один замкнутый многоугольник без отдельных дырок."""
    net: str | None
    layers: tuple[str, ...]
    outline: list[Contour]
    filled: dict[str, list[Contour]] = field(default_factory=dict)
    name: str = ''
    priority: int = 0
    clearance: float | None = None
    min_thickness: float | None = None
    connect_pads: ZoneConnection = ZoneConnection.THERMAL
    thermal_gap: float | None = None
    thermal_bridge_width: float | None = None
    fill_mode: str = 'solid'                 # solid | hatch
    island_removal_mode: int = 0             # 0 — всегда удалять островки, 1 — никогда, 2 — меньше площади
    island_area_min: float | None = None     # мм²
    keepout: dict[str, bool] | None = None   # для области правил: {'tracks': запрещено?, …}
    teardrop: bool = False                   # зона-«капля» (teardrop)
    footprint: str | None = None
    uid: str | None = None

    @property
    def is_rule_area(self) -> bool:
        return self.keepout is not None


@dataclass(slots=True)
class NetClass:
    """Класс цепей из .kicad_pro. None — параметр в этом классе не задан."""
    name: str
    priority: int = 2 ** 31 - 1
    clearance: float | None = None
    track_width: float | None = None
    via_diameter: float | None = None
    via_drill: float | None = None
    microvia_diameter: float | None = None
    microvia_drill: float | None = None
    diff_pair_width: float | None = None
    diff_pair_gap: float | None = None

    FIELDS = ('clearance', 'track_width', 'via_diameter', 'via_drill', 'microvia_diameter',
              'microvia_drill', 'diff_pair_width', 'diff_pair_gap')


@dataclass(slots=True)
class DesignRules:
    """Правила проекта: ограничения платы (Board Setup → Constraints), классы цепей и
    параметры маски/пасты из .kicad_pcb. Пользовательские правила .kicad_dru не разбираются
    (has_custom_rules = True, если файл есть, — тогда возможны расхождения с DRC KiCad)."""
    min_clearance: float = 0.0
    min_track_width: float = 0.0
    min_connection: float = 0.0
    min_via_diameter: float = 0.0
    min_via_annular_width: float = 0.0
    min_through_hole_diameter: float = 0.0
    min_hole_to_hole: float = 0.0
    min_hole_clearance: float = 0.0
    min_copper_edge_clearance: float = 0.0
    min_microvia_diameter: float = 0.0
    min_microvia_drill: float = 0.0
    min_groove_width: float = 0.0
    solder_mask_to_copper_clearance: float = 0.0
    max_error: float = 0.005                  # допуск KiCad при переводе дуг в ломаные, мм
    allow_blind_buried_vias: bool = True
    allow_microvias: bool = True
    # из (setup …) в .kicad_pcb
    pad_to_mask_clearance: float = 0.0        # отступ окна маски от площадки по умолчанию
    solder_mask_min_width: float = 0.0
    pad_to_paste_clearance: float = 0.0
    pad_to_paste_clearance_ratio: float = 0.0
    allow_soldermask_bridges_in_footprints: bool = False
    tenting: dict[str, bool] = field(default_factory=lambda: {'front': True, 'back': True})
    zone_min_clearance: float | None = None
    netclasses: dict[str, NetClass] = field(default_factory=dict)
    netclass_patterns: list[tuple[str, str]] = field(default_factory=list)   # (шаблон, класс)
    netclass_assignments: dict[str, list[str]] = field(default_factory=dict)  # цепь → классы
    has_project_file: bool = False
    has_custom_rules: bool = False

    def netclass_of(self, net: str | None) -> NetClass:
        """Действующий класс цепи, как в KiCad 8+: цепь может попасть в несколько классов
        (явное назначение и шаблоны); каждый параметр берётся из класса с наивысшим
        приоритетом (меньшее число), где он задан, а недостающие — из класса Default."""
        default = self.netclasses.get('Default') or NetClass('Default')
        if not net:
            return default
        names = list(self.netclass_assignments.get(net, []))
        names += [cls for pat, cls in self.netclass_patterns if _match(pat, net)]
        classes = sorted({n: self.netclasses[n] for n in names if n in self.netclasses}.values(),
                         key=lambda c: c.priority)
        if not classes:
            return default
        if len(classes) == 1:
            eff = NetClass(classes[0].name, classes[0].priority)
        else:
            eff = NetClass(','.join(c.name for c in classes), classes[0].priority)
        for f in NetClass.FIELDS:
            v = next((getattr(c, f) for c in classes if getattr(c, f) is not None), None)
            setattr(eff, f, getattr(default, f) if v is None else v)
        return eff


def _match(pattern: str, net: str) -> bool:
    """Шаблон класса цепи: подстановочные знаки * и ? (с учётом регистра);
    шаблон вида /…/ — регулярное выражение (так их понимает KiCad)."""
    if len(pattern) > 2 and pattern.startswith('/') and pattern.endswith('/'):
        import re
        try:
            return re.search(pattern[1:-1], net) is not None
        except re.error:
            return False
    return fnmatch.fnmatchcase(net, pattern)


@dataclass(slots=True, eq=False)
class Design:
    """Всё, что прочитано из проекта KiCad."""
    version: int                         # версия формата файла (20241229 — KiCad 9, 20260206 — KiCad 10)
    generator_version: str
    footprints: list[Footprint] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    arcs: list[ArcTrack] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)      # зоны платы (зоны компонентов — в Footprint.zones)
    graphics: list[Graphic] = field(default_factory=list)   # графика платы (без графики компонентов)
    texts: list[Text] = field(default_factory=list)      # тексты на меди и маске
    rules: DesignRules = field(default_factory=DesignRules)
    layer_types: dict[str, str] = field(default_factory=dict)   # медь: signal / power / mixed / jumper
    layer_user_names: dict[str, str] = field(default_factory=dict)

    @property
    def pads(self) -> list[Pad]:
        return [p for fp in self.footprints for p in fp.pads]

    def all_graphics(self) -> list[Graphic]:
        """Графика платы и компонентов вместе."""
        return self.graphics + [g for fp in self.footprints for g in fp.graphics]

    def all_zones(self) -> list[Zone]:
        return self.zones + [z for fp in self.footprints for z in fp.zones]

    def footprint(self, ref: str) -> Footprint:
        for fp in self.footprints:
            if fp.ref == ref:
                return fp
        raise KeyError(ref)
