"""Уровень P из уровня D для платы KiCad: медь слоёв, окна маски, отверстия, контур.

    build_physical(board)            заполняет board.layers[*].features, board.holes, board.outline
    stitch_outline(graphics, tol)    контур платы из графики слоя Edge.Cuts

Что откуда берётся (так же KiCad выводит Gerber, проверено сверкой — tests/test_board_gerber.py):
  * медь слоя — площадки, переходные отверстия, дорожки и дуги, залитые зоны и графика на медном
    слое. Зоны берутся уже залитыми (filled_polygon): в заливке KiCad учёл зазоры, термобарьеры
    и удалённые островки, своя заливка здесь не строится;
  * площадка отверстия есть не на каждом слое — см. _Links и флажок remove_unused_layers;
  * окно маски — площадка, расширенная на отступ маски (отступ площадки → компонента → платы);
    переходное отверстие, если оно не закрыто маской (tenting); графика и зоны на слое маски;
  * текст на меди и маске в геометрию не переводится (для этого нужны шрифты KiCad) — об этом
    пишется предупреждение. В Gerber такой текст приходит уже готовой геометрией;
  * контур — линии, дуги, окружности, прямоугольники и многоугольники слоя Edge.Cuts (платы и
    компонентов), сшитые в замкнутые контуры. Чётная вложенность — плата, нечётная — вырезы.

Формы в Feature — точные (см. shapes.py); один и тот же Feature площадки лежит на слое и в
падстеке отверстия (Hole.padstack).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import Point

from ..geometry.polygons import to_geometry
from .design import Footprint, Graphic, Pad, PadShape, PadType, Via, ViaType
from .kicad_shapes import bezier_points, graphic_shape, hole_ends, pad_shape
from .layers import LayerKind, copper_span, kind_of
from .model import Board, Feature, Hole, Role
from .outline import CHAIN_TOL, stitch
from .shapes import Circle, Contour, Region, Stroke

__all__ = ['build_physical', 'stitch_outline', 'CHAIN_TOL']

_GEOM_TOL = 0.001       # мм: точность ломаных в проверках «точка внутри формы»


# ---------- подключения на слое (для удаления неиспользуемых площадок) ----------

class _Links:
    """Есть ли у площадки или переходного отверстия подключение на медном слое.

    Повторяет KiCad (PAD::FlashLayer, PCB_VIA::FlashLayer и граф связности): объект подключён на
    слое, если там есть объект ТОЙ ЖЕ цепи, у которого якорь лежит внутри формы нашего объекта,
    или наш якорь лежит внутри его формы. Якоря дорожки и дуги — их концы, площадки и
    переходного отверстия — центр отверстия. Для площадки учитываются дорожки, дуги, переходные
    отверстия и другие площадки; для переходного отверстия — дорожки, дуги и площадки.
    Подключение зоной KiCad сам определяет при заливке и сохраняет в zone_layer_connections."""

    def __init__(self, design, copper: list[str]):
        self.copper = copper
        self.tracks = defaultdict(list)       # (слой, цепь) → [(форма, концы)]
        for t in design.tracks:
            self.tracks[(t.layer, t.net)].append(
                (Stroke(np.array([t.start, t.end]), t.width), (t.start, t.end)))
        for a in design.arcs:
            self.tracks[(a.layer, a.net)].append(
                (Stroke(np.array([a.start, a.end]), a.width, {0: a.mid}), (a.start, a.end)))
        self.vias = defaultdict(list)          # цепь → [Via]
        for v in design.vias:
            self.vias[v.net].append(v)
        self.pads = defaultdict(list)          # цепь → [Pad]
        for p in design.pads:
            self.pads[p.net].append(p)
        self._trees = {}
        self._pad_geom = {}

    def _track_tree(self, layer, net):
        key = (layer, net)
        if key not in self._trees:
            items = self.tracks.get(key, [])
            geoms = [to_geometry(s, _GEOM_TOL) for s, _ in items]
            self._trees[key] = (shapely.STRtree(geoms) if geoms else None, geoms, items)
        return self._trees[key]

    def pad_geom(self, p: Pad, layer: str):
        key = (id(p), layer)
        if key not in self._pad_geom:
            s = pad_shape(p.shape_on(layer, self.copper), p.pos, p.angle)
            self._pad_geom[key] = to_geometry(s, _GEOM_TOL) if s is not None else None
        return self._pad_geom[key]

    def _tracks_touch(self, geom, anchor, layer, net) -> bool:
        tree, geoms, items = self._track_tree(layer, net)
        if tree is None:
            return False
        for i in tree.query(geom, predicate='intersects'):
            if any(shapely.intersects_xy(geom, *e) for e in items[i][1]):
                return True
            if shapely.intersects_xy(geoms[i], *anchor):
                return True
        return False

    def pad_connected(self, p: Pad, layer: str) -> bool:
        if p.net is None:
            return False
        geom = self.pad_geom(p, layer)
        if geom is None:
            return False
        if self._tracks_touch(geom, p.pos, layer, p.net):
            return True
        for v in self.vias[p.net]:
            if layer in _span(v, self.copper) and (
                    shapely.intersects_xy(geom, *v.pos)
                    or math.dist(v.pos, p.pos) <= v.size_on(layer, self.copper) / 2):
                return True
        for q in self.pads[p.net]:
            if q is p or layer not in q.layers:
                continue
            g = self.pad_geom(q, layer)
            if g is not None and (shapely.intersects_xy(geom, *q.pos) or shapely.intersects_xy(g, *p.pos)):
                return True
        return False

    def via_connected(self, v: Via, layer: str) -> bool:
        if v.net is None:
            return False
        geom = Point(v.pos).buffer(v.size_on(layer, self.copper) / 2, quad_segs=32)
        if self._tracks_touch(geom, v.pos, layer, v.net):
            return True
        for q in self.pads[v.net]:
            if layer not in q.layers:
                continue
            g = self.pad_geom(q, layer)
            if g is not None and (shapely.intersects_xy(geom, *q.pos) or shapely.intersects_xy(g, *v.pos)):
                return True
        return False


def _span(v: Via, copper: list[str]) -> list[str]:
    """Медные слои, через которые проходит переходное отверстие (сверху вниз)."""
    a, b = v.layers
    if a not in copper or b not in copper:
        return []
    return copper_span(a, b, copper)


# ---------- сборка ----------

class _Builder:
    def __init__(self, board: Board):
        self.b = board
        self.d = board.design
        self.cu = board.copper
        self.cuset = set(self.cu)
        self.rules = self.d.rules
        self._links = None

    @property
    def links(self) -> _Links:
        if self._links is None:            # строится только если на плате есть удаление площадок
            self._links = _Links(self.d, self.cu)
        return self._links

    def add(self, layer: str, f: Feature) -> bool:
        lay = self.b.layers.get(layer)
        if lay is None:                    # слоя нет в таблице слоёв — об этом сообщит validate()
            return False
        lay.features.append(f)
        return True

    # ----- площадки -----
    def pad_flashed(self, p: Pad, layer: str) -> bool:
        """Есть ли медь площадки на медном слое из её списка слоёв (как PAD::FlashLayer)."""
        if p.type is PadType.NPTH:
            # неметаллизированное отверстие не меньше площадки — меди нет
            ps = p.shape_on(layer, self.cu)
            if p.drill is not None and ps.offset == (0.0, 0.0):
                dx, dy = p.drill.size
                if ps.shape == 'circle' and not p.drill.oval and dx >= ps.size[0]:
                    return False
                if ps.shape == 'oval' and p.drill.oval and dx >= ps.size[0] and dy >= ps.size[1]:
                    return False
            return True
        if p.type is not PadType.THRU_HOLE or not p.remove_unused_layers:
            return True
        if p.keep_end_layers and layer in (self.cu[0], self.cu[-1]):
            return True
        if layer in p.zone_layer_connections:
            return True
        return self.links.pad_connected(p, layer)

    def mask_margin(self, p: Pad, fp: Footprint, cu_layer: str) -> float:
        """Отступ окна маски от площадки (как PAD::GetSolderMaskExpansion): свой у площадки,
        иначе у компонента, иначе у платы. Площадка без меди (только на маске) — без отступа.
        Отрицательный отступ не уменьшает площадку больше чем до нуля."""
        if not any(kind_of(x) is LayerKind.COPPER for x in p.layers):
            return 0.0
        m = p.solder_mask_margin
        if m is None:
            m = fp.solder_mask_margin
        if m is None:
            m = self.rules.pad_to_mask_clearance
        if m < 0:
            m = max(m, -min(p.shape_on(cu_layer, self.cu).size) / 2)
        return m

    def pads(self):
        front, back = self.cu[0], self.cu[-1]
        for fp in self.d.footprints:
            for p in fp.pads:
                meta = dict(uid=p.uid, component=fp.ref, pin=p.number)
                stack = {}
                for layer in p.layers:
                    kind = kind_of(layer)
                    if kind is LayerKind.COPPER:
                        if layer not in self.cuset or not self.pad_flashed(p, layer):
                            continue
                        s = pad_shape(p.shape_on(layer, self.cu), p.pos, p.angle)
                        if s is not None:
                            f = Feature(s, p.net, Role.PAD, **meta)
                            self.add(layer, f)
                            stack[layer] = f
                    elif kind is LayerKind.MASK:
                        # Форма окна маски (KiCad 10.0, проверено по Gerber на t04_padstacks и
                        # синтетической плате): у падстека front_inner_back нижнее окно строится по
                        # форме B.Cu; у падстека custom оба окна — КРУГ диаметром по ширине основной
                        # площадки, в центре отверстия (форма и смещение основной площадки не учитываются).
                        cu_layer = front if layer.startswith('F.') else back
                        if p.padstack_mode == 'custom':
                            main = p.shape
                            ps = PadShape('circle', (main.size[0], main.size[0]))      # в центре отверстия
                        else:
                            shape_layer = back if (cu_layer == back and p.padstack_mode == 'front_inner_back') \
                                else front
                            ps = p.shape_on(shape_layer, self.cu)
                        s = pad_shape(ps, p.pos, p.angle, self.mask_margin(p, fp, cu_layer))
                        if s is not None:
                            self.add(layer, Feature(s, p.net, Role.PAD, **meta))
                if p.drill is not None:
                    start, end, dia = hole_ends(p.drill, p.pos, p.angle)
                    self.b.holes.append(Hole(start, dia, p.plated, (front, back), p.net, Role.PAD, end,
                                             stack, **meta))

    # ----- переходные отверстия -----
    def via_flashed(self, v: Via, layer: str, span: list[str]) -> bool:
        """Есть ли площадка переходного отверстия на слое (как PCB_VIA::FlashLayer)."""
        if not v.remove_unused_layers:
            return True
        if v.keep_end_layers and layer in (span[0], span[-1]):
            return True
        if layer in v.zone_layer_connections:
            return True
        return self.links.via_connected(v, layer)

    def vias(self):
        r = self.rules
        for v in self.d.vias:
            span = _span(v, self.cu)
            stack = {}
            for layer in span:
                if self.via_flashed(v, layer, span):
                    f = Feature(Circle(v.pos, v.size_on(layer, self.cu) / 2), v.net, Role.VIA, uid=v.uid)
                    self.add(layer, f)
                    stack[layer] = f
            self.b.holes.append(Hole(v.pos, v.drill, True, (span[0], span[-1]) if span else v.layers,
                                     v.net, Role.VIA, padstack=stack, uid=v.uid,
                                     microvia=v.via_type == ViaType.MICRO))
            # окно маски — только на наружной стороне, до которой доходит отверстие
            for side, cu_layer, mask in (('front', self.cu[0], 'F.Mask'), ('back', self.cu[-1], 'B.Mask')):
                if cu_layer not in span:
                    continue
                tented = v.tenting.get(side)
                if tented is None:
                    tented = r.tenting.get(side, True)
                if tented:
                    continue
                radius = v.size_on(cu_layer, self.cu) / 2 + r.pad_to_mask_clearance
                if radius > 0:
                    self.add(mask, Feature(Circle(v.pos, radius), v.net, Role.VIA, uid=v.uid))

    # ----- дорожки, зоны, графика -----
    def tracks(self):
        for t in self.d.tracks:
            if t.layer in self.cuset:
                pts = np.array([t.start] if t.start == t.end else [t.start, t.end])
                self.add(t.layer, Feature(Stroke(pts, t.width), t.net, Role.TRACK, uid=t.uid))
        for a in self.d.arcs:
            if a.layer in self.cuset:
                s = Stroke(np.array([a.start]), a.width) if a.start == a.end else \
                    Stroke(np.array([a.start, a.end]), a.width, {0: a.mid})
                self.add(a.layer, Feature(s, a.net, Role.TRACK, uid=a.uid))

    def zones(self):
        for z in self.d.all_zones():
            if z.is_rule_area:
                continue
            for layer, contours in z.filled.items():
                kind = kind_of(layer)
                if kind not in (LayerKind.COPPER, LayerKind.MASK):
                    continue
                net = z.net if kind is LayerKind.COPPER else None
                for c in contours:
                    if len(c) >= 3:
                        self.add(layer, Feature(Region(c), net, Role.ZONE, uid=z.uid, component=z.footprint))

    def graphics(self) -> list[Graphic]:
        """Графика на меди и маске → объекты слоёв; возвращает графику контура платы."""
        edge = []
        for g in self.d.all_graphics():
            kind = kind_of(g.layer)
            if kind is LayerKind.OUTLINE:
                edge.append(g)
                continue
            if kind not in (LayerKind.COPPER, LayerKind.MASK):
                continue
            s = graphic_shape(g)
            if s is not None:
                net = g.net if kind is LayerKind.COPPER else None
                self.add(g.layer, Feature(s, net, Role.GRAPHIC, uid=g.uid, component=g.footprint))
        return edge


def build_physical(board: Board) -> None:
    """Строит уровень P платы из её уровня D (board.design)."""
    if board.design is None or not board.copper:
        return
    bld = _Builder(board)
    bld.pads()
    bld.vias()
    bld.tracks()
    bld.zones()
    edge = bld.graphics()
    board.outline, problems = stitch_outline(edge)
    board.warnings.extend(problems)
    texts = board.design.texts
    if texts:
        on_cu = sum(kind_of(t.layer) is LayerKind.COPPER for t in texts)
        board.warnings.append(
            f'текст на меди ({on_cu} шт.) и маске ({len(texts) - on_cu} шт.) не переведён в геометрию '
            f'(нужны шрифты KiCad)')


# ---------- контур платы ----------

def _piece(g: Graphic):
    """Графика контура → ('closed', Contour) или ('open', (точки, дуги)), None — пропустить."""
    k = g.kind
    if k == 'line':
        a, b = g.points
        return None if a == b else ('open', (np.array([a, b], dtype=float), {}))
    if k == 'arc':
        s, m, e = g.points
        return None if s == e else ('open', (np.array([s, e], dtype=float), {0: m}))
    if k == 'curve' and len(g.points) == 4:
        return 'open', (bezier_points(*g.points), {})
    if k == 'circle':
        (cx, cy), e = g.points
        r = math.dist((cx, cy), e)
        if r <= 0:
            return None
        return 'closed', Contour(np.array([(cx + r, cy), (cx - r, cy)]), {0: (cx, cy + r), 1: (cx, cy - r)})
    if k == 'poly' and g.contour is not None and (len(g.contour) >= 3 or g.contour.arcs):
        return 'closed', g.contour
    return None


def stitch_outline(graphics, tol: float = CHAIN_TOL) -> tuple[list[Region], list[str]]:
    """Контур платы из графики слоя Edge.Cuts → (области платы с вырезами, замечания)."""
    closed, pieces = [], []
    for g in graphics:
        r = _piece(g)
        if r is None:
            continue
        (closed if r[0] == 'closed' else pieces).append(r[1])
    return stitch(closed, pieces, tol)
