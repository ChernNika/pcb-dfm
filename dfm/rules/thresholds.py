"""Технологические пороги ширины и зазора по слоям (документ A1, раздел 5).

    th = layer_thresholds(board, profile)      {слой: LayerThresholds}
    th['F.Cu'].clearance.target, th['In1.Cu'].width.target   (мм)

Зазор (минимум, который протравится, формула 2):    s_min = g_res + 2t/EF
Ширина с разбросом фактора травления (формулы 4–5):  w_min = w_fin + 2δ,  δ = t/(EF − ΔEF) − t/EF
t — толщина меди слоя: фольга из стека платы, у наружных слоёв плюс гальваника (формула 3).
Если в профиле есть таблица завода «толщина → зазор», зазор берётся линейной интерполяцией по ней;
вне таблицы — по прямой МНК через её точки (с пометкой «экстраполяция», документ A1, раздел 5.5).

Уточнённая модель (если в профиле задан slot_aspect_max; отчёт «Исправление спорных параметров DFM»):
    s_min = max(g_res, (t_e + t_r)/AR) + 2·t_e/EF + m
t_e — стравливаемая медь (фольга, у наружных слоёв плюс outer_etched_plating_um), t_r — толщина резиста,
AR — предельное отношение (медь + резист)/щель (узкая щель не протравливается), m — запас завода. Модель
формулы 2 ошибается против опубликованных таблиц 11 заводов на 39 мкм (с. к. о.), эта — на 12 мкм.
Таблицы clearance_table_inner_um / clearance_table_outer_um — по ФОЛЬГЕ (как публикуют заводы), вне таблицы —
моделью с запасом m, подогнанным под таблицу.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profile import Profile

__all__ = ['Threshold', 'LayerThresholds', 'layer_thresholds', 'copper_thickness_um', 'copper_foil_um',
           'etched_copper_um']

DEFAULT_FOIL_UM = 35.0


@dataclass(frozen=True)
class Threshold:
    target: float                # мм — порог целевого уровня
    level_a: float | None = None
    level_c: float | None = None
    source: str = 'модель'       # модель / таблица / экстраполяция / профиль
    note: str = ''

    def as_dict(self) -> dict:
        d = {'цел': round(self.target, 6)}
        if self.level_a is not None:
            d['A'] = self.level_a
        if self.level_c is not None:
            d['C'] = self.level_c
        return d


@dataclass(frozen=True)
class LayerThresholds:
    layer: str
    copper_um: float
    outer: bool
    clearance: Threshold
    width: Threshold


def copper_thickness_um(board, layer: str, profile: Profile) -> tuple[float, bool, bool]:
    """Толщина меди слоя, мкм → (толщина, наружный ли, взята ли из стека)."""
    cu = board.copper
    outer = layer in (cu[0], cu[-1])
    foil, from_stack = DEFAULT_FOIL_UM, False
    if board.stackup is not None:
        for item in board.stackup.copper:
            if item.name == layer and item.thickness:
                foil, from_stack = item.thickness * 1000, not board.stackup.synthetic
    etch = profile['etch']
    t = foil + (etch['plating_um'] if outer and not etch.get('outer_includes_plating') else 0.0)
    return t, outer, from_stack


def copper_foil_um(board, layer: str) -> tuple[float, bool]:
    """Толщина фольги слоя по стеку, мкм → (толщина, взята ли из стека)."""
    if board.stackup is not None:
        for item in board.stackup.copper:
            if item.name == layer and item.thickness:
                return item.thickness * 1000, not board.stackup.synthetic
    return DEFAULT_FOIL_UM, False


def etched_copper_um(board, layer: str, profile: Profile) -> float:
    """Стравливаемая медь слоя, мкм (от неё подтрав u = t/EF): в уточнённой модели (slot_aspect_max в профиле) —
    фольга плюс у наружных слоёв outer_etched_plating_um (гальваника под маской травления не вся
    стравливается); в модели формулы 3 — вся толщина слоя. Общая для A1, B1 и Д9."""
    etch = profile['etch']
    t, outer, _ = copper_thickness_um(board, layer, profile)
    if 'slot_aspect_max' not in etch:
        return t
    foil, _ = copper_foil_um(board, layer)
    if outer and etch.get('outer_includes_plating'):
        foil = max(foil - etch.get('plating_um', 0.0), 0.0)
    return foil + (etch.get('outer_etched_plating_um', etch.get('plating_um', 0.0)) if outer else 0.0)


def _model_clearance(etch, t_e: float, margin: float | None = None) -> float:
    """Уточнённая модель зазора, мкм (см. описание модуля)."""
    m = etch.get('process_margin_um', 0.0) if margin is None else margin
    gap = max(etch['resist_resolution_um'], (t_e + etch.get('resist_thickness_um', 35.0)) / etch['slot_aspect_max'])
    return gap + 2 * t_e / etch['etch_factor'] + m


def _table_model(table, etch, t_e_of, foil: float) -> tuple[float, str]:
    """Таблица по фольге: внутри — интерполяция, вне — модель с запасом, подогнанным под таблицу."""
    pts = np.array(sorted(table), dtype=float)
    fs, ss = pts[:, 0], pts[:, 1]
    if fs[0] <= foil <= fs[-1]:
        return float(np.interp(foil, fs, ss)), 'таблица'
    m_fit = float(np.mean([s_ - _model_clearance(etch, t_e_of(f_), 0.0) for f_, s_ in zip(fs, ss)]))
    return _model_clearance(etch, t_e_of(foil), m_fit), 'экстраполяция моделью'


def _table_clearance(table, t: float) -> tuple[float, str]:
    pts = np.array(sorted(table), dtype=float)
    ts, ss = pts[:, 0], pts[:, 1]
    if ts[0] <= t <= ts[-1]:
        return float(np.interp(t, ts, ss)), 'таблица'
    k, a = np.polyfit(ts, ss, 1)                 # МНК: s = a + k·t
    return float(a + k * t), 'экстраполяция'


def layer_thresholds(board, profile: Profile) -> dict[str, LayerThresholds]:
    etch = profile['etch']
    ef, d_ef = etch['etch_factor'], etch.get('etch_factor_tolerance', 0.0)
    out = {}
    lvl_a, lvl_c = etch.get('level_A', {}), etch.get('level_C', {})
    refined = 'slot_aspect_max' in etch
    for layer in board.copper:
        t, outer, from_stack = copper_thickness_um(board, layer, profile)
        note = '' if from_stack else f'толщина меди не задана в стеке — принято {DEFAULT_FOIL_UM:g} мкм фольги'
        foil, _ = copper_foil_um(board, layer)
        if outer and etch.get('outer_includes_plating'):
            foil = max(foil - etch.get('plating_um', 0.0), 0.0)
        etched = etch.get('outer_etched_plating_um', etch.get('plating_um', 0.0)) if outer else 0.0
        t_e_of = lambda f_, e_=etched: f_ + e_          # стравливаемая медь при фольге f_
        table = etch.get('clearance_table_outer_um' if outer else 'clearance_table_inner_um')
        if table and refined:
            s, src = _table_model(table, etch, t_e_of, foil)
        elif etch.get('clearance_table_um'):
            s, src = _table_clearance(etch['clearance_table_um'], t)
        elif refined:
            s, src = _model_clearance(etch, t_e_of(foil)), 'модель (уточнённая)'
        else:
            s, src = etch['resist_resolution_um'] + 2 * t / ef, 'модель'
        t_d = etched_copper_um(board, layer, profile)         # подтрав — по стравливаемой меди
        delta = t_d / (ef - d_ef) - t_d / ef if d_ef else 0.0
        w = etch['finished_width_min_um'] + 2 * delta
        mm = lambda v: v / 1000 if v is not None else None
        out[layer] = LayerThresholds(
            layer, t, outer,
            Threshold(s / 1000, mm(lvl_a.get('clearance_um')), mm(lvl_c.get('clearance_um')), src, note),
            Threshold(w / 1000, mm(lvl_a.get('width_um')), mm(lvl_c.get('width_um')), 'модель', note))
    return out
