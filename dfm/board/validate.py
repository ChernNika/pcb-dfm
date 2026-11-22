"""Проверка согласованности модели платы и краткая сводка.

Это не DFM-правила, а контроль данных: всё ли, на что ссылаются объекты, существует,
нет ли невозможных размеров. Одинаковые проблемы собираются в одну строку с числом
случаев и первым примером — чтобы 300 одинаковых замечаний не заслоняли остальное.
"""
from __future__ import annotations

import math
from collections import Counter

from .layers import LayerKind

__all__ = ['validate_board', 'summarize']


class _Issues:
    def __init__(self):
        self.items: dict[str, list[str]] = {}

    def add(self, msg: str, where: str = ''):
        self.items.setdefault(msg, []).append(where)

    def lines(self) -> list[str]:
        out = []
        for msg, wh in self.items.items():
            ex = next((w for w in wh if w), '')
            n = f' — {len(wh)} шт.' if len(wh) > 1 else ''
            out.append(f'{msg}{n}' + (f' (например, {ex})' if ex else ''))
        return out


def _at(p) -> str:
    return f'({p[0]:.4f}; {p[1]:.4f})'


def _finite(*vals) -> bool:
    return all(math.isfinite(v) for v in vals)


def validate_board(b) -> list[str]:
    iss = _Issues()
    copper = b.copper
    names = set(b.layers)
    if not copper:
        iss.add('на плате нет медных слоёв')
    if b.stackup is not None and not b.stackup.synthetic:
        sc = [i.name for i in b.stackup.copper]
        if sc != copper:
            iss.add(f'медные слои стека {sc} не совпадают со слоями платы {copper}')
        for i in b.stackup.items:
            if i.thickness is not None and i.thickness <= 0:
                iss.add('слой стека с нулевой толщиной', i.name)
    if b.thickness is not None and b.stackup is not None and b.stackup.thickness > 0:
        if abs(b.stackup.thickness - b.thickness) > 0.05 * b.thickness:
            iss.add(f'толщина платы {b.thickness} мм отличается от суммы толщин стека '
                    f'{b.stackup.thickness:.3f} мм больше чем на 5 %')

    # уровень P
    for h in b.holes:
        if not h.diameter > 0:
            iss.add('отверстие с неположительным диаметром', _at(h.pos))
        if h.span[0] not in copper or h.span[1] not in copper:
            iss.add('отверстие ссылается на несуществующий медный слой', f'{h.span} в {_at(h.pos)}')
        for lay in h.padstack:
            if lay not in copper:
                iss.add('падстек содержит не медный слой', lay)
    if not b.outline:
        iss.add('контур платы не построен (см. предупреждения чтения)')
    for lay in b.layers.values():
        for f in lay.features:
            if f.net is not None and f.net not in b.nets:
                iss.add('объект слоя ссылается на неизвестную цепь', f'{lay.name}: {f.net}')

    d = b.design
    if d is None:
        return iss.lines()

    def layer_ok(name, what, where):
        if name not in names:
            iss.add(f'{what} на несуществующем слое {name}', where)
            return False
        return True

    for t in d.tracks + d.arcs:
        where = f'{t.layer} {_at(t.start)}'
        if layer_ok(t.layer, 'дорожка', where) and b.layers[t.layer].kind is not LayerKind.COPPER:
            iss.add('дорожка не на медном слое', where)
        if not t.width > 0:
            iss.add('дорожка нулевой ширины', where)
        if not _finite(*t.start, *t.end):
            iss.add('дорожка с нечисловыми координатами', where)
    for t in d.tracks:
        if t.start == t.end:
            iss.add('дорожка нулевой длины', f'{t.layer} {_at(t.start)}')

    for v in d.vias:
        where = _at(v.pos)
        for lay in v.layers:
            layer_ok(lay, 'переходное отверстие', where)
        if v.drill <= 0.001:
            iss.add(f'переходное отверстие с отверстием {v.drill:g} мм (вырожденное)', where)
        elif v.drill >= v.size:
            iss.add('переходное отверстие: отверстие не меньше площадки', where)
        if v.layers[0] == v.layers[1]:
            iss.add('переходное отверстие на одном слое', where)
        if copper and v.via_type != 'through' and v.layers == (copper[0], copper[-1]):
            iss.add(f'переходное отверстие типа {v.via_type} проходит через все слои (фактически сквозное)', where)

    refs = Counter(fp.ref for fp in d.footprints)
    for ref, n in refs.items():
        if n > 1 and ref and not ref.startswith(('REF**', '#')):
            iss.add('повторяющееся позиционное обозначение', f'{ref} ×{n}')
    for fp in d.footprints:
        for p in fp.pads:
            where = f'{fp.ref}.{p.number}'
            for lay in p.layers:
                if lay not in names:
                    iss.add(f'площадка на несуществующем слое {lay}', where)
            if p.type in ('thru_hole', 'np_thru_hole'):
                if p.drill is None:
                    iss.add('выводная площадка без отверстия', where)
                elif p.type == 'thru_hole' and p.shape.shape != 'custom' and \
                        min(p.drill.size) >= min(p.shape.size) and not p.layer_shapes:
                    iss.add('металлизированное отверстие не меньше площадки (пояска нет)', where)
            elif p.drill is not None:
                iss.add('у SMD-площадки задано отверстие', where)
            if p.shape.shape != 'custom' and not (p.shape.size[0] > 0 and p.shape.size[1] > 0) \
                    and p.type != 'np_thru_hole':
                iss.add('площадка нулевого размера', where)
            if p.type == 'thru_hole' and copper and not any(l in copper for l in p.layers):
                iss.add('выводная площадка без медных слоёв', where)

    for z in d.all_zones():
        where = z.name or z.uid or ''
        for lay in z.layers:
            if lay not in names and lay != 'Inner':
                iss.add(f'зона на несуществующем слое {lay}', where)
        if not z.outline or any(len(c) < 3 for c in z.outline):
            iss.add('зона с контуром меньше трёх точек', where)
        for lay in z.filled:
            if lay not in z.layers:
                iss.add(f'заливка зоны на слое {lay}, которого нет среди слоёв зоны', where)
        if z.is_rule_area and z.filled:
            iss.add('у области правил (keepout) есть заливка', where)

    for g in d.all_graphics():
        if g.layer and g.layer not in names:
            iss.add(f'графика на несуществующем слое {g.layer}', g.footprint or '')
    return iss.lines()


def summarize(b) -> dict:
    """Сводка платы: что прочитано и сколько."""
    s = {
        'плата': b.name,
        'источник': b.source,
        'медных слоёв': len(b.copper),
        'медь': ', '.join(b.copper),
        'толщина, мм': b.thickness,
        'цепей': len(b.nets),
    }
    if b.stackup is not None:
        st = b.stackup
        s['стек'] = (f'элементов {len(st.items)}, сумма толщин {st.thickness:.3f} мм'
                     + (', построен по умолчанию' if st.synthetic else '')
                     + (f', покрытие {st.finish}' if st.finish else ''))
    else:
        s['стек'] = 'не задан в файле'
    if b.holes or any(l.features for l in b.layers.values()):
        s['отверстий'] = (f'{len(b.holes)}: металлизированных {sum(h.plated for h in b.holes)}, '
                          f'пазов {sum(h.slot_end is not None for h in b.holes)}')
        s['объектов на слоях'] = {n: len(l.features) for n, l in b.layers.items() if l.features}
        cu = b.copper
        removed = sum(len([x for x in cu[cu.index(h.span[0]):cu.index(h.span[1]) + 1] if x not in h.padstack])
                      for h in b.holes if h.plated and h.span[0] in cu and h.span[1] in cu)
        s['площадок отверстий удалено на слоях'] = removed
    s['контур'] = (f'областей {len(b.outline)}, вырезов {sum(len(r.holes) for r in b.outline)}'
                   if b.outline else 'не построен')
    d = b.design
    if d is not None:
        pads = d.pads
        pt = Counter(str(p.type) for p in pads)
        vt = Counter(str(v.via_type) for v in d.vias)
        zones = d.all_zones()
        s.update({
            'формат': f'{d.version} (KiCad {d.generator_version})',
            'компонентов': len(d.footprints),
            'площадок': f'{len(pads)}: ' + ', '.join(f'{k} {v}' for k, v in pt.most_common()),
            'падстеков (разные формы по слоям)': sum(1 for p in pads if p.padstack_mode != 'normal'),
            'удаление площадок на неиспользуемых слоях': f'выводов {sum(1 for p in pads if p.remove_unused_layers)}, '
                f'переходных отверстий {sum(1 for v in d.vias if v.remove_unused_layers)}',
            'отрезков дорожек': len(d.tracks),
            'дуг дорожек': len(d.arcs),
            'переходных отверстий': f'{len(d.vias)}' + (': ' + ', '.join(f'{k} {v}' for k, v in vt.most_common())
                                                        if d.vias else ''),
            'зон': f'{len(zones)} (заливок {sum(1 for z in zones if not z.is_rule_area)}, '
                   f'областей правил {sum(1 for z in zones if z.is_rule_area)})',
            'точек в залитых зонах': sum(len(c) for z in zones for cs in z.filled.values() for c in cs),
            'графики на меди': sum(1 for g in d.all_graphics() if b.layers.get(g.layer) is not None
                                   and b.layers[g.layer].kind is LayerKind.COPPER),
            'графики контура (Edge.Cuts)': sum(1 for g in d.all_graphics() if g.layer == 'Edge.Cuts'),
            'текстов на меди и маске': len(d.texts),
            'классов цепей': len(d.rules.netclasses),
            'правила': 'из .kicad_pro' if d.rules.has_project_file else 'по умолчанию (нет .kicad_pro)',
        })
    return s
