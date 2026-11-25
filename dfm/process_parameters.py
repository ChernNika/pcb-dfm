"""Параметры производства, доступные для уточнения в интерфейсе плагина."""

from .rules.profile import product_class


# Поля с теми же единицами, что и в TOML. Каждый параметр передаётся в DFM через --set.
_FIELDS = (
    ('etch.etch_factor', 'Фактор травления', True),
    ('etch.resist_resolution_um', 'Разрешение фоторезиста, мкм', True),
    ('etch.finished_width_min_um', 'Минимальная ширина проводника, мкм', True),
    ('drilling.sigma_drill_um', 'СКО положения сверла, мкм', False),
    ('drilling.sigma_layer_um', 'СКО общего смещения слоя, мкм', False),
    ('drilling.sigma_outer_pattern_um', 'СКО положения наружного рисунка, мкм', False),
    ('drilling.scale_sigma_ppm', 'СКО масштаба слоя, ppm', False),
    ('annular_ring.required_um.class3.outer', 'Поясок класса 3 на наружном слое, мкм', False),
    ('annular_ring.required_um.class3.inner', 'Поясок класса 3 на внутреннем слое, мкм', False),
    ('yield.litho_sigma_um', 'СКО ширины после экспонирования, мкм', False),
)


def process_parameters(profile):
    definitions = list(_FIELDS)
    # Некоторые заводы задают суммарный допуск вместо СКО отдельных операций.
    if profile['drilling'].get('registration_tol_um'):
        definitions = [item for item in definitions if item[0] not in
                       ('drilling.sigma_layer_um', 'drilling.sigma_outer_pattern_um')]
        definitions.extend((
            ('drilling.registration_tol_um.outer', 'Допуск совмещения наружного слоя, ±мкм', False),
            ('drilling.registration_tol_um.inner', 'Допуск совмещения внутреннего слоя, ±мкм', False),
        ))
        if 'registration_k' in profile['drilling']:
            definitions.append(('drilling.registration_k', 'Число СКО в допуске совмещения', True))
    fields = {}
    for key, label, positive in definitions:
        value = profile
        for part in key.split('.'):
            value = value[part]
        fields[key] = dict(label=label, value=value, positive=positive)
    return dict(fields=fields, product_class=product_class(profile), width_loss=profile['yield']['k_width'],
                breakout=profile['annular_ring'].get('breakout_allowed_deg', {}))
