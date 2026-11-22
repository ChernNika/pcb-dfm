"""Геометрические признаки кармана для обучения и проверки платы."""
import math

import numpy as np
from shapely.geometry import Point

from ..rules.b1 import POCKET_DELTA_UM, theta_from_area

DESC_NAMES = ('θ', 'θ_ℓ', 'θ_min', 'θ_эфф', 'A/δ²', 'H/δ', 'w_мин', 'w_макс', 'd_чуж', 'есть_чуж')


def pocket_descriptors(info, walls, own, foreign, pocket_delta_um=POCKET_DELTA_UM):
    """θ_эфф — по площади кармана при том δ, при котором она измерена (не при δ профиля: иначе сдвиг
    на 2–9°, а перетрав острого клина очень чувствителен к углу)."""
    delta_um = info['profile']['delta']
    position, width = walls['s'], walls['w']
    # Знак координаты вдоль стенки различает две стороны кармана.
    sides = (width[position > 0], width[position < 0])
    narrow, wide = sorted(float(np.min(side if len(side) else width)) for side in sides)
    tip = Point(*info['v'])
    distance = min(float(foreign.distance(tip)), 1.0) if foreign is not None else 1.0

    theta = info['theta']
    theta_local = info['theta_l']
    area = info['area_mm2']
    theta_eff = theta_from_area(area, pocket_delta_um / 1000)
    return [
        theta / 90, theta_local / 90, min(theta, theta_local) / 90, theta_eff / 90,
        math.log(max(area * 1e6 / delta_um ** 2, 1e-4)), info['depth_mm'] * 1000 / delta_um,
        narrow, wide, distance, float(foreign is not None),
    ]
