"""Профиль изготовителя: возможности завода, из которых выводятся пороги правил.

    p = load_profile()                    # условный завод из документов (profiles/illustrative.toml)
    p = load_profile('мой_завод.toml')    # свой профиль: недостающее берётся из условного
    p['etch']['etch_factor'], p.illustrative

Профиль — TOML (читается встроенным tomllib). Свой профиль накладывается на условный по ключам,
так что в нём достаточно указать то, что известно о заводе. Основа может быть и другой: meta.base =
"open_data.toml" (цепочкой: у основы может быть своя основа). Происхождение каждого ключа запоминается
(meta.kind профиля, задавшего его последним); в находке (data_quality) — виды профилей, из которых
взяты числа раздела, а флаг illustrative показывает, что пороги иллюстративные, а не данные завода.
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

__all__ = ['Profile', 'load_profile', 'product_class', 'DEFAULT_PROFILE', 'OPEN_DATA_PROFILE']

DEFAULT_PROFILE = Path(__file__).with_name('profiles') / 'illustrative.toml'
OPEN_DATA_PROFILE = Path(__file__).with_name('profiles') / 'open_data.toml'   # ГОСТ, ECSS, нормы заводов


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Profile(dict):
    """Словарь разделов профиля плюс сведения о происхождении чисел."""

    def __init__(self, data: dict, source: str, origin: dict | None = None):
        super().__init__(data)
        self.source = source
        # {раздел: {ключ: вид профиля, задавшего ключ}} — только для ключей не из условного профиля
        self.origin = origin or {}

    @property
    def illustrative(self) -> bool:
        return bool(self.get('meta', {}).get('illustrative', False))

    def quality(self, section: str) -> str:
        """Откуда числа раздела: профиль завода, лабораторная линия, открытые данные (ГОСТ, ECSS, нормы
        заводов) или условный профиль документов; если раздел собран из нескольких — все через запятую
        (первым — вид, задавший больше ключей)."""
        own = self.origin.get(section, {})
        kinds = list(own.values())
        if self.illustrative or not kinds:
            return 'условный профиль'
        names = {'open_data': 'открытые данные', 'lab': 'лабораторная линия'}
        order = sorted(set(kinds), key=lambda k: (-kinds.count(k), str(k)))
        out = list(dict.fromkeys(names.get(k, 'профиль завода') for k in order))
        sec = self.get(section)
        if isinstance(sec, dict) and any(k not in own for k in sec):   # часть ключей раздела — из условного
            out.append('условный профиль')
        return ', '.join(out)


def product_class(profile) -> int:
    """Класс изделия (ГОСТ Р 55693 табл. 2, IPC-2221C п. 1.6.2) — один на все правила: etch.product_class.
    annular_ring.product_class — старое место ключа: учитывается, если etch.product_class не задан своим
    профилем (в условном он есть всегда); заданы оба и различаются — ошибка, а не тихий выбор."""
    e = profile['etch'].get('product_class')
    a = profile.get('annular_ring', {}).get('product_class')
    if a is None or (e is not None and int(e) == int(a)):
        return int(2 if e is None else e)
    own_e = 'product_class' in getattr(profile, 'origin', {}).get('etch', {})
    if e is None or not own_e:
        return int(a)
    raise ValueError(f'класс изделия задан по-разному: etch.product_class = {e}, annular_ring.product_class = {a}')


def _chain(path: Path, seen: tuple = ()) -> list[dict]:
    """Профиль и его основы (meta.base) → список от самой дальней основы к самому профилю."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f'цикл meta.base: {" → ".join(str(x.name) for x in (*seen, path))}')
    data = tomllib.loads(path.read_text(encoding='utf-8'))
    parent = data.get('meta', {}).get('base')
    if not parent:
        return [data]
    pp = Path(parent) if Path(parent).is_absolute() else path.parent / parent
    if not pp.exists():
        pp = DEFAULT_PROFILE.parent / parent
    if pp.resolve() == DEFAULT_PROFILE.resolve():
        return [data]
    return _chain(pp, (*seen, path)) + [data]


def load_profile(path=None) -> Profile:
    data = tomllib.loads(DEFAULT_PROFILE.read_text(encoding='utf-8'))
    if path is None:
        return Profile(data, str(DEFAULT_PROFILE))
    chain = _chain(Path(path))
    origin: dict = {}
    for layer in chain:
        kind = layer.get('meta', {}).get('kind')
        data = _merge(data, layer)
        for sec, v in layer.items():
            if sec == 'meta':
                continue
            keys = v.keys() if isinstance(v, dict) else ['']
            origin.setdefault(sec, {}).update({k: kind for k in keys})
    data.setdefault('meta', {})['illustrative'] = bool(chain[-1].get('meta', {}).get('illustrative', False))
    return Profile(data, str(path), origin)
