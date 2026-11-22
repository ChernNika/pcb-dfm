"""Проверка структуры профиля и применение параметров --set до чтения платы."""
import math
import tomllib

from .rules.profile import load_profile


def check_shape(prof, ref=None, where=''):
    """Что в условном профиле — таблица, то и в действующем таблица (b1 = 1 вместо [b1] — ошибка профиля,
    а не сбой правила)."""
    ref = load_profile() if ref is None else ref
    for k, v in ref.items():
        if isinstance(v, dict) and k in prof:
            if not isinstance(prof[k], dict):
                raise ValueError(f'{where}{k} = {prof[k]!r}: ожидается таблица [{where}{k}]')
            check_shape(prof[k], v, f'{where}{k}.')


class SetError(ValueError):
    """Неверное --set раздел.ключ=значение."""


# у целых по смыслу ключей (класс изделия, счётчики) дробное не принимается; у величин с единицей в имени
# (…_um, …_mm, …_ppm …) целое в профиле — просто запись числа, дробное допустимо
_UNITS = ('_um', '_mm', '_ppm', '_deg', '_c', '_kv', '_per_mm')


def set_value(prof, kv: str):
    """«раздел.ключ=значение» → профиль. Ключ должен быть в профиле (с учётом meta.base и условного профиля),
    значение читается по типу прежнего: true/false, число (можно с десятичной запятой), строка (кавычки вокруг
    снимаются), список (TOML). b1.tau — "class" или число в (0; 1); b1.method и b1.process — только из
    допустимых; meta.* (имя, основа профиля) так не меняется."""
    key, sep, raw = kv.partition('=')
    key, raw = key.strip(), raw.strip()
    parts = [k.strip() for k in key.split('.')]
    if not sep or len(parts) < 2 or not all(parts):
        raise SetError(f'--set: ожидается раздел.ключ=значение, получено {kv!r}')
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in '"\'':
        raw = raw[1:-1].strip()
    if not raw:
        raise SetError(f'--set {key}: пустое значение')
    if parts[0] == 'meta':
        raise SetError(f'--set {key}: раздел meta (имя, основа, вид профиля) через --set не меняется')
    section = prof
    for depth, part in enumerate(parts):
        parent_key = '.'.join(parts[:depth])
        if not isinstance(section, dict):
            raise SetError(f'--set {key}: {parent_key} — значение, а не раздел')
        if part not in section:
            where = f'в разделе {parent_key}' if parent_key else 'в профиле'
            raise SetError(f'--set {key}: {where} нет ключа {part!r}; есть: {", ".join(map(str, section)) or "—"}')
        if depth < len(parts) - 1:
            section = section[part]
    old = section[parts[-1]]
    if isinstance(old, dict):
        raise SetError(f'--set {key}: это раздел, укажите ключ: {", ".join(map(str, old))}')
    if parts == ['b1', 'tau']:                         # "class" или число: тип в профиле бывает любой из двух
        if raw.lower() == 'class':
            value = 'class'
        else:
            value = _number(raw, key)
            if not 0 < value < 1:
                raise SetError(f'--set {key}: ожидается "class" или число в (0; 1), например 0,2; получено {raw!r}')
            value = float(value)
    else:
        value = _parse_like(old, raw, key)
    if parts == ['b1', 'method']:
        from .rules.b1 import METHODS
        if value not in METHODS:
            raise SetError(f'--set {key}: ожидается одно из {", ".join(METHODS)}, получено {raw!r}')
    if parts == ['b1', 'process']:
        b1 = prof['b1']
        # Старые профили перечисляли процессы только в таблице перетрава.
        known = dict.fromkeys((*b1.get('process_params', {}), *b1.get('overetch_um', {})))
        if value not in known:
            raise SetError(f'--set {key}: нет процесса {raw!r} в [b1.process_params]; есть: {", ".join(known) or "—"}')
    section[parts[-1]] = value


def _number(raw: str, key: str):
    """Число из строки (можно с десятичной запятой); nan/inf — SetError."""
    s = raw.replace(',', '.') if raw.count(',') == 1 and '.' not in raw else raw   # десятичная запятая
    try:
        v = int(s)
    except ValueError:
        try:
            v = float(s)
        except ValueError:
            raise SetError(f'--set {key}: ожидается число, получено {raw!r}') from None
    if not math.isfinite(v):
        raise SetError(f'--set {key}: нужно конечное число, получено {raw!r}')
    return v


def _parse_like(old, raw: str, key: str):
    """Строка → значение того же типа, что old; nan/inf и чужой тип — SetError."""
    if isinstance(old, bool):
        v = {'true': True, 'false': False}.get(raw.lower())
        if v is None:
            raise SetError(f'--set {key}: ожидается true или false, получено {raw!r}')
        return v
    if isinstance(old, (int, float)):
        v = _number(raw, key)
        if isinstance(old, float):
            return float(v)
        if isinstance(v, float):                       # было целое
            if v.is_integer():
                return int(v)
            if not any(p.endswith(_UNITS) for p in key.split('.')):
                raise SetError(f'--set {key}: ожидается целое число, получено {raw!r}')
        return v
    if isinstance(old, str):
        return raw
    if isinstance(old, list):
        try:
            v = tomllib.loads(f'v = {raw}')['v']
        except tomllib.TOMLDecodeError:
            v = None
        if not isinstance(v, list):
            raise SetError(f'--set {key}: ожидается список TOML, например [1, 2], получено {raw!r}')

        def finite(x):
            if isinstance(x, list):
                return all(map(finite, x))
            return not isinstance(x, float) or math.isfinite(x)
        if not finite(v):
            raise SetError(f'--set {key}: в списке нужны конечные числа, получено {raw!r}')
        return v
    raise SetError(f'--set {key}: значение типа {type(old).__name__} так не задаётся')
