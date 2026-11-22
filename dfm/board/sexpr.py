"""Чтение и запись S-выражений — текстового формата файлов KiCad.

Файл .kicad_pcb состоит из вложенных скобок:

    (kicad_pcb (version 20260206)
        (via (at 189.865 110.49) (size 1.6) (drill 0.6) (layers "F.Cu" "B.Cu") (net "VCC")))

Модуль ничего не знает о платах — только о скобках, словах, строках и числах.
Результат разбора — дерево из узлов Node (это список: первый элемент — имя узла).

Атомы превращаются так:
    123, -4, 1.6, 2e-3  → int или float
    "GND", "F.Cu"       → str (строка была в кавычках)
    yes, thru_hole      → Sym (слово без кавычек; Sym — подкласс str, сравнивается как обычная строка)

Различие между Sym и str нужно только для записи обратно: KiCad требует, чтобы ключевые
слова (thru_hole, yes, signal) были без кавычек, а имена (цепей, слоёв) — в кавычках.

Скорость (KiCad 10, демо-платы): обычная плата 1–6 МБ — 0,05–0,5 с; самая большая
(Jetson, 85 МБ, 15 млн токенов) — 8–14 с и до 1,2 ГБ памяти в пике (дерево — около 0,5 ГБ).
"""
import gc
import re

__all__ = ['Node', 'Sym', 'SExprError', 'parse', 'load', 'dumps', 'dump']

# Полное выражение токена: скобки, строка в кавычках (с экранированием \" и \\), слово.
# Быстрый разбор его не использует (см. _tokenize); оно нужно, чтобы при ошибке найти
# строку и столбец. Одиночная кавычка — признак незакрытой строки.
_TOKEN = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+|"', re.S)
_NUMBER = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\Z')
_UNESCAPE = re.compile(r'\\(.)', re.S)
# строка в кавычках целиком (в скобках — чтобы re.split вернул и сами строки)
_STRING = re.compile(r'("[^"\\]*(?:\\.[^"\\]*)*")', re.S)
# Пропущенная открывающая скобка: «)» сразу за которой буква, например
#   (curved_edges no)filter_ratio 0.9)
# Так записаны параметры капель (teardrops) в части файлов KiCad 9.0 — в демо-плате
# RoyalBlue54L-Feather 349 таких мест. KiCad их читает, потому что у него «(» перед ключевым
# словом необязательна. Правильный файл KiCad всегда ставит после «)» пробел, перевод строки или «)».
# Ищется только вне строк в кавычках (внутри имени цепи «)A» — обычный текст).
_MISSING_OPEN_OUTSIDE = re.compile(r'\)(?=[A-Za-z_])')
_MISSING_OPEN = re.compile(r'"(?:[^"\\]|\\.)*"|\)(?=[A-Za-z_])', re.S)   # для сообщений об ошибках
_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r'}


class SExprError(ValueError):
    """Ошибка синтаксиса с указанием строки и столбца."""


class Sym(str):
    """Слово без кавычек (ключевое слово KiCad)."""
    __slots__ = ()

    def __repr__(self):
        return f'Sym({str.__repr__(self)})'


class Node(list):
    """Узел дерева: [имя, аргумент1, аргумент2, ...]; аргументы — атомы или вложенные узлы."""
    __slots__ = ()

    @property
    def tag(self):
        return self[0] if self else None

    @property
    def args(self):
        """Все элементы после имени."""
        return self[1:]

    def get(self, tag, default=None):
        """Первый вложенный узел с данным именем."""
        for x in self:
            if type(x) is Node and x and x[0] == tag:
                return x
        return default

    def all(self, tag):
        """Все вложенные узлы с данным именем (по порядку)."""
        return [x for x in self if type(x) is Node and x and x[0] == tag]

    def children(self):
        """Все вложенные узлы."""
        return [x for x in self if type(x) is Node]

    def val(self, tag, default=None):
        """Первый аргумент вложенного узла: node.val('drill') для (drill 0.6) → 0.6."""
        n = self.get(tag)
        return n[1] if n is not None and len(n) > 1 else default

    def vals(self, tag):
        """Все аргументы вложенного узла: node.vals('at') для (at 1 2 90) → [1, 2, 90]."""
        n = self.get(tag)
        return n[1:] if n is not None else []

    def has(self, word):
        """Есть ли среди аргументов слово-флаг: (via blind ...) → has('blind')."""
        return any(type(x) is Sym and x == word for x in self[1:])

    def __repr__(self):
        return f'Node({list.__repr__(self)})'


def _position(text, token_index):
    """Строка и столбец токена с данным номером (медленно, только для сообщений об ошибках)."""
    for i, m in enumerate(_TOKEN.finditer(text)):
        if i == token_index:
            line = text.count('\n', 0, m.start()) + 1
            col = m.start() - (text.rfind('\n', 0, m.start()) + 1) + 1
            return f'строка {line}, столбец {col}'
    return 'конец текста'


def _locate_error(text):
    """Медленный повторный проход, который находит место синтаксической ошибки."""
    depth = 0
    for i, tok in enumerate(_TOKEN.findall(text)):
        if tok == '(':
            depth += 1
        elif tok == ')':
            depth -= 1
            if depth < 0:
                return SExprError(f'лишняя закрывающая скобка: {_position(text, i)}')
        elif tok == '"' or (tok[0] == '"' and (len(tok) == 1 or tok[-1] != '"')):
            return SExprError(f'незакрытая кавычка: {_position(text, i)}')
    if depth > 0:
        return SExprError(f'не хватает закрывающих скобок: {depth}')
    return None


def _repair_missing_open(text):
    """Вставляет пропущенные «(» (см. _MISSING_OPEN); строки в кавычках не трогает."""
    count = 0

    def fix(m):
        nonlocal count
        if m.group(0) == ')':
            count += 1
            return ') ('
        return m.group(0)

    return _MISSING_OPEN.sub(fix, text), count


def _tokenize(text):
    """Токены в порядке следования и число исправленных пропусков «(».

    Строки в кавычках выделяет регулярное выражение, всё остальное делится встроенным split():
    так вдвое быстрее, чем одним выражением. Текст между строками склеивается через символ NUL
    (в файлах KiCad его не бывает), чтобы замены и ремонт «(» делать одним проходом по всему тексту."""
    parts = _STRING.split(text)          # чётные элементы — текст между строками, нечётные — строки
    out = []
    ext = out.extend
    app = out.append
    if _NUL in text:                     # на всякий случай — медленный путь по частям
        fixed = 0
        for i in range(0, len(parts), 2):
            seg, n = _MISSING_OPEN_OUTSIDE.subn(') (', parts[i])
            fixed += n
            ext(seg.replace('(', ' ( ').replace(')', ' ) ').split())
            if i + 1 < len(parts):
                app(parts[i + 1])
        return out, fixed
    outside, fixed = _MISSING_OPEN_OUTSIDE.subn(') (', _NUL.join(parts[0::2]))
    segs = outside.replace('(', ' ( ').replace(')', ' ) ').split(_NUL)
    for seg, string in zip(segs, parts[1::2]):
        ext(seg.split())
        app(string)
    ext(segs[-1].split())
    return out, fixed


_OPEN, _CLOSE = object(), object()
_BOM = chr(0xFEFF)   # метка порядка байтов в начале файла
_NUL = chr(0)        # разделитель частей текста в _tokenize


def _unescape(s):
    return _UNESCAPE.sub(lambda m: _ESCAPES.get(m.group(1), m.group(1)), s)


def parse(text, warnings=None):
    """Разбирает текст с одним S-выражением верхнего уровня и возвращает его корневой узел.

    warnings — необязательный список, куда добавляются сообщения об исправленных неточностях
    файла (тех, которые прощает и сам KiCad)."""
    if text.startswith(_BOM):   # метка BOM, которую ставят некоторые редакторы
        text = text[len(_BOM):]
    tokens, fixed = _tokenize(text)
    if fixed and warnings is not None:
        warnings.append(f'пропущена открывающая скобка перед ключевым словом: {fixed} мест(а); '
                        'исправлено так же, как это делает KiCad')

    # Кэш значений: каждый разный токен превращается в значение один раз
    # (одни и те же слова, числа и имена слоёв встречаются тысячи раз).
    cache = {'(': _OPEN, ')': _CLOSE}
    get = cache.get
    root = Node()
    cur = root
    stack = []
    push = stack.append
    pop = stack.pop
    # Сборщик мусора на время разбора выключается: дерево создаёт миллионы списков, мусора при этом
    # нет, а сборщик раз за разом обходит всё растущее дерево (на больших платах это ~треть времени).
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for tok in tokens:
            a = get(tok)
            if a is None:
                if '"' in tok:
                    # строкой считается только токен, целиком совпавший с шаблоном строки;
                    # кавычка в любом другом месте — незакрытая строка
                    if not _STRING.fullmatch(tok):
                        raise SExprError('незакрытая кавычка')
                    a = tok[1:-1]
                    if '\\' in a:
                        a = _unescape(a)
                elif _NUMBER.match(tok):
                    a = int(tok) if tok.lstrip('+-').isdigit() else float(tok)
                else:
                    a = Sym(tok)
                cache[tok] = a
            if a is _OPEN:
                node = Node()
                cur.append(node)
                push(cur)
                cur = node
            elif a is _CLOSE:
                cur = pop()
            else:
                cur.append(a)
        if stack:
            raise SExprError('не хватает закрывающих скобок')
    except (IndexError, SExprError):
        # IndexError — pop() из пустого стека, то есть лишняя «)»; уточняем место ошибки
        # место ищется в тексте с исправленными пропусками «(», чтобы не указать на них
        raise _locate_error(_repair_missing_open(text)[0]) or SExprError('синтаксическая ошибка') from None
    finally:
        if gc_was_enabled:
            gc.enable()
    if len(root) != 1 or type(root[0]) is not Node:
        raise SExprError(f'ожидалось одно выражение в скобках, найдено элементов верхнего уровня: {len(root)}')
    return root[0]


def load(path, warnings=None):
    """Читает файл KiCad (UTF-8) и возвращает корневой узел."""
    with open(path, encoding='utf-8-sig') as f:
        return parse(f.read(), warnings)


# ---------- запись ----------

def _atom(x):
    if type(x) is Sym:
        return str(x)
    if isinstance(x, str):
        return '"' + x.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == int(x) and abs(x) < 1e15:
            return str(int(x))
        r = repr(x)
        if 'e' in r or 'E' in r:      # KiCad пишет числа без экспоненты
            r = format(x, '.12f').rstrip('0').rstrip('.')
        return r
    raise TypeError(f'нельзя записать {type(x).__name__}: {x!r}')


def _write(node, depth, out):
    flat = all(type(x) is not Node for x in node)
    if flat:   # узел только из атомов пишем в одну строку: (at 1 2 90)
        out.append('(' + ' '.join(_atom(x) for x in node) + ')')
        return
    # атомы до первого вложенного узла — в строке с именем; дальше каждый элемент с новой строки,
    # порядок элементов сохраняется полностью
    out.append('(')
    pad = '\n' + '\t' * (depth + 1)
    inline = True
    for k, x in enumerate(node):
        if type(x) is Node:
            inline = False
            out.append(pad)
            _write(x, depth + 1, out)
        elif inline:
            out.append((' ' if k else '') + _atom(x))
        else:
            out.append(pad + _atom(x))
    out.append('\n' + '\t' * depth + ')')


def dumps(node):
    """Текст S-выражения в стиле KiCad: табуляции, вложенные узлы с новой строки, порядок сохраняется."""
    out = []
    _write(node, 0, out)
    return ''.join(out) + '\n'


def dump(node, path):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(dumps(node))
