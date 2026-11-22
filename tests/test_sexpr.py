"""Проверки читателя S-выражений (dfm.board.sexpr)."""
import pytest

from dfm.board.sexpr import Node, Sym, SExprError, parse, dumps

VIA = '(via (at 189.865 110.49) (size 1.6) (drill 0.6) (layers "F.Cu" "B.Cu") (net "VCC"))'


# ---------- разбор ----------

def test_simple_node():
    n = parse(VIA)
    assert n.tag == 'via'
    assert n.vals('at') == [189.865, 110.49]
    assert n.val('drill') == 0.6
    assert n.vals('layers') == ['F.Cu', 'B.Cu']
    assert n.val('net') == 'VCC'
    assert n.val('nothing') is None and n.val('nothing', 0) == 0


def test_numbers():
    n = parse('(x 0 -4 +7 1.6 -0.25 .5 3. 2e-3 1E5 20260206)')
    assert n.args == [0, -4, 7, 1.6, -0.25, 0.5, 3.0, 0.002, 1e5, 20260206]
    assert [type(v) for v in n.args[:3]] == [int, int, int]
    assert type(n.args[3]) is float


def test_symbol_vs_string():
    n = parse('(pad "1" thru_hole circle (layers "*.Cu" "*.Mask") (remove_unused_layers no))')
    num, kind, shape = n.args[:3]
    assert num == '1' and type(num) is str          # номер вывода — строка, даже если похож на число
    assert kind == 'thru_hole' and type(kind) is Sym
    assert shape == 'circle' and type(shape) is Sym
    assert n.has('thru_hole') and not n.has('smd')
    assert not n.has('1')                            # строка в кавычках — не флаг
    assert n.val('remove_unused_layers') == 'no'


def test_not_numbers_stay_symbols():
    n = parse('(x - . 0x1F 1.2.3 inf nan 12ab F.Cu)')
    assert all(type(v) is Sym for v in n.args)


def test_escapes_and_unicode():
    n = parse(r'(net "D\"1\\2" "Земля" "a b (c)" "line\nnext" "")')
    assert n.args == ['D"1\\2', 'Земля', 'a b (c)', 'line\nnext', '']


def test_whitespace_and_nesting():
    n = parse(chr(0xFEFF) + '  (a\n\t(b (c 1)\r\n)  (b 2) )  ')
    assert n.tag == 'a'
    assert len(n.all('b')) == 2
    assert n.get('b').get('c').val('c') is None      # (c 1): val ищет вложенный узел c внутри c
    assert n.get('b').get('c').args == [1]
    assert n.all('b')[1].args == [2]


def test_empty_list():
    n = parse('(a () (b))')
    assert n.children()[0] == [] and n.get('b') == ['b']


# ---------- ошибки ----------

def test_missing_open_paren_kicad9():
    """Ошибка записи из KiCad 9: «(curved_edges no)filter_ratio 0.9)» — KiCad такой файл читает."""
    bad = '(pad "1" (teardrops (max_width 2) (curved_edges no)filter_ratio 0.9) (enabled yes)) (uuid "x"))'
    good = '(pad "1" (teardrops (max_width 2) (curved_edges no) (filter_ratio 0.9) (enabled yes)) (uuid "x"))'
    w = []
    assert parse(bad, w) == parse(good)
    assert len(w) == 1 and '1 мест' in w[0]


def test_missing_open_paren_ignores_strings():
    w = []
    n = parse('(net "a)b" "(x)y")', w)
    assert n.args == ['a)b', '(x)y'] and w == []


@pytest.mark.parametrize('text, words', [
    ('(a (b 1)', 'не хватает'),
    ('(a (b 1)))', 'лишняя'),
    ('(a "незакрытая)', 'кавычка'),
    ('(a) (b)', 'одно выражение'),
    ('', 'одно выражение'),
    ('abc', 'одно выражение'),
])
def test_errors(text, words):
    with pytest.raises(SExprError, match=words):
        parse(text)


def test_error_position():
    # первая ')' в строке 3 закрывает (a, лишняя — вторая, в столбце 6
    with pytest.raises(SExprError, match='лишняя закрывающая скобка: строка 3, столбец 6'):
        parse('(a\n  (b 1)\n    ))')


# ---------- запись и «туда-обратно» ----------

def test_dumps_roundtrip_small():
    src = r'(kicad_pcb (version 20260206) (net 0 "") (pad "1" smd roundrect (at 1.5 -2 90) (roundrect_rratio 0.25) (net "Сигнал \"А\"")) (effects (font (size 1 1)) hide))'
    n = parse(src)
    text = dumps(n)
    assert parse(text) == n
    assert '\t(pad "1" smd roundrect' in text           # ключевые слова без кавычек, номер в кавычках
    assert '(at 1.5 -2 90)' in text                      # узел из одних атомов — в одну строку
    assert 'hide' in text.split('(effects')[1]           # слово после вложенного узла не потерялось


def test_dumps_types():
    n = Node([Sym('x'), 1, 2.0, 0.1, 1e-7, 'str', Sym('yes')])
    assert dumps(n).strip() == '(x 1 2 0.1 0.0000001 "str" yes)'
