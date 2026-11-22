"""Чтение Gerber и Excellon (dfm/board/gerber_format.py, excellon.py) на файлах, записанных в тесте."""
import math

import numpy as np
import pytest

from dfm.board.excellon import read_excellon
from dfm.board.gerber_format import GerberError, _eval, read_gerber
from dfm.board.shapes import Circle, Compound, Region, Stroke, arc_geometry
from dfm.geometry.polygons import to_geometry

HEAD = '%FSLAX46Y46*%\n%MOMM*%\n%LPD*%\nG01*\nG75*\n'
TOL = 1e-4


def gerber(tmp_path, body, head=HEAD, name='t.gbr'):
    p = tmp_path / name
    p.write_text(head + body + 'M02*\n', encoding='utf-8')
    return read_gerber(p)


def c(v):
    """мм → целое число формата 4.6."""
    return str(int(round(v * 1e6)))


def area(s):
    return to_geometry(s, TOL).area


def test_macro_expressions():
    assert _eval('$1+$1', {1: 0.25}) == 0.5
    assert _eval('-$2x2+(1-0.5)/2', {2: 1.0}) == -1.75
    assert _eval('3X$1', {1: 2}) == 6


def test_standard_apertures(tmp_path):
    gf = gerber(tmp_path, '%ADD10C,1.0*%\n%ADD11R,2X1*%\n%ADD12O,2X1*%\n%ADD13P,2X6X30*%\n%ADD14C,1.0X0.4*%\n'
                          f'D10*\nX{c(1)}Y{c(2)}D03*\nD11*\nX{c(5)}Y0D03*\nD12*\nX{c(10)}Y0D03*\n'
                          f'D13*\nX{c(15)}Y0D03*\nD14*\nX{c(20)}Y0D03*\n')
    s = [o.shape for o in gf.objects]
    assert isinstance(s[0], Circle) and s[0].center == (1, 2) and s[0].radius == 0.5
    assert math.isclose(area(s[1]), 2)
    assert isinstance(s[2], Stroke) and math.isclose(area(s[2]), math.pi * 0.25 + 1, rel_tol=1e-3)
    assert math.isclose(area(s[3]), 3 * math.sqrt(3) / 2, rel_tol=1e-9)        # шестиугольник, описанный d = 2
    assert math.isclose(area(s[4]), math.pi * (0.25 - 0.04), rel_tol=1e-3)      # круг с отверстием
    assert all(o.kind == 'flash' and o.dark for o in gf.objects)


def test_macro_roundrect_and_exposure_off(tmp_path):
    """Макрос KiCad RoundRect (контур + 4 круга + 4 линии) и макрос с вычитанием."""
    rr = ('%AMRoundRect*\n0 Rectangle with rounded corners*\n4,1,4,$2,$3,$4,$5,$6,$7,$8,$9,$2,$3,0*\n'
          '1,1,$1+$1,$2,$3*\n1,1,$1+$1,$4,$5*\n1,1,$1+$1,$6,$7*\n1,1,$1+$1,$8,$9*\n'
          '20,1,$1+$1,$2,$3,$4,$5,0*\n20,1,$1+$1,$4,$5,$6,$7,0*\n20,1,$1+$1,$6,$7,$8,$9,0*\n20,1,$1+$1,$8,$9,$2,$3,0*%\n'
          '%AMRing*\n1,1,2,0,0*\n1,0,1,0,0*%\n')
    gf = gerber(tmp_path, rr + '%ADD10RoundRect,0.25X-0.5X-1X0.5X-1X0.5X1X-0.5X1X0*%\n%ADD11Ring*%\n'
                               f'D10*\nX0Y0D03*\nD11*\nX{c(5)}Y0D03*\n')
    r = 0.25
    assert isinstance(gf.objects[0].shape, Compound)
    # скруглённый прямоугольник 1,5 × 2,5 с радиусом 0,25
    assert math.isclose(area(gf.objects[0].shape), 1.5 * 2.5 - (4 - math.pi) * r * r, rel_tol=1e-4)
    assert math.isclose(area(gf.objects[1].shape), math.pi * (1 - 0.25), rel_tol=1e-3)


def test_arcs_keep_exact_geometry(tmp_path):
    """Дуги G02/G03 хранятся как дуги: середина на окружности, направление верное; полная окружность."""
    gf = gerber(tmp_path, '%ADD10C,0.2*%\nD10*\n'
                          f'X{c(1)}Y0D02*\nG03*\nX{c(-1)}Y0I{c(-1)}J0D01*\n'          # верхняя полуокружность
                          f'X{c(3)}Y0D02*\nG02*\nX{c(1)}Y0I{c(-1)}J0D01*\n'            # по часовой: нижняя
                          f'X{c(6)}Y0D02*\nX{c(6)}Y0I{c(-1)}J0D01*\n')                # полная окружность
    up, down, full = (o.shape for o in gf.objects)
    assert np.allclose(up.arcs[0], (0, 1))
    assert np.allclose(down.arcs[0], (2, -1))
    (cx, cy), radius, _, sweep = arc_geometry(tuple(full.pts[0]), full.arcs[0], tuple(full.pts[1]))
    assert np.allclose((cx, cy, radius), (5, 0, 1)) and math.isclose(abs(sweep), 2 * math.pi)


def test_region_with_arc_polarity_and_attributes(tmp_path):
    body = ('%TA.AperFunction,Conductor*%\n%TO.N,GND*%\nG36*\n'
            f'X0Y0D02*\nG01*\nX{c(2)}Y0D01*\nG03*\nX{c(2)}Y{c(2)}I0J{c(1)}D01*\nG01*\nX0Y{c(2)}D01*\nX0Y0D01*\nG37*\n'
            '%TD*%\n%LPC*%\n%ADD10C,0.5*%\nD10*\n%TO.P,U1,3,VCC*%\n%TO.N,VCC*%\n'
            f'X{c(1)}Y{c(1)}D03*\n%TD*%\n')
    gf = gerber(tmp_path, body)
    reg, hole = gf.objects
    assert reg.kind == 'region' and reg.attrs == {'AperFunction': 'Conductor', 'N': 'GND'}
    assert isinstance(reg.shape, Region) and reg.shape.outer.arcs
    assert math.isclose(area(reg.shape), 4 + math.pi / 2, rel_tol=1e-4)     # квадрат + полукруг
    assert not hole.dark and hole.attrs['P'] == 'U1,3,VCC' and hole.attrs['N'] == 'VCC'


def test_consecutive_draws_are_merged(tmp_path):
    """Подряд идущие отрезки одной апертуры без подъёма пера — одна линия (Stroke с тремя точками)."""
    gf = gerber(tmp_path, f'%ADD10C,0.25*%\nD10*\nX0Y0D02*\nX{c(1)}Y0D01*\nX{c(1)}Y{c(1)}D01*\n'
                          f'X{c(5)}Y0D02*\nX{c(6)}Y0D01*\n')
    assert len(gf.objects) == 2 and len(gf.objects[0].shape.pts) == 3


def test_inch_units(tmp_path):
    gf = gerber(tmp_path, '%ADD10C,0.1*%\nD10*\nX1000000Y0D03*\n', head='%FSLAX26Y26*%\n%MOIN*%\n')
    s = gf.objects[0].shape
    assert np.allclose(s.center, (25.4, 0)) and math.isclose(s.radius, 1.27)


def test_unsupported_commands_fail_loudly(tmp_path):
    with pytest.raises(GerberError, match='SR'):
        gerber(tmp_path, '%SRX2Y2I1J1*%\n')
    with pytest.raises(GerberError, match='G74'):
        gerber(tmp_path, 'G74*\n')


def test_file_attributes(tmp_path):
    gf = gerber(tmp_path, '', head='%TF.FileFunction,Copper,L2,Inr*%\n%TF.FilePolarity,Positive*%\n' + HEAD)
    assert gf.function == 'Copper,L2,Inr' and gf.polarity == 'Positive'


# ---------- Excellon ----------

def test_excellon_kicad_style(tmp_path):
    p = tmp_path / 'b-PTH.drl'
    p.write_text('M48\n; FORMAT={-:-/ absolute / metric / decimal}\n; #@! TF.FileFunction,Plated,1,4,PTH\n'
                 'FMAT,2\nMETRIC\n'
                 '; #@! TA.AperFunction,Plated,PTH,ViaDrill\nT1C0.300\n'
                 '; #@! TA.AperFunction,Plated,PTH,ComponentDrill\nT2C0.600\n%\nG90\nG05\n'
                 'T1\nX25.0Y-46.0\nX26.5Y-46.0\nT2\nX27.202Y-26.154G85X27.602Y-26.846\n'
                 'G00X1Y1\nM15\nG01X2Y1\nM16\nG05\nM30\n', encoding='utf-8')
    df = read_excellon(p)
    assert df.plated and df.layers == (1, 4) and df.kind == 'PTH' and df.attributes
    assert [(h.pos, h.diameter, h.function) for h in df.hits[:2]] == [((25, -46), 0.3, 'ViaDrill'),
                                                                      ((26.5, -46), 0.3, 'ViaDrill')]
    assert df.hits[2].end == (27.602, -26.846) and df.hits[2].function == 'ComponentDrill'
    assert df.hits[3].pos == (1, 1) and df.hits[3].end == (2, 1)


def test_excellon_integer_formats(tmp_path):
    """Целые координаты: TZ — ведущие нули опущены (12000 = 12,000 мм в формате 3.3);
    LZ — хвостовые опущены (012 = 12 мм); дюймы 2.4."""
    p = tmp_path / 'npth.drl'
    p.write_text('M48\nMETRIC,TZ\nT1C1.0\n%\nT1\nX12000Y-5500\nM30\n', encoding='utf-8')
    df = read_excellon(p)
    assert df.hits[0].pos == (12.0, -5.5) and not df.plated          # NPTH в имени файла
    p.write_text('M48\nMETRIC,LZ\nT1C1.0\n%\nT1\nX012Y-0055\nM30\n', encoding='utf-8')
    assert read_excellon(p).hits[0].pos == (12.0, -5.5)
    p.write_text('M48\nINCH,TZ\nT1C0.04\n%\nT1\nX10000Y5000\nM30\n', encoding='utf-8')
    h = read_excellon(p).hits[0]
    assert np.allclose(h.pos, (25.4, 12.7)) and math.isclose(h.diameter, 1.016)
