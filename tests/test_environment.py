"""Проверка окружения: библиотеки, пакет dfm, KiCad 10."""
import pytest

from dfm import kicad_cli


def test_libraries():
    import numpy, scipy, shapely, matplotlib  # noqa: F401
    assert int(shapely.__version__.split('.')[0]) >= 2


def test_package_layout():
    import dfm.board, dfm.geometry, dfm.rules, dfm.physics, dfm.ml, dfm.report  # noqa: F401


def test_kicad_cli_version():
    try:
        v = kicad_cli.version()
    except FileNotFoundError:
        pytest.skip('KiCad не установлен')
    assert v.startswith('10.')
