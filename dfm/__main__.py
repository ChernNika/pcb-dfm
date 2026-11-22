"""Точка входа для python -m dfm."""
from .cli import main
from .profile_options import SetError, check_shape, set_value

__all__ = ['main', 'SetError', 'check_shape', 'set_value']


if __name__ == '__main__':
    raise SystemExit(main())
