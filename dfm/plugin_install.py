"""Установка плагина KiCad: `python -m dfm plugin-install [--dest папка]`.

Копирует code/plugin в каталог плагинов KiCad (по умолчанию «Документы»/KiCad/<версия>/plugins/
ru.dfm.check) и пишет config.json: путь к интерпретатору, в котором установлен пакет dfm (тот, что
запустил команду), и папку отчётов. Плагину самому нужна только библиотека kicad-python — её
KiCad ставит в окружение плагина из requirements.txt при первом запуске.

Перед первым запуском в KiCad: Настройки → Плагины → «Включить сервер IPC API», затем перезапуск.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from importlib.metadata import files
from pathlib import Path

IDENT = 'ru.dfm.check'


def documents_dir() -> Path:
    """Папка «Документы» (на Windows — через Known Folders: с OneDrive она бывает не в профиле)."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [('a', wintypes.DWORD), ('b', wintypes.WORD), ('c', wintypes.WORD), ('d', ctypes.c_ubyte * 8)]
        docs = GUID(0xFDD39AD0, 0x238F, 0x46AF, (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7))
        p = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(docs), 0, None, ctypes.byref(p)) == 0:
            path = Path(p.value)
            ctypes.windll.ole32.CoTaskMemFree(p)
            return path
    return Path.home() / 'Documents'


def default_dest(version: str = '10.0') -> Path:
    base = os.environ.get('KICAD_DOCUMENTS_HOME')
    root = Path(base) if base else documents_dir() / 'KiCad'
    return root / version / 'plugins' / IDENT


class InstallError(RuntimeError):
    """Папка назначения — не папка плагина: очищать её нельзя."""


def _plugin_source() -> Path:
    """В рабочей копии файлы лежат в plugin/, при установке wheel — в share/pcb-dfm/plugin/."""
    checkout = Path(__file__).resolve().parents[1] / 'plugin'
    if (checkout / 'dfm_plugin.py').is_file():
        return checkout
    # Пути из метаданных учитывают установку как в venv, так и в пользовательский каталог.
    for entry in files('pcb-dfm') or ():
        if entry.match('share/pcb-dfm/plugin/plugin.json'):
            return Path(entry.locate()).parent
    raise InstallError('В установленном пакете отсутствуют файлы плагина KiCad.')


def is_plugin_dir(dest: Path) -> bool:
    """Пустая папка или папка нашего плагина (plugin.json с identifier = IDENT)."""
    if not dest.exists():
        return True
    if not dest.is_dir():
        return False
    if not any(dest.iterdir()):
        return True
    try:
        return json.loads((dest / 'plugin.json').read_text(encoding='utf-8')).get('identifier') == IDENT
    except (OSError, ValueError, AttributeError):
        return False


def install(dest: Path | None = None, python: str | None = None) -> Path:
    source = _plugin_source()
    dest = Path(dest) if dest else default_dest()
    if dest.exists() and not dest.is_dir():
        raise InstallError(f'{dest} — файл, а не папка; укажите папку плагина (пустую или прежней установки)')
    if not is_plugin_dir(dest):
        raise InstallError(f'{dest} — не папка плагина {IDENT} (нет plugin.json с этим identifier) и не пуста; '
                           f'её содержимое удалилось бы. Укажите пустую папку или папку прежней установки.')
    old = {}
    cfg_path = dest / 'config.json'
    if cfg_path.exists():                              # выбор на странице настроек и свои ключи — сохранить
        try:
            old = json.loads(cfg_path.read_text(encoding='utf-8'))
            if not isinstance(old, dict):
                raise ValueError('ожидается объект JSON {…}')
        except (OSError, ValueError) as e:             # молча заменить — потерять свои ключи (reports, профили)
            raise InstallError(f'{cfg_path} не читается ({e}); исправьте его или удалите (тогда настройки '
                               f'плагина будут по умолчанию) и повторите установку') from None
    if dest.exists():
        keep = {(dest / 'reports').resolve()}
        if isinstance(old.get('reports'), str) and old['reports']:
            rep = Path(old['reports'])
            keep.add((rep if rep.is_absolute() else dest / rep).resolve())   # своя папка отчётов
        root = dest.resolve()
        if any(k == root or k in root.parents for k in keep):
            # отчёты — сама папка плагина или выше: удаляются только файлы плагина
            names = {f.name for f in source.iterdir()} | {'__pycache__'}
            gone = [f for f in dest.iterdir() if f.name in names and f.name not in ('reports', 'config.json')]
        else:                                          # папка отчётов и папки на пути к ней остаются
            gone = [f for f in dest.iterdir()
                    if not any(f.resolve() == k or f.resolve() in k.parents for k in keep)]
        for f in gone:
            shutil.rmtree(f) if f.is_dir() and not f.is_symlink() else f.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    for f in source.iterdir():
        if f.name in ('__pycache__', 'reports', 'config.json'):
            continue
        if f.is_dir():
            shutil.copytree(f, dest / f.name, dirs_exist_ok=True)   # могла остаться: на пути к папке отчётов
        else:
            shutil.copy(f, dest / f.name)
    cfg = dict(old)                                    # все прежние ключи, кроме пути к интерпретатору
    cfg['python'] = python or sys.executable
    cfg.setdefault('reports', str(dest / 'reports'))
    cfg.setdefault('extra_args', [])
    cfg.setdefault('profiles_dirs', [])
    (dest / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return dest
