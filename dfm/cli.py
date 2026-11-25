"""Командная строка: python -m dfm <команда> …

    python -m dfm inspect плата.kicad_pcb               сводка модели платы и проверка её согласованности
    python -m dfm inspect плата.kicad_pcb --geometry    + медь по слоям и связность по геометрии
    python -m dfm inspect плата.kicad_pcb --pictures папка   + картинки слоёв (PNG)
    python -m dfm compare плата.kicad_pcb [ещё …] -o папка  сверка двух путей и нарушений с DRC KiCad
    python -m dfm check плата [--profile завод.toml] [--classes A1,A7] [--set b1.process=О] [-o папка]
                                                        проверка технологичности
    python -m dfm profiles [--dir папка]                 профили завода (JSON) — для страницы настроек плагина
    python -m dfm plugin-install [--dest папка]                                       плагин для KiCad
"""
import argparse
import json
import sys
import time
from pathlib import Path

from .profile_options import SetError, check_shape, set_value


def _check(args):
    from .board import load
    from .report.check import summary_lines, write_report
    from .rules.engine import Context, available_classes, run
    from .rules.findings import Severity
    from .rules.profile import load_profile

    started = time.perf_counter()
    try:                                               # профиль, --set и классы — до чтения платы: ошибка сразу
        profile = load_profile(args.profile)
        check_shape(profile)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        args.parser.error(f'профиль: {e}')             # в т. ч. цикл meta.base, раздел — не таблица
    try:
        for kv in args.set or []:
            set_value(profile, kv)
    except SetError as e:
        args.parser.error(str(e))
    classes = None
    if args.classes:
        available = available_classes()
        classes = list(dict.fromkeys(c.strip().upper() for c in args.classes.split(',')))
        unknown = [c for c in classes if c not in available]
        if unknown:
            args.parser.error(f'--classes: нет классов {", ".join(repr(c) for c in unknown)}; есть: {", ".join(available)}')
    board = load(args.board)
    ctx = Context(board, profile)
    findings = run(board, classes=classes, ctx=ctx)
    out = Path(args.out or f'check_{board.name}')
    write_report(board, ctx, findings, out, pictures=not args.no_pictures)
    print(f'{board.name}: находок {len(findings)} за {time.perf_counter() - started:.1f} с'
          + (' (профиль условный — числа документов, не завода)' if ctx.profile.illustrative else ''))
    if findings:
        print('\n'.join(summary_lines(findings)))
    print(f'Отчёт: {out / "report.html"}')
    if args.open:
        import webbrowser
        webbrowser.open((out / 'report.html').resolve().as_uri())
    return 1 if any(f.severity is Severity.ERROR for f in findings) else 0


def _export_gerber(args):
    from . import kicad_cli

    kicad_cli.export_gerbers(args.board, args.out, check_zones=True)
    kicad_cli.export_drill(args.board, args.out)
    return 0


def _profiles(args):
    """Встроенные профили и профили из --dir → JSON: имя, путь, условный ли, процесс B1, есть ли нейросеть."""
    from .process_parameters import process_parameters
    from .rules.profile import DEFAULT_PROFILE, load_profile
    files = sorted(DEFAULT_PROFILE.parent.glob('*.toml'))
    for d in args.dir or []:
        files += sorted(Path(d).glob('*.toml'))
    out = []
    for f in files:
        path = '' if f == DEFAULT_PROFILE else str(f)
        try:                                           # один испорченный профиль не ломает весь список
            pr = load_profile(path or None)
            check_shape(pr)
            out.append(dict(path=path, name=str(pr.get('meta', {}).get('name', f.stem)), illustrative=pr.illustrative,
                            parameters=process_parameters(pr), b1_process=str(pr['b1']['process']), b1_processes=list(pr['b1'].get('process_params', {}))))
        except Exception as e:
            out.append(dict(path=path, name=f.stem, error=f'{type(e).__name__}: {e}'))
    try:
        from .ml import infer
        nn = infer.available()
    except ImportError:
        nn = False
    from .rules.engine import available_classes
    print(json.dumps(dict(profiles=out, nn=nn, classes=available_classes()), ensure_ascii=False))
    return 0


def _plugin_install(args):
    from .plugin_install import InstallError, install
    try:
        dest = install(args.dest)
    except InstallError as e:
        args.parser.error(str(e))
    print(f'плагин установлен: {dest}')
    print('в KiCad: Настройки → Плагины → включить сервер IPC API, перезапустить KiCad; в редакторе плат '
          'появится кнопка «Проверить технологичность» (при первом запуске KiCad ставит kicad-python)')
    return 0


def _inspect(args):
    from .board import load

    started = time.perf_counter()
    board = load(args.board)
    elapsed = time.perf_counter() - started
    for k, v in board.summary().items():
        print(f'{k:>45}: {v}')
    print(f'{"время чтения, с":>45}: {elapsed:.2f}')
    if board.warnings:
        print('\nПредупреждения чтения:')
        for warning in board.warnings:
            print('  -', warning)
    problems = board.validate()
    print('\nПроверка модели:', 'замечаний нет' if not problems else '')
    for p in problems:
        print('  -', p)
    if args.geometry:
        _geometry(board)
    if args.pictures:
        from .draw import draw_layers
        from .geometry.copper import BoardGeometry
        paths = draw_layers(board, BoardGeometry(board), args.pictures)
        print(f'\nКартинки слоёв: {len(paths)} в {args.pictures}')
    return 0


def _compare(args):
    from .compare import compare_board, summary_table
    out = Path(args.out)
    reports = []
    for board in args.boards:
        name = Path(board).stem
        print(f'{name} …', flush=True)
        r = compare_board(board, out / name, pictures=not args.no_pictures)
        reports.append(r)
        print(f'   {"совпадает" if r.ok else "ЕСТЬ РАСХОЖДЕНИЯ"}: отклонение края до {r.max_dev * 1000:.2f} мкм, '
              f'отверстий {r.holes}, {r.drc_note}')
    table = summary_table(reports)
    (out / 'summary.md').write_text('# Сверка ядра: два пути и DRC KiCad\n\n' + table, encoding='utf-8')
    print('\n' + table)
    print(f'Отчёты: {out}')
    return 0 if all(r.ok for r in reports) else 1


def _geometry(board, limit=10):
    from .geometry.connectivity import connectivity
    from .geometry.copper import BoardGeometry

    started = time.perf_counter()
    geo = BoardGeometry(board)
    print('\nМедь по слоям:')
    for name in board.copper:
        lc = geo.layer(name)
        print(f'  {name:8} площадь {lc.area:10.2f} мм², островов {len(lc.islands):5}, цепей {len(lc.nets):5}')
    connections = connectivity(geo)
    print(f'\nСвязность по геометрии ({time.perf_counter() - started:.1f} с): компонент {connections.count}')

    def show(title, items, fmt):
        print(f'  {title}: {len(items)}')
        for x in items[:limit]:
            print('     ', fmt(x))
        if len(items) > limit:
            print(f'      … ещё {len(items) - limit}')

    def at(point):
        # В текстовом отчёте ось Y направлена вниз, как в KiCad.
        return f'({point[0]:.3f}; {-point[1] + 0.0:.3f})'

    show('замыкания', connections.shorts, lambda s: f'{s.nets[0]} — {s.nets[1]}: '
         + ('через отверстие ' if s.via_hole else f'{s.layer} ') + at(s.pos))
    show(f'разрывы (недостающих соединений {connections.missing_connections})', connections.opens,
         lambda o: f'{o.net}: кусков {len(o.fragments)}, ' + ', '.join(f'{lay} {at(p)}' for lay, p in o.fragments[:3]))
    show('изолированные островки заливки', connections.islands, lambda i: f'{i.net} {i.layer} {at(i.pos)}, {i.area:.2f} мм²')
    show('перемычки медью без цепи', connections.bridges, lambda x: f'{x[0][0]} — {x[0][1]}: {x[1]} {at(x[2])}')
    show('медь без цепи касается цепи', connections.unnamed_touch, lambda x: f'{x[0]}: {x[1]} {at(x[2])}')


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(prog='python -m dfm', description='Анализ технологичности печатных плат')
    commands = parser.add_subparsers(dest='cmd', required=True)

    inspect_cmd = commands.add_parser('inspect', help='сводка модели платы')
    inspect_cmd.add_argument('board')
    inspect_cmd.add_argument('--geometry', action='store_true', help='медь по слоям и связность по геометрии')
    inspect_cmd.add_argument('--pictures', metavar='ПАПКА', help='картинки слоёв (PNG) в папку')
    inspect_cmd.set_defaults(func=_inspect)

    compare_cmd = commands.add_parser('compare', help='сверка двух путей (.kicad_pcb и Gerber) и нарушений с DRC KiCad')
    compare_cmd.add_argument('boards', nargs='+')
    compare_cmd.add_argument('-o', '--out', default='compare_out', help='папка для отчётов (по умолчанию compare_out)')
    compare_cmd.add_argument('--no-pictures', action='store_true', help='без картинок (быстрее)')
    compare_cmd.set_defaults(func=_compare)

    check_cmd = commands.add_parser('check', help='проверка технологичности (A1, A3, A5, A7, B1)')
    check_cmd.add_argument('board', help='.kicad_pcb или папка с Gerber')
    check_cmd.add_argument('--profile', help='профиль изготовителя .toml (по умолчанию — условный завод документов)')
    check_cmd.add_argument('--classes', help='через запятую, например A1,A7 (по умолчанию все)')
    check_cmd.add_argument('-o', '--out', help='папка отчёта (по умолчанию check_<плата>)')
    check_cmd.add_argument('--no-pictures', action='store_true', help='без картинок')
    check_cmd.add_argument('--open', action='store_true', help='открыть report.html в браузере')
    check_cmd.add_argument('--set', action='append', metavar='РАЗДЕЛ.КЛЮЧ=ЗНАЧЕНИЕ',
                           help='изменить значение профиля, например b1.process=О или b1.method=formula '
                                '(ключ должен быть в профиле; число можно с запятой)')
    check_cmd.set_defaults(func=_check, parser=check_cmd)

    export_cmd = commands.add_parser('export-gerber', help='выгрузить слои Gerber и сверловку Excellon через KiCad')
    export_cmd.add_argument('board')
    export_cmd.add_argument('--out', required=True)
    export_cmd.set_defaults(func=_export_gerber)

    profiles_cmd = commands.add_parser('profiles', help='профили завода (JSON) для страницы настроек плагина')
    profiles_cmd.add_argument('--dir', action='append', help='ещё папка с профилями .toml')
    profiles_cmd.set_defaults(func=_profiles)

    install_cmd = commands.add_parser('plugin-install', help='установить плагин KiCad (кнопка «Проверить технологичность»)')
    install_cmd.add_argument('--dest', help='папка плагина (по умолчанию Документы/KiCad/10.0/plugins/ru.dfm.check)')
    install_cmd.set_defaults(func=_plugin_install, parser=install_cmd)

    args = parser.parse_args(argv)
    return args.func(args)
