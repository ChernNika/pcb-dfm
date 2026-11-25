"""Плагин KiCad (IPC API): проверка технологичности открытой платы и показ находок в редакторе.

KiCad запускает этот файл кнопкой «Проверить технологичность». Плагин поднимает локальный сервер
(только 127.0.0.1) и открывает браузер:
  1. страница настроек: профиль завода, процесс травления и способ оценки B1, классы проверок
     (список профилей — `python -m dfm profiles`; последний выбор запоминается в config.json);
  2. «Проверить»: плата берётся из редактора (get_as_string — с несохранёнными правками, файл не
     трогается) вместе с правилами проекта (.kicad_pro, .kicad_dru) и проверяется командой
     `python -m dfm check` в окружении пакета dfm (путь — config.json, его пишет `python -m dfm plugin-install`);
  3. страница хода, потом отчёт report.html. «Показать в KiCad» у находки выделяет её объекты на плате
     (по uuid или по позиционному обозначению) и приближает выделенное; «Проверить снова» — новая
     проверка текущего состояния платы (исправил в KiCad — перепроверил, не нажимая кнопку снова).
Сервер завершается, если им не пользуются 30 минут (идущая проверка доводится до конца). Запросы с чужим Host
отклоняются (DNS rebinding), запуск проверки и выделение — только с ключом сеанса, который знают отданные сервером
страницы. Отчёт пишется в <reports>/<имя платы>; очищаются там только файлы отчёта и только в папке с меткой
.dfm-report (её создаёт плагин) — чужая папка не трогается.
"""
from __future__ import annotations

import hmac
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
CONFIG = HERE / 'config.json'
IDLE_LIMIT_S = 30 * 60
ZOOM_ACTION = 'common.Control.zoomFitSelection'   # действие KiCad «приблизить выделенное» (может смениться)


REQUEST_TIMEOUT_S = 10                              # медленный или недосланный запрос не держит поток сервера
MAX_BODY = 64 * 1024                                # больше — 413
REPORT_MARK = '.dfm-report'                         # метка папки отчёта, созданной плагином
REPORT_FILES = ('report.html', 'findings.json', 'report.md')   # что пишет `dfm check` (+ layers/*.png)
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}


def load_config() -> dict:
    p = CONFIG
    if p.exists():
        cfg = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(cfg, dict):
            raise ValueError('ожидается объект JSON {…}')
        return cfg
    return {}


def save_last(choice: dict):
    """Последний выбор → config.json (через временный файл: оборванная запись не портит настройки)."""
    p = CONFIG
    cfg = load_config()
    cfg['last'] = choice
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def safe_stem(name: str) -> str:
    """Имя папки отчёта из имени платы: без разделителей пути и запрещённых знаков, не «.», «..» и не пустое."""
    n = Path(name).name
    n = n[:-len('.kicad_pcb')] if n.lower().endswith('.kicad_pcb') else Path(n).stem
    s = ''.join('_' if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in n)
    s = s.strip(' .')
    if not s:
        return 'board'
    return '_' + s if s.split('.')[0].upper() in _RESERVED else s


def reports_root(cfg: dict) -> Path:
    """Папка отчётов из config.json; относительный путь — от папки плагина."""
    p = Path(cfg.get('reports') or 'reports')
    return p if p.is_absolute() else HERE / p


def prepare_out(out: Path):
    """Папка отчёта к новой проверке: старый отчёт не должен сойти за новый. Удаляются только файлы, которые
    пишет `dfm check` (report.html, findings.json, report.md, layers/*.png), и только в папке с меткой
    .dfm-report или в папке, где кроме них ничего нет. Чужая папка (например, reports указывает на папку
    проектов и папка проекта совпала с именем платы) не очищается и не используется — ошибка."""
    if out.exists() and not out.is_dir():
        raise RuntimeError(f'{out} — файл, а не папка отчёта')
    if out.exists():
        names = {f.name for f in out.iterdir()}
        foreign = names - {*REPORT_FILES, 'layers', REPORT_MARK}
        if REPORT_MARK not in names and foreign:
            raise RuntimeError(f'папка {out} создана не плагином (нет метки {REPORT_MARK}) и в ней есть другие файлы '
                               f'({", ".join(sorted(foreign)[:5])}) — она не очищается; укажите в config.json '
                               f'другую папку reports')
        for n in REPORT_FILES:
            if (out / n).is_file():
                (out / n).unlink()                     # не удалился — исключение, причина в журнале
        lay = out / 'layers'
        if lay.is_dir() and not lay.is_symlink() and not getattr(lay, 'is_junction', lambda: False)():   # 3.12+
            for f in lay.glob('*.png'):
                f.unlink()
            try:
                lay.rmdir()
            except OSError:                            # там есть что-то ещё — не наше, остаётся
                pass
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_MARK).write_text('папка отчёта плагина DFM: файлы отчёта здесь перезаписываются\n',
                                   encoding='utf-8')


B1_METHODS = {'nn': 'гибрид: формула клина + поправка нейросети', 'formula': 'формула клина по эффективному углу',
              'doc_table': 'таблица документа B1 (медь 35 мкм)'}


def options(python: str, dirs: list[str]) -> dict:
    """Профили завода, классы и доступность гибрида B1 — от пакета dfm (`python -m dfm profiles`)."""
    cmd = [python, '-m', 'dfm', 'profiles', *sum((['--dir', d] for d in dirs), [])]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    r = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', creationflags=flags,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8'), timeout=120)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr).strip()[-2000:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    out['b1_methods'] = B1_METHODS if out.get('nn') else {k: v for k, v in B1_METHODS.items() if k != 'nn'}
    return out


def check_args(choice: dict, opts: dict) -> list[str]:
    """Выбор со страницы настроек → ключи `dfm check`; только значения из списков (не произвольные ключи)."""
    prof = {p['path']: p for p in opts['profiles'] if 'error' not in p}
    path = choice.get('profile', '')
    if path not in prof:
        raise ValueError(f'неизвестный профиль: {path!r}')
    args = ['--profile', path] if path else []
    classes = list(dict.fromkeys(c for c in choice.get('classes', []) if c in opts['classes']))   # без повторов
    if not classes:
        raise ValueError('не выбран ни один класс проверок')
    if set(classes) != set(opts['classes']):
        args += ['--classes', ','.join(classes)]
    if 'B1' in classes:
        proc = choice.get('b1_process') or prof[path]['b1_process']
        if proc not in prof[path]['b1_processes']:
            raise ValueError(f'неизвестный процесс травления: {proc!r}')
        method = choice.get('b1_method', 'nn')
        if method not in opts['b1_methods']:
            raise ValueError(f'неизвестный способ оценки B1: {method!r}')
        args += ['--set', f'b1.process={proc}', '--set', f'b1.method={method}']
    changes = choice.get('parameters', {})
    allowed = prof[path].get('parameters', {}).get('fields', {})
    if not isinstance(changes, dict):
        raise ValueError('ожидается список значений параметров производства')
    for key, value in changes.items():
        if key not in allowed or type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f'недопустимый параметр производства: {key}')
        if value < 0 or (allowed[key]['positive'] and value == 0):
            raise ValueError(f'значение вне диапазона: {key}')
        args.extend(['--set', f'{key}={value}'])
    return args


# ---------- работа с KiCad ----------

class KiCadSide:
    """Связь с открытой платой через IPC API (kipy)."""

    def __init__(self):
        from kipy import KiCad
        self.kicad = KiCad(client_name='dfm-check')
        self.board = self.kicad.get_board()
        self.lock = threading.Lock()

    def snapshot(self, dst: Path) -> Path:
        """Плата из редактора + правила проекта → dst/<имя>.kicad_pcb."""
        name = Path(self.board.name).name or 'board.kicad_pcb'
        pcb = dst / name
        with self.lock:                                # вызовы IPC — по одному (select идёт из другого потока)
            text = self.board.get_as_string()
            try:
                proj_dir = Path(self.board.get_project().path)
            except Exception:
                proj_dir = None
        pcb.write_text(text, encoding='utf-8')
        if proj_dir and proj_dir.is_dir():
            for ext in ('.kicad_pro', '.kicad_dru'):
                src = proj_dir / (Path(name).stem + ext)
                if src.exists():
                    shutil.copy(src, pcb.with_suffix(ext))
        return pcb

    def select(self, uids: list[str], refs: list[str]) -> int:
        from kipy.proto.common.types import KIID
        with self.lock:
            items = []
            if uids:
                items += list(self.board.get_items_by_id([KIID(value=u) for u in uids]))
            if refs:
                want = set(refs)
                items += [fp for fp in self.board.get_footprints() if fp.reference_field.text.value in want]
            self.board.clear_selection()
            if items:
                self.board.add_to_selection(items)
                try:
                    self.kicad.run_action(ZOOM_ACTION)
                except Exception:
                    pass
            return len(items)


# ---------- проверка в фоне ----------

class Job:
    def __init__(self, side, python: str, out_dir: Path, extra: list[str]):
        self.side = side
        self.python = python
        self.out = out_dir
        self.extra = extra
        self.state = 'не начата'
        self.log = []
        self.started = time.time()
        self.code = None
        self.proc = None
        self.lock = threading.Lock()

    def stop(self):
        """Сервер завершается — идущая проверка прерывается, а не остаётся сиротой."""
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()

    def start(self, args: list[str]) -> bool:
        """Новая проверка в фоне (False — предыдущая ещё идёт)."""
        with self.lock:
            if self.state == 'идёт':
                return False
            self.state = 'идёт'
            self.log = []
            self.started = time.time()
            self.code = None
        threading.Thread(target=self.run, args=(args,), daemon=True).start()
        return True

    def run(self, args: list[str]):
        report = self.out / 'report.html'
        try:
            prepare_out(self.out)                      # старый отчёт не должен сойти за новый
            t0 = time.time()
            with tempfile.TemporaryDirectory(prefix='dfm_') as tmp:
                pcb = self.side.snapshot(Path(tmp))
                self.log.append(f'плата: {Path(self.side.board.name).name}')
                cmd = [self.python, '-m', 'dfm', 'check', str(pcb), '-o', str(self.out), *args, *self.extra]
                self.log.append('dfm check ' + ' '.join(args))
                env = dict(os.environ, PYTHONIOENCODING='utf-8')
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                p = self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                                 encoding='utf-8', errors='replace', creationflags=flags)
                for line in p.stdout:
                    self.log.append(line.rstrip())
                self.code = p.wait()
            fresh = report.exists() and report.stat().st_mtime >= t0 - 2   # запас на грубые часы ФС
            # `dfm check`: 0 — без ошибок, 1 — есть находки-ошибки; иное (2 — неверные ключи, сбой) — не отчёт
            ok = fresh and self.code in (0, 1)
            self.state = 'готово' if ok else 'ошибка'
            if not ok:
                self.log.append(f'dfm check завершился с кодом {self.code}'
                                + ('' if fresh else ', отчёт не создан') + ' — см. журнал выше')
        except Exception as e:                        # покажем причину на странице хода
            self.state = 'ошибка'
            self.log.append(f'{type(e).__name__}: {e}')


UI_DIR = HERE / 'ui'
PAGE = (UI_DIR / 'settings.html').read_text(encoding='utf-8')
AGAIN = (UI_DIR / 'report-actions.html').read_text(encoding='utf-8')
TOKEN_HEADER = 'X-DFM-Token'


CALCULATOR_FILES = {
    'gerber.html', 'via.html', 'gerber.css', 'via.css', 'analysis-theme.css', 'gerber-engine.js',
    'gerber.js', 'via-view.js', 'via.js', 'calc-profile.js', 'jszip.min.js',
}


def board_files(side, python):
    """Экспортируем снимок редактора, включая несохранённую трассировку."""
    with tempfile.TemporaryDirectory(prefix='dfm_gerber_') as temp:
        board = side.snapshot(Path(temp))
        out = Path(temp) / 'layers'
        command = [python, '-m', 'dfm', 'export-gerber', str(board), '--out', str(out)]
        result = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace',
                                timeout=180, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[-1500:])
        files = []
        for path in sorted(out.iterdir()):
            # KiCad пишет медь как .gtl/.gbl/.g1 или .gbr. Назначение задано в заголовке X2.
            with path.open(encoding='utf-8', errors='replace') as stream:
                head = stream.read(4096)
                if path.suffix.lower() == '.drl' or '%TF.FileFunction,Copper,' in head:
                    files.append(dict(name=path.name, text=head + stream.read()))
        return files


def make_handler(job: Job, side, touch, cfg, token: str | None = None):
    """Обработчик запросов. Чужие страницы отсекаются: Host только 127.0.0.1/localhost с нашим портом
    (защита от DNS rebinding), а /start и /select — только с ключом сеанса из отданных нами страниц
    (заголовок X-DFM-Token; чужая страница его не знает, а свой заголовок требует CORS-разрешения)."""
    cache = {}
    token = token or secrets.token_urlsafe(24)

    def get_options():
        if 'opts' not in cache:
            cache['opts'] = options(job.python, cfg.get('profiles_dirs', []))
        return cache['opts']

    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_S                    # недосланное тело (большой Content-Length) не вешает поток

        def log_message(self, *a):
            pass

        def end_headers(self):                         # во всех ответах, и в send_error тоже
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "frame-ancestors 'none'")
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data):
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
            return self._send(200, body, 'application/json')

        def _host_ok(self) -> bool:
            port = self.server.server_address[1]
            return self.headers.get('Host', '').strip().lower() in (f'127.0.0.1:{port}', f'localhost:{port}')

        def _token_ok(self) -> bool:
            return hmac.compare_digest(self.headers.get(TOKEN_HEADER, '').encode('utf-8', 'replace'),
                                       token.encode('ascii'))

        def _deny(self, why: str):
            return self._send(403, why.encode('utf-8'), 'text/plain; charset=utf-8')

        def do_GET(self):
            if not self._host_ok():
                return self._deny('чужой Host')
            touch()
            u = urlparse(self.path)
            if u.path in ('/', '/index.html'):
                return self._send(200, PAGE.replace('__DFM_TOKEN__', token).encode('utf-8'), 'text/html; charset=utf-8')
            name = u.path.removeprefix('/')
            if name in CALCULATOR_FILES:
                data = (UI_DIR / name).read_bytes()
                suffix = Path(name).suffix
                mime = {'.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript'}[suffix]
                if suffix == '.html':
                    data = data.replace(b'__DFM_TOKEN__', token.encode('ascii'))
                return self._send(200, data, mime + '; charset=utf-8')
            if u.path == '/options':
                try:
                    last = load_config().get('last')
                    body = dict(get_options(), last=last if isinstance(last, dict) else {}, running=job.state == 'идёт')
                except Exception as e:
                    body = {'error': f'{type(e).__name__}: {e}'}
                return self._json(body)
            if u.path == '/status':
                s = {'state': job.state, 'elapsed': int(time.time() - job.started), 'log': job.log[-200:]}
                return self._json(s)
            if u.path == '/select':
                if not self._token_ok():
                    return self._deny('нет ключа сеанса')
                q = parse_qs(u.query)
                uids = [x for x in (q.get('uids', [''])[0]).split(',') if x]
                refs = [x for x in (q.get('refs', [''])[0]).split(',') if x]
                try:
                    n = side.select(uids, refs)
                    body = {'ok': True, 'selected': n}
                except Exception as e:
                    body = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
                return self._json(body)
            f = (job.out / u.path.lstrip('/')).resolve()
            if job.out.resolve() in f.parents and f.is_file():   # только файлы отчёта
                ctype = {'.html': 'text/html; charset=utf-8', '.json': 'application/json', '.png': 'image/png',
                         '.md': 'text/plain; charset=utf-8'}.get(f.suffix, 'application/octet-stream')
                body = f.read_bytes()
                if f.name == 'report.html':
                    body = body.replace(b'</body>', AGAIN.replace('__DFM_TOKEN__', token).encode('utf-8')
                                        + b'</body>', 1)
                return self._send(200, body, ctype)
            return self._send(404, 'нет такого файла'.encode('utf-8'), 'text/plain; charset=utf-8')

        def do_POST(self):
            try:
                n = max(int(self.headers.get('Content-Length') or 0), 0)
            except ValueError:
                n = 0
            if n > MAX_BODY:                           # не читаем: соединение закроется после ответа
                self.close_connection = True
                return self._send(413, 'слишком большой запрос'.encode('utf-8'), 'text/plain; charset=utf-8')
            raw = self.rfile.read(n)                   # тело читаем сразу: отказ до чтения рвёт соединение
            if not self._host_ok():
                return self._deny('чужой Host')
            endpoint = urlparse(self.path).path
            if endpoint not in ('/start', '/board-gerbers'):
                return self._send(404, b'', 'text/plain')
            # только JSON: простые межсайтовые запросы так не отправить (защита от запуска чужой страницей)
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                return self._send(415, b'', 'text/plain')
            if not self._token_ok():
                return self._deny('нет ключа сеанса')
            touch()
            try:
                choice = json.loads(raw.decode('utf-8'))
                if not isinstance(choice, dict):
                    raise ValueError('ожидается объект JSON')
                if endpoint == '/board-gerbers':
                    return self._json({'files': board_files(side, job.python)})
                args = check_args(choice, get_options())
                if not job.start(args):
                    raise RuntimeError('проверка уже идёт')
                try:                                   # не записался выбор — проверка всё равно идёт
                    save_last({k: choice.get(k) for k in ('profile', 'classes', 'b1_process', 'b1_method')})
                except Exception as e:
                    job.log.append(f'выбор не сохранён в config.json: {type(e).__name__}: {e}')
                body = {'ok': True}
            except Exception as e:
                body = {'ok': False, 'error': f'{e}'}
            return self._json(body)
    return Handler


HINT_IPC = ('Проверьте: KiCad → Настройки → Плагины → «Включить сервер IPC API» (после включения перезапустить '
            'KiCad); в редакторе плат открыта плата.')


def show_error(text: str, hint: str = HINT_IPC):
    """Не удалось начать (нет связи с KiCad, испорчен config.json) — страница с причиной вместо молчаливого выхода."""
    page = Path(tempfile.gettempdir()) / 'dfm_plugin_error.html'
    esc = lambda s: s.replace('&', '&amp;').replace('<', '&lt;')
    page.write_text(PAGE.split('<h1>')[0] + '<h1>Проверка технологичности: не удалось начать</h1><pre>'
                    + esc(text) + '</pre><p>' + esc(hint) + '</p></body></html>', encoding='utf-8')
    webbrowser.open(page.as_uri())


def main():
    try:
        cfg = load_config()
    except Exception as e:
        show_error(f'{CONFIG}: {type(e).__name__}: {e}',
                   'config.json плагина не читается: исправьте его или переустановите плагин '
                   '(python -m dfm plugin-install).')
        return
    python = cfg.get('python') or os.environ.get('DFM_PYTHON') or sys.executable
    try:
        side = KiCadSide()
    except Exception as e:
        show_error(f'{type(e).__name__}: {e}')
        return
    out = reports_root(cfg) / safe_stem(side.board.name)
    job = Job(side, python, out, cfg.get('extra_args', []))
    last = [time.time()]
    server = ThreadingHTTPServer(('127.0.0.1', 0),
                                 make_handler(job, side, lambda: last.__setitem__(0, time.time()), cfg))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webbrowser.open(f'http://127.0.0.1:{server.server_address[1]}/')
    while time.time() - last[0] < IDLE_LIMIT_S or job.state == 'идёт':   # идущую проверку не бросаем
        time.sleep(5)
    server.shutdown()
    job.stop()


if __name__ == '__main__':
    main()
