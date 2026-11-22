"""Плагин KiCad (шаги 3.2, 4.5) без KiCad: сторона KiCad подменена, проверяются страница настроек, запуск
проверки, сервер отчёта, выделение по запросу из отчёта, защита папки и установка."""
import importlib.util
import json
import re
import shutil
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A7 = ROOT / 'boards' / 'demo' / 'a7_demo.kicad_pcb'


def _plugin():
    spec = importlib.util.spec_from_file_location('dfm_plugin', ROOT / 'plugin' / 'dfm_plugin.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeBoard:
    name = 'a7_demo.kicad_pcb'


class FakeSide:
    """Вместо KiCad: снимок — копия демо-платы, выделение — запоминание запроса."""
    board = FakeBoard()

    def __init__(self):
        self.calls = []

    def snapshot(self, dst):
        p = dst / 'a7_demo.kicad_pcb'
        shutil.copy(A7, p)
        return p

    def select(self, uids, refs):
        self.calls.append((uids, refs))
        return len(uids) + len(refs)


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as r:
        return r.status, r.read().decode('utf-8')


def _post(url, body, ctype='application/json', token=None, headers=None):
    h = {'Content-Type': ctype, **({'X-DFM-Token': token} if token else {}), **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=h)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


def _code(fn, *a, **kw):
    """HTTP-код отказа (или 200, если запрос принят)."""
    try:
        fn(*a, **kw)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def _token(page):
    return re.search(r"const TOKEN\s*=\s*'([^']+)'", page).group(1)


def test_plugin_flow(tmp_path):
    """Настройки → «Проверить» → ход → отчёт с кнопкой повторной проверки; выбор запоминается."""
    import time
    mod = _plugin()
    mod.CONFIG = tmp_path / 'config.json'
    side = FakeSide()
    job = mod.Job(side, sys.executable, tmp_path / 'out', [])
    server = ThreadingHTTPServer(('127.0.0.1', 0), mod.make_handler(job, side, lambda: None, {}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        code, page = _get(base + '/')
        assert code == 200 and 'Проверка технологичности' in page and 'id="profile"' in page
        tok = _token(page)
        assert "join('\\n')" in page                              # перевод строки в JS — экранирован, не сырой
        opts = json.loads(_get(base + '/options')[1])
        assert {'A1', 'A7', 'B1'} <= set(opts['classes']) and opts['profiles'][0]['path'] == ''
        assert set(opts['b1_methods']) >= {'formula', 'doc_table'}
        bad = _post(base + '/start', dict(profile='C:/чужой.toml', classes=['A7']), token=tok)
        assert not bad['ok'] and 'профиль' in bad['error']                # только профили из списка
        # не JSON — отказ (защита от чужих страниц)
        assert _code(_post, base + '/start', dict(profile='', classes=['A7']), ctype='text/plain', token=tok) == 415
        assert _post(base + '/start', dict(profile='', classes=['A7', 'B1'], b1_process='О', b1_method='formula'),
                     token=tok)['ok']
        for _ in range(600):
            st = json.loads(_get(base + '/status')[1])
            if st['state'] != 'идёт':
                break
            time.sleep(0.2)
        assert st['state'] == 'готово', st['log']
        assert any('b1.process=О' in line for line in st['log']) and any('находок' in line for line in st['log'])
        code, rep = _get(base + '/report.html')
        assert code == 200 and 'button class="kicad"' in rep and 'Проверить снова' in rep
        assert re.search(rf"\.set\('X-DFM-Token',\s*'{re.escape(tok)}'\)", rep)             # кнопки отчёта шлют ключ сеанса
        assert re.search(r'\.origin\s*===\s*location\.origin', rep)                  # …только на свой адрес
        assert (tmp_path / 'out' / '.dfm-report').exists()        # папка отчёта помечена как своя
        last = json.loads(mod.CONFIG.read_text(encoding='utf-8'))['last']
        assert last == dict(profile='', classes=['A7', 'B1'], b1_process='О', b1_method='formula')
        assert json.loads(_get(base + '/options')[1])['last']['b1_process'] == 'О'
        res = json.loads(_get(base + '/select?uids=a,b&refs=U3', {'X-DFM-Token': tok})[1])
        assert res == {'ok': True, 'selected': 3} and side.calls == [(['a', 'b'], ['U3'])]
        try:                                                      # за пределы папки отчёта — нельзя
            _get(base + '/..%2F..%2Fplugin.json')
            assert False, 'файл вне отчёта отдан'
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()


def test_plugin_install(tmp_path):
    from dfm.plugin_install import install
    dest = install(tmp_path / 'ru.dfm.check', python='C:/py/python.exe')
    assert (dest / 'plugin.json').exists() and (dest / 'dfm_plugin.py').exists()
    cfg = json.loads((dest / 'config.json').read_text(encoding='utf-8'))
    assert cfg['python'] == 'C:/py/python.exe'
    cfg['last'] = {'b1_process': 'О'}                           # выбор на странице настроек переживает переустановку
    (dest / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    dest = install(dest, python='C:/py/python.exe')
    assert json.loads((dest / 'config.json').read_text(encoding='utf-8'))['last'] == {'b1_process': 'О'}
    from kipy.packaging import validate
    assert validate(dest).errors == []                          # схема plugin.json KiCad


def test_plugin_install_keeps_keys(tmp_path):
    """Переустановка меняет только python: своя папка отчётов и прочие ключи остаются, отчёты не удаляются."""
    from dfm.plugin_install import install
    dest = install(tmp_path / 'ru.dfm.check', python='C:/py/old.exe')
    cfg = json.loads((dest / 'config.json').read_text(encoding='utf-8'))
    cfg.update(reports=str(dest / 'мои_отчёты'), extra_args=['--no-pictures'], my_key=1)
    (dest / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    (dest / 'мои_отчёты').mkdir()
    (dest / 'мои_отчёты' / 'r.html').write_text('x', encoding='utf-8')
    (dest / 'лишний.txt').write_text('x', encoding='utf-8')
    install(dest, python='C:/py/new.exe')
    new = json.loads((dest / 'config.json').read_text(encoding='utf-8'))
    assert new == dict(cfg, python='C:/py/new.exe')
    assert (dest / 'мои_отчёты' / 'r.html').exists() and not (dest / 'лишний.txt').exists()


def test_plugin_install_refuses_foreign_dir(tmp_path):
    """--dest с чужими файлами (например, «Документы» по ошибке) — отказ, ничего не удалено."""
    import pytest
    from dfm.plugin_install import InstallError, install
    docs = tmp_path / 'Документы'
    docs.mkdir()
    (docs / 'диплом.docx').write_text('x', encoding='utf-8')
    with pytest.raises(InstallError, match='не папка плагина'):
        install(docs, python='C:/py/python.exe')
    other = tmp_path / 'other'                                  # plugin.json чужого плагина — тоже отказ
    other.mkdir()
    (other / 'plugin.json').write_text(json.dumps({'identifier': 'com.other'}), encoding='utf-8')
    with pytest.raises(InstallError):
        install(other, python='C:/py/python.exe')
    assert (docs / 'диплом.docx').exists() and (other / 'plugin.json').exists()
    empty = tmp_path / 'empty'                                  # пустая папка — можно
    empty.mkdir()
    assert (install(empty, python='C:/py/python.exe') / 'plugin.json').exists()
    from dfm.__main__ import main                               # из командной строки — ошибка argparse, код 2
    with pytest.raises(SystemExit) as e:
        main(['plugin-install', '--dest', str(docs)])
    assert e.value.code == 2 and (docs / 'диплом.docx').exists()


OPTS = dict(profiles=[dict(path='', name='усл', b1_process='З', b1_processes=['С', 'О', 'З'])],
            classes=['A1', 'A3', 'A5', 'A7', 'B1'], b1_methods={'nn': '', 'formula': ''})


def test_check_args_classes():
    """Повторы классов не превращаются в «все классы» (B1 без --set)."""
    mod = _plugin()
    assert mod.check_args(dict(profile='', classes=['A1'] * 5), OPTS) == ['--classes', 'A1']
    assert mod.check_args(dict(profile='', classes=['A1', 'A1', 'A3', 'A5', 'A7']), OPTS) == \
        ['--classes', 'A1,A3,A5,A7']
    all5 = mod.check_args(dict(profile='', classes=['B1', 'A1', 'A3', 'A5', 'A7', 'B1']), OPTS)
    assert '--classes' not in all5 and all5[-4:] == ['--set', 'b1.process=З', '--set', 'b1.method=nn']
    for bad in (dict(profile='', classes=['X1']), dict(profile='', classes=['B1'], b1_process='Ж'),
                dict(profile='', classes=['B1'], b1_method='rm -rf')):
        try:
            mod.check_args(bad, OPTS)
            assert False, bad
        except ValueError:
            pass


def test_plugin_server_guards(tmp_path):
    """Чужой Host (DNS rebinding) — 403 на всё; /start и /select без ключа сеанса — 403."""
    mod = _plugin()
    mod.CONFIG = tmp_path / 'config.json'
    side = FakeSide()
    job = mod.Job(side, sys.executable, tmp_path / 'out', [])
    server = ThreadingHTTPServer(('127.0.0.1', 0), mod.make_handler(job, side, lambda: None, {}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    base = f'http://127.0.0.1:{port}'
    try:
        tok = _token(_get(base + '/')[1])
        assert _get(base + '/', {'Host': f'localhost:{port}'})[0] == 200
        evil = {'Host': f'evil.example:{port}'}
        for path in ('/', '/status', '/report.html', '/findings.json', '/select?refs=U1'):
            assert _code(_get, base + path, evil) == 403, path
        assert _code(_get, base + '/select?refs=U1', {'Host': f'127.0.0.1:{port}', 'X-DFM-Token': tok}) == 200
        assert _code(_get, base + '/', {'Host': '127.0.0.1:1'}) == 403            # чужой порт
        assert _code(_post, base + '/start', dict(profile='', classes=['A7']), token=tok, headers=evil) == 403
        assert _code(_post, base + '/start', dict(profile='', classes=['A7'])) == 403            # без ключа
        assert _code(_post, base + '/start', dict(profile='', classes=['A7']), token='x' * 32) == 403
        assert _code(_get, base + '/select?refs=U1') == 403
        assert _code(_get, base + '/select?refs=U1', {'X-DFM-Token': 'wrong\xe9'}) == 403
        assert side.calls == [([], ['U1'])] and job.state == 'не начата'
    finally:
        server.shutdown()


def test_job_stale_report(tmp_path):
    """Проверка упала — старый report.html не выдаётся за новый."""
    mod = _plugin()
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'report.html').write_text('старый', encoding='utf-8')
    job = mod.Job(FakeSide(), sys.executable, out, ['--нет-такого-ключа'])
    job.run([])
    assert job.state == 'ошибка' and not (out / 'report.html').exists()


def test_safe_stem():
    """Имя папки отчёта: «.», «..», пустое, разделители пути и имена устройств Windows — безопасно."""
    mod = _plugin()
    assert mod.safe_stem('..kicad_pcb') == 'board' and mod.safe_stem('...kicad_pcb') == 'board'
    assert mod.safe_stem('') == 'board' and mod.safe_stem('.kicad_pcb') == 'board'
    assert mod.safe_stem('C:/проекты/плата.kicad_pcb') == 'плата' and mod.safe_stem('a\\b\\x.kicad_pcb') == 'x'
    assert mod.safe_stem('con.kicad_pcb') == '_con' and mod.safe_stem('a?b.kicad_pcb') == 'a_b'


def test_report_dir_protected(tmp_path):
    """Чужая папка (папка проекта с именем платы) не очищается; в своей удаляются только файлы отчёта."""
    mod = _plugin()
    proj = tmp_path / 'проект'
    proj.mkdir()
    for n in ('a7_demo.kicad_pro', 'a7_demo.kicad_pcb', 'report.html'):
        (proj / n).write_text('x', encoding='utf-8')
    job = mod.Job(FakeSide(), sys.executable, proj, [])
    job.run([])
    assert job.state == 'ошибка' and any('не очищается' in s for s in job.log)
    assert sorted(f.name for f in proj.iterdir()) == ['a7_demo.kicad_pcb', 'a7_demo.kicad_pro', 'report.html']
    own = tmp_path / 'свой'                                     # помечена — чужое внутри остаётся
    (own / 'layers').mkdir(parents=True)
    for n in ('.dfm-report', 'report.html', 'findings.json', 'report.md', 'заметки.txt', 'layers/F_Cu.png',
              'layers/моё.txt'):
        (own / n).write_text('x', encoding='utf-8')
    mod.prepare_out(own)
    assert sorted(str(f.relative_to(own)).replace('\\', '/') for f in own.rglob('*')) == \
        ['.dfm-report', 'layers', 'layers/моё.txt', 'заметки.txt']
    old = tmp_path / 'старый'                                   # без метки, но только файлы отчёта — очищается
    (old / 'layers').mkdir(parents=True)
    for n in ('report.html', 'layers/F_Cu.png'):
        (old / n).write_text('x', encoding='utf-8')
    mod.prepare_out(old)
    assert [f.name for f in old.iterdir()] == ['.dfm-report']
    f = tmp_path / 'файл'
    f.write_text('x', encoding='utf-8')
    try:
        mod.prepare_out(f)
        assert False
    except RuntimeError:
        assert f.read_text(encoding='utf-8') == 'x'


def test_job_exit_code(tmp_path, monkeypatch):
    """Свежий report.html — удача только при коде 0 или 1; иначе (2 — ключи, сбой) — ошибка."""
    mod = _plugin()

    class FakePopen:
        code = 0

        def __init__(self, cmd, **kw):
            out = Path(cmd[cmd.index('-o') + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / 'report.html').write_text('новый', encoding='utf-8')
            self.stdout = iter(['журнал\n'])

        def wait(self):
            return self.code

        def poll(self):
            return self.code

    for code, state in ((0, 'готово'), (1, 'готово'), (2, 'ошибка'), (3, 'ошибка'), (-1, 'ошибка')):
        monkeypatch.setattr(mod.subprocess, 'Popen', type('P', (FakePopen,), {'code': code}))
        job = mod.Job(FakeSide(), sys.executable, tmp_path / 'out', [])
        job.run([])
        assert job.state == state and job.code == code, (code, job.log)
        if state == 'ошибка':
            assert f'кодом {code}' in job.log[-1]


def test_request_limits(tmp_path):
    """Огромный Content-Length — 413 сразу; недосланное тело не держит поток дольше тайм-аута;
    заголовки против встраивания страницы во фрейм и подмены типа — во всех ответах."""
    import socket
    import time
    mod = _plugin()
    mod.REQUEST_TIMEOUT_S = 1
    mod.CONFIG = tmp_path / 'config.json'
    side = FakeSide()
    job = mod.Job(side, sys.executable, tmp_path / 'out', [])
    server = ThreadingHTTPServer(('127.0.0.1', 0), mod.make_handler(job, side, lambda: None, {}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def raw(length, body):
        s = socket.create_connection(('127.0.0.1', port), timeout=15)
        s.sendall(f'POST /start HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n'
                  f'Content-Length: {length}\r\n\r\n'.encode() + body)
        data = b''
        while chunk := s.recv(4096):
            data += chunk
        s.close()
        return data
    try:
        assert raw(10 ** 9, b'{}').split(b'\r\n')[0].split()[1] == b'413'
        t0 = time.time()
        raw(100, b'{"a"')                                         # тело короче заявленного
        assert time.time() - t0 < 8
        for path in ('/', '/nope'):
            try:
                r = urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=10)
            except urllib.error.HTTPError as e:
                r = e
            assert r.headers['X-Frame-Options'] == 'DENY' and r.headers['X-Content-Type-Options'] == 'nosniff'
            assert r.headers['Content-Security-Policy'] == "frame-ancestors 'none'"
        assert job.state == 'не начата'
    finally:
        server.shutdown()


def test_plugin_install_nested_reports(tmp_path):
    """Папка отчётов глубже в папке плагина (и путь к ней) переживает переустановку; испорченный config.json
    не заменяется молча; файл вместо папки — понятное сообщение."""
    import pytest
    from dfm.plugin_install import InstallError, install
    dest = install(tmp_path / 'ru.dfm.check', python='C:/py/python.exe')
    cfg = json.loads((dest / 'config.json').read_text(encoding='utf-8'))
    for reports in (str(dest / 'data' / 'reps'), 'rel/reps'):  # абсолютный и относительный (от папки плагина)
        cfg['reports'] = reports
        (dest / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
        rep = dest / reports
        (rep / 'плата').mkdir(parents=True, exist_ok=True)
        (rep / 'плата' / 'report.html').write_text('x', encoding='utf-8')
        install(dest, python='C:/py/python.exe')
        assert (rep / 'плата' / 'report.html').exists() and (dest / 'plugin.json').exists()
    (dest / 'config.json').write_text('{испорчен', encoding='utf-8')
    with pytest.raises(InstallError, match='не читается'):
        install(dest, python='C:/py/python.exe')
    assert (dest / 'config.json').read_text(encoding='utf-8') == '{испорчен'
    f = tmp_path / 'файл'
    f.write_text('x', encoding='utf-8')
    with pytest.raises(InstallError, match='файл, а не папка'):
        install(f, python='C:/py/python.exe')


def test_page_scripts_parse(tmp_path):
    """Скрипты страницы настроек и вставки в отчёт — синтаксически верный JS (если есть node)."""
    import subprocess
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('нет node')
    mod = _plugin()
    for i, src in enumerate((mod.PAGE, mod.AGAIN)):
        for j, js in enumerate(re.findall(r'<script>(.*?)</script>', src, re.S)):
            p = tmp_path / f's{i}_{j}.js'
            p.write_text(js, encoding='utf-8')
            r = subprocess.run([node, '--check', str(p)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
