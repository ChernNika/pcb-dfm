"""Генератор данных для нейросети B1 (документ B1, глава 7; шаг 4.2).

Пример = карман + профиль процесса; метку даёт численная модель (dfm/ml/etch.py), а не угол.

Источники карманов:
  * синтетика — простые ситуации Л1–Л6 со случайными параметрами (угол 10–90°, смещённый к острым, ширина трассы
    0,1–0,3 мм, длина хвоста, размер и форма площадки, соседняя чужая медь);
  * реальные платы — все кандидаты B1 (`trap_candidates`, тот же перебор, что в проверке).
Угол синтетики: 60 % — 10–40°, остальные — 40–90° (острые карманы редки и важнее всего).
Профиль: Da — логарифмически равномерно 0,1–30, δ — 30–80 мкм, медь 18 / 35 / 70 мкм, фактор
травления 2–4 (u = t/EF). Промывка u_r в расчёт не входит: сохраняются стенки a(s), b(s), w(s),
метка для любого u_r — max (a + u_r·b)/w (`etch.labels_for`).

Предел времени пребывания: в метках y T на стенках обрезано до 1,25·(1 + A(θ)) — предела клина того же
угла (`etch.t_cap`); метки без предела — в y_raw. Причина: у 12 из 2949 примеров первого прогона с
углом 32–84° T_max = 14–40 (в 2,6–19 раз больше, чем у клина) из-за узких щелей своей цепи вдоль стенки,
и член u_r·(T − 1) давал «опасный» перетрав, которого не даёт сам угол. `relabel` пересчитывает метки
готовых примеров по сохранённым стенкам.
Острые углы: в первом прогоне измеренный угол кармана обычно больше заданного, и θ < 25° вышло лишь
у 218 примеров, хотя опасность почти вся там; `--sharp N` добавляет N синтетических примеров
с заданным углом 8–28° (id a…, группы sharp…).

Каждый пример — отдельный файл .npz (без pickle) в папке данных: прерванный прогон продолжается
с места остановки. Щели при δ примера (`is_slot`) хранятся с признаком `slot` и в обучение
не идут. Запуск:

    python -m dfm.ml.dataset run --out C:/Users/<…>/dfm_data/b1 --synth 2000 --real --workers 6
    python -m dfm.ml.dataset run --out … --synth 2000 --real --sharp 800 --workers 6   # + острые углы
    python -m dfm.ml.dataset relabel --out …
    python -m dfm.ml.dataset summary --out …
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import unary_union

from ..rules.b1 import POCKET_GEOM

__all__ = ['sample_profile', 'synth_scene', 'board_candidates', 'patch', 'make_tasks', 'run_task', 'is_slot',
           'load_example']

WIN = 0.7                                  # полуразмер окна меди, сохраняемого с примером, мм
B1_GEOM = dict(POCKET_GEOM, theta_max_deg=90.0)      # поиск кандидатов — как в проверке (dfm/rules/b1.py)


# ---------- профиль процесса ----------

def sample_profile(rng):
    t = float(rng.choice([18.0, 35.0, 70.0]))
    ef = float(rng.uniform(2.0, 4.0))
    return dict(Da=float(math.exp(rng.uniform(math.log(0.1), math.log(30.0)))),
                delta=float(rng.uniform(30.0, 80.0)), t=t, ef=ef, u=t / ef)


# ---------- синтетические ситуации ----------

def _trace(pts, w):
    return shapely.LineString(pts).buffer(w / 2, quad_segs=16)


def _polar(p, deg, L):
    return (p[0] + L * math.cos(math.radians(deg)), p[1] + L * math.sin(math.radians(deg)))


KINDS = ('Л1', 'Л2', 'Л3', 'Л4', 'Л5', 'Л6')


def synth_scene(rng, kind=None, theta_range=None):
    """Случайная ситуация → (своя медь, чужая медь или None, параметры). Стык — около начала координат.
    theta_range — заданный угол (для Л5 — шаг веера) равномерно в этом диапазоне, градусы."""
    kind = kind or str(rng.choice(KINDS))
    if theta_range is not None:
        th = float(rng.uniform(*theta_range))
    else:                       # острые карманы редки и важнее всего: 60 % углов — 10–40°, остальные — 40–90°
        th = float(rng.uniform(10, 40) if rng.random() < 0.6 else rng.uniform(40, 90))
    w = float(rng.uniform(0.1, 0.3))
    # у острого клина вершина кармана (встреча краёв трасс) уходит от стыка на ≈ (w₁ + w₂)/(2 sin θ) —
    # острая партия удлиняет трассы и площадки на эту величину (обычные сцены — без изменений, ext = 0)
    ext = 0.3 / math.sin(math.radians(th)) if theta_range is not None else 0.0
    par = dict(kind=kind, theta_set=th, w=w)
    if kind == 'Л1':                                   # трасса к прямоугольной площадке
        pw, ph = float(rng.uniform(0.4, 1.6)), float(rng.uniform(0.3, 1.2))
        x0 = float(rng.uniform(-0.3, 0.3)) * pw
        own = unary_union([shapely.box(-pw / 2, -ph, pw / 2 + ext, 0),
                           _trace([(x0, -w / 2), _polar((x0, -w / 2), th, 1.4 + ext)], w)])
        par.update(pad=(pw, ph))
    elif kind == 'Л6':                                 # трасса к круглой площадке
        r = float(rng.uniform(0.2, 0.6))
        off = float(rng.uniform(0.0, 0.8)) * r
        p0 = (-off * math.sin(math.radians(th)), off * math.cos(math.radians(th)))
        p1 = _polar((0, 0), th, 1.5)
        if theta_range is not None:
            # угол кармана задаёт касание: острый карман — между ВНУТРЕННИМ (ближним к центру) краем трассы
            # и окружностью, его угол arccos(p/r) при расстоянии края p = r·cos θ; трасса идёт по своей
            # прямой (направление θ), а не к точке луча через центр
            off = r * math.cos(math.radians(th)) + w / 2
            p0 = (-off * math.sin(math.radians(th)), off * math.cos(math.radians(th)))
            p1 = _polar(p0, th, 1.5 + ext)
        own = unary_union([shapely.Point(0, 0).buffer(r, quad_segs=32), _trace([p0, p1], w)])
        par.update(r=r, off=off)
    elif kind in ('Л2', 'Л3'):                         # ответвление / хвост за стыком
        w2 = float(rng.uniform(0.1, 0.3))
        tail = float(rng.uniform(0.6, 1.2)) if kind == 'Л2' else float(rng.uniform(0.02, 0.4))
        tail_l = tail + ext if kind == 'Л2' else tail  # Л3 — короткий хвост по определению
        own = unary_union([_trace([(-1.2, 0), (tail_l, 0)], w2), _trace([(0, 0), _polar((0, 0), th, 1.4 + ext)], w)])
        par.update(w2=w2, tail=tail)
    elif kind == 'Л4':                                 # вход трассы в заливку
        own = unary_union([shapely.box(-1.5, -1.2, 1.5 + ext, 0), _trace([(0, -w), _polar((0, -w), th, 1.4 + ext)], w)])
    else:                                              # Л5: веер из одной площадки
        n = int(rng.integers(2, 4))
        r = float(rng.uniform(0.15, 0.35))
        angs = [float(rng.uniform(20, 160))]
        for _ in range(n - 1):
            angs.append(angs[-1] + float(rng.uniform(*(theta_range or (10, 60)))))
        if theta_range is not None:
            ext = 0.3 / math.sin(math.radians(min(np.diff(angs))))
        own = unary_union([shapely.Point(0, 0).buffer(r, quad_segs=32)] +
                          [_trace([(0, 0), _polar((0, 0), a, 1.4 + ext)], float(rng.uniform(0.1, 0.3))) for a in angs])
        par.update(r=r, fan=angs)
        th = min(np.diff(angs)) if n > 1 else th
        par['theta_set'] = float(th)
    par['ext'] = ext
    foreign = None
    if rng.random() < 0.4:                             # соседняя чужая медь со стороны кармана
        bis = (par.get('fan', [0])[0] + th / 2) if kind == 'Л5' else th / 2
        d = float(rng.uniform(0.1, 0.45))
        c = _polar((0, 0), bis, ext + d + float(rng.uniform(0.15, 0.35)))
        L = float(rng.uniform(0.3, 1.2))
        ang = float(rng.uniform(0, 180))
        foreign = _trace([_polar(c, ang, -L / 2), _polar(c, ang, L / 2)], float(rng.uniform(0.1, 0.3))).difference(
            own.buffer(0.1))
        foreign = None if foreign.is_empty else foreign
        par.update(foreign=True)
    return own, foreign, par


# ---------- реальные платы ----------

def board_candidates(path, max_per_board=None, seed=0):
    """Кандидаты B1 со всех медных слоёв платы → список (своя медь, чужая медь, кандидат, метаданные)."""
    from ..board import load
    from ..geometry.copper import BoardGeometry
    from ..rules.b1 import _situation, trap_candidates
    b = load(path)
    geo = BoardGeometry(b)
    out = []
    for layer in b.copper:
        lc = geo.layer(layer)
        geom_of = {id(f): g for f, g in zip(lc.features, lc.feature_geoms)}
        for net, g in lc.nets.items():
            if net is None:
                continue
            for c in trap_candidates(g, B1_GEOM):
                if c['slot'] is not None:
                    continue
                v = c['v']
                box = shapely.box(v[0] - WIN, v[1] - WIN, v[0] + WIN, v[1] + WIN)
                own = g.intersection(box)
                foreign = unary_union([gg.intersection(box) for n, gg in lc.nets.items()
                                       if n != net and gg.intersects(box)] or [shapely.Polygon()])
                code, w = _situation(lc, geom_of, v, net)
                out.append((own, None if foreign.is_empty else foreign, c,
                            dict(kind=code, w=w, layer=layer, net=net, board=Path(path).stem)))
    if max_per_board and len(out) > max_per_board:
        rng = np.random.default_rng(seed)
        out = [out[i] for i in sorted(rng.choice(len(out), max_per_board, replace=False))]
    return out


# ---------- растровый фрагмент (вход сети) ----------

def patch(own, foreign, v, beta_deg, delta_um=50.0, size=0.5, n=64):
    """Фрагмент size × size мм вокруг вершины v, повёрнутый биссектрисой кармана вверх (документ B1,
    раздел 8.2): каналы — своя медь, чужая медь, расстояние до ближайшей меди / δ (до 4)."""
    xs = (np.arange(n) + 0.5) / n * size - size / 2
    Xp, Yp = np.meshgrid(xs, xs[::-1])                 # строка 0 — верх фрагмента (+y')
    g = math.radians(beta_deg - 90.0)
    c, s = math.cos(g), math.sin(g)
    pts = shapely.points(v[0] + c * Xp.ravel() - s * Yp.ravel(), v[1] + s * Xp.ravel() + c * Yp.ravel())
    out = np.zeros((3, n, n), np.float32)
    allcu = own if foreign is None else unary_union([own, foreign])
    for k, geom in enumerate((own, foreign)):
        if geom is not None and not geom.is_empty:
            shapely.prepare(geom)
            out[k] = shapely.contains(geom, pts).reshape(n, n)
    out[2] = np.minimum(shapely.distance(allcu, pts).reshape(n, n) * 1000 / delta_um, 4.0)
    return out


# ---------- задачи и выполнение ----------

SHARP_RANGE = (8.0, 28.0)


def make_tasks(n_synth, boards=(), per_real=2, max_per_board=150, seed=2026, n_sharp=0, sharp_range=SHARP_RANGE):
    """Детерминированный список задач: синтетика (по зерну), карманы реальных плат (по нескольку профилей)
    и добавочная синтетика острых углов (id a…, своё зерно — не меняет остальные задачи)."""
    rng = np.random.default_rng(seed)
    tasks = [dict(id=f's{i:05d}', src='synth', seed=int(rng.integers(2 ** 31)), group=f'synth{i % 50:02d}')
             for i in range(n_synth)]
    for path in boards:
        cands = board_candidates(path, max_per_board, seed)
        for j, (own, foreign, c, meta) in enumerate(cands):
            for k in range(per_real):
                tasks.append(dict(id=f'{meta["board"]}_{j:04d}_{k}', src='real', seed=int(rng.integers(2 ** 31)),
                                  group=meta['board'], own=own, foreign=foreign, cand=c, meta=meta))
    rs = np.random.default_rng([seed, 1])
    tasks += [dict(id=f'a{i:05d}', src='synth', seed=int(rs.integers(2 ** 31)), group=f'sharp{i % 25:02d}',
                   theta_range=tuple(sharp_range)) for i in range(n_sharp)]
    return tasks


def wall_labels(walls, theta_deg, urs=(0.0, 0.5, 2.0)):
    """Метки примера при нескольких u_r: с пределом времени пребывания (`etch.t_cap`) и без него (y_raw)."""
    from .etch import labels_for, t_cap
    tc = t_cap(theta_deg)
    lab = {ur: labels_for(walls, ur, t_max=tc) for ur in urs}
    raw = {ur: labels_for(walls, ur) for ur in urs}
    return dict(T_cap=tc, y={str(k): x['y'] for k, x in lab.items()},
                notch={str(k): x['notch'] for k, x in lab.items()},
                y_raw={str(k): x['y'] for k, x in raw.items()})


def _wkb(g):
    return np.frombuffer(shapely.to_wkb(g), np.uint8) if g is not None and not g.is_empty else np.zeros(0, np.uint8)


def run_task(task, out_dir):
    """Один расчёт: карман + профиль → файл <id>.npz. Возвращает краткую сводку или None."""
    from ..rules.b1 import theta_from_area, trap_candidates
    from .etch import simulate
    path = Path(out_dir) / f'{task["id"]}.npz'
    if path.exists():
        return None
    rng = np.random.default_rng(task['seed'])
    if task['src'] == 'synth':
        for _ in range(20):                            # сцена без кармана — повторить
            own, foreign, meta = synth_scene(rng, theta_range=task.get('theta_range'))
            cands = [c for c in trap_candidates(own, B1_GEOM)
                     if c['slot'] is None and math.hypot(*c['v']) < 0.6 + meta['ext']]
            if task.get('theta_range'):                # острая партия: только если острый карман и вышел
                cands = [c for c in cands if min(c['theta'], c['theta_l']) <= task['theta_range'][1] + 5]
            if cands:
                break
        else:
            return None
        cand = min(cands, key=lambda c: c['theta'])
        meta.update(board='synth')
    else:
        own, foreign, cand, meta = task['own'], task['foreign'], task['cand'], dict(task['meta'])
    prof = sample_profile(rng)
    v = np.asarray(cand['v'], float)
    allcu = own if foreign is None else unary_union([own, foreign])
    sim = simulate(own, allcu, v, min(cand['theta'], cand['theta_l']), prof['Da'], prof['delta'], prof['u'])
    if len(sim['s']) == 0:
        return None
    lab = wall_labels(sim, min(cand['theta'], cand['theta_l']))
    th_e = theta_from_area(cand['area'], prof['delta'] / 1000)
    info = dict(id=task['id'], src=task['src'], group=task['group'], **{k: meta[k] for k in meta if k != 'fan'},
                v=[float(v[0]), float(v[1])], theta=cand['theta'], theta_l=cand['theta_l'], beta=cand['beta'],
                area_mm2=cand['area'], depth_mm=cand['depth'], theta_eff_d=th_e, profile=prof,
                rw_um=sim['rw_um'], h_um=sim['h_um'], sec=round(sim['sec'], 1),
                T_max=float(np.nanmax(sim['b']) + 1),       # наибольшее время пребывания на стенках
                **lab)
    info['slot'] = is_slot(info, own)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, info=np.array(json.dumps(info, ensure_ascii=False, default=float)),
                        s=sim['s'].astype(np.float32), a=sim['a'].astype(np.float32), b=sim['b'].astype(np.float32),
                        w=sim['w'].astype(np.float32), own=_wkb(own), foreign=_wkb(foreign),
                        patch=patch(own, foreign, v, cand['beta'], prof['delta']).astype(np.float16))
    os.replace(tmp, path)
    return dict(id=task['id'], kind=meta.get('kind'), theta=round(cand['theta'], 1), sec=info['sec'],
                y2=round(lab['y']['2.0'], 3))


def relabel(out):
    """Пересчитать метки готовых примеров из сохранённых стенок (без нового расчёта травления)."""
    n = 0
    for p in sorted(Path(out).glob('*.npz')):
        if p.name.endswith('.tmp.npz'):
            continue
        z = dict(np.load(p))
        info = json.loads(str(z['info']))
        info.update(wall_labels(dict(s=z['s'], a=z['a'].astype(float), b=z['b'].astype(float), w=z['w'].astype(float)),
                                min(info['theta'], info['theta_l'])))
        z['info'] = np.array(json.dumps(info, ensure_ascii=False, default=float))
        tmp = p.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **z)
        os.replace(tmp, p)
        n += 1
    print(f'метки пересчитаны: {n} примеров')


def is_slot(info, own):
    """Щель при δ примера: тот же признак, что в проверке B1 (`_pocket`), но с δ профиля, а не 50 мкм.
    В узкой щели время пребывания раствора T во много раз больше, чем у плоской стенки, и член промывки
    u_r·(T − 1) модели документа, не учитывающей расход задержанного раствора, завышает перетрав.
    Такие примеры не идут в обучение: щели проверка B1 сообщает отдельно, без оценки перетрава.
    Порог по T для этого не годится — у вершины острого клина T тоже велико (≈ 1/(2 sin²(θ/2)))."""
    from ..rules.b1 import _pocket
    pk = _pocket(own, np.asarray(info['v'], float), info['profile']['delta'] / 1000,
                 min(info['theta'], info['theta_l']))
    return pk is None or pk[2] is not None


def load_example(path):
    z = np.load(path)
    info = json.loads(str(z['info']))
    own = shapely.from_wkb(z['own'].tobytes()) if len(z['own']) else None
    foreign = shapely.from_wkb(z['foreign'].tobytes()) if len(z['foreign']) else None
    return info, dict(s=z['s'], a=z['a'], b=z['b'], w=z['w']), own, foreign, z['patch'].astype(np.float32)


def _worker_init():
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'


def _run_one(args):
    task, out_dir = args
    try:
        return run_task(task, out_dir)
    except Exception as e:                             # один неудачный пример не должен ронять прогон
        return dict(id=task['id'], error=f'{type(e).__name__}: {e}')


def demo_boards():
    root = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/KiCad/10.0/share/kicad/demos'
    names = ['pic_programmer', 'StickHub', 'One-Air-Max', 'RoyalBlue54L-Feather', 'CM5_MINIMA_3', 'video',
             'tinytapeout-demo', 'multichannel_mixer', 'interf_u', 'complex_hierarchy']
    out = [p for n in names for p in list(root.rglob(n + '.kicad_pcb'))[:1]]
    here = Path(__file__).resolve().parents[2] / 'boards' / 'demo' / 'b1_demo.kicad_pcb'
    return [here, *out] if here.exists() else out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m dfm.ml.dataset')
    ap.add_argument('cmd', choices=['run', 'summary', 'relabel'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--synth', type=int, default=0)
    ap.add_argument('--real', action='store_true', help='карманы демо-плат KiCad и b1_demo')
    ap.add_argument('--per-real', type=int, default=2)
    ap.add_argument('--max-per-board', type=int, default=150)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--sharp', type=int, default=0, help='добавочная синтетика острых углов (id a…)')
    ap.add_argument('--sharp-range', type=float, nargs=2, default=SHARP_RANGE, metavar=('ОТ', 'ДО'))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'summary':
        return summary(out)
    if a.cmd == 'relabel':
        relabel(out)
        return summary(out)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.perf_counter()
    tasks = make_tasks(a.synth, demo_boards() if a.real else (), a.per_real, a.max_per_board, a.seed,
                       a.sharp, a.sharp_range)
    todo = [t for t in tasks if not (out / f'{t["id"]}.npz').exists()]
    print(f'задач {len(tasks)}, осталось {len(todo)} (подготовка {time.perf_counter() - t0:.0f} с)', flush=True)
    done = err = 0
    with ProcessPoolExecutor(a.workers, initializer=_worker_init) as ex:
        for r in ex.map(_run_one, [(t, str(out)) for t in todo], chunksize=1):
            if r is None:
                continue
            if 'error' in r:
                err += 1
                print('ошибка', r, flush=True)
                continue
            done += 1
            if done % 25 == 0 or done <= 5:
                el = time.perf_counter() - t0
                print(f'{done}/{len(todo)}  {el / 60:.1f} мин  ≈ {el / done * (len(todo) - done) / 60:.0f} мин осталось '
                      f'| {r}', flush=True)
    print(f'готово: {done}, ошибок {err}, {(time.perf_counter() - t0) / 60:.1f} мин', flush=True)
    summary(out)


def summary(out):
    rows = []
    for p in sorted(Path(out).glob('*.npz')):
        if p.name.endswith('.tmp.npz'):
            continue
        z = np.load(p)
        r = json.loads(str(z['info']))
        if 'slot' not in r:                            # примеры первой версии генератора — досчитать
            r['slot'] = is_slot(r, shapely.from_wkb(z['own'].tobytes()))
        rows.append(r)
    if not rows:
        print('нет примеров')
        return
    from collections import Counter
    n_slot = sum(bool(r.get('slot')) for r in rows)
    secs = np.array([r['sec'] for r in rows])
    print(f'примеров {len(rows)}: синтетика {sum(r["src"] == "synth" for r in rows)}, '
          f'реальные {sum(r["src"] == "real" for r in rows)}; расчёт {secs.mean():.1f} с в среднем, '
          f'{secs.max():.0f} с максимум')
    print(f'щелей при δ примера (в обучение не идут): {n_slot}')
    print('ситуации:', dict(sorted(Counter(r['kind'] for r in rows).items())))
    for ur in ('0.0', '0.5', '2.0'):
        y = np.array([r['y'][ur] for r in rows], float)
        print(f'u_r = {ur}: y > τ (0,2) — {np.mean(y > 0.2) * 100:.1f} %, y > 0,1 — {np.mean(y > 0.1) * 100:.1f} %, '
              f'медиана y {np.nanmedian(y):.3f}')
    if all('y_raw' in r for r in rows):
        yr = np.array([r['y_raw']['2.0'] for r in rows], float)
        y2 = np.array([r['y']['2.0'] for r in rows], float)
        cut = yr - y2 > 1e-3
        from .etch import T_CAP_K
        print(f'предел T ({T_CAP_K}·(1 + A(θ))): снизил y при u_r = 2 у {cut.sum()} примеров, из опасных стали безопасными '
              f'{int(((yr > 0.2) & (y2 <= 0.2)).sum())}')
    th = np.array([min(r['theta'], r['theta_l']) for r in rows])
    print('острые углы θ < 25°:', int((th < 25).sum()), '| θ < 15°:', int((th < 15).sum()))
    json.dump(dict(n=len(rows), rows=rows), open(Path(out) / 'summary.json', 'w', encoding='utf-8'),
              ensure_ascii=False, default=float)


if __name__ == '__main__':
    sys.exit(main())
