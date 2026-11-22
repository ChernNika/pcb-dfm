"""Базовые линии для нейросети B1 (документ B1, разделы 5.3 и 9.5; шаг 4.4).

1. Угловое правило: предупреждение у каждого кандидата с θ < 90° (так работает большинство инструментов).
2. Формула клина по углу: перетрав идеального клина Δ̂ = f_z(θ) при профиле процесса примера, ŷ = Δ̂ / w.
3. Формула клина по эффективному углу: то же с θ_эфф (по площади кармана).
4. Логистическая регрессия по дескрипторам и профилю (только классификация).

Таблица f_z в документе посчитана для трёх профилей С/О/З при одной толщине меди (u = 12 мкм, δ = 50 мкм).
Примеры генератора имеют произвольные Da, δ и u, поэтому таблица здесь считается той же численной
моделью (`etch.simulate`, с тем же пределом времени пребывания) для идеального клина — ответвление
трассы 0,15 мм от трассы 0,15 мм (ситуация Л2 документа) — на сетке θ × Da × δ × u и хранится для сетки
u_r, а значения между узлами — линейная интерполяция (по ln Da). Узел (Da 3, δ 50, u 12) совпадает
с профилем З документа — проверка таблицы.

    python -m dfm.ml.baselines table --out C:/Users/<…>/dfm_data/b1_wedge --workers 10
    python -m dfm.ml.baselines evaluate --data …/dfm_data/b1 --table …/dfm_data/b1_wedge --out research/b1_nn/baselines.json

Оценка — на тех же пяти проверочных частях, что и сеть (train.folds): для каждой части и u_r = 0,5 и 2 —
MAE (кроме углового правила и регрессии), AP, точность при полноте 0,95, а для формул ещё и их
естественное решение «ŷ > τ» (точность и полнота). Логистическая регрессия обучается на обучающих
частях той же разбивки (u_r — узлы сетки 0; 0,5; 1; 2).
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

THETAS = (8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0)
DAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
DELTAS = (30.0, 50.0, 80.0)
US = (4.5, 9.0, 12.0, 20.0, 35.0)
URS = tuple(np.round(np.arange(0, 2.01, 0.1), 2))
W_WEDGE = 0.15


def wedge(theta, w=W_WEDGE):
    """Идеальный клин: трасса w вдоль оси x и ответвление w под углом θ (вершина кармана далеко от
    стыка при остром угле — трассы удлинены, как в острой партии генератора)."""
    import shapely
    ext = 0.3 / math.sin(math.radians(theta))
    a = shapely.LineString([(-1.2, 0), (1.0 + ext, 0)]).buffer(w / 2, quad_segs=16)
    L = 1.4 + ext
    b = shapely.LineString([(0, 0), (L * math.cos(math.radians(theta)), L * math.sin(math.radians(theta)))]).buffer(
        w / 2, quad_segs=16)
    return a.union(b)


def _one(args):
    theta, Da, delta, u, out_dir = args
    path = Path(out_dir) / f'w_{theta:g}_{Da:g}_{delta:g}_{u:g}.json'
    if path.exists():
        return None
    from ..rules.b1 import trap_candidates
    from .dataset import B1_GEOM
    from .etch import simulate, t_cap
    own = wedge(theta)
    geom = dict(B1_GEOM, theta_max_deg=120.0)          # узел 90° — тоже кандидат (в проверке θ < 90°)
    cands = [c for c in trap_candidates(own, geom) if c['slot'] is None]
    c = min(cands, key=lambda c: abs(min(c['theta'], c['theta_l']) - theta))
    th = min(c['theta'], c['theta_l'])
    sim = simulate(own, own, c['v'], th, Da, delta, u)
    b = np.minimum(sim['b'], t_cap(th) - 1)
    d = [float(np.max(sim['a'] + ur * b)) for ur in URS]
    rec = dict(theta=theta, theta_meas=th, Da=Da, delta=delta, u=u, overetch=d, sec=round(sim['sec'], 1))
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec), encoding='utf-8')
    os.replace(tmp, path)
    return rec


def _init():
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'


def build_table(out_dir, workers=10):
    from concurrent.futures import ProcessPoolExecutor
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(t, Da, d, u, str(out_dir)) for t in THETAS for Da in DAS for d in DELTAS for u in US]
    todo = [t for t in tasks if not (out_dir / f'w_{t[0]:g}_{t[1]:g}_{t[2]:g}_{t[3]:g}.json').exists()]
    print(f'узлов {len(tasks)}, осталось {len(todo)}', flush=True)
    t0, done = time.perf_counter(), 0
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for r in ex.map(_one, todo, chunksize=1):
            done += 1
            if r and (done % 50 == 0 or done <= 3):
                el = time.perf_counter() - t0
                print(f'{done}/{len(todo)}  {el / 60:.1f} мин  ≈ {el / done * (len(todo) - done) / 60:.0f} мин осталось '
                      f'| θ {r["theta"]:g} Da {r["Da"]:g} δ {r["delta"]:g} u {r["u"]:g}: Δ(2) {r["overetch"][-1]:.2f}',
                      flush=True)
    return load_table(out_dir)


def load_table(out_dir):
    """Узлы → массив D[θ, Da, δ, u, u_r] (мкм)."""
    out_dir = Path(out_dir)
    D = np.full((len(THETAS), len(DAS), len(DELTAS), len(US), len(URS)), np.nan)
    for p in out_dir.glob('w_*.json'):
        r = json.loads(p.read_text(encoding='utf-8'))
        D[THETAS.index(r['theta']), DAS.index(r['Da']), DELTAS.index(r['delta']), US.index(r['u'])] = r['overetch']
    return D


TABLE_NPZ = Path(__file__).with_name('tables') / 'b1_wedge.npz'


def table_sha1(path=TABLE_NPZ) -> str | None:
    """Отпечаток таблицы клина: гибрид обучается с ней как с входом и должен работать с той же."""
    import hashlib
    path = Path(path)
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else None


def save_table(D, path=TABLE_NPZ):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, D=D.astype(np.float32), thetas=THETAS, das=DAS, deltas=DELTAS, us=US, urs=URS,
                        w_wedge=W_WEDGE)


class WedgeFormula:
    """Δ̂(θ, Da, δ, u, u_r) — линейная интерполяция таблицы (ln Da), за краями сетки — ближайшее значение."""

    def __init__(self, D):
        from scipy.interpolate import RegularGridInterpolator
        if np.isnan(D).any():
            raise ValueError(f'таблица клина неполная: {int(np.isnan(D).any(axis=-1).sum())} узлов без расчёта')
        self.f = RegularGridInterpolator((THETAS, np.log(DAS), DELTAS, US, URS), D)
        self.lo = np.array([THETAS[0], math.log(DAS[0]), DELTAS[0], US[0], URS[0]])
        self.hi = np.array([THETAS[-1], math.log(DAS[-1]), DELTAS[-1], US[-1], URS[-1]])

    @classmethod
    def load(cls, path=TABLE_NPZ):
        z = np.load(path)
        assert tuple(z['thetas']) == THETAS and tuple(z['urs']) == URS, 'сетка таблицы клина изменилась'
        return cls(z['D'].astype(float))

    def __call__(self, theta, Da, delta, u, u_r):
        X = np.stack(np.broadcast_arrays(theta, np.log(Da), delta, u, u_r), axis=-1).astype(float)
        return self.f(np.clip(X, self.lo, self.hi))


def _logreg(Xtr, ytr, l2=1e-3):
    import torch
    X = torch.as_tensor(Xtr, dtype=torch.float64)
    y = torch.as_tensor(ytr, dtype=torch.float64)
    w = torch.zeros(X.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = X @ w[:-1] + w[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def _cls(y, score, tau):
    from .train import average_precision, precision_at_recall
    pos = y > tau
    pr, thr, nw = precision_at_recall(pos, score)
    return dict(ap=average_precision(pos, score), precision_at_r95=pr, warnings_at_r95=nw, threshold_r95=thr)


def evaluate(data_dir, table_dir, urs=(0.5, 2.0)):
    """Базовые линии по частям → {метод: {u_r: [метрики части]}} + объединённые прогнозы при u_r = 2."""
    from .train import N_FOLDS, TAU, UR_GRID, Norm, _split, capped_b, labels, load_data
    data = load_data(data_dir)
    F = WedgeFormula(load_table(table_dir))
    desc, prof = data['desc'], data['prof']
    th_min, w_min = desc[:, 2] * 90, desc[:, 6]
    Da, dl, uu = np.exp(prof[:, 0]), prof[:, 1], prof[:, 4]
    # θ_эфф — дескриптор 3: по площади кармана при δ = 50 мкм, при котором она измерена (B1_GEOM), как в проверке
    th_eff = np.clip(desc[:, 3] * 90, 0, 90)
    bc = capped_b(data)
    Y = {u: labels(data['a'], bc, data['w'], data['mask'], u) for u in set(urs) | set(UR_GRID)}
    meta = data['meta']
    src = np.array([r['source'] for r in meta])
    res, pooled = {}, {}
    for k in range(N_FOLDS):
        tr, _, te, _ = _split(meta, k)
        # логистическая регрессия: дескрипторы + профиль + u_r, обучение на узлах сетки u_r
        dn, pn = Norm(desc[tr]), Norm(prof[tr])
        feat = lambda idx, u: np.c_[dn(desc[idx]), pn(prof[idx]), np.full(len(idx), (u - 1.0) / 0.577)]
        Xtr = np.concatenate([feat(tr, u) for u in UR_GRID])
        ytr = np.concatenate([Y[u][tr] > TAU for u in UR_GRID]).astype(float)
        wlr = _logreg(Xtr, ytr)
        for u in urs:
            y = Y[u][te]
            out = {}
            n, npos = len(te), int((y > TAU).sum())
            out['угловое правило'] = dict(n=n, n_pos=npos, precision_flag=npos / n, recall_flag=1.0, warnings_flag=n,
                                          **_cls(y, -th_min[te], TAU))
            for name, th in (('формула по θ', th_min), ('формула по θ_эфф', th_eff)):
                yh = F(th[te], Da[te], dl[te], uu[te], u) / (w_min[te] * 1000)
                flag = yh > TAU
                out[name] = dict(n=n, n_pos=npos, mae=float(np.mean(np.abs(y - yh))),
                                 mae_sharp=float(np.mean(np.abs(y - yh)[th_min[te] < 30])),
                                 precision_flag=float((flag & (y > TAU)).sum() / max(flag.sum(), 1)),
                                 recall_flag=float((flag & (y > TAU)).sum() / max(npos, 1)), warnings_flag=int(flag.sum()),
                                 **_cls(y, yh, TAU))
                if u == 2.0:
                    pooled.setdefault(name, dict(ids=[], y=[], yh=[], source=[], theta=[]))
                    for key, val in (('ids', [meta[i]['id'] for i in te]), ('y', y.tolist()), ('yh', yh.tolist()),
                                     ('source', src[te].tolist()), ('theta', th_min[te].tolist())):
                        pooled[name][key] += val
            z = feat(te, u) @ wlr[:-1] + wlr[-1]
            out['логистическая регрессия'] = dict(n=n, n_pos=npos, **_cls(y, z, TAU))
            for name, m in out.items():
                res.setdefault(name, {}).setdefault(str(u), []).append(m)
    return dict(folds=res, pooled_ur2=pooled)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m dfm.ml.baselines')
    ap.add_argument('cmd', choices=['table', 'evaluate'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--data')
    ap.add_argument('--table')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args(argv)
    if a.cmd == 'evaluate':
        r = evaluate(a.data, a.table)
        json.dump(r, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        for name, by in r['folds'].items():
            for u, ms in by.items():
                keys = [k for k in ('mae', 'ap', 'precision_at_r95', 'precision_flag', 'recall_flag') if k in ms[0]]
                print(f'{name:24s} u_r {u}: ' + ', '.join(f'{k} {np.nanmean([m[k] for m in ms]):.3f}' for k in keys))
        return 0
    D = build_table(a.out, a.workers)
    if not np.isnan(D).any():
        save_table(D)
        print('таблица сохранена:', TABLE_NPZ)
    i = (THETAS.index(15.0), DAS.index(3.0), DELTAS.index(50.0), US.index(12.0))
    print('проверка по документу (профиль З: Da 3, δ 50, u 12, u_r 2): '
          + ', '.join(f'{t:g}° {D[THETAS.index(t), i[1], i[2], i[3], -1]:.1f} мкм' for t in (15.0, 20.0, 30.0, 45.0)
                      if t in THETAS) + ' (документ: 15° 52,2; 20° 24,9; 30° 5,3)')


if __name__ == '__main__':
    sys.exit(main())
