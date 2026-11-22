"""Обучение и проверка нейросети B1 (документ B1, главы 8–9; шаг 4.3).

Данные — примеры генератора (dfm/ml/dataset.py) без щелей. Входы:
  * фрагмент 3×64×64 (своя медь, чужая медь, расстояние / δ), выровненный биссектрисой вверх;
    при обучении случайно отражается относительно биссектрисы (физика от отражения не зависит);
  * дескрипторы раздела 4.7, только геометрия: θ, θ_ℓ, min(θ, θ_ℓ), θ_эфф, A/δ², H/δ, ширины меди у
    двух стенок (меньшая, большая), расстояние от вершины до чужой меди, признак чужой меди;
  * профиль z = (ln Da, δ, t, EF, u, u_r).
Метка для любого u_r — из сохранённых стенок: y(u_r) = max_s (a + u_r·min(b, T_пред − 1))/w
(`etch.t_cap`). Поэтому u_r при обучении берётся случайно, U(0; 2) мкм, для каждого примера в каждой
эпохе, а проверка — на сетке u_r = 0; 0,5; 1; 2.

Потеря: пинбол по квантилям 0,1/0,5/0,9 + λ·фокальная по признаку y > τ (α = 0,75 — вес опасного
класса). Вес редкого класса — только в классификационной части: вес по метке в пинболе сдвинул бы
квантили (интервалы перестали бы покрывать 80 %).

Разбиение — по группам (плата целиком; партия синтетики целиком), 5 частей: проверка — часть k,
ранняя остановка — часть k + 1, обучение — остальные. Демонстрационная плата документа (b1_demo)
в обучение не идёт никогда и проверяется отдельно. Вероятность калибруется температурой на части
ранней остановки.

    python -m dfm.ml.train --data C:/Users/<…>/dfm_data/b1 --out C:/Users/<…>/dfm_data/models/b1 --cv
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .calibration import CQR_BINS, CQR_MIN, TAU, UR_GRID, apply_cqr, cqr_offsets, fit_cqr, prob_from_quantiles
from .features import DESC_NAMES, POCKET_DELTA_UM, pocket_descriptors

UR_MAX = 2.0
UR_REPORT = (0.0, 0.5, 2.0)
N_FOLDS = 5
DEMO_GROUP = 'b1_demo'
Z_NAMES = ('ln Da', 'δ', 't', 'EF', 'u', 'u_r')
_descriptors = pocket_descriptors  # прежнее имя для исследовательских скриптов


# ---------- данные ----------

def _source(info):
    if info['group'] == DEMO_GROUP:
        return 'demo'
    if info['src'] == 'real':
        return 'real'
    return 'sharp' if info['id'].startswith('a') else 'synth'


CACHE_VERSION = 3          # менять при изменении состава примеров или дескрипторов


def _code_hash():
    """Отпечаток кода, от которого зависит кэш: дескрипторы, источник, щели, эффективный угол, поиск карманов."""
    import inspect
    from ..rules.b1 import _pocket, corner_area, theta_from_area
    from .dataset import B1_GEOM, is_slot
    src = ''.join(inspect.getsource(f) for f in (pocket_descriptors, _source, is_slot, theta_from_area, corner_area, _pocket))
    return hashlib.sha1((src + repr(sorted(B1_GEOM.items()))).encode('utf-8')).hexdigest()


def _fingerprint(files):
    h = hashlib.sha1(f'v{CACHE_VERSION}:{_code_hash()}'.encode())
    for p in files:
        h.update(f'{p.name}:{p.stat().st_mtime_ns}'.encode())
    return h.hexdigest()


def load_data(data_dir, cache=None):
    """Все примеры без щелей → словарь массивов (кэш в <data>/../b1_cache.npz, пересобирается при изменении)."""
    from .dataset import is_slot, load_example
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.glob('*.npz') if not p.name.endswith('.tmp.npz'))
    fp = _fingerprint(files)
    cache = Path(cache) if cache else data_dir.parent / f'{data_dir.name}_cache.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:          # закрыть: иначе Windows не даст заменить файл
            if str(z['fingerprint']) == fp:
                out = {k: z[k] for k in z.files if k not in ('fingerprint', 'meta')}
                out['meta'] = json.loads(str(z['meta']))
                return out
    rows, patches, desc, prof, A, B, W, theta = [], [], [], [], [], [], [], []
    for p in files:
        info, walls, own, foreign, patch = load_example(p)
        if 'slot' not in info:                         # примеры первой версии генератора — как в summary
            info['slot'] = is_slot(info, own)
        if info['slot']:
            continue
        rows.append(dict(id=info['id'], group=info['group'], source=_source(info), kind=info.get('kind'),
                         theta=min(info['theta'], info['theta_l'])))
        patches.append(patch.astype(np.float16))
        desc.append(pocket_descriptors(info, walls, own, foreign))
        pr = info['profile']
        prof.append([math.log(pr['Da']), pr['delta'], pr['t'], pr['ef'], pr['u']])
        A.append(walls['a']), B.append(walls['b']), W.append(walls['w'] * 1000)
        theta.append(min(info['theta'], info['theta_l']))
    L = max(len(a) for a in A)
    pad = lambda arrs, fill: np.stack([np.pad(a.astype(np.float32), (0, L - len(a)), constant_values=fill) for a in arrs])
    out = dict(patch=np.stack(patches), desc=np.array(desc, np.float32), prof=np.array(prof, np.float32),
               a=pad(A, 0.0), b=pad(B, 0.0), w=pad(W, np.inf), theta=np.array(theta, np.float32))
    out['mask'] = np.isfinite(out['w'])
    tmp = cache.with_name(cache.name + '.tmp')
    with open(tmp, 'wb') as fh:                      # запись целиком, затем замена: читатель не увидит половину
        np.savez(fh, fingerprint=np.array(fp), meta=np.array(json.dumps(rows, ensure_ascii=False)), **out)
    import os
    os.replace(tmp, cache)
    out['meta'] = rows
    return out


def formula_matrix(data, n=1):
    """Сужение по формуле клина (dfm/ml/baselines.py) для каждого примера в узлах u_r таблицы → (N, len(URS))
    при n = 1 (по эффективному углу θ_эфф) или (N, len(URS), 2) при n = 2 (по θ_эфф и по углу вершины
    min(θ, θ_ℓ)); ширина — меньшая ширина меди у стенок."""
    from .baselines import TABLE_NPZ, URS, WedgeFormula
    if not TABLE_NPZ.exists():
        return None
    F = WedgeFormula.load()
    desc, prof = data['desc'], data['prof']
    w_min = desc[:, 6]
    Da, dl, uu = np.exp(prof[:, 0]), prof[:, 1], prof[:, 4]
    cols = [np.clip(desc[:, 3] * 90, 0, 90), np.clip(desc[:, 2] * 90, 0, 90)][:n]
    M = np.stack([np.stack([F(th, Da, dl, uu, u) / (w_min * 1000) for u in URS], axis=1) for th in cols], axis=2)
    return (M[:, :, 0] if n == 1 else M).astype(np.float32)


def formula_at(Fm, u_r):
    """Линейная интерполяция formula_matrix по u_r (torch; u_r — вектор по примерам, мкм) → (N,) или (N, 2)."""
    import torch
    from .baselines import URS
    step = URS[1] - URS[0]
    x = torch.clamp(u_r / step, 0, len(URS) - 1 - 1e-6)
    i = x.floor().long()
    t = x - i
    if Fm.ndim == 3:
        t = t[:, None]
    ar = torch.arange(len(i))
    return Fm[ar, i] * (1 - t) + Fm[ar, i + 1] * t


def capped_b(data):
    from .etch import t_cap
    tc = np.array([t_cap(float(t)) for t in data['theta']], np.float32)
    return np.minimum(data['b'], tc[:, None] - 1)


def labels(a, b, w, mask, u_r):
    """y(u_r) по стенкам; работает и с numpy, и с torch (u_r — число или вектор по примерам)."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            u = u_r[:, None] if isinstance(u_r, torch.Tensor) and u_r.ndim else u_r
            rel = torch.where(mask, (a + u * b) / w, torch.full_like(a, -1e9))
            return rel.max(dim=1).values
    except ImportError:
        pass
    u = np.asarray(u_r)[:, None] if np.ndim(u_r) else u_r
    return np.where(mask, (a + u * b) / w, -1e9).max(axis=1)


def folds(meta, n=N_FOLDS, seed=0):
    """Номер части для каждого примера: группы каждого источника перемешаны и разложены по кругу; демо — −1."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for src in ('synth', 'sharp', 'real'):
        gs = sorted({r['group'] for r in meta if r['source'] == src})
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold_of[g] = i % n
    return np.array([fold_of.get(r['group'], -1) for r in meta])


# ---------- метрики ----------

def binom_ci(k, n, level=0.95):
    """Точный доверительный интервал доли k/n (Клоппер — Пирсон, по биномиальному распределению):
    (нижняя, верхняя) граница; при n = 0 — (nan, nan). 30 из 31 → 0,833…0,999."""
    from scipy.stats import beta
    k, n = int(k), int(n)
    if n <= 0:
        return float('nan'), float('nan')
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def fmt_ci(k, n, digits=3):
    """«k/n = доля [нижняя; верхняя]» с десятичной запятой."""
    if not n:
        return '—'
    lo, hi = binom_ci(k, n)
    f = lambda x: f'{x:.{digits}f}'.replace('.', ',')
    return f'{k}/{n} = {f(k / n)} [{f(lo)}; {f(hi)}]'


def _pr_curve(y_true, score):
    """Точки кривой «точность — полнота» только на концах блоков равных оценок (порог не может пройти
    внутри блока равных значений): (порог, верных, всего выдано, всего опасных)."""
    score = np.asarray(score, float)
    o = np.argsort(-score, kind='stable')
    s, t = score[o], np.asarray(y_true, bool)[o]
    end = np.r_[s[1:] != s[:-1], True]
    return s[end], np.cumsum(t)[end], np.arange(1, len(t) + 1)[end], int(t.sum())


def average_precision(y_true, score):
    """AP = Σ (R_k − R_{k−1})·P_k по порогам (как sklearn.metrics.average_precision_score)."""
    _, tp, n, pos = _pr_curve(y_true, score)
    if not pos:
        return float('nan')
    rec = tp / pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * tp / n))


def precision_at_recall(y_true, score, r_min=0.95):
    thr, tp, n, pos = _pr_curve(y_true, score)
    if not pos:
        return float('nan'), float('nan'), 0
    rec, prec = tp / pos, tp / n
    k = int(np.argmax(rec >= r_min))                   # с этого порога полнота ≥ r_min
    k += int(np.argmax(prec[k:]))                      # наибольшая точность среди таких порогов
    return float(prec[k]), float(thr[k]), int(n[k])


def metrics(y, q, p, theta, q_raw=None, tau=TAU):
    """q — интервалы после калибровки CQR, q_raw — выход сети (покрытие до калибровки)."""
    pos = y > tau
    m = dict(n=int(len(y)), n_pos=int(pos.sum()),
             mae=float(np.mean(np.abs(y - q[:, 1]))),
             mae_sharp=float(np.mean(np.abs(y - q[:, 1])[theta < 30])) if (theta < 30).any() else float('nan'),
             coverage80=float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))),
             coverage80_raw=float(np.mean((y >= q_raw[:, 0]) & (y <= q_raw[:, 2]))) if q_raw is not None else float('nan'),
             width80=float(np.mean(q[:, 2] - q[:, 0])),
             ap=average_precision(pos, p))
    pr, thr, n_warn = precision_at_recall(pos, p)
    m.update(precision_at_r95=pr, p_threshold_r95=thr, warnings_at_r95=n_warn)
    pred = p > 0.5
    m.update(precision_p50=float((pred & pos).sum() / max(pred.sum(), 1)),
             recall_p50=float((pred & pos).sum() / max(pos.sum(), 1)) if pos.any() else float('nan'))
    return m


# ---------- обучение ----------

class Norm:
    def __init__(self, x):
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0) + 1e-6

    def __call__(self, x):
        return (x - self.mu) / self.sd

    def state(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist())


def _z(prof_n, u_r):
    import torch
    return torch.cat([prof_n, ((u_r - UR_MAX / 2) / (UR_MAX / math.sqrt(12)))[:, None]], dim=1)


def prepare(x, variant, idx=None, rng=None):
    """Вход для варианта абляции: no_foreign — без канала чужой меди; res32 — фрагмент 32×32 (среднее
    2×2); no_align — фрагмент повёрнут на случайный угол вокруг вершины (при проверке угол задан номером
    примера — воспроизводимо), т. е. без выравнивания по биссектрисе."""
    import torch
    from torch.nn import functional as F
    if variant == 'no_foreign':
        x = x.clone()
        x[:, 1] = 0
    elif variant == 'res32':
        x = F.avg_pool2d(x, 2)
    elif variant == 'no_align':
        if rng is not None:
            ang = torch.as_tensor(rng.uniform(0, 2 * math.pi, len(x)), dtype=torch.float32)
        else:
            ang = torch.as_tensor(np.random.default_rng(np.asarray(idx) + 12345).uniform(0, 2 * math.pi)
                                  if np.ndim(idx) == 0 else
                                  np.array([np.random.default_rng(int(i) + 12345).uniform(0, 2 * math.pi)
                                            for i in idx]), dtype=torch.float32)
        c, s_ = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([c, -s_, torch.zeros_like(c)], 1),
                             torch.stack([s_, c, torch.zeros_like(c)], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return x


def without_foreign(data):
    """Данные для абляции no_foreign: чужая медь убрана отовсюду — канал чужой меди обнулён, канал
    расстояния пересчитан до своей меди (евклидово расстояние по растру, шаг 0,5 мм / 64, в долях δ, до 4),
    дескрипторы «расстояние до чужой меди» = 1 мм и «есть чужая» = 0, как у примеров без соседей.
    Ограничение: своя медь за краем фрагмента на растре не видна — у края расстояние завышено (до ≈ 0,9δ)."""
    from scipy.ndimage import distance_transform_edt
    out = dict(data)
    patch = data['patch'].astype(np.float32).copy()
    px_um = 500.0 / patch.shape[-1]
    for i in range(len(patch)):
        own = patch[i, 0] > 0.5
        dist = distance_transform_edt(~own) * px_um if own.any() else np.full(own.shape, 1e9)
        patch[i, 2] = np.minimum(dist / data['prof'][i, 1], 4.0)
    patch[:, 1] = 0
    out['patch'] = patch.astype(np.float16)
    desc = data['desc'].copy()
    desc[:, DESC_NAMES.index('d_чуж')] = 1.0
    desc[:, DESC_NAMES.index('есть_чуж')] = 0.0
    out['desc'] = desc
    return out


def train_fold(data, fold, out_dir, epochs=150, patience=20, batch=128, lr=1e-3, lam=1.0, seed=0, threads=8,
               log=print, variant='full', hybrid=False):
    import torch
    from .model import B1Net, focal, n_params, pinball
    if variant == 'ab_head' and hybrid:
        # в гибриде оценка формулы f(u_r) входит в сеть и в выход — u_r «протекает», абляция теряет смысл
        log('ab_head: без гибрида (иначе u_r входит в сеть через оценку формулы)')
        hybrid = False
    if variant == 'no_foreign':
        data = without_foreign(data)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    tr, val, test, demo = _split(data['meta'], fold)
    dn, pn = Norm(data['desc'][tr]), Norm(data['prof'][tr])
    T = lambda x: torch.as_tensor(x)
    X = T(data['patch'])                                # float16, переводится по партиям
    D = T(dn(data['desc']).astype(np.float32))
    P = T(pn(data['prof']).astype(np.float32))
    bc = capped_b(data)
    A, B, Wt, M = T(data['a']), T(bc), T(data['w']), T(data['mask'])
    model = B1Net(D.shape[1], z_dim=P.shape[1] + 1, variant=variant, hybrid=hybrid)
    Fm = T(formula_matrix(data, int(hybrid))) if hybrid else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
    log(f'часть {fold} ({"гибрид, " if hybrid else ""}{variant}): обучение {len(tr)}, ранняя остановка {len(val)}, проверка {len(test)}, демо {len(demo)}; '
        f'параметров {n_params(model)}')

    def run(idx, u_r, train):
        x = X[idx].float()
        if train:
            flip = torch.as_tensor(rng.random(len(idx)) < 0.5)
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        x = prepare(x, variant, idx.numpy(), rng if train else None)
        y = labels(A[idx], B[idx], Wt[idx], M[idx], u_r)
        q, logit = model(x, D[idx], _z(P[idx], u_r), u_r, formula_at(Fm[idx], u_r) if hybrid else None)
        return q, logit, y

    def eval_loss(idx):
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for u in UR_GRID:
                for i in range(0, len(idx), 512):
                    j = T(idx[i:i + 512])
                    q, logit, y = run(j, torch.full((len(j),), u), False)
                    tot += (pinball(y, q) + lam * focal(logit, (y > TAU).float())).item() * len(j)
        return tot / (len(idx) * len(UR_GRID))

    best, best_state, bad, hist = float('inf'), None, 0, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        tl = 0.0
        for i in range(0, len(perm), batch):
            j = T(perm[i:i + batch])
            u = torch.as_tensor(rng.uniform(0, UR_MAX, len(j)), dtype=torch.float32)
            q, logit, y = run(j, u, True)
            loss = pinball(y, q) + lam * focal(logit, (y > TAU).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(j)
        vl = eval_loss(val)
        sched.step(vl)
        hist.append(dict(epoch=ep + 1, train=tl / len(perm), val=vl, lr=opt.param_groups[0]['lr']))
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f'  эпоха {ep + 1}: обучение {hist[-1]["train"]:.4f}, проверка {vl:.4f} '
                f'(лучшая {best:.4f}), {time.perf_counter() - t0:.0f} с')
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    train_info = dict(epochs=len(hist), best_val=best, sec=time.perf_counter() - t0, history=hist)
    return evaluate(data, fold, model, dn, pn, out_dir, train_info)


def _split(meta, fold):
    f = folds(meta)
    return (np.where((f >= 0) & (f != fold) & (f != (fold + 1) % N_FOLDS))[0], np.where(f == (fold + 1) % N_FOLDS)[0],
            np.where(f == fold)[0], np.where(f == -1)[0])


def evaluate(data, fold, model, dn, pn, out_dir, train_info):
    """Калибровка (температура вероятности и CQR интервалов — на части ранней остановки) и метрики
    на проверочной части и демо-плате; модель и метрики сохраняются."""
    import torch
    model.eval()
    _, val, test, demo = _split(data['meta'], fold)
    T = lambda x: torch.as_tensor(x)
    X = T(data['patch'])
    D = T(dn(data['desc']).astype(np.float32))
    P = T(pn(data['prof']).astype(np.float32))
    A, B, Wt, M = T(data['a']), T(capped_b(data)), T(data['w']), T(data['mask'])
    Fm = T(formula_matrix(data, model.n_f)) if model.hybrid else None

    def predict(idx, u):
        qs, ls, ys = [], [], []
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                j = T(idx[i:i + 512])
                uu = torch.full((len(j),), float(u))
                q, logit = model(prepare(X[j].float(), model.variant, j.numpy()), D[j], _z(P[j], uu), uu,
                                 formula_at(Fm[j], uu) if model.hybrid else None)
                qs.append(q.numpy()), ls.append(logit.numpy()), ys.append(labels(A[j], B[j], Wt[j], M[j], uu).numpy())
        return np.concatenate(ys), np.concatenate(qs), np.concatenate(ls)

    pv = {u: predict(val, u) for u in UR_GRID}
    temp = fit_temperature(np.concatenate([pv[u][2] for u in UR_GRID]), np.concatenate([pv[u][0] for u in UR_GRID]) > TAU)
    cqr = fit_cqr(pv, data['theta'][val])

    res = dict(fold=fold, variant=model.variant, hybrid=model.n_f, epochs=train_info['epochs'], best_val=train_info['best_val'], temperature=temp, cqr=cqr,
               sec=train_info['sec'], by_ur={}, by_source={}, by_kind={})
    meta = data['meta']
    src = np.array([r['source'] for r in meta])
    kind = np.array([r['kind'] for r in meta])
    th = data['theta']
    for u in UR_REPORT:
        y, q_raw, lg = predict(test, u)
        q = apply_cqr(q_raw, u, cqr, th[test])
        p_head = 1 / (1 + np.exp(-lg / temp))
        p = prob_from_quantiles(q)
        res['by_ur'][str(u)] = metrics(y, q, p, th[test], q_raw)
        mh = metrics(y, q, p_head, th[test], q_raw)
        res['by_ur'][str(u)].update(ap_head=mh['ap'], precision_at_r95_head=mh['precision_at_r95'])
        if u == UR_MAX:
            for s in ('synth', 'sharp', 'real'):
                k = src[test] == s
                if k.any():
                    res['by_source'][s] = metrics(y[k], q[k], p[k], th[test][k], q_raw[k])
            for kd in sorted(set(kind[test])):
                k = kind[test] == kd
                res['by_kind'][kd] = metrics(y[k], q[k], p[k], th[test][k], q_raw[k])
            res['pred'] = dict(ids=[meta[i]['id'] for i in test], y=y.tolist(), q=q.tolist(), q_raw=q_raw.tolist(),
                               p=p.tolist(), p_head=p_head.tolist(), theta=th[test].tolist(), source=src[test].tolist(),
                               kind=kind[test].tolist())
    if len(demo):                                      # демо-плата документа: профиль — как в примере
        dres = {}
        for u in UR_REPORT:
            y, q_raw, lg = predict(demo, u)
            q = apply_cqr(q_raw, u, cqr, th[demo])
            dres[str(u)] = [dict(id=meta[i]['id'], kind=meta[i]['kind'], theta=float(th[i]), y=float(y[k]),
                                 q50=float(q[k, 1]), q10=float(q[k, 0]), q90=float(q[k, 2]),
                                 p=float(prob_from_quantiles(q[k:k + 1])[0]),
                                 p_head=float(1 / (1 + np.exp(-lg[k] / temp)))) for k, i in enumerate(demo)]
        res['demo'] = dres
    with torch.no_grad():                              # скорость: партия и один фрагмент
        j = T(test[:256])
        uu = torch.full((len(j),), 1.0)
        z = _z(P[j], uu)
        x, d = prepare(X[j].float(), model.variant, j.numpy()), D[j]
        fj = formula_at(Fm[j], uu) if model.hybrid else torch.zeros(len(j))
        model(x, d, z, uu, fj)
        t1 = time.perf_counter()
        for _ in range(5):
            model(x, d, z, uu, fj)
        res['ms_per_example_batch'] = (time.perf_counter() - t1) / 5 / len(j) * 1000
        t1 = time.perf_counter()
        for k in range(50):
            model(x[k:k + 1], d[k:k + 1], z[k:k + 1], uu[k:k + 1], fj[k:k + 1])
        res['ms_single'] = (time.perf_counter() - t1) / 50 * 1000
    res['history'] = train_info['history']
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from .baselines import table_sha1
    meta = data['meta']
    torch.save(dict(state=model.state_dict(), desc_norm=dn.state(), prof_norm=pn.state(), temperature=temp, cqr=cqr,
                    desc_names=DESC_NAMES, z_names=Z_NAMES, tau=TAU, ur_max=UR_MAX, fold=fold,
                    n_desc=len(DESC_NAMES), z_dim=len(Z_NAMES), variant=model.variant, hybrid=model.n_f,
                    # для сверки при --eval-only: разбиение групп по частям и таблица клина, с которой обучен гибрид
                    fold_groups=_fold_groups(meta), table_sha1=table_sha1() if model.n_f else None),
               out_dir / f'b1_fold{fold}.pt')
    json.dump(res, open(out_dir / f'metrics_fold{fold}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return res


def load_model(path):
    """Сохранённая модель части → (модель, нормировка дескрипторов, нормировка профиля, контрольная точка)."""
    import torch
    from .model import B1Net
    ck = torch.load(path, weights_only=False)
    model = B1Net(len(ck['desc_names']), z_dim=len(ck['z_names']), variant=ck.get('variant', 'full'),
                  hybrid=ck.get('hybrid', False))
    model.load_state_dict(ck['state'])
    model.eval()
    norm = lambda st: _NormState(np.array(st['mu'], np.float32), np.array(st['sd'], np.float32))
    return model, norm(ck['desc_norm']), norm(ck['prof_norm']), ck


class _NormState(Norm):
    def __init__(self, mu, sd):
        self.mu, self.sd = mu, sd


def _fold_groups(meta):
    f = folds(meta)
    return {r['group']: int(k) for r, k in zip(meta, f)}


def reevaluate(data, fold, out_dir):
    """Пересчитать калибровку и метрики сохранённой модели части (без обучения). Модель обучена на других
    данных или с другой таблицей клина — ошибка, а не тихая проверка на обучающих платах."""
    from .baselines import table_sha1
    out_dir = Path(out_dir)
    model, dn, pn, ck = load_model(out_dir / f'b1_fold{fold}.pt')
    if ck.get('fold_groups') is None:
        print(f'внимание: в b1_fold{fold}.pt нет разбиения групп — совпадение частей не проверено '
              '(research/b1_nn/stamp_checkpoints.py)', file=sys.stderr)
    elif ck['fold_groups'] != _fold_groups(data['meta']):
        raise ValueError('разбиение групп по частям изменилось с обучения модели — переоценка смешала бы части')
    if model.n_f and not ck.get('table_sha1'):
        print(f'внимание: в b1_fold{fold}.pt нет отпечатка таблицы клина — не проверено', file=sys.stderr)
    elif ck.get('table_sha1') and ck['table_sha1'] != table_sha1():
        raise ValueError('таблица клина изменилась с обучения гибрида')
    if model.variant == 'no_foreign':
        data = without_foreign(data)
    old = json.load(open(out_dir / f'metrics_fold{fold}.json', encoding='utf-8'))
    return evaluate(data, fold, model, dn, pn, out_dir,
                    dict(epochs=old['epochs'], best_val=old['best_val'], sec=old['sec'], history=old['history']))


def fit_temperature(logits, y):
    import torch
    lg = torch.as_tensor(logits, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def summarize(results):
    """Среднее и разброс метрик по частям перекрёстной проверки."""
    out = {}
    for u in UR_REPORT:
        keys = results[0]['by_ur'][str(u)].keys()
        out[str(u)] = {k: dict(mean=float(np.nanmean([r['by_ur'][str(u)][k] for r in results])),
                               sd=float(np.nanstd([r['by_ur'][str(u)][k] for r in results])))
                       for k in keys}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m dfm.ml.train')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--cv', action='store_true', help=f'все {N_FOLDS} частей')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--eval-only', action='store_true', help='пересчитать калибровку и метрики сохранённых моделей')
    ap.add_argument('--hybrid', type=int, nargs='?', const=1, default=0,
                    help='гибрид: ŷ = формула клина по θ_эфф + поправка сети; --hybrid 2 — на входе и формула по θ')
    ap.add_argument('--variant', default='full',
                    help='абляция: full, no_raster, no_desc, no_foreign, z_input, no_align, ab_head, res32')
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    data = load_data(a.data)
    print(f'примеров {len(data["meta"])} (загрузка {time.perf_counter() - t0:.0f} с)', flush=True)
    log = lambda s: print(s, flush=True)
    ks = range(N_FOLDS) if a.cv else [a.fold]
    if a.eval_only:
        import torch
        torch.set_num_threads(a.threads)
        res = [reevaluate(data, k, a.out) for k in ks]
    else:
        res = [train_fold(data, k, a.out, a.epochs, a.patience, lam=a.lam, threads=a.threads, log=log,
                          variant=a.variant, hybrid=a.hybrid) for k in ks]
    for r in res:
        m = r['by_ur'][str(UR_MAX)]
        print(f'часть {r["fold"]}: эпох {r["epochs"]}, u_r = 2: MAE {m["mae"]:.4f}, острые {m["mae_sharp"]:.4f}, '
              f'покрытие {m["coverage80_raw"]:.2f} → {m["coverage80"]:.2f} (CQR), AP {m["ap"]:.3f}, точность@R95 {m["precision_at_r95"]:.3f} '
              f'({m["n_pos"]} опасных из {m["n"]})', flush=True)
    if a.cv:
        s = summarize(res)
        json.dump(s, open(Path(a.out) / 'metrics_cv.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        for u, m in s.items():
            print(f'u_r = {u}: ' + ', '.join(f'{k} {v["mean"]:.3f}±{v["sd"]:.3f}' for k, v in m.items()
                                            if k in ('mae', 'mae_sharp', 'coverage80', 'ap', 'precision_at_r95')))


if __name__ == '__main__':
    sys.exit(main())
