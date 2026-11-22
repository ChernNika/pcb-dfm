"""Гибрид B1 в проверке (шаг 4.5): формула клина по эффективному углу + поправка сети на дескрипторах.

По итогам шага 4.4 (research/b1_nn) свёрточная ветвь ничего не добавляет: гибрид только на дескрипторах
(вариант no_raster, 8 тыс. параметров) не хуже полного. Поэтому в проверке работает он, и без PyTorch:
веса пяти моделей перекрёстной проверки выгружены в `weights/b1_hybrid.npz`, прямой проход — numpy.

    дескрипторы ⊕ f ─ 2 слоя → 32 ─ FiLM(z) ─ ⊕ f ─ 64 ─ (ŷ₀,₅ − f, ln…, ln…)
    ŷ₀,₅ = f + o₀,  ŷ₀,₁ = ŷ₀,₅ − softplus(o₁),  ŷ₀,₉ = ŷ₀,₅ + softplus(o₂)

f — сужение по формуле клина (таблица `tables/b1_wedge.npz`) при θ_эфф и профиле процесса. Интервалы каждой
модели калиброваны конформно по группам угла на её части ранней остановки (CQR); ансамбль — среднее
калиброванных квантилей пяти моделей; вероятность P(y > τ) — из квантилей (двухчастное нормальное).

    python -m dfm.ml.infer export dfm_data/models/b1h_no_raster    # выгрузить веса (нужен torch)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import unary_union

from .calibration import apply_cqr, prob_from_quantiles
from .features import POCKET_DELTA_UM, pocket_descriptors

WEIGHTS = Path(__file__).with_name('weights') / 'b1_hybrid.npz'
WIN = 0.7                     # полуразмер окна меди вокруг вершины, мм (как dataset.WIN)
_LAYERS = ('desc.0', 'desc.2', 'film.net.0', 'film.net.2', 'head.0', 'head.2')
# диапазон профилей обучающих данных (dataset.sample_profile); вне его — экстраполяция
TRAIN_RANGE = {'Da': (0.1, 30.0), 'δ_мкм': (30.0, 80.0), 't_мкм': (18.0, 70.0), 'EF': (2.0, 4.0), 'u_r_мкм': (0.0, 2.0)}


# ---------- выгрузка весов (один раз, нужен torch) ----------

def export(models_dir, out=WEIGHTS):
    import torch
    files = sorted(Path(models_dir).glob('b1_fold*.pt'))
    arrays, meta = {}, []
    for k, f in enumerate(files):
        ck = torch.load(f, weights_only=False)
        assert ck['variant'] == 'no_raster' and int(ck['hybrid']) == 1, (f, ck['variant'], ck['hybrid'])
        for name in _LAYERS:
            arrays[f'm{k}/{name}.weight'] = ck['state'][f'{name}.weight'].numpy()
            arrays[f'm{k}/{name}.bias'] = ck['state'][f'{name}.bias'].numpy()
        for nm in ('desc_norm', 'prof_norm'):
            arrays[f'm{k}/{nm}.mu'] = np.array(ck[nm]['mu'], np.float32)
            arrays[f'm{k}/{nm}.sd'] = np.array(ck[nm]['sd'], np.float32)
        meta.append(dict(fold=ck['fold'], cqr=ck['cqr']))
        tsha = ck.get('table_sha1')
        head = dict(desc_names=list(ck['desc_names']), z_names=list(ck['z_names']), tau=ck['tau'], ur_max=ck['ur_max'])
    from .baselines import table_sha1
    info = dict(head, models=meta, source=str(Path(models_dir).name), table_sha1=tsha or table_sha1(),
                note='гибрид B1 на дескрипторах (шаг 4.4, вариант no_raster), 5 моделей перекрёстной проверки')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, info=np.array(json.dumps(info, ensure_ascii=False)), **arrays)
    return out


# ---------- модель ----------

def _relu(x):
    return np.maximum(x, 0.0)


def _softplus(x):
    return np.logaddexp(0.0, x)


class Hybrid:
    """Ансамбль гибридов B1 (numpy)."""

    def __init__(self, path=WEIGHTS):
        with np.load(path, allow_pickle=False) as weights:
            self.info = json.loads(str(weights['info']))
            self.models = []
            for index, metadata in enumerate(self.info['models']):
                def array(name):
                    return weights[f'm{index}/{name}'].astype(np.float64)

                self.models.append({
                    'W': {name: (array(f'{name}.weight'), array(f'{name}.bias')) for name in _LAYERS},
                    'dn': (array('desc_norm.mu'), array('desc_norm.sd')),
                    'pn': (array('prof_norm.mu'), array('prof_norm.sd')),
                    'cqr': metadata['cqr'],
                })
        self.tau = float(self.info['tau'])
        self.ur_max = float(self.info['ur_max'])

    def _forward(self, model, descriptors, profile, u_r, formula):
        def linear(name, inputs):
            weights, bias = model['W'][name]
            return inputs @ weights.T + bias

        desc_mean, desc_std = model['dn']
        profile_mean, profile_std = model['pn']
        normalized_desc = (descriptors - desc_mean) / desc_std
        normalized_profile = (profile - profile_mean) / profile_std
        normalized_ur = (u_r - self.ur_max / 2) / (self.ur_max / math.sqrt(12))

        hidden = _relu(linear('desc.0', np.c_[normalized_desc, formula]))
        hidden = _relu(linear('desc.2', hidden))
        # Профиль травления задаёт масштаб и сдвиг каждого геометрического признака (FiLM).
        process = _relu(linear('film.net.0', np.c_[normalized_profile, normalized_ur]))
        scale, shift = np.split(linear('film.net.2', process), 2, axis=1)
        hidden = (1 + scale) * hidden + shift

        hidden = _relu(linear('head.0', np.c_[hidden, formula]))
        output = linear('head.2', hidden)
        median = formula + output[:, 0]
        return np.stack([median - _softplus(output[:, 1]), median, median + _softplus(output[:, 2])], axis=1)

    @staticmethod
    def formula_y(d, prof, u_r):
        return formula_y(d, prof, u_r)

    def predict(self, d, prof, u_r, tau=None):
        """d — (N, 10) дескрипторы, prof — (N, 5) = (ln Da, δ, t, EF, u), u_r — мкм, tau — допуск сужения
        (по умолчанию — τ обучения) → (квантили (N, 3) после CQR, среднее по моделям; P(y > τ); f — оценка
        формулы). Квантили — величины y, поэтому P(y > τ) годится для любого τ, не только обучающего."""
        d, prof = np.asarray(d, float), np.asarray(prof, float)
        u = np.full(len(d), float(u_r))
        f = self.formula_y(d, prof, u)
        theta = np.minimum(d[:, 0], d[:, 1]) * 90       # группы угла CQR — по min(θ, θ_ℓ), как в обучении
        qs = [apply_cqr(self._forward(m, d, prof, u, f), float(u_r), m['cqr'], theta) for m in self.models]
        q = np.mean(qs, axis=0)
        return q, prob_from_quantiles(q, self.tau if tau is None else float(tau)), f


_MODEL = None
_FORMULA = None


def formula_y(d, prof, u_r):
    """Сужение по формуле клина (таблица численной модели) при θ_эфф (дескриптор 3) и меньшей ширине меди
    у стенок (дескриптор 6); prof — (ln Da, δ, t, EF, u), как в train.formula_matrix."""
    global _FORMULA
    if _FORMULA is None:
        from .baselines import WedgeFormula
        _FORMULA = WedgeFormula.load()
    d, prof = np.asarray(d, float), np.asarray(prof, float)
    th = np.clip(d[:, 3] * 90, 0, 90)
    return _FORMULA(th, np.exp(prof[:, 0]), prof[:, 1], prof[:, 4], u_r) / (d[:, 6] * 1000)


def available(path=WEIGHTS) -> bool:
    """Есть веса и таблица клина, и таблица — та, с которой гибрид обучен (иначе основа f другая)."""
    from .baselines import TABLE_NPZ, table_sha1
    if not (Path(path).exists() and TABLE_NPZ.exists()):
        return False
    with np.load(path, allow_pickle=False) as weights:
        want = json.loads(str(weights['info'])).get('table_sha1')
    return want is None or want == table_sha1()


def model() -> Hybrid:
    global _MODEL
    if _MODEL is None:
        _MODEL = Hybrid()
    return _MODEL


def out_of_range(params: dict) -> list[str]:
    """Предупреждения, если профиль вне диапазона обучающих данных."""
    out = []
    for k, (lo, hi) in TRAIN_RANGE.items():
        v = params.get(k)
        if v is not None and not lo - 1e-9 <= v <= hi + 1e-9:
            out.append(f'{k} = {v:g} вне диапазона обучающих данных [{lo:g}; {hi:g}] — экстраполяция')
    return out


# ---------- дескрипторы кармана (общие признаки для анализатора и генератора) ----------

def window(lc, net, v, win=WIN):
    """Медь своей цепи и чужая медь слоя в квадрате ±win мм у вершины (как dataset.board_candidates)."""
    box = shapely.box(v[0] - win, v[1] - win, v[0] + win, v[1] + win)
    own = lc.nets[net].intersection(box)
    foreign = unary_union([g.intersection(box) for n, g in lc.nets.items() if n != net and g.intersects(box)]
                          or [shapely.Polygon()])
    return own, (None if foreign.is_empty else foreign)


def wall_widths(own, foreign, v, theta_deg, delta_um):
    """Ширина меди у обеих стенок кармана: s (мм, знак — сторона) и w (мм), точки — как в etch.simulate."""
    from .etch import _local_width, _walls, wall_length_mm
    Lw = wall_length_mm(theta_deg, delta_um)
    allcu = own if foreign is None else unary_union([own, foreign])
    shapely.prepare(allcu)
    s, w = [], []
    for side, si, p, n in _walls(own, np.asarray(v, float), Lw):
        s.append(side * si)
        w.append(_local_width(allcu, p, n))
    return np.array(s), np.array(w)


def descriptors(own, foreign, cand, delta_um, pocket_delta_um=None):
    """10 дескрипторов из features.DESC_NAMES; None, если ширину меди у стенок определить не удалось."""
    theta = min(cand['theta'], cand['theta_l'])
    s, w = wall_widths(own, foreign, cand['v'], theta, delta_um)
    ok = np.isfinite(w)
    if not ok.any():
        return None, None
    s, w = s[ok], w[ok]
    info = dict(theta=cand['theta'], theta_l=cand['theta_l'], area_mm2=cand['area'], depth_mm=cand['depth'],
                v=list(cand['v']), profile=dict(delta=delta_um))
    return pocket_descriptors(info, dict(s=s, w=w), own, foreign, pocket_delta_um or POCKET_DELTA_UM), float(np.min(w))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m dfm.ml.infer')
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export', help='выгрузить веса гибрида (папка моделей b1h_no_raster) в numpy')
    e.add_argument('models')
    e.add_argument('--out', default=str(WEIGHTS))
    a = ap.parse_args(argv)
    print(export(a.models, a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
