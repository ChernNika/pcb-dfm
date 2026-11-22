"""Шаг 4.3: нейросеть B1 (модель, потери, метки по стенкам, разбиение, метрики)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from dfm.ml.dataset import load_example, run_task                 # noqa: E402
from dfm.ml.etch import labels_for, t_cap                         # noqa: E402
from dfm.ml.model import B1Net, focal, pinball                    # noqa: E402
from dfm.ml.train import (apply_cqr, average_precision, cqr_offsets, fit_cqr, folds, labels,  # noqa: E402
                          precision_at_recall)

torch.set_num_threads(2)


def test_pinball_matches_document_example():
    """Документ B1, раздел 8.4: y = 10, q = 0,9 — прогноз 8 даёт штраф 1,8, прогноз 12 — 0,2."""
    y = torch.tensor([10.0])
    assert pinball(y, torch.tensor([[8.0]]), (0.9,)).item() == pytest.approx(1.8)
    assert pinball(y, torch.tensor([[12.0]]), (0.9,)).item() == pytest.approx(0.2)


def test_focal_downweights_easy_examples():
    t = torch.tensor([1.0])
    assert focal(torch.tensor([4.0]), t) < 0.01 * focal(torch.tensor([-1.0]), t)


def test_model_shapes_order_and_film():
    torch.manual_seed(0)
    m = B1Net(n_desc=10, z_dim=6).eval()
    x, d = torch.rand(5, 3, 64, 64), torch.randn(5, 10)
    z1, z2 = torch.zeros(5, 6), torch.ones(5, 6)
    q, logit = m(x, d, z1)
    assert q.shape == (5, 3) and logit.shape == (5,)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()       # квантили не пересекаются
    with torch.no_grad():
        assert torch.allclose(m(x, d, z2)[0], q)       # FiLM в начале обучения тождественна
        m.film.net[-1].weight.normal_()
        assert not torch.allclose(m(x, d, z2)[0], m(x, d, z1)[0])   # профиль меняет ответ только через FiLM


def test_labels_from_walls_match_teacher(tmp_path):
    run_task(dict(id='t0', src='synth', seed=5, group='synth00'), tmp_path)
    info, walls, *_ = load_example(tmp_path / 't0.npz')
    th = min(info['theta'], info['theta_l'])
    tc = t_cap(th)
    a = walls['a'][None].astype(np.float32)
    b = np.minimum(walls['b'], tc - 1)[None].astype(np.float32)
    w = (walls['w'] * 1000)[None].astype(np.float32)
    m = np.ones_like(a, bool)
    for u in (0.0, 0.5, 2.0):
        ref = labels_for(walls, u, t_max=tc)['y']
        assert labels(a, b, w, m, u)[0] == pytest.approx(ref, abs=1e-5)
        assert labels(*map(torch.as_tensor, (a, b, w, m)), torch.tensor([u]))[0].item() == pytest.approx(ref, abs=1e-5)
        assert info['y'][str(u)] == pytest.approx(ref, abs=1e-5)


def test_folds_keep_groups_whole():
    meta = ([dict(group=f'synth{i % 10:02d}', source='synth') for i in range(100)] +
            [dict(group=f'sharp{i % 5:02d}', source='sharp') for i in range(30)] +
            [dict(group=f'board{i % 5}', source='real') for i in range(50)] + [dict(group='b1_demo', source='demo')] * 4)
    f = folds(meta)
    for g in {r['group'] for r in meta}:
        assert len({int(f[i]) for i, r in enumerate(meta) if r['group'] == g}) == 1
    assert (f[-4:] == -1).all()
    for src in ('synth', 'sharp', 'real'):
        assert len({int(f[i]) for i, r in enumerate(meta) if r['source'] == src}) == 5   # каждый источник во всех частях


def test_precision_at_recall_and_ap():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    # чтобы найти все 3 опасных, надо выдать 6 предупреждений: точность 3/6
    assert precision_at_recall(y, s, 0.95)[0] == pytest.approx(0.5)
    assert precision_at_recall(y, s, 0.95)[2] == 6
    assert precision_at_recall(y, s, 0.6)[0] == pytest.approx(2 / 3)
    assert average_precision(y, s) == pytest.approx((1 + 2 / 3 + 0.5) / 3)


def test_metrics_with_ties():
    """Равные оценки — один порог: результат не зависит от порядка примеров внутри блока (как sklearn)."""
    y1, y2 = np.array([1, 0, 1, 0, 0]), np.array([0, 1, 1, 0, 0])
    s = np.array([1.0, 1.0, 0.5, 0.2, 0.2])
    for y in (y1, y2):
        assert average_precision(y, s) == pytest.approx(0.5 * 0.5 + 0.5 * 2 / 3)
        assert precision_at_recall(y, s, 0.95) == (pytest.approx(2 / 3), 0.5, 3)


def test_split_parts_disjoint_demo_excluded():
    """Обучение, ранняя остановка, проверка не пересекаются; демо-плата — только отдельно."""
    from dfm.ml.train import _split
    meta = ([dict(group=f'synth{i % 10:02d}', source='synth') for i in range(100)] +
            [dict(group=f'board{i % 5}', source='real') for i in range(50)] + [dict(group='b1_demo', source='demo')] * 4)
    for k in range(5):
        tr, va, te, demo = (set(map(int, x)) for x in _split(meta, k))
        assert not (tr & va or tr & te or va & te) and demo == {150, 151, 152, 153}
        assert not demo & (tr | va | te) and len(tr | va | te) == 150


def test_capped_b():
    """Время пребывания в метках ограничено пределом клина того же угла: b ≤ t_cap(θ) − 1."""
    from dfm.ml.train import capped_b
    th = np.array([20.0, 60.0], np.float32)
    b = np.array([[0.1, 100.0], [0.0, 0.5]], np.float32)
    out = capped_b(dict(theta=th, b=b))
    assert out[0, 0] == pytest.approx(0.1) and out[0, 1] == pytest.approx(t_cap(20.0) - 1)
    assert out[1, 1] == pytest.approx(min(0.5, t_cap(60.0) - 1))


def test_model_can_fit_tiny_batch():
    torch.manual_seed(1)
    m = B1Net(n_desc=4, z_dim=3)
    x, d, z = torch.rand(16, 3, 64, 64), torch.randn(16, 4), torch.randn(16, 3)
    y = d[:, 0] * 0.1 + z[:, 0] * 0.05
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        q, logit = m(x, d, z)
        loss = pinball(y, q) + focal(logit, (y > 0).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first


def test_cqr_restores_coverage():
    """Узкие интервалы (покрытие ≈ 50 %) после конформной калибровки покрывают ≥ 80 % новой выборки."""
    rng = np.random.default_rng(0)
    def sample(n):
        y = rng.normal(0, 1, n)
        q = np.stack([np.full(n, -0.67), np.zeros(n), np.full(n, 0.67)], axis=1)
        return y, q
    y, q = sample(2000)
    e = {'0.0': cqr_offsets(y, q), '2.0': cqr_offsets(y, q)}
    assert e['0.0'] > 0.5
    y2, q2 = sample(5000)
    c = apply_cqr(q2, 1.0, e)
    cov = np.mean((y2 >= c[:, 0]) & (y2 <= c[:, 2]))
    assert 0.78 <= cov <= 0.84
    wide = np.stack([np.full(10, -5.0), np.zeros(10), np.full(10, 5.0)], axis=1)
    c = apply_cqr(wide, 0.0, {'0.0': -10.0, '2.0': -10.0})       # сужение не переворачивает интервал
    assert (c[:, 0] <= c[:, 1]).all() and (c[:, 1] <= c[:, 2]).all()


def test_cqr_by_angle_group():
    """Острые карманы (θ < 20°) с разбросом втрое больше: калибровка по группам даёт ≈ 80 % в каждой."""
    rng = np.random.default_rng(1)
    def sample(n):
        th = np.where(rng.random(n) < 0.3, 12.0, 60.0)
        y = rng.normal(0, 1, n) * np.where(th < 20, 3.0, 1.0)
        q = np.stack([np.full(n, -0.67), np.zeros(n), np.full(n, 0.67)], axis=1)
        return y, q, th
    y, q, th = sample(4000)
    cqr = fit_cqr({0.0: (y, q, None), 2.0: (y, q, None)}, th, grid=(0.0, 2.0))
    y2, q2, th2 = sample(8000)
    c = apply_cqr(q2, 1.0, cqr, th2)
    for m in (th2 < 20, th2 >= 20):
        cov = np.mean((y2[m] >= c[m, 0]) & (y2[m] <= c[m, 2]))
        assert 0.76 <= cov <= 0.85


def test_wedge_geometry_gives_requested_angle():
    from dfm.ml.baselines import wedge
    from dfm.ml.dataset import B1_GEOM
    from dfm.rules.b1 import trap_candidates
    for th in (8.0, 15.0, 30.0, 60.0):
        cs = [c for c in trap_candidates(wedge(th), B1_GEOM) if c['slot'] is None]
        assert min(abs(min(c['theta'], c['theta_l']) - th) for c in cs) < 0.5


def test_wedge_formula_interpolates_nodes():
    from dfm.ml.baselines import DAS, DELTAS, THETAS, URS, US, WedgeFormula
    rng = np.random.default_rng(0)
    D = rng.random((len(THETAS), len(DAS), len(DELTAS), len(US), len(URS)))
    F = WedgeFormula(D)
    i = (3, 2, 1, 4, 20)
    assert F(THETAS[3], DAS[2], DELTAS[1], US[4], URS[20]) == pytest.approx(D[i])
    mid = F((THETAS[3] + THETAS[4]) / 2, DAS[2], DELTAS[1], US[4], URS[20])     # линейно между узлами по θ
    assert mid == pytest.approx((D[i] + D[(4, 2, 1, 4, 20)]) / 2)
    assert F(200.0, DAS[2], DELTAS[1], US[4], URS[20]) == pytest.approx(D[(-1, 2, 1, 4, 20)])   # за краем — край


def test_ablation_inputs():
    from dfm.ml.train import prepare
    x = torch.rand(3, 3, 64, 64)
    assert prepare(x, 'no_foreign')[:, 1].abs().sum() == 0
    assert prepare(x, 'res32').shape == (3, 3, 32, 32)
    r1, r2 = prepare(x, 'no_align', np.arange(3)), prepare(x, 'no_align', np.arange(3))
    assert torch.equal(r1, r2) and not torch.allclose(r1, x)        # при проверке поворот воспроизводим
    assert torch.equal(prepare(x, 'full'), x)


def test_hybrid_starts_from_formula():
    """Гибрид: до обучения ŷ₀,₅ = оценка формулы; профиль и растр меняют только поправку."""
    torch.manual_seed(0)
    m = B1Net(n_desc=10, z_dim=6, hybrid=True).eval()
    f = torch.tensor([0.0, 0.1, 0.35, 1.2])
    q, _ = m(torch.rand(4, 3, 64, 64), torch.randn(4, 10), torch.randn(4, 6), torch.ones(4), f)
    assert torch.allclose(q[:, 1], f)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()


def test_formula_at_interpolates_ur():
    from dfm.ml.baselines import URS
    from dfm.ml.train import formula_at
    Fm = torch.as_tensor(np.tile(np.array(URS, np.float32) * 3.0, (2, 1)))       # f = 3·u_r
    u = torch.tensor([0.25, 1.95])
    assert torch.allclose(formula_at(Fm, u), 3 * u, atol=1e-5)
    assert torch.allclose(formula_at(Fm, torch.tensor([5.0, -1.0])), torch.tensor([6.0, 0.0]))  # за краями — край


def test_prob_from_quantiles():
    from dfm.ml.train import prob_from_quantiles
    q = np.array([[0.1, 0.2, 0.3], [0.0, 0.05, 0.1], [0.3, 0.4, 0.6], [0.12, 0.15, 0.2]])
    p = prob_from_quantiles(q, 0.2)
    assert p[0] == pytest.approx(0.5)                      # медиана на пороге
    assert p[1] < 1e-3 and p[2] > 0.9                       # интервал целиком ниже / выше порога
    assert p[3] == pytest.approx(0.1, abs=1e-6)             # порог на ŷ₀,₉ → 10 %


def test_binom_ci_clopper_pearson():
    """Точный интервал доли: 30/31 → 0,833…0,999; 0 из n и n из n — односторонние границы."""
    from dfm.ml.train import binom_ci, fmt_ci
    assert binom_ci(30, 31) == (pytest.approx(0.8330, abs=1e-3), pytest.approx(0.9992, abs=1e-3))
    assert binom_ci(42, 42) == (pytest.approx(0.9159, abs=1e-3), 1.0)
    assert binom_ci(0, 1133)[0] == 0.0 and binom_ci(0, 1133)[1] == pytest.approx(0.00325, abs=1e-4)
    assert fmt_ci(30, 31) == '30/31 = 0,968 [0,833; 0,999]'
