"""Шаг 4.2: учитель (численная модель травления), растровый фрагмент и генератор данных B1."""
import json
import math

import numpy as np
import pytest
import shapely

from dfm.ml.dataset import (B1_GEOM, KINDS, is_slot, load_example, make_tasks, patch, relabel, run_task,
                            synth_scene)
from dfm.ml.etch import labels_for, simulate, t_cap
from dfm.rules.b1 import trap_candidates


def _wedge_branch(th=30.0, w=0.15):
    """Ситуация Л2 документа: трасса 0,15 мм и ответвление под углом θ."""
    a = shapely.LineString([(-1.2, 0), (1.0, 0)]).buffer(w / 2, quad_segs=16)
    b = shapely.LineString([(0, 0), (1.4 * math.cos(math.radians(th)), 1.4 * math.sin(math.radians(th)))]).buffer(
        w / 2, quad_segs=16)
    return a.union(b)


def test_teacher_matches_document():
    """Карман 30° (Л2): профиль З (Da 3, u_r 2) — перетрав ≈ 5,4 мкм, О (u_r 0,5) — ≈ 0,1 мкм
    (документ B1, таблица раздела 5.3); недотрав в глубине тупых мест — отрицательные смещения."""
    own = _wedge_branch()
    c = min(trap_candidates(own, B1_GEOM), key=lambda c: c['theta'])
    assert c['theta'] == pytest.approx(30, abs=0.5)
    sim = simulate(own, own, c['v'], c['theta'], 3.0, 50.0, 12.0)
    assert labels_for(sim, 2.0)['notch'] == pytest.approx(5.4, abs=0.4)
    assert labels_for(sim, 0.5)['notch'] == pytest.approx(0.1, abs=0.3)
    z = labels_for(sim, 2.0)
    assert z['y'] == pytest.approx(z['notch'] / 150, rel=0.15)    # сужение относительно ширины трассы 0,15 мм
    assert labels_for(sim, 0.0)['notch'] < z['notch']             # промывка только добавляет перетрав
    # предел времени пребывания клина не меняет карман документа
    assert labels_for(sim, 2.0, t_max=t_cap(c['theta']))['notch'] == pytest.approx(z['notch'], abs=1e-6)


def test_t_cap_cuts_only_long_residence():
    """Предел T: 1,25·(1 + A(θ)); стенки с T выше предела обрезаются, остальные не меняются."""
    assert t_cap(30.0) == pytest.approx(1.25 * (1 + 1 / (2 * math.sin(math.radians(15)) ** 2)))
    assert t_cap(10.0) > t_cap(30.0) > t_cap(90.0)
    tc = t_cap(70.0)
    sim = dict(s=np.array([0.1, 0.2]), a=np.array([0.5, 0.5]), b=np.array([1.0, 30.0]), w=np.array([0.1, 0.1]))
    lab = labels_for(sim, 2.0, t_max=tc)
    assert lab['notch'] == pytest.approx(0.5 + 2.0 * (tc - 1))
    assert labels_for(sim, 2.0)['notch'] == pytest.approx(0.5 + 2.0 * 30)
    sim['b'] = np.array([1.0, 1.5])
    assert labels_for(sim, 2.0, t_max=tc)['notch'] == labels_for(sim, 2.0)['notch']


def test_patch_points_bisector_up():
    """Фрагмент повёрнут биссектрисой вверх: над вершиной — пустота кармана, под ней — медь."""
    own = _wedge_branch().buffer(0)
    for rot in (0, 73, 200):
        g = shapely.affinity.rotate(own, rot, origin=(0, 0))
        c = min(trap_candidates(g, B1_GEOM), key=lambda c: c['theta'])
        p = patch(g, None, c['v'], c['beta'])
        n = p.shape[1]
        assert p[0, : n // 2 - 4, n // 2].mean() == 0                # столбец над вершиной — пустота
        assert p[0, n // 2 + 2:, n // 2].mean() > 0.5               # под вершиной — медь
        assert p[2].min() == 0 and p[2].max() > 1


@pytest.mark.parametrize('kind', KINDS)
def test_synthetic_scenes_have_pockets(kind):
    """Как в run_task: карман (не щель, у стыка) находится не более чем за 20 сцен. Сцена без кармана
    бывает часто (Л3 — короткий хвост закрывает острый карман скруглением), поэтому генератор повторяет."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for _ in range(20):
            own, foreign, par = synth_scene(rng, kind)
            assert own.is_valid and par['kind'] == kind
            if any(c['slot'] is None and math.hypot(*c['v']) < 0.6 for c in trap_candidates(own, B1_GEOM)):
                break
        else:
            pytest.fail(f'{kind}: нет кармана за 20 сцен (seed {seed})')


def test_run_task_writes_example(tmp_path):
    task = dict(id='t0', src='synth', seed=5, group='synth00')
    r = run_task(task, tmp_path)
    assert r is not None and (tmp_path / 't0.npz').exists()
    assert run_task(task, tmp_path) is None                          # уже посчитан — пропуск
    info, walls, own, _, p = load_example(tmp_path / 't0.npz')
    assert own is not None and p.shape == (3, 64, 64)
    assert set(info['profile']) == {'Da', 'delta', 't', 'ef', 'u'}
    assert 0.1 <= info['profile']['Da'] <= 30 and len(walls['a']) == len(walls['w'])


def test_slot_flag_depends_on_delta():
    """Трасса вдоль края площадки своей цепи в 120 мкм: при δ = 50 мкм — клин (щель шире 2δ),
    при δ = 75 мкм — щель, в обучение не идёт; карман ответвления 30° — не щель при любом δ."""
    pad = shapely.box(-0.5, -0.5, 0.5, 0.5)
    x = 0.5 + 0.12 + 0.075
    tr = shapely.LineString([(0.3, 0.8), (x, 0.8 - (x - 0.3)), (x, -0.8)]).buffer(0.075, quad_segs=16)
    own = pad.union(tr)
    c = min(trap_candidates(own, B1_GEOM), key=lambda c: c['theta'])
    info = dict(v=list(c['v']), theta=c['theta'], theta_l=c['theta_l'], profile=dict(delta=50.0))
    assert not is_slot(info, own)
    info['profile']['delta'] = 75.0
    assert is_slot(info, own)
    br = _wedge_branch()
    c = min(trap_candidates(br, B1_GEOM), key=lambda c: c['theta'])
    for d in (30.0, 80.0):
        assert not is_slot(dict(v=list(c['v']), theta=c['theta'], theta_l=c['theta_l'], profile=dict(delta=d)), br)


def test_sharp_tasks_do_not_change_main_tasks():
    base = make_tasks(20, seed=7)
    more = make_tasks(20, seed=7, n_sharp=5)
    assert more[:20] == base
    assert [t['id'] for t in more[20:]] == [f'a{i:05d}' for i in range(5)]
    assert all(t['theta_range'] == (8.0, 28.0) for t in more[20:])


@pytest.mark.parametrize('kind', ['Л1', 'Л2', 'Л4', 'Л5', 'Л6'])
def test_sharp_scenes_have_sharp_pockets(kind):
    """Острая партия: измеренный угол кармана близок к заданному (Л6 — на ~1,5° больше из-за кривизны)."""
    rng = np.random.default_rng(3)
    good = 0
    for _ in range(8):
        own, _, par = synth_scene(rng, kind, theta_range=(8, 28))
        cs = [c for c in trap_candidates(own, B1_GEOM)
              if c['slot'] is None and math.hypot(*c['v']) < 0.6 + par['ext']]
        m = min((min(c['theta'], c['theta_l']) for c in cs), default=None)
        good += m is not None and abs(m - par['theta_set']) < 3
    assert good >= 6, kind


def test_relabel_applies_cap(tmp_path):
    task = dict(id='t0', src='synth', seed=5, group='synth00')
    run_task(task, tmp_path)
    info, walls, *_ = load_example(tmp_path / 't0.npz')
    assert {'T_cap', 'y', 'y_raw', 'notch'} <= set(info)
    assert info['y']['2.0'] <= info['y_raw']['2.0'] + 1e-9
    z = dict(np.load(tmp_path / 't0.npz'))
    old = json.loads(str(z['info']))
    for k in ('T_cap', 'y_raw'):
        old.pop(k)
    old['y'] = {'2.0': 99.0}
    z['info'] = np.array(json.dumps(old, ensure_ascii=False))
    np.savez_compressed(tmp_path / 't0.npz', **z)
    relabel(tmp_path)
    info2, walls2, *_ = load_example(tmp_path / 't0.npz')
    assert info2['y'] == pytest.approx(info['y']) and info2['T_cap'] == pytest.approx(info['T_cap'])
    assert np.array_equal(walls2['b'], walls['b']) and walls2['b'].dtype == walls['b'].dtype


def test_teacher_window_covers_pocket():
    """Окно учителя должно вмещать диффузионный слой кармана: 1,25·δ/sin(θ/2) + 100 мкм. Обрезанное окно
    (раньше ≤ 600 мкм) занижало перетрав острых клиньев до 5 раз; предел 2000 мкм покрывает генератор (8°, δ 80)."""
    import inspect
    rw_max = inspect.signature(simulate).parameters['rw_max'].default
    assert 1.25 * 80 / math.sin(math.radians(8) / 2) + 100 <= rw_max
    from dfm.ml.baselines import wedge                     # удлинённый клин: стенки длиннее окна
    own = wedge(15.0)
    c = min(trap_candidates(own, B1_GEOM), key=lambda c: c['theta'])
    a = labels_for(simulate(own, own, c['v'], c['theta'], 3.0, 80.0, 12.0, rw_max=900.0), 2.0)['notch']
    b = labels_for(simulate(own, own, c['v'], c['theta'], 3.0, 80.0, 12.0, rw_max=1200.0), 2.0)['notch']
    assert a == pytest.approx(b, rel=0.01)                 # окно больше нужного (866 мкм) — ответ не меняется
    c = labels_for(simulate(own, own, c['v'], c['theta'], 3.0, 80.0, 12.0, rw_max=600.0), 2.0)['notch']
    assert c < 0.9 * b                                     # обрезанное окно занижает перетрав (так было до 20.11)
