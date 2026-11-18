import numpy as np

from seismoe.data.dataset import gaussian_labels, normalize
from seismoe.eval import evaluate_picks, summarise, top1_picks


def test_gaussian_labels():
    y = gaussian_labels(1000, 2000)
    assert y.shape == (3, 6000)
    assert y[0, 1000] == 1.0
    assert y[1, 2000] == 1.0
    assert y[2].sum() == 0.0
    yn = gaussian_labels(-1, -1)
    assert yn[2].all()
    assert yn[0].sum() == 0.0


def test_normalize_per_channel():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(6000, 3)).astype(np.float32)
    y = normalize(x)
    np.testing.assert_allclose(y.mean(axis=0), np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(y.std(axis=0), np.ones(3), atol=1e-5)


def test_top1_picks_respects_order():
    p = np.zeros(6000)
    s = np.zeros(6000)
    p[1000] = 0.9
    s[900] = 0.95  # before P -> should not be selected as S
    s[2000] = 0.8
    pi, si = top1_picks(p, s)
    assert pi == 1000
    assert si == 2000


def test_evaluate_perfect():
    p = np.zeros((2, 6000), dtype=np.float32)
    s = np.zeros((2, 6000), dtype=np.float32)
    p[0, 1000] = 1.0
    s[0, 2000] = 1.0
    # second trace is noise and should stay silent
    p_true = np.array([1000, -1])
    s_true = np.array([2000, -1])
    is_noise = np.array([False, True])
    result = evaluate_picks(p, s, p_true, s_true, is_noise, 0.3, 0.3)
    metrics = summarise(result, 50)
    assert metrics["P_f1"] == 1.0
    assert metrics["S_f1"] == 1.0
    assert metrics["P_mae"] == 0.0
