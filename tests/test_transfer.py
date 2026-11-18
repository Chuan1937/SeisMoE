import numpy as np
import torch

from seismoe.transfer import TransferDataset, TransferSet, normalize


def test_normalize_modes():
    x = np.random.default_rng(0).normal(2.0, 3.0, size=(3, 6000)).astype(np.float32)
    y = normalize(x, "std")
    np.testing.assert_allclose(y.mean(axis=1), np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(y.std(axis=1), np.ones(3), atol=1e-5)
    z = normalize(x, "max")
    np.testing.assert_allclose(np.abs(z).max(axis=1), np.ones(3), atol=1e-5)


def test_transfer_dataset_shapes():
    waves = np.random.default_rng(0).normal(size=(4, 3, 6000)).astype(np.float32)
    data = TransferSet(
        waveforms=waves,
        p=np.array([500, 800, 1000, 1200]),
        s=np.array([1500, 1800, 2000, 2200]),
        names=["a", "b", "c", "d"],
    )
    dataset = TransferDataset(data, train=True, seed=0, label_sigma=10)
    x, y = dataset[0]
    assert x.shape == (3, 6000)
    assert y.shape == (3, 6000)
    assert torch.isfinite(x).all()
    # P/S peaks stay near the (possibly shifted) labels
    assert y[0].argmax() > 200 and y[1].argmax() > y[0].argmax()
