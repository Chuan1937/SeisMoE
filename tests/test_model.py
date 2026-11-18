import torch

from seismoe.models import SeisMoE


def test_forward_shapes():
    model = SeisMoE(channels=(8, 16, 16, 32, 32, 64), moe_layers=1, num_experts=4)
    x = torch.randn(2, 3, 6000)
    logits, aux = model(x)
    assert logits.shape == (2, 3, 6000)
    assert torch.isfinite(aux)


def test_backward():
    model = SeisMoE(channels=(8, 16, 16, 32, 32, 64), moe_layers=1, num_experts=4)
    x = torch.randn(2, 3, 6000)
    logits, aux = model(x)
    loss = logits.square().mean() + aux
    loss.backward()
    assert model.head[-1].weight.grad is not None
    assert torch.isfinite(model.head[-1].weight.grad).all()


def test_no_moe_matches_dense():
    model = SeisMoE(channels=(8, 16, 16, 32, 32, 64), moe_layers=0)
    x = torch.randn(1, 3, 6000)
    logits, aux = model(x)
    assert logits.shape == (1, 3, 6000)
    assert float(aux) == 0.0
