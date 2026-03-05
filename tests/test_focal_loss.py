import torch
import torch.nn.functional as F

from src.utils.focal_loss import FocalLoss


def test_focal_loss_matches_cross_entropy_when_gamma_zero():
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.7, -0.2]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    focal = FocalLoss(gamma=0.0)
    focal_value = focal(logits, targets)
    ce_value = F.cross_entropy(logits, targets)

    assert torch.isclose(focal_value, ce_value, atol=1e-7)


def test_focal_loss_matches_manual_formula_with_alpha():
    logits = torch.tensor(
        [[1.0, -0.5, 0.2], [0.3, 0.1, -0.4], [-0.2, 0.4, 1.2]], dtype=torch.float32
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long)
    alpha = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    gamma = 2.0

    focal = FocalLoss(alpha=alpha, gamma=gamma)
    focal_value = focal(logits, targets)

    ce = F.cross_entropy(logits, targets, reduction="none", weight=alpha)
    pt = torch.exp(-ce)
    expected = (((1 - pt) ** gamma) * ce).mean()

    assert torch.isclose(focal_value, expected, atol=1e-7)


def test_focal_loss_backprop_produces_finite_gradients():
    torch.manual_seed(0)
    logits = torch.randn(5, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)

    loss = FocalLoss(gamma=2.0)(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
