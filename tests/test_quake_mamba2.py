import importlib
import sys
import types

import pytest
import torch
import torch.nn as nn


class _StubMamba2(nn.Module):
    instances = []

    def __init__(self, d_model, d_state, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.kwargs = kwargs
        self.last_input_shape = None
        _StubMamba2.instances.append(self)

    def forward(self, x):
        self.last_input_shape = tuple(x.shape)
        return x


@pytest.fixture
def quake_module(monkeypatch):
    _StubMamba2.instances.clear()
    monkeypatch.setitem(sys.modules, "mamba_ssm", types.SimpleNamespace(Mamba2=_StubMamba2))

    sys.modules.pop("src.models.quake_mamba2", None)
    return importlib.import_module("src.models.quake_mamba2")


def test_quake_mamba2_forward_shape(quake_module):
    QuakeMamba2 = quake_module.QuakeMamba2
    model = QuakeMamba2(d_model=32, d_state=8, input_dim=64, num_classes=4, num_patches=16)

    x = torch.randn(2, 5, 16, 4)  # P * F = 64
    y = model(x)

    assert y.shape == (2, 16, 4)


def test_quake_mamba2_passes_kwargs_to_mamba2(quake_module):
    QuakeMamba2 = quake_module.QuakeMamba2
    model = QuakeMamba2(d_model=48, d_state=12, input_dim=64, expand=2)

    assert len(_StubMamba2.instances) == 1
    stub = _StubMamba2.instances[0]
    assert stub.d_model == 48
    assert stub.d_state == 12
    assert stub.kwargs["expand"] == 2

    x = torch.randn(1, 3, 16, 4)
    _ = model(x)
    assert stub.last_input_shape == (1, 3, 48)


def test_quake_mamba2_uses_last_sequence_step(quake_module):
    QuakeMamba2 = quake_module.QuakeMamba2
    model = QuakeMamba2(d_model=16, d_state=4, input_dim=8, num_classes=3, num_patches=2)

    with torch.no_grad():
        # Zero all Linear weights and biases in the Sequential proj_in
        for layer in model.proj_in:
            if hasattr(layer, 'weight'):
                layer.weight.zero_()
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.zero_()
        model.proj_out.weight.zero_()
        model.proj_out.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    x = torch.randn(4, 7, 2, 4)  # input_dim = 2 * 4 = 8
    y = model(x)

    expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.allclose(y, expected.unsqueeze(0).expand(4, -1, -1))
