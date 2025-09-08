import os
import sys

import pytest
import torch

from src.inference import InferenceReuseModel
from src.utils.weight_adapter import map_training_to_inference_keys
from src.models.videoinstruct_module import VideoinstructLitModule


class TinyToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


def test_weight_adapter_state_dict_parity():
    # Create a toy model and a toy checkpoint with extra prefixes
    model = TinyToy()
    sd = model.state_dict()
    ckpt = {f"net.{k}": v.clone() for k, v in sd.items()}

    remapped = map_training_to_inference_keys(ckpt)
    # Load with strict=False should not error and have no unexpected
    res = model.load_state_dict(remapped, strict=False)
    assert res.unexpected_keys == []
    # Should fill all parameters
    assert set(res.missing_keys) == set()


def test_kernel_availability_fallback_import():
    """
    If Triton is unavailable, importing the module may fail. This test ensures
    we handle absence gracefully by skipping.
    """
    try:
        import triton  # noqa: F401
        has_triton = True
    except Exception:
        has_triton = False

    if not has_triton:
        pytest.skip("Triton not available on CPU; kernel tests skipped.")

    # Sanity: if present, ensure callable exists
    from src.models.components.stage_states import stage_states_local  # noqa: F401


def test_architecture_parity_shapes(monkeypatch):
    """
    Verify that inference wrapper exposes consistent attributes.
    Use a toy base model to avoid heavy dependencies.
    """
    base = TinyToy()

    # Monkeypatch attributes used by the wrapper for shape exposure
    class _Proj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_features = 4

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_projection = _Proj()
            self.vision_model = torch.nn.Module()
            self.vision_model.encoder = torch.nn.Module()
            self.vision_model.encoder.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])

    base.model = _Model()

    infer = InferenceReuseModel(base)  # type: ignore[arg-type]
    assert len(infer.encoder_layers) == 2
    assert infer.embed_dim == 4


def test_numerical_sanity_forward(monkeypatch):
    """
    Monkeypatch the base model's forward and forward_eval to deterministic transforms,
    then compare outputs with small tolerance to emulate parity.
    """
    class _Base(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=-1, keepdim=True)

        def forward_eval(self, pixel_values, **kwargs):
            out = pixel_values.sum(dim=-1, keepdim=True)
            reuse = torch.zeros(1, 1, 1)
            return out, reuse, None, None

    base = _Base()
    infer = InferenceReuseModel(base)  # type: ignore[arg-type]

    x = torch.randn(2, 3)
    out_train = base(x)
    out_infer, _, _, _ = infer(pixel_values=x)
    assert torch.allclose(out_train, out_infer, atol=1e-5, rtol=1e-4)


def test_iwl_compatibility_checks():
    # Build a tiny module to exercise the compatibility helper
    class _DummySim(torch.nn.Module):
        pass

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            class _RM(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.similarity_module = _DummySim()
            self.reuse_module = _RM()

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            class _Enc(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
            class _VM(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = _Enc()
                def forward(self, x, **kwargs):
                    return x
            self.model = type('M', (), {})()
            self.model.vision_model = _VM()
            self.orig_model = torch.nn.Identity()

    m = VideoinstructLitModule(
        net=_Net(),
        loss=torch.nn.Identity(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        blobnet_lr=0.0,
        restoration_lr=0.0,
        decision_lr=0.0,
        rloss_pattern=[True],
        sloss_pattern=[True],
        compile=False,
        gating_hard=False,
    )
    ok, reason = m._check_iwl_compatibility("src.models.components.reuse.similarity.LocalCosineSimilarity")
    assert not ok and "mismatch" in reason
    ok2, _ = m._check_iwl_compatibility(_DummySim.__name__)
    assert ok2
