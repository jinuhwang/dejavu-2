import os
import pytest
import torch

import hydra
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.inference import InferenceReuseModel
from src.utils.weight_adapter import map_training_to_inference_keys, load_with_report


def _compose_net_cfg():
    # Compose full train config then pull resolved model.net
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(
            config_name="train.yaml",
            overrides=[
                # Use large CLIP net preset (24 layers, safetensors available)
                "model/net=clip-vit-large-patch14",
                # Keep CPU-friendly trainer to avoid GPU assumptions
                "trainer=cpu",
                # Avoid WANDB network
                "logger=csv",
            ],
        )
    return cfg.model.net


@pytest.mark.slow
def test_forward_vs_infer_eval_parity_cpu():
    net_cfg = _compose_net_cfg()
    model = instantiate(net_cfg)
    model.eval()

    # Tiny batch
    B, F = 1, 1
    pixel_values = torch.randn(B, F, 3, 224, 224)
    # Match BlobNet default input_shape [C,T,H,W] = [4,4,14,14]
    compressed = torch.randn(B, F, 4, 4, 16, 16)
    ref_mask = torch.zeros(B, F, F, dtype=torch.bool)
    for f in range(F):
        ref_mask[:, f, : f + 1] = True

    # Reference type: simple constant one-hot per frame (B,F,3)
    ref_type = torch.zeros(B, F, 3)
    ref_type[..., 0] = 1.0

    # Training path
    out_train, _, _ = model(pixel_values=pixel_values, compressed=compressed, ref_mask=ref_mask, ref_type=ref_type)

    # Inference path with reuse disabled to match forward
    out_infer, _, _, _ = model.forward_eval(
        pixel_values=pixel_values,
        compressed=compressed,
        ref_mask=ref_mask,
        ref_type=ref_type,
        disable_reuse=True,
    )

    assert out_train.shape == out_infer.shape
    assert torch.allclose(out_train, out_infer, atol=1e-5, rtol=1e-4)


def test_real_state_dict_adapter_parity_cpu():
    net_cfg = _compose_net_cfg()
    model = instantiate(net_cfg)
    model.eval()
    infer = InferenceReuseModel(model)

    # Create a mock checkpoint from the model's own weights with legacy-style prefixes
    raw_sd = model.state_dict()
    ckpt = {f"net.{k}": v.clone() for k, v in raw_sd.items()}

    report, res = load_with_report(infer, ckpt, strict=False)
    # Strongest guarantee: no unexpected/missing after mapping
    assert report["unexpected"] == []
    assert report["missing"] == []


@pytest.mark.skipif("TRITON_SKIP" in os.environ, reason="Skipping Triton kernel check by env override")
def test_triton_stage_states_executes_if_available():
    try:
        import triton  # noqa: F401
        has_triton = True
    except Exception:
        has_triton = False

    import torch
    if not (has_triton and torch.cuda.is_available()):
        pytest.skip("Triton GPU execution not available; skipping kernel run")

    from src.models.components.stage_states import stage_states_local

    # Minimal, consistent shapes on GPU
    device = torch.device("cuda")
    B, N, dim = 1, 8, 16
    reuse_map = torch.zeros(B, N, dtype=torch.bool, device=device)
    pre_proj = torch.randn(B, N, dim, device=device)
    pre_proj_norm = torch.randn(B, N, dim, device=device)
    hidden_states = torch.randn(B, N, dim, device=device)
    ref = torch.randn(B, N, dim, device=device)
    ref_norm = torch.randn(B, N, dim, device=device)
    ref_gather_idx = torch.zeros(B, N, dtype=torch.long, device=device)
    diff_pre_proj = torch.empty_like(pre_proj, device=device)
    compute_cache = torch.empty((B*4*N, dim), device=device)
    hidden_cache = torch.empty_like(compute_cache, device=device)
    compute_cache_len = torch.zeros((1,), dtype=torch.long, device=device)
    gather_idxs = torch.zeros(B, N, dtype=torch.long, device=device)

    # Should run without error
    stage_states_local(
        reuse_map,
        pre_proj,
        pre_proj_norm,
        hidden_states,
        ref,
        ref_norm,
        ref_gather_idx,
        diff_pre_proj,
        compute_cache,
        hidden_cache,
        compute_cache_len,
        gather_idxs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU E2E test")
def test_reusevit_hard_gpu_end_to_end_runs():
    """GPU E2E: Ensure hard (Triton) model runs forward with F=4 on CUDA."""
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(
            config_name="train.yaml",
            overrides=[
                "model/net=clip-vit-large-patch14-hard",
                "trainer=cpu",
                "logger=csv",
            ],
        )
    model = instantiate(cfg.model.net)
    model = model.cuda().eval()

    # Inputs on CUDA (F must be 4 for hard pipeline)
    B, F = 1, 4
    pixel_values = torch.randn(B, F, 3, 224, 224, device="cuda")
    compressed = torch.randn(B, F, 4, 4, 16, 16, device="cuda")
    ref_mask = torch.zeros(B, F, F, dtype=torch.bool, device="cuda")
    for f in range(F):
        ref_mask[:, f, : f + 1] = True
    ref_type = torch.zeros(B, F, 3, device="cuda")
    ref_type[..., 0] = 1.0
    # Hard pipeline expects reference_type indexed by frame first: [F, B, C]
    ref_type_hard = ref_type.permute(1, 0, 2).contiguous()

    # Build warmup caches (per-layer), shapes inferred from config
    vcfg = model.model.vision_model.config
    hidden_size = vcfg.hidden_size
    tokens = (vcfg.image_size // vcfg.patch_size) ** 2 + 1
    L = len(model.model.vision_model.encoder.layers)
    reference_caches = [torch.zeros(B, tokens, hidden_size, device="cuda") for _ in range(L)]
    hqkv_caches = [torch.zeros(4, B, tokens, hidden_size, device="cuda") for _ in range(L)]

    # Run forward (should exercise Triton kernels via stage_states_local)
    with torch.inference_mode():
        out = model(
            pixel_values=pixel_values,
            compressed=compressed,
            ref_mask=ref_mask,
            reference_type=ref_type_hard,
            reference_caches=reference_caches,
            hqkv_caches=hqkv_caches,
        )

    # Expected image_embeds shape from CLIPVisionModelWithProjection: [F, B, embed_dim]
    embeds = out.image_embeds
    assert embeds.shape[0] == F and embeds.shape[1] == B and embeds.ndim == 3
