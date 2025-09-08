import torch

from src.models.components.reuse.gating import AdafuseGating


def test_uncertainty_scales_gate_map():
    B, N = 2, 3
    gating = AdafuseGating(tau=1.0, gating_scheduling=False)

    # Decision favoring reuse strongly, but with different uncertainties
    main_logits = torch.tensor([[[ 4.0, -4.0]]]).expand(B, N, 2)  # strong reuse
    low_uncert = torch.full((B, N, 1), -4.0)  # sigmoid ~ 0.018 (low uncertainty)
    high_uncert = torch.full((B, N, 1),  4.0)  # sigmoid ~ 0.982 (high uncertainty)

    dec_low = torch.cat([main_logits, low_uncert], dim=-1)
    dec_high = torch.cat([main_logits, high_uncert], dim=-1)

    upper = [torch.ones(B, N, 1)]
    lower = [torch.zeros(B, N, 1)]

    gate_low, _ = gating(dec_low, upper, lower, hard=False, tau=1.0)
    gate_high, _ = gating(dec_high, upper, lower, hard=False, tau=1.0)

    # Gate map should be significantly reduced by high uncertainty
    assert (gate_low.mean() > gate_high.mean()).item() is True
