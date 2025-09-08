#!/usr/bin/env python
"""CPU-only probe for decision layer and loss wiring.

Runs a synthetic forward through ReuseLoss and prints shapes and which losses supervise them.
No datasets or external downloads.
"""
from __future__ import annotations

import torch
from torch import nn

from ..models.components.loss import ReuseLoss


def main():
    torch.manual_seed(0)

    B, F, N, D = 2, 2, 1 + 8, 16  # CLS + 8 patches
    L = 2  # layers contributing to hidden loss

    # Synthetic hidden states: list of L tensors shaped (B, F, N, D)
    hidden_states = [torch.randn(B, F, N, D) for _ in range(L)]
    original_hidden_states = [h + 0.01 * torch.randn_like(h) for h in hidden_states]

    # Synthetic pooled outputs (CLS embeddings): (B, F, D)
    output = torch.randn(B, F, D)
    original_output = output + 0.01 * torch.randn_like(output)

    # Synthetic reuse maps: (B, F-1, num_layers, N)
    reuse_maps = torch.rand(B, F, L, N) > 0.5

    loss_fn = ReuseLoss(
        target_reuse_rate=0.5,
        use_min_hloss=False,
        use_min_sloss=True,
        hloss_scaler=[0.0, 1.5],
        sloss_scaler=1.0,
        rloss_scaler=0.35,
    )

    loss, hidden_error, hh_error, cls_error, reuse_rate, reuse_rate_per_frame = loss_fn(
        hidden_states,
        output,
        original_hidden_states,
        original_output,
        reuse_maps,
    )

    print("Tokens: N=", N, "(CLS idx=0, patches idx=1..N-1)")
    print("hidden_states[L,B,F,N,D] sample:", torch.stack(hidden_states).shape)
    print("output[B,F,D] shape:", output.shape)
    print("reuse_maps[B,F,L,N] shape:", reuse_maps.shape)
    print("Loss components:")
    print(" - sloss (CLS pooled cosine): contributes via sloss_scaler")
    print(" - hloss (token cosine over last L layers): contributes via hloss_scaler")
    print(" - rloss (reuse rate target): contributes via rloss_scaler")
    print("Scalars (float):")
    for k, v in {
        'loss': loss,
        'hidden_error': hidden_error,
        'hh_error': hh_error,
        'cls_error': cls_error,
        'reuse_rate': reuse_rate,
    }.items():
        print(f"  {k}: {float(v):.6f}")


if __name__ == "__main__":
    main()

