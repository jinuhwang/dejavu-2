# Decision Layer: Architecture and Loss (dejavu-2)

This brief summarizes where the reuse/decision layer sits, how class vs non-class tokens flow, and how the current losses supervise them. Pointers reference concrete code paths in this repository for reproducibility.

## Overview (Modules & Config)
- Reuse layer wrapper: `src/models/components/reusevit.py` → `ReuseEncoderLayer`
- Core reuse block: `src/models/components/reuse/module.py` → `ReuseModule`
  - Similarity backends: `src/models/components/reuse/similarity.py` (e.g., `LocalCosineSimilarity`)
  - Decision MLP: `src/models/components/reuse/decision.py` → `ReuseMLP`
  - Gating: `src/models/components/reuse/gating.py` (e.g., `AdafuseGating`)
  - Restoration: `src/models/components/reuse/restoration.py` (e.g., `MLPRestoration`)
- Loss: `src/models/components/loss.py` → `ReuseLoss`
- Typical configs: `configs/model/reusevit.yaml`, `configs/model/reuse/decision/mlp.yaml`, `configs/model/reuse/similarity/local_cosine.yaml`, overrides in `configs/experiment/nextqa*.yaml`.

## Decision MLP (current used config)
- Base MLP config: `configs/model/reuse/decision/mlp.yaml`
  - `out_dim: 2` ([reuse, recompute])
  - `inner_dim: 160`
  - `use_compressed_info: true`, `use_reference_type: true`, `use_norm: false`
  - Input per token: `[importance, most_similar_score, compressed_map, ref_type_onehot(3)]` → 6 dims total
  - Default `layer_pattern: 'lbrlbrl'`
    - Linear(6→160) → BatchNorm1d → ReLU → Linear(160→160) → BatchNorm1d → ReLU → Linear(160→2)
  - Weight init: `initialize: 'adafuse'` (Linear weights ~ N(0, 0.001), bias=0)
- In `configs/experiment/nextqa*.yaml` the pattern is overridden:
  - `layer_pattern: 'ldrl'` → Linear(6→160) → Dropout(0.25) → ReLU → Linear(160→2)
  - This is the current effective MLP in most runs.

## Forward sketch (shapes, CLS vs others)
- Per encoder layer, hidden states `h_l` have shape `(B, N, D)`, `N = 1 + P` (CLS + P patches).
- `ReuseModule.forward()` (see `reuse/module.py`):
  - If `skip_cls=True`, split CLS and tokens (CLS excluded from decision):
    - `cls = h_l[:, :1]`, `tokens = h_l[:, 1:]`
  - Compute similarity between `tokens` and cached tokens with the selected backend (e.g., `LocalCosineSimilarity`), mask by reference graph, then pool to the most similar index per token with the decision MLP.
  - Restoration MLP adjusts reused states; gating merges [restored vs recomputed] per token.
  - Re‑attach CLS unchanged; decision logits shape `(B, N-1, 2)`; updated tokens `(B, N-1, D)`.

## Current Losses (`src/models/components/loss.py`)
- Hidden-state cosine (all tokens across last L layers):
  - `hidden_states_sim = cos(h_l, h_l_true)` → aggregated via `use_min_hloss` and `hloss_scaler`.
- CLS pooled cosine (output):
  - `output_sim = cos(z, z_true)` where `z` is pooled/visual projection (CLS‑driven).
- Reuse‑rate regularizer:
  - Encourages target reuse `target_reuse_rate` using `reuse_maps` per layer/frame.
- Optional delta‑consistency (residual MSE; default off):
  - `delta_pred = h_l - h_{l-1}`, `delta_true = h_l_true - h_{l-1,true}`
  - Penalize residual: `|| delta_true - delta_pred ||^2` averaged over layers/tokens.
- Final: `loss = sloss_scaler*sloss + sum(hloss_scaler*hloss) + rloss_scaler*rloss + dloss_scaler*dloss`.
- Example weights: see `configs/model/reusevit.yaml` (e.g., `hloss_scaler: [0,1.5]`, `sloss_scaler: 1.0`, `rloss_scaler: 0.35`, `dloss_scaler: 0.0`).

## Observations
- CLS remains stable (high pooled cosine); patch/hidden tokens drift despite hidden‑state cosine.

## Hypotheses (why patches drift)
- Strong CLS supervision via pooled cosine vs diffused per‑token hidden supervision; `hloss` may be underweighted.
- `skip_cls=True` excludes CLS from reuse; non‑CLS are frequently reused/restored, compounding error.
- Local similarity limits reference; non‑local dependencies accumulate drift.
- Restoration MLP capacity/weighting insufficient to preserve fine‑grained token transforms.

## Suggested Experiments
- Increase `hloss_scaler` for last/penultimate layers; reduce reuse aggressiveness early.
- Compare similarity backends: `LocalCosineSimilarity` vs `LowRankCosineSimilarity` / headwise.
- Add small token‑wise auxiliary loss (e.g., MSE on selected patch tokens).
- Enable delta‑consistency with `dloss_scaler=0.5` to protect the per‑layer transform.
- Scheduling: enable reuse on fewer layers initially; ramp over epochs.

## Code Pointers
- Decision MLP: `src/models/components/reuse/decision.py` (MLP built from `layer_pattern`).
- CLS handling & gating: `src/models/components/reuse/module.py` (`skip_cls`, re‑attach).
- Loss composition: `src/models/components/loss.py`.
- Layer replacement: `src/models/components/reusevit.py`.
