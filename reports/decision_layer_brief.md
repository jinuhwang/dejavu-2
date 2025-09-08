# Decision Layer: Architecture and Loss (dejavu-2)

This brief summarizes where the reuse/decision layer sits in the model, how class vs non-class tokens flow, and how the current losses supervise them. Pointers reference concrete files to ensure reproducibility.

## Overview
- Base backbone: CLIP ViT vision encoder with projection.
- Reuse block: per-encoder-layer module that decides per-token reuse vs recompute and merges accordingly.
- Class token handling: decision runs on non-class tokens; class token (CLS) is skipped and re-attached unchanged.

Key code:
- Reuse module: `src/models/components/reuse/module.py` (class `ReuseModule`)
  - Similarity backends: `src/models/components/reuse/similarity.py` (e.g., `LocalCosineSimilarity`)
  - Decision MLP: `src/models/components/reuse/decision.py` (`ReuseMLP`)
  - Gating: `src/models/components/reuse/gating.py` (e.g., `AdafuseGating`)
  - Restoration: `src/models/components/reuse/restoration.py` (e.g., `MLPRestoration`)
- Loss: `src/models/components/loss.py` (`ReuseLoss`)
- Net wiring: `src/models/components/reusevit.py` (`ReuseEncoderLayer`, layer replacement)
- Config examples: `configs/model/reuse/default.yaml`, `configs/model/reuse/decision/mlp.yaml`, `configs/model/reuse/similarity/local_cosine.yaml`.

## Forward sketch (shapes)
- Hidden states per layer: `(B, N, D)` with `N = 1 + P` (CLS + patches).
- In `ReuseModule.forward()`:
  - If `skip_cls=True`, split: `cls` = `[:, :1]`, `tokens` = `[:, 1:]` (skip branch; CLS excluded).
  - Compute similarity between `tokens` and cached tokens → `(B, N-1, K)` (K: cached window) using selected similarity backend.
  - Decision MLP consumes `[importance, similarity, compressed_map, ref_type]` slices.
  - Gating mixes restored vs recomputed states per token (excluding CLS) and re-attaches CLS unchanged.

## Losses (current)
- File: `src/models/components/loss.py` (`ReuseLoss.forward`)
- Inputs: last L hidden states, pooled outputs, and reuse maps.
- Cosine similarity on hidden states (all tokens): `hidden_states_sim = cos(hidden_states, original_hidden_states)`.
- Cosine similarity on pooled outputs (CLS embeddings): `output_sim = cos(output, original_output)`; supervises CLS.
- Reuse-rate regularizer: encourages target reuse under `target_reuse_rate`.
- Final: `loss = sloss_scaler*sloss + sum(hloss_scaler*hloss) + rloss_scaler*rloss`.

Weights (example config):
- See `configs/model/reusevit.yaml` → `loss` block; e.g., `hloss_scaler: [0, 1.5]`, `sloss_scaler: 1.`, `rloss_scaler: 0.35`.

## Observed phenomenon
- Empirically: CLS path stays stable (pooled cosine similarity remains high); other tokens (patch/hidden states) drift despite hidden-state cosine loss.

## Hypotheses
- CLS is directly pooled and strongly supervised by `sloss`; per-token hidden-state supervision diffuses across layers/tokens and may be underweighted.
- `skip_cls` excludes CLS from reuse decisions; non-CLS tokens are more frequently gated/reused/restored, amplifying drift.
- Local similarity restricts reference scope; errors accumulate for non-local dependencies.
- Restoration MLP capacity / loss balancing insufficient to reconstruct fine-grained token states.

## Suggested experiments
- Increase `hloss_scaler` for deeper layers (last/penultimate) to emphasize final hidden integrity.
- Switch similarity: try `LowRankCosineSimilarity` or head-wise variants; compare reuse/loss.
- Add token-wise auxiliary loss (e.g., MSE on selected patch tokens) with small weight.
- Adjust gating hardness/temperature to reduce over-aggressive reuse early.
- Layer-wise scheduling: start with fewer reused layers and ramp up.

## Pointers
- Decision MLP: `src/models/components/reuse/decision.py` (`ReuseMLP.forward_inner`).
- CLS handling: `src/models/components/reuse/module.py` (`skip_cls` branch; re-attach at end of forward).
- Loss composition: `src/models/components/loss.py`.
- Net replacement: `src/models/components/reusevit.py` (encoder layer wrapping).

## Variants tried
- Decision policies: MLP, thresholds/top-K.
- Similarity: local cosine (default), low-rank cosine.
- Hidden-state weighting: multi-layer `hloss_scaler` arrays (emphasize penultimate/last layers).
- Scheduling: reuse on all but first layer; skip last-layer reuse variants via config.
