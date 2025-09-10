#!/usr/bin/env bash
set -euo pipefail

# Systematic ablation runner for ReuseViT variants.
# Now robust to different environments (CPU/GPU, W&B on/off).
#
# Usage:
#   tmux new -s ablate
#   bash scripts/ablate_reuse.sh
#
# Key env overrides (examples):
#   EPOCHS=3 SEEDS="123 231 456" START_PHASE=1 END_PHASE=5 \
#   DATA=default ACCELERATOR=gpu DEVICES=4 DEBUG=0 DRY_RUN=0 \
#   LOGGER=none|wandb|csv WANDB_PROJECT=myproj WANDB_ENTITY=myself
#
# Notes:
# - By default, we auto-detect GPUs and pick ACCELERATOR/DEVICES accordingly.
# - LOGGER defaults to 'none' to avoid unexpected W&B errors. Set LOGGER=wandb
#   and configure WANDB_PROJECT/WANDB_ENTITY to enable W&B logging.
# - Set DEBUG=1 to use configs/debug/default.yaml (disables loggers/callbacks,
#   forces CPU, 1 epoch) regardless of LOGGER/ACCELERATOR/DEVICES.

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  grep -E '^(# |#\t|# -|#   |# Now|# Key|# Notes|# - )' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

EPOCHS=${EPOCHS:-3}
SEEDS_STR=${SEEDS:-"123 231 456"}
START_PHASE=${START_PHASE:-1}
END_PHASE=${END_PHASE:-5}

# Data config group under configs/data/*.yaml
# Default to NextQA per project usage
DATA=${DATA:-nextqa}

# Debug mode (overrides logger/accelerator/devices)
DEBUG=${DEBUG:-0}

# Logger selection
# - none: no explicit logger override (or disabled by debug)
# - csv: use CSV logger
# - wandb: use W&B (requires WANDB_PROJECT/WANDB_ENTITY)
# Default to wandb per team preference
LOGGER=${LOGGER:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-dejavu}
WANDB_ENTITY=${WANDB_ENTITY:-casys-kaist}

# Dry-run: print commands only
DRY_RUN=${DRY_RUN:-0}

# Devices/accelerator: auto-detect if not provided
ACCELERATOR=${ACCELERATOR:-}
DEVICES=${DEVICES:-}
if [[ -z "${ACCELERATOR}" || -z "${DEVICES}" ]]; then
  # Try to detect GPUs with nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_CNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || echo 0)
  else
    GPU_CNT=0
  fi
  if [[ -z "${ACCELERATOR}" ]]; then
    if [[ ${GPU_CNT} -ge 1 ]]; then ACCELERATOR=gpu; else ACCELERATOR=cpu; fi
  fi
  if [[ -z "${DEVICES}" ]]; then
    if [[ ${ACCELERATOR} == gpu ]]; then
      # Cap to 4 by default, but at least 1
      DEVICES=$(( GPU_CNT > 4 ? 4 : (GPU_CNT < 1 ? 1 : GPU_CNT) ))
    else
      DEVICES=1
    fi
  fi
fi

IFS=' ' read -r -a SEEDS <<< "$SEEDS_STR"

# Resolve repo root relative to this script and set PYTHONPATH once
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Build base hydra overrides
BASE_CMD=(python "${REPO_ROOT}/src/train.py" data=${DATA})

# Double the batch size for ablation runs (based on data group)
read -r BASE_BS < <(python - "${DATA}" << 'PY'
from omegaconf import OmegaConf
import sys
data_group = sys.argv[1]
try:
    dcfg = OmegaConf.load(f'configs/data/{data_group}.yaml')
    bs = dcfg.get('batch_size')
    if bs is None:
        bs = OmegaConf.load('configs/data/default.yaml').get('batch_size', 3)
except Exception:
    bs = OmegaConf.load('configs/data/default.yaml').get('batch_size', 3)
print(int(bs))
PY
)
if [[ -n "${BASE_BS}" ]]; then
  QUAD_BS=$(( BASE_BS * 4 ))
  BASE_CMD+=(data.batch_size=${QUAD_BS})
fi

# Base model override
# Prefer base-patch16 for quick iteration unless explicitly overridden via BASE_MODEL
BASE_MODEL=${BASE_MODEL:-openai/clip-vit-base-patch16}
BASE_CMD+=("model.net.base_model_name=${BASE_MODEL}")

# Fix gating/restoration to adafuse/mlp for this sweep
BASE_CMD+=(model/reuse/gating@reuse_module.gating=adafuse)
BASE_CMD+=(model/reuse/restoration@reuse_module.restoration=mlp)

if [[ ${DEBUG} == 1 ]]; then
  # Force debug config (disables loggers/callbacks, cpu, 1 epoch)
  BASE_CMD+=(debug=default)
else
  # Trainer
  BASE_CMD+=(trainer=ddp trainer.accelerator=${ACCELERATOR} trainer.devices=${DEVICES})
  BASE_CMD+=(trainer.max_epochs=${EPOCHS})
  # Ensure compile is disabled (compile adds long warmup)
  BASE_CMD+=(model.compile=false)
  # Optional path overrides for datasets
  if [[ -n "${PATHS_DATA_DIR:-}" ]]; then
    BASE_CMD+=(paths.data_dir="${PATHS_DATA_DIR}")
  fi
  if [[ -n "${PATHS_ROOT_DIR:-}" ]]; then
    BASE_CMD+=(paths.root_dir="${PATHS_ROOT_DIR}")
  fi
  # Logger
  case "${LOGGER}" in
    none)
      # keep defaults from config (can still be disabled via debug)
      ;;
    csv)
      BASE_CMD+=(logger=csv)
      ;;
    wandb)
      BASE_CMD+=(logger=wandb logger.wandb.project=${WANDB_PROJECT} logger.wandb.entity=${WANDB_ENTITY})
      ;;
    *)
      echo "Unknown LOGGER=${LOGGER}. Use: none|csv|wandb" >&2
      exit 1
      ;;
  esac
fi

run_one() {
  local group="$1"; shift
  local extra=("$@")
  for s in "${SEEDS[@]}"; do
    echo -e "\n>>> Running group=${group} seed=${s} overrides=${extra[*]}"
    if [[ ${DRY_RUN} == 1 ]]; then
      echo "CMD: ${BASE_CMD[*]} seed=${s} logger.wandb.group=${group} ${extra[*]}"
    else
      # Only pass wandb.group if using wandb logger
      if [[ ${LOGGER} == "wandb" && ${DEBUG} != 1 ]]; then
        "${BASE_CMD[@]}" seed=${s} logger.wandb.group=${group} "${extra[@]}"
      else
        "${BASE_CMD[@]}" seed=${s} "${extra[@]}"
      fi
    fi
  done
}

phase1() {
  # Phase 1 — Loss Path (token supervision)
  run_one abl-loss-hidden model.loss.hidden_loss_type=cosine model.loss.output_loss_type=cosine model.loss.detach_targets=true
  run_one abl-loss-hidden model.loss.hidden_loss_type=mse    model.loss.output_loss_type=cosine model.loss.detach_targets=true

  run_one abl-loss-cls    model.loss.hidden_loss_type=cosine model.loss.output_loss_type=mse    model.loss.detach_targets=true

  run_one abl-loss-detach model.loss.hidden_loss_type=mse model.loss.output_loss_type=cosine model.loss.detach_targets=false

  run_one abl-loss-min    model.loss.use_min_hloss=true  model.loss.use_min_sloss=true
  run_one abl-loss-min    model.loss.use_min_hloss=false model.loss.use_min_sloss=true

  run_one abl-loss-delta  model.loss.dloss_scaler=0.0
  run_one abl-loss-delta  model.loss.dloss_scaler=0.1
}

phase2() {
  # Phase 2 — Decision Features (local context)
  run_one abl-ctx-mode reuse_module.decision.local_ctx_mode=none
  run_one abl-ctx-mode reuse_module.decision.local_ctx_mode=conv
  run_one abl-ctx-mode reuse_module.decision.local_ctx_mode=neighbors

  # Conv detail sweep
  run_one abl-ctx-conv reuse_module.decision.local_ctx_mode=conv reuse_module.decision.local_ctx_rank=16 reuse_module.decision.local_ctx_kernel=3
  run_one abl-ctx-conv reuse_module.decision.local_ctx_mode=conv reuse_module.decision.local_ctx_rank=16 reuse_module.decision.local_ctx_kernel=5
  run_one abl-ctx-conv reuse_module.decision.local_ctx_mode=conv reuse_module.decision.local_ctx_rank=32 reuse_module.decision.local_ctx_kernel=3
  run_one abl-ctx-conv reuse_module.decision.local_ctx_mode=conv reuse_module.decision.local_ctx_rank=32 reuse_module.decision.local_ctx_kernel=5

  # Neighbors single baseline (avoid explosion)
  run_one abl-ctx-neigh reuse_module.decision.local_ctx_mode=neighbors reuse_module.decision.local_ctx_rank=32 reuse_module.decision.local_ctx_kernel=3

  # Decision module variant (exclude baseline topk/threshold)
  run_one abl-decision model/reuse/decision@reuse_module.decision=mlp_uncertainty
}

phase3() {
  # Phase 3 — Similarity Module (no deformable offsets)
  run_one abl-sim model/reuse/similarity@reuse_module.similarity=cosine
  run_one abl-sim model/reuse/similarity@reuse_module.similarity=local_cosine
  run_one abl-sim model/reuse/similarity@reuse_module.similarity=local_l2
  run_one abl-sim model/reuse/similarity@reuse_module.similarity=local_sad
  run_one abl-sim model/reuse/similarity@reuse_module.similarity=sad

  # LowRankCosine and HeadSpaceCosine with small dims
  run_one abl-sim-lowrank model/reuse/similarity@reuse_module.similarity=low_rank_cosine model.net.reuse_modules.0.similarity.rank=32
  run_one abl-sim-head    model/reuse/similarity@reuse_module.similarity=headspace_cosine model.net.reuse_modules.0.similarity.proj_dim=64 model.net.reuse_modules.0.similarity.num_heads=2
}

phase4() {
  # Phase 4 — Architecture/dim ablations (keep gating=adafuse, restoration=mlp)
  # Decision MLP dims
  run_one abl-dec-dim reuse_module.decision.inner_dim=96
  run_one abl-dec-dim reuse_module.decision.inner_dim=160
  run_one abl-dec-dim reuse_module.decision.inner_dim=256
  # Decision layer patterns
  run_one abl-dec-lpat reuse_module.decision.layer_pattern=lbrlbrl
  run_one abl-dec-lpat reuse_module.decision.layer_pattern=ldrl
  # Restoration MLP width
  run_one abl-rest-dim reuse_module.restoration.inner_dim=32
  run_one abl-rest-dim reuse_module.restoration.inner_dim=64
  run_one abl-rest-dim reuse_module.restoration.inner_dim=128

  # Importance weighting ablations
  run_one abl-imp model/reuse/importance@reuse_module.importance=none
  run_one abl-imp model/reuse/importance@reuse_module.importance=mean
}

phase5() {
  # Phase 5 — Reuse-Rate Targets
  run_one abl-rrate model.loss.max_reuse_per_layer=0.9 model.loss.rloss_duplicate_final_frame=true
  run_one abl-rrate model.loss.max_reuse_per_layer=1.0 model.loss.rloss_duplicate_final_frame=false
}

combine_best() {
  # Example combined run — edit after inspecting W&B results
  run_one abl-combined \
    model.loss.hidden_loss_type=mse model.loss.output_loss_type=cosine model.loss.detach_targets=true model.loss.dloss_scaler=0.1 \
    reuse_module.decision.local_ctx_mode=conv reuse_module.decision.local_ctx_rank=32 reuse_module.decision.local_ctx_kernel=3 \
    model/reuse/similarity@reuse_module.similarity=local_cosine \
    model/reuse/gating@reuse_module.gating=adafuse \
    model/reuse/restoration@reuse_module.restoration=mlp
}

main() {
  for p in $(seq ${START_PHASE} ${END_PHASE}); do
    echo "==== Phase ${p} ===="
    phase"${p}" || true
  done
}

main "$@"
