#!/usr/bin/env bash
# Steer-DrivoR stage 1 on DLC (8 devices): frozen checkpoint D, train SteerViT CA/connector/seg head
# with the GT-projection grounding loss on per-camera stage-1 texts. Environment setup follows
# run_drivor_vitb_8gpu_dlc.sh.
#   bash scripts/training/run_steer_stage1_8gpu_dlc.sh             # prepare (once) + train
#   bash scripts/training/run_steer_stage1_8gpu_dlc.sh --dry-run   # print the commands only
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"
DRIVOR_CHECKPOINT="${DRIVOR_CHECKPOINT:-$REPO_ROOT/exp/ke/training_drivoR_Nav1_vitb_8gpu_b16_25epochs/11.07_03.36/lightning_logs/version_0/checkpoints/best-epoch=23-step=19344.ckpt}"
STEERVIT_CHECKPOINT="${STEERVIT_CHECKPOINT:-$REPO_ROOT/SteerViT_checkpoint/steervit_dinov2_base.pth}"
ROBERTA_PATH="${ROBERTA_PATH:-/mnt/zhaozc_workspace/project/LQH/RoBERTa}"
STAGE1_BUNDLE="${STAGE1_BUNDLE:-/mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage1_navtrain_8gpu_full}"
GROUNDING_OVERRIDE="${GROUNDING_OVERRIDE:-}"     # optional dir of corrected <token>.npz labels
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/exp/steer_stage1}"
SAMPLES="${SAMPLES:-$OUTPUT_DIR/stage1_samples.jsonl}"
DEVICES="${DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"                   # camera rows per device
MAX_EPOCHS="${MAX_EPOCHS:-2}"
LR="${LR:-1e-4}"

DRY_RUN=false
[[ "${1:-}" == --dry-run ]] && DRY_RUN=true

for f in "$DRIVOR_CHECKPOINT" "$STEERVIT_CHECKPOINT" "$STAGE1_BUNDLE/training_index.jsonl" "$STAGE1_BUNDLE/scene_text.json"; do
    [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 1; }
done
[[ -d "$ROBERTA_PATH" ]] || { echo "Missing RoBERTa directory: $ROBERTA_PATH" >&2; exit 1; }
# Git LFS pointer files are ~130 bytes; the real checkpoint is ~576 MB.
(( $(stat -c %s "$DRIVOR_CHECKPOINT") > 100000000 )) || { echo "checkpoint D looks like an LFS pointer; run git lfs pull" >&2; exit 1; }

set +u
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then source /usr/local/PPU_SDK/envsetup.sh; fi
source "$ENV_ROOT/bin/activate"
set -u
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/weights/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor_steer_matplotlib}"
cd "$REPO_ROOT"

prepare=(python scripts/training/train_steer_stage1.py prepare --bundle "$STAGE1_BUNDLE" --output "$SAMPLES")
[[ -n "$GROUNDING_OVERRIDE" ]] && prepare+=(--grounding-override "$GROUNDING_OVERRIDE")
train=(python scripts/training/train_steer_stage1.py train --samples "$SAMPLES" --output "$OUTPUT_DIR"
    --drivor-checkpoint "$DRIVOR_CHECKPOINT" --steervit-checkpoint "$STEERVIT_CHECKPOINT"
    --roberta "$ROBERTA_PATH" --devices "$DEVICES" --batch-size "$BATCH_SIZE"
    --max-epochs "$MAX_EPOCHS" --lr "$LR")
if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${prepare[@]}"; printf '\n'; printf '%q ' "${train[@]}"; printf '\n'; exit 0
fi
mkdir -p "$OUTPUT_DIR"
if [[ ! -f "$SAMPLES" ]]; then
    "${prepare[@]}"
else
    echo "Reusing sample list $SAMPLES (delete it to rebuild)"
fi
# Lightning's DDP strategy launches the processes; do not wrap in torchrun.
exec "${train[@]}"
