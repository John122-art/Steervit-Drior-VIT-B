#!/usr/bin/env bash
# Steer-DrivoR stage 2 (and the D+ control) on DLC: eight devices, batch per device, no accumulation.
# Same NAVSIM inputs, metric cache and DrivoR overrides as run_drivor_vitb_8gpu_dlc.sh (checkpoint D).
#   STAGE2_BUNDLE=<stage2_focus output> STAGE1_CHECKPOINT=<stage-1 best .ckpt> bash $0
#   STEER_ENABLED=false bash $0          # D+ : same init/lr/steps, no language, no grounding
#   ... bash $0 --dry-run
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REFERENCE_ROOT="${REFERENCE_ROOT:-/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy}"
ENV_ROOT="${ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
STEER_ENABLED="${STEER_ENABLED:-true}"
DRIVOR_CHECKPOINT="${DRIVOR_CHECKPOINT:-$REPO_ROOT/exp/ke/training_drivoR_Nav1_vitb_8gpu_b16_25epochs/11.07_03.36/lightning_logs/version_0/checkpoints/best-epoch=23-step=19344.ckpt}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
STEERVIT_CHECKPOINT="${STEERVIT_CHECKPOINT:-$REPO_ROOT/SteerViT_checkpoint/steervit_dinov2_base.pth}"
ROBERTA_PATH="${ROBERTA_PATH:-/mnt/zhaozc_workspace/project/LQH/RoBERTa}"
STAGE2_BUNDLE="${STAGE2_BUNDLE:-}"
D_LR_SCALE="${D_LR_SCALE:-0.1}"
STEER_LR_SCALE="${STEER_LR_SCALE:-0.01}"
if [[ "$STEER_ENABLED" == true ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-steer_stage2_vitb_8gpu_b${BATCH_SIZE}_${MAX_EPOCHS}ep}"
else
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-dplus_vitb_8gpu_b${BATCH_SIZE}_${MAX_EPOCHS}ep}"
fi

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$REFERENCE_ROOT/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
DRIVOR_TRAIN_METRIC_CACHE_PATH="${DRIVOR_TRAIN_METRIC_CACHE_PATH:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be a positive integer" >&2; exit 2; }
[[ "$STEER_ENABLED" =~ ^(true|false)$ ]] || { echo "STEER_ENABLED must be true or false" >&2; exit 2; }
for v in "$DRIVOR_CHECKPOINT" "$STAGE1_CHECKPOINT" "$STEERVIT_CHECKPOINT" "$ROBERTA_PATH" "$STAGE2_BUNDLE"; do
    [[ "$v" != *"'"* ]] || { echo "Paths must not contain single quotes: $v" >&2; exit 2; }
done
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi

for input_dir in "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" "$NUPLAN_MAPS_ROOT"; do
    [[ -d "$input_dir" ]] || { echo "Missing NAVSIM input directory: $input_dir" >&2; exit 1; }
done
[[ -f "$DRIVOR_CHECKPOINT" ]] || { echo "Missing checkpoint D: $DRIVOR_CHECKPOINT" >&2; exit 1; }
(( $(stat -c %s "$DRIVOR_CHECKPOINT") > 100000000 )) || { echo "checkpoint D looks like an LFS pointer; run git lfs pull" >&2; exit 1; }
steer_args=("agent.config.steer.enabled=$STEER_ENABLED")
if [[ "$STEER_ENABLED" == true ]]; then
    [[ -n "$STAGE2_BUNDLE" ]] || { echo "Set STAGE2_BUNDLE to the stage2_focus output directory" >&2; exit 1; }
    for f in "$STAGE2_BUNDLE/train/dataset_index.jsonl" "$STAGE2_BUNDLE/val/dataset_index.jsonl" \
             "$STAGE2_BUNDLE/train/COMPLETE.json" "$STAGE2_BUNDLE/val/COMPLETE.json"; do
        [[ -f "$f" ]] || { echo "Missing stage-2 data file: $f" >&2; exit 1; }
    done
    [[ -d "$ROBERTA_PATH" ]] || { echo "Missing RoBERTa directory: $ROBERTA_PATH" >&2; exit 1; }
    if [[ -n "$STAGE1_CHECKPOINT" ]]; then
        [[ -f "$STAGE1_CHECKPOINT" ]] || { echo "Missing stage-1 checkpoint: $STAGE1_CHECKPOINT" >&2; exit 1; }
    else
        [[ -f "$STEERVIT_CHECKPOINT" ]] || { echo "Missing SteerViT checkpoint: $STEERVIT_CHECKPOINT" >&2; exit 1; }
        echo "WARNING: STAGE1_CHECKPOINT empty; steer modules start from the official SteerViT weights" >&2
    fi
    steer_args+=("agent.config.steer.text_encoder='$ROBERTA_PATH'"
        "agent.init_args.stage1_checkpoint='$STAGE1_CHECKPOINT'"
        "agent.init_args.steervit_checkpoint='$STEERVIT_CHECKPOINT'"
        "+steer_data.train_index='$STAGE2_BUNDLE/train/dataset_index.jsonl'"
        "+steer_data.val_index='$STAGE2_BUNDLE/val/dataset_index.jsonl'")
fi

metric_cache_link="$NAVSIM_EXP_ROOT/train_metric_cache"
if [[ ! -e "$metric_cache_link" && ! -L "$metric_cache_link" ]]; then
    [[ -d "$DRIVOR_TRAIN_METRIC_CACHE_PATH/metadata" ]] || { echo "Missing training metric-cache metadata" >&2; exit 1; }
    mkdir -p "$NAVSIM_EXP_ROOT"
    ln -s "$DRIVOR_TRAIN_METRIC_CACHE_PATH" "$metric_cache_link"
fi
[[ -d "$metric_cache_link/metadata" ]] || { echo "Invalid training metric cache: $metric_cache_link" >&2; exit 1; }

set +u
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then source /usr/local/PPU_SDK/envsetup.sh; fi
source "$ENV_ROOT/bin/activate"
set -u
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export NAVSIM_EXP_ROOT OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/weights/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor_steer_matplotlib}"
cd "$REPO_ROOT"

printf 'Devices=8, batch/device=%s, global batch=%s, epochs=%s, steer=%s\n' \
    "$BATCH_SIZE" "$((8 * BATCH_SIZE))" "$MAX_EPOCHS" "$STEER_ENABLED"
printf 'Peak LR: DrivoR params 0.0002*sqrt(gbs/64)*%s, steer modules *%s\n' "$D_LR_SCALE" "$STEER_LR_SCALE"

# Paths are single-quoted for Hydra: Lightning checkpoint names contain '='.
# Overrides below are the ones of run_drivor_vitb_8gpu_dlc.sh (checkpoint D), plus the steer ones.
command=(python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_steer.py" \
    agent=drivoR_steer \
    "experiment_name=$EXPERIMENT_NAME" \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    "trainer.params.max_epochs=$MAX_EPOCHS" \
    +trainer.params.devices=8 \
    trainer.params.num_nodes=1 \
    trainer.params.strategy=ddp \
    trainer.params.accumulate_grad_batches=1 \
    dataloader.params.prefetch_factor=1 \
    "dataloader.params.batch_size=$BATCH_SIZE" \
    "agent.batch_size=$BATCH_SIZE" \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.lr_args.base_batch_size=64 \
    "agent.lr_args.d_lr_scale=$D_LR_SCALE" \
    "agent.lr_args.steer_lr_scale=$STEER_LR_SCALE" \
    agent.num_gpus=8 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    "agent.init_args.drivor_checkpoint='$DRIVOR_CHECKPOINT'" \
    "${steer_args[@]}" \
    seed=2 \
    "$@")
if [[ "$DRY_RUN" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
