#!/usr/bin/env bash
# DLC: eight accelerators; batch size is per device, without gradient accumulation.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REFERENCE_ROOT="${REFERENCE_ROOT:-/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy}"
ENV_ROOT="${ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_drivoR_Nav1_vitb_8gpu_b${BATCH_SIZE}_25epochs}"

# Reuse the original NAVSIM dataset layout and metric cache from the reference
# launcher; keep this model's training outputs in its own experiment directory.
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$REFERENCE_ROOT/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
DRIVOR_TRAIN_METRIC_CACHE_PATH="${DRIVOR_TRAIN_METRIC_CACHE_PATH:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer (samples per device)." >&2
    exit 2
fi
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
    DRY_RUN=true
    shift
fi

for input_dir in "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
                 "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" "$NUPLAN_MAPS_ROOT"; do
    if [[ ! -d "$input_dir" ]]; then
        echo "Missing NAVSIM input directory: $input_dir" >&2
        exit 1
    fi
done

# The original agent reads NAVSIM_EXP_ROOT/train_metric_cache directly. Link the
# existing cache into that location without changing the agent or copying data.
metric_cache_link="$NAVSIM_EXP_ROOT/train_metric_cache"
if [[ ! -e "$metric_cache_link" && ! -L "$metric_cache_link" ]]; then
    if [[ ! -d "$DRIVOR_TRAIN_METRIC_CACHE_PATH/metadata" ]]; then
        echo "Missing training metric-cache metadata: $DRIVOR_TRAIN_METRIC_CACHE_PATH/metadata" >&2
        exit 1
    fi
    mkdir -p "$NAVSIM_EXP_ROOT"
    ln -s "$DRIVOR_TRAIN_METRIC_CACHE_PATH" "$metric_cache_link"
fi
if [[ ! -d "$metric_cache_link/metadata" ]]; then
    echo "Invalid training metric cache: $metric_cache_link" >&2
    exit 1
fi

# Check mounted inputs without importing the training entrypoint or using GPUs.
if [[ "${1:-}" == --check-inputs ]]; then
    printf 'Training logs: %s\nSensor blobs: %s\nMaps: %s\nTraining metric cache: %s\nExperiment root: %s\n' \
        "$(readlink -f "$OPENSCENE_DATA_ROOT/navsim_logs/trainval")" \
        "$(readlink -f "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval")" \
        "$(readlink -f "$NUPLAN_MAPS_ROOT")" \
        "$(readlink -f "$metric_cache_link")" "$NAVSIM_EXP_ROOT"
    exit 0
fi

# The optional SDK setup matches the existing DLC/PPU environment. NVIDIA-only
# images without this SDK use their existing CUDA runtime.
set +u
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
    source /usr/local/PPU_SDK/envsetup.sh
fi
source "$ENV_ROOT/bin/activate"
set -u

export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export NAVSIM_EXP_ROOT OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/weights/hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor_vitb_matplotlib}"
export HF_HUB_OFFLINE=1
cd "$REPO_ROOT"

printf 'Devices=8, batch/device=%s, gradient accumulation=1, global batch=%s\n' \
    "$BATCH_SIZE" "$((8 * BATCH_SIZE))"
printf 'AdamW peak LR=0.0002 * sqrt(global batch / 64); batch/device=16 gives peak LR=0.0002828427125.\n'

# Lightning's DDP strategy launches the eight processes. Do not wrap this
# entrypoint in an additional eight-process torchrun command.
command=(python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
    agent=drivoR \
    "experiment_name=$EXPERIMENT_NAME" \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
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
    seed=2 \
    "$@")
if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
