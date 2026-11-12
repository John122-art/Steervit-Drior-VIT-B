#!/usr/bin/env bash
# Steer-DrivoR / D+ on navtest; one model on one device, original NAVSIM CPU PDM scoring.
#   CHECKPOINT=<steer .ckpt> TEXT_INDEX=<stage2_focus>/test/dataset_index.jsonl TEXT_MODE=correct|empty|shuffle|swap_f0_b0 bash $0
#   CHECKPOINT=<D+ .ckpt> TEXT_MODE=none bash $0
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-run}"
if [[ $# -gt 1 || ! "$MODE" =~ ^(run|--dry-run|--check-inputs|--check-model)$ ]]; then
    echo "Usage: bash $0 [--dry-run|--check-inputs|--check-model]" >&2; exit 2
fi
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a Steer-DrivoR or D+ checkpoint}"
TEXT_INDEX="${TEXT_INDEX:-}"
TEXT_MODE="${TEXT_MODE:-correct}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/drivor_official_navtest_full_cache_20261007/metric_cache}"
ENV_ROOT="${ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"
TEST_LIMIT="${TEST_LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/exp/drivor_steer_navtest_${TEXT_MODE}_1gpu_b${BATCH_SIZE}_$(date -u +%Y%m%dT%H%M%S)}"
[[ "$TEXT_MODE" =~ ^(correct|empty|shuffle|swap_f0_b0|none)$ ]] || { echo "Invalid TEXT_MODE" >&2; exit 2; }
[[ "$TEST_LIMIT" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid TEST_LIMIT" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be positive" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT" NAVSIM_EXP_ROOT="$REPO_ROOT/exp"
export OPENSCENE_DATA_ROOT="$DATASET_ROOT" NUPLAN_MAPS_ROOT="$DATASET_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/weights/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor_vitb_matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO_ROOT"
command=("$ENV_ROOT/bin/python" -B scripts/evaluation/evaluate_drivor_steer.py
    --checkpoint "$CHECKPOINT" --dataset "$DATASET_ROOT" --metric-cache "$METRIC_CACHE_PATH"
    --output "$OUTPUT_DIR" --limit "$TEST_LIMIT" --batch-size "$BATCH_SIZE")
if [[ "$TEXT_MODE" != none ]]; then
    [[ -f "$TEXT_INDEX" ]] || { echo "Set TEXT_INDEX to the navtest dataset_index.jsonl" >&2; exit 2; }
    command+=(--text-index "$TEXT_INDEX" --text-mode "$TEXT_MODE" --shuffle-seed "$SHUFFLE_SEED")
fi
if [[ "$MODE" == --dry-run ]]; then
    printf 'Single device, FP32, batch=%s; TEST_LIMIT=0 means full navtest. CUDA_VISIBLE_DEVICES=%s\n' "$BATCH_SIZE" "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
if [[ "$MODE" != run ]]; then command+=("$MODE"); fi
set +u
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then source /usr/local/PPU_SDK/envsetup.sh; fi
source "$ENV_ROOT/bin/activate"
set -u
exec "${command[@]}"
