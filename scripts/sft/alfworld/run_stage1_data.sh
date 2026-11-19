#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    printf 'Usage: bash %s [smoke|full]\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-seed}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-$CONDA_ENV}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs}"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    printf 'Conda activation script not found: %s\n' "$CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda run -n "$CONDA_ENV" python --version >/dev/null 2>&1; then
    printf 'Conda environment is not ready: %s\n' "$CONDA_ENV" >&2
    exit 1
fi
if [[ ! -e /dev/nvidia0 ]]; then
    printf 'No NVIDIA device found. Start the AutoDL instance with a GPU before running Stage 1.\n' >&2
    exit 1
fi

python3 - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index_path = root / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit(f"Model index not found: {index_path}")
index = json.loads(index_path.read_text(encoding="utf-8"))
weight_files = sorted(set(index["weight_map"].values()))
missing = [name for name in weight_files if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Model weight files are missing or incomplete: {missing}")
PY

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ]]; then
    printf 'ALFWorld train split not found: %s\n' "$ALFWORLD_DATA/json_2.1.1/train" >&2
    exit 1
fi

if [[ "$MODE" == "smoke" ]]; then
    TASKS_PER_TYPE="${TASKS_PER_TYPE:-1}"
    ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-1}"
    MAX_CANDIDATES="${MAX_CANDIDATES:-2}"
    DEFAULT_OUTPUT_DIR="$OUTPUT_ROOT/alfworld_stage1_smoke"
else
    TASKS_PER_TYPE="${TASKS_PER_TYPE:-30}"
    ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-8}"
    MAX_CANDIDATES="${MAX_CANDIDATES:-}"
    DEFAULT_OUTPUT_DIR="$OUTPUT_ROOT/alfworld_stage1_full"
fi

export RUN_MODE="$MODE"
export CONDA_ENV VLLM_CONDA_ENV MODEL_PATH ALFWORLD_DATA
export POLICY_MODEL="${POLICY_MODEL:-$(basename "$MODEL_PATH")}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
export VLLM_LOG_DIR="${VLLM_LOG_DIR:-$OUTPUT_ROOT/vllm_logs}"
export ENV_FILE="${ENV_FILE:-/dev/null}"
export TASKS_PER_TYPE ROLLOUTS_PER_TASK MAX_CANDIDATES
export TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-8}"
export REQUEST_WORKERS="${REQUEST_WORKERS:-16}"
export SKILL_GEN_WORKERS="${SKILL_GEN_WORKERS:-4}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export STOP_VLLM_AFTER_API="${STOP_VLLM_AFTER_API:-1}"
export KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"
export CHECK_ALFWORLD_DATA=true
export PROGRESS_MONITOR=1

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="$NO_PROXY"

printf 'Mode: %s\n' "$MODE"
printf 'Policy model: %s\n' "$MODEL_PATH"
printf 'Rollout scale: six task types x %s tasks x %s rollouts\n' "$TASKS_PER_TYPE" "$ROLLOUTS_PER_TASK"
printf 'Skill candidates: %s\n' "${MAX_CANDIDATES:-all}"
printf 'Output directory: %s\n' "$OUTPUT_DIR"
printf 'External skill API: required after rollout; SFT training: disabled\n'

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
    read -r -p 'Skill API base URL: ' OPENAI_BASE_URL
fi
if [[ -z "${OPENAI_MODEL:-}" ]]; then
    read -r -p 'Skill API model: ' OPENAI_MODEL
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    read -r -s -p 'Skill API key (input hidden): ' OPENAI_API_KEY
    printf '\n'
fi

: "${OPENAI_BASE_URL:?Skill API base URL is required}"
: "${OPENAI_MODEL:?Skill API model is required}"
: "${OPENAI_API_KEY:?Skill API key is required}"
export OPENAI_BASE_URL OPENAI_MODEL OPENAI_API_KEY
trap 'unset OPENAI_API_KEY SKILL_API_KEY SKILL_OPENAI_API_KEY' EXIT

bash "$PROJECT_ROOT/scripts/sft/alfworld/prepare_data.sh"
