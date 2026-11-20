#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-seed}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/root/autodl-tmp/alfworld_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/alfworld_stage1_rollouts_full}"

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
    printf 'No NVIDIA device found. Switch the instance to GPU mode before rollout.\n' >&2
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
missing = [
    name
    for name in sorted(set(index["weight_map"].values()))
    if not (root / name).is_file()
]
if missing:
    raise SystemExit(f"Model weight files are missing: {missing}")
PY

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ]]; then
    printf 'ALFWorld train split not found: %s\n' "$ALFWORLD_DATA/json_2.1.1/train" >&2
    exit 1
fi

export RUN_MODE=full
export CONDA_ENV MODEL_PATH ALFWORLD_DATA
export POLICY_MODEL="${POLICY_MODEL:-$(basename "$MODEL_PATH")}"
export OUTPUT_DIR
export VLLM_LOG_DIR="${VLLM_LOG_DIR:-$OUTPUT_ROOT/vllm_logs}"
export ENV_FILE="${ENV_FILE:-/dev/null}"
export TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-16}"
export REQUEST_WORKERS="${REQUEST_WORKERS:-128}"
export SKILL_GEN_WORKERS=1
export DATA_PARALLEL_SIZE=1
export TENSOR_PARALLEL_SIZE=1
export MAX_CANDIDATES=0
export STOP_VLLM_AFTER_API=1
export KEEP_VLLM_ALIVE=0
export CHECK_ALFWORLD_DATA=true
export PROGRESS_MONITOR=1
export RESUME="${RESUME:-false}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---gpu-memory-utilization 0.90 --max-num-seqs 256 --max-num-batched-tokens 16384 --disable-log-requests}"

export OPENAI_BASE_URL="http://127.0.0.1:9/v1"
export OPENAI_API_KEY="EMPTY"
export OPENAI_MODEL="rollout-only-no-skill-generation"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="$NO_PROXY"

printf 'Phase: rollout only\n'
printf 'Policy model: %s\n' "$MODEL_PATH"
printf 'Rollout scale: six task types x 30 tasks x 8 rollouts = 1440\n'
printf 'Environment batch: %s tasks, request workers: %s\n' "$TASK_BATCH_SIZE" "$REQUEST_WORKERS"
printf 'vLLM extra args: %s\n' "$VLLM_EXTRA_ARGS"
printf 'Skill API calls: disabled for this phase\n'
printf 'Output directory: %s\n' "$OUTPUT_DIR"

bash "$PROJECT_ROOT/scripts/sft/alfworld/prepare_data.sh"

baseline_file="$OUTPUT_DIR/baseline_rollouts.jsonl"
if [[ ! -f "$baseline_file" ]]; then
    printf 'Rollout file not found after run: %s\n' "$baseline_file" >&2
    exit 1
fi
printf 'ROLLOUT_COMPLETE %s\n' "$baseline_file"
