#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash %s <baseline_rollouts.jsonl>\n' "$0" >&2
    exit 2
fi

ROLLOUT_FILE="$1"
if [[ ! -f "$ROLLOUT_FILE" ]]; then
    printf 'Rollout file not found: %s\n' "$ROLLOUT_FILE" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/alfworld_stage1_skills}"
SKILL_GEN_WORKERS="${SKILL_GEN_WORKERS:-8}"
SKILL_MAX_COMPLETION_TOKENS="${SKILL_MAX_COMPLETION_TOKENS:-4096}"
SKILL_TIMEOUT="${SKILL_TIMEOUT:-180}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
SEED="${SEED:-2026}"

ROLLOUT_COUNT="$("$PYTHON_BIN" - "$ROLLOUT_FILE" <<'PY'
import sys
from pathlib import Path

with Path(sys.argv[1]).open("r", encoding="utf-8") as file:
    print(sum(1 for line in file if line.strip()))
PY
)"

"$PYTHON_BIN" -c "import numpy, openai, pandas, pyarrow, yaml" >/dev/null

if [[ -z "${SKILL_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}" ]]; then
    read -r -p 'Skill API base URL: ' SKILL_OPENAI_BASE_URL
fi
SKILL_OPENAI_BASE_URL="${SKILL_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}"

if [[ -z "${SKILL_OPENAI_MODEL:-${OPENAI_MODEL:-}}" ]]; then
    read -r -p 'Skill API model: ' SKILL_OPENAI_MODEL
fi
SKILL_OPENAI_MODEL="${SKILL_OPENAI_MODEL:-${OPENAI_MODEL:-}}"

if [[ -z "${SKILL_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
    read -r -s -p 'Skill API key (input hidden): ' SKILL_OPENAI_API_KEY
    printf '\n'
fi
SKILL_OPENAI_API_KEY="${SKILL_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}"

: "${SKILL_OPENAI_BASE_URL:?Skill API base URL is required}"
: "${SKILL_OPENAI_MODEL:?Skill API model is required}"
: "${SKILL_OPENAI_API_KEY:?Skill API key is required}"
export SKILL_OPENAI_BASE_URL SKILL_OPENAI_MODEL SKILL_OPENAI_API_KEY
trap 'unset SKILL_OPENAI_API_KEY OPENAI_API_KEY' EXIT

printf 'Phase: Skill generation from existing rollouts\n'
printf 'Input rollouts: %s (%s records)\n' "$ROLLOUT_FILE" "$ROLLOUT_COUNT"
printf 'Skill API model: %s\n' "$SKILL_OPENAI_MODEL"
printf 'Skill workers: %s, max completion tokens: %s, timeout: %s\n' "$SKILL_GEN_WORKERS" "$SKILL_MAX_COMPLETION_TOKENS" "$SKILL_TIMEOUT"
printf 'Output directory: %s\n' "$OUTPUT_DIR"
printf 'GPU and local policy model: not required\n'

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sft/alfworld/pipeline.py" \
    --env-file /dev/null \
    --output-dir "$OUTPUT_DIR" \
    --baseline-rollouts "$ROLLOUT_FILE" \
    --skill-gen-workers "$SKILL_GEN_WORKERS" \
    --skill-max-completion-tokens "$SKILL_MAX_COMPLETION_TOKENS" \
    --skill-timeout "$SKILL_TIMEOUT" \
    --sft-val-ratio "$SFT_VAL_RATIO" \
    --seed "$SEED"
