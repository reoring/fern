#!/usr/bin/env bash
# Full teacher labeling run (resumable; re-run to continue). Teacher must be up on :8080.
set -euo pipefail
cd "$(dirname "$0")/.."
C=${CONCURRENCY:-32}
uv run fern label data/hf.jsonl   --source hf        --concurrency "$C"
uv run fern label data/syn.jsonl  --source synthetic --concurrency "$C" --n "${SYN_N:-5000}"
uv run fern label data/eval_mmlu_pro.jsonl --source eval --concurrency "$C"
