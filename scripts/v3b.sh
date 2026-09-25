#!/usr/bin/env bash
# v3b: one more epoch from runs/v3 (v1->v2 recipe: lr 5e-6), same data mix. No teacher needed.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=runs/pipeline.log
log() { printf '%s %s\n' "$(date +%FT%T)" "$*" | tee -a "$LOG"; }
DATA=(--data data/syn.jsonl --data data/syn_multi.jsonl --data data/xnli.jsonl --data data/wide.jsonl
      --data data/hf_knowledge_think.jsonl --data data/hf_aux_think.jsonl --data data/hf.jsonl)
WEIGHTS=(--data-weight data/hf_knowledge_think.jsonl=3 --data-weight data/hf_aux_think.jsonl=3
         --data-weight data/wide.jsonl=2 --data-weight data/syn_multi.jsonl=2 --data-weight data/xnli.jsonl=2)
if [ ! -f runs/v3b/train_meta.json ]; then
  log "=== train runs/v3b (continued from v3) ==="
  uv run fern train runs/v3b --model runs/v3 --epochs 1 --lr 5e-6 "${DATA[@]}" "${WEIGHTS[@]}" >> "$LOG" 2>&1 || { log "train failed"; exit 1; }
fi
log "=== eval runs/v3b ==="
uv run fern eval runs/v3b --labels data/eval_mmlu_pro_think512.jsonl --labels data/eval_wide.jsonl \
  --labels data/eval_xnli.jsonl --labels data/eval_multi.jsonl --out runs/v3b/eval.json >> "$LOG" 2>&1 || { log "eval failed"; exit 1; }
log "=== v3b complete ==="
