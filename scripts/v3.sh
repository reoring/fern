#!/usr/bin/env bash
# v3 (issue 009): 007 thinking labels + 008-a 26-way choice + 008-g non-English, one labeling pass,
# one training run from the base model. Resumable; re-run after a crash.
#   THINK_COMMONSENSE=1 scripts/v3.sh   # after Phase 0 gate passes: relabel hf choice sources with thinking
set -uo pipefail
cd "$(dirname "$0")/.."
export QUANT=${QUANT:-UD-IQ2_M}
THINK=${THINK:-512}
C=${CONCURRENCY:-32}
PORT=8080
LOG=runs/pipeline.log
TNAME="DeepSeek-V4-Flash-$QUANT"
TNAME_THINK="$TNAME-think$THINK"
log() { printf '%s %s\n' "$(date +%FT%T)" "$*" | tee -a "$LOG"; }
teacher_ok() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; }

teacher_start() {
  teacher_ok && return 0
  pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
  log "starting teacher $QUANT"
  scripts/teacher_up.sh "$QUANT" "$PORT" >> runs/teacher.log 2>&1 &
  for _ in $(seq 120); do teacher_ok && return 0; sleep 5; done
  return 1
}

label_retry() {
  for try in $(seq 20); do
    teacher_start || continue
    log "label attempt $try: $*"
    uv run fern label "$@" >> "$LOG" 2>&1 && return 0
    log "label failed; retry in 30s"; sleep 30
  done
  return 1
}

log "=== v3 start (THINK_COMMONSENSE=${THINK_COMMONSENSE:-0}) ==="
# --- non-thinking labels (~6h) ---
label_retry data/syn_multi.jsonl --source synthetic --n 10000 --langs ja,zh,de,es,fr,ko --concurrency "$C" --teacher-name "$TNAME" || exit 1
label_retry data/eval_multi.jsonl --source synthetic --n 600 --langs ja,zh,de,es,fr,ko --seed 1 --concurrency "$C" --teacher-name "$TNAME" || exit 1
label_retry data/xnli.jsonl      --source hf --names xnli --split train --concurrency "$C" --teacher-name "$TNAME" || exit 1
label_retry data/eval_xnli.jsonl --source hf --names xnli --split eval  --concurrency "$C" --teacher-name "$TNAME" || exit 1
label_retry data/wide.jsonl      --source hf --names wide --split train --concurrency "$C" --teacher-name "$TNAME" || exit 1
label_retry data/eval_wide.jsonl --source hf --names wide --split eval  --concurrency "$C" --teacher-name "$TNAME" || exit 1
# --- thinking labels ---
label_retry data/hf_aux_think.jsonl --source hf --names mmlu_aux --think "$THINK" --concurrency "$C" --teacher-name "$TNAME_THINK" || exit 1
DATA=(--data data/syn.jsonl --data data/syn_multi.jsonl --data data/xnli.jsonl --data data/wide.jsonl
      --data data/hf_knowledge_think.jsonl --data data/hf_aux_think.jsonl)
WEIGHTS=(--data-weight data/hf_knowledge_think.jsonl=3 --data-weight data/hf_aux_think.jsonl=3
         --data-weight data/wide.jsonl=2 --data-weight data/syn_multi.jsonl=2 --data-weight data/xnli.jsonl=2)
if [ "${THINK_COMMONSENSE:-0}" = 1 ]; then
  # gate passed: choice-type hf sources get thinking labels; keep the non-choice ones (boolq/yelp/mnli/tweet) as is
  label_retry data/hf_choice_think.jsonl --source hf --names arc,hellaswag,csqa,mmlu_pro_val --think "$THINK" \
    --concurrency "$C" --teacher-name "$TNAME_THINK" || exit 1
  label_retry data/hf_rest.jsonl --source hf --names boolq,yelp,mnli,tweet_emotion --concurrency "$C" --teacher-name "$TNAME" || exit 1
  DATA+=(--data data/hf_choice_think.jsonl --data data/hf_rest.jsonl)
  WEIGHTS+=(--data-weight data/hf_choice_think.jsonl=2)
else
  DATA+=(--data data/hf.jsonl)
fi

log "stopping teacher"
pkill -f "llama-server.*--port $PORT" 2>/dev/null
for _ in $(seq 30); do pgrep -f "llama-server.*--port $PORT" >/dev/null || break; sleep 2; done

if [ ! -f runs/v3/train_meta.json ]; then
  log "=== train runs/v3 (from base) ==="
  uv run fern train runs/v3 --epochs 2 --lr 1e-5 "${DATA[@]}" "${WEIGHTS[@]}" >> "$LOG" 2>&1 || { log "train failed"; exit 1; }
fi
log "=== eval runs/v3 ==="
uv run fern eval runs/v3 --labels data/eval_mmlu_pro_think512.jsonl --labels data/eval_wide.jsonl \
  --labels data/eval_xnli.jsonl --labels data/eval_multi.jsonl --out runs/v3/eval.json >> "$LOG" 2>&1 || { log "eval failed"; exit 1; }
log "=== v3 complete ==="
