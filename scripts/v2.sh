#!/usr/bin/env bash
# v2: thinking teacher (IQ2_M, --think 512) on knowledge-heavy sources, continued training from v1.
# Resumable; re-run after a crash. Issues 001/002/003.
set -uo pipefail
cd "$(dirname "$0")/.."
export QUANT=${QUANT:-UD-IQ2_M}
THINK=${THINK:-512}
C=${CONCURRENCY:-32}
PORT=8080
LOG=runs/pipeline.log
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

# retry a resumable label command until it exits 0 (max 20), restarting the teacher if unhealthy
label_retry() {
  for try in $(seq 20); do
    teacher_start || continue
    log "label attempt $try: $*"
    uv run fern label "$@" >> "$LOG" 2>&1 && return 0
    log "label failed; retry in 30s"; sleep 30
  done
  return 1
}

log "=== v2 start ==="
label_retry data/hf_knowledge_think.jsonl --source hf --names mmlu,openbookqa,sciq,gsm8k --think "$THINK" \
  --concurrency "$C" --teacher-name "DeepSeek-V4-Flash-$QUANT-think$THINK" || exit 1
label_retry data/eval_mmlu_pro_think512.jsonl --source eval --limit 1500 --think "$THINK" \
  --concurrency "$C" --teacher-name "DeepSeek-V4-Flash-$QUANT-think$THINK" || exit 1

log "stopping teacher"
pkill -f "llama-server.*--port $PORT" 2>/dev/null
for _ in $(seq 30); do pgrep -f "llama-server.*--port $PORT" >/dev/null || break; sleep 2; done

if [ ! -f runs/v2/train_meta.json ]; then
  log "=== train runs/v2 ==="
  uv run fern train runs/v2 --model runs/v1 --epochs 1 --lr 5e-6 \
    --data data/hf.jsonl --data data/syn.jsonl --data data/hf_knowledge_think.jsonl \
    --data-weight data/hf_knowledge_think.jsonl=3 >> "$LOG" 2>&1 || { log "train failed"; exit 1; }
fi
log "=== eval runs/v2 ==="
uv run fern eval runs/v2 --labels data/eval_mmlu_pro_think512.jsonl --out runs/v2/eval.json >> "$LOG" 2>&1 || { log "eval failed"; exit 1; }
log "=== pipeline complete ==="
