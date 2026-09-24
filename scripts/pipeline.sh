#!/usr/bin/env bash
# Self-healing end-to-end run: teacher up → label (hf, syn, eval) with stall detection and
# retries → teacher down → train → eval. Everything is resumable; re-run after a crash.
#   scripts/pipeline.sh [RUN_NAME]        env: QUANT, SYN_N, CONCURRENCY, STALL_MIN, MAX_RETRY
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-v1}
QUANT=${QUANT:-UD-IQ2_XXS}
PORT=8080
STALL_MIN=${STALL_MIN:-20}
MAX_RETRY=${MAX_RETRY:-20}
export CONCURRENCY=${CONCURRENCY:-32} SYN_N=${SYN_N:-5000}
LOG=runs/pipeline.log
mkdir -p runs data
TEACHER_PID=""

log() { printf '%s %s\n' "$(date +%FT%T)" "$*" | tee -a "$LOG"; }

teacher_ok() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; }

teacher_start() {
  if teacher_ok; then log "teacher already healthy"; return; fi
  [ -n "$TEACHER_PID" ] && kill "$TEACHER_PID" 2>/dev/null
  pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
  log "starting teacher $QUANT"
  scripts/teacher_up.sh "$QUANT" "$PORT" >> runs/teacher.log 2>&1 &
  TEACHER_PID=$!
  for _ in $(seq 120); do teacher_ok && { log "teacher ready pid=$TEACHER_PID"; return; }; sleep 5; done
  log "teacher failed to become ready"; return 1
}

teacher_stop() {
  log "stopping teacher"
  [ -n "$TEACHER_PID" ] && kill "$TEACHER_PID" 2>/dev/null
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
  for _ in $(seq 30); do pgrep -f "llama-server.*--port $PORT" >/dev/null || break; sleep 2; done
  TEACHER_PID=""
}

lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# run_stage <out-file> <cmd...>: retries on failure, restarts teacher on stall/unhealthy.
run_stage() {
  local out=$1; shift
  local try=0
  while [ "$try" -lt "$MAX_RETRY" ]; do
    try=$((try + 1))
    teacher_start || continue
    log "stage $out attempt $try: $*"
    "$@" >> "$LOG" 2>&1 &
    local pid=$! last=$(lines "$out") stall=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
      local now; now=$(lines "$out")
      if [ "$now" -gt "$last" ]; then last=$now; stall=0; else stall=$((stall + 1)); fi
      if [ "$stall" -ge "$STALL_MIN" ]; then
        log "stall: $out unchanged for ${STALL_MIN}m (rows=$now); killing stage and restarting teacher"
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        teacher_stop
        continue 2
      fi
      if ! teacher_ok; then
        log "teacher unhealthy during $out; killing stage"
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        teacher_stop
        continue 2
      fi
    done
    wait "$pid" && { log "stage $out done (rows=$(lines "$out"))"; return 0; }
    log "stage $out exited non-zero; retrying in 30s"; sleep 30
  done
  log "stage $out gave up after $MAX_RETRY attempts"; return 1
}

log "=== pipeline start run=$RUN ==="
run_stage data/hf.jsonl            uv run fern label data/hf.jsonl            --source hf        --concurrency "$CONCURRENCY" || exit 1
run_stage data/syn.jsonl           uv run fern label data/syn.jsonl           --source synthetic --concurrency "$CONCURRENCY" --n "$SYN_N" || exit 1
run_stage data/eval_mmlu_pro.jsonl uv run fern label data/eval_mmlu_pro.jsonl --source eval      --concurrency "$CONCURRENCY" || exit 1
teacher_stop

if [ ! -f "runs/$RUN/train_meta.json" ]; then
  log "=== train runs/$RUN ==="
  uv run fern train "runs/$RUN" --data data/hf.jsonl --data data/syn.jsonl >> "$LOG" 2>&1 || { log "train failed"; exit 1; }
fi
log "=== eval runs/$RUN ==="
uv run fern eval "runs/$RUN" --labels data/eval_mmlu_pro.jsonl --out "runs/$RUN/eval.json" >> "$LOG" 2>&1 || { log "eval failed"; exit 1; }
log "=== pipeline complete ==="
