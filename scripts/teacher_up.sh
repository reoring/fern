#!/usr/bin/env bash
# Start the teacher llama-server fully on GPU.
#   scripts/teacher_up.sh [QUANT] [PORT]     e.g. scripts/teacher_up.sh UD-IQ2_XXS 8080
# Downloads unsloth/DeepSeek-V4-Flash-GGUF/<QUANT> if missing. Fall back to UD-IQ1_M on OOM.
set -euo pipefail
QUANT=${1:-UD-IQ2_XXS}
PORT=${2:-8080}
REPO=unsloth/DeepSeek-V4-Flash-GGUF
LLAMA=${LLAMA_SERVER:-$HOME/.unsloth/llama.cpp/llama-server}

cd "$(dirname "$0")/.."
uv run hf download "$REPO" --include "$QUANT/*" >/dev/null
FIRST=$(uv run python -c "
from huggingface_hub import snapshot_download; import glob
d = snapshot_download('$REPO', allow_patterns=['$QUANT/*'])
print(sorted(glob.glob(f'{d}/$QUANT/*.gguf'))[0])")

exec "$LLAMA" -m "$FIRST" --port "$PORT" -ngl 999 -c 32768 --parallel 32 -b 4096 -ub 2048 -fa on --jinja \
  --no-warmup --cache-reuse 256 "${@:3}"
