#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export TMPDIR="$ROOT/.tmp"
export MPLCONFIGDIR="$ROOT/.matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

source "$ROOT/c2xg/.venv/bin/activate"
python "$ROOT/learn_2_steps_params_live.py" \
  /Users/kamalaboumikhael/g/UBC/dissertation/c2xg/codex-c2xg/data/IN/embeddings-input.txt \
  /Users/kamalaboumikhael/g/UBC/dissertation/c2xg/codex-c2xg/data/IN/grammar-input.txt \
  /Users/kamalaboumikhael/g/UBC/dissertation/c2xg/codex-c2xg/data/OUT/
