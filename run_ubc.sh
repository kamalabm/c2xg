#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export TMPDIR="$ROOT/.tmp"
export MPLCONFIGDIR="$ROOT/.matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

source "$ROOT/c2xg/.venv/bin/activate"
python "$ROOT/learn_ubc.py"

