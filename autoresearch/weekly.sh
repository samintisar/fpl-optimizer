#!/usr/bin/env bash
# The weekly live monitor (autoresearch/program.md), on fplopt-server, run from the pinned
# harness in ~/fpl-live (main, the code the bot runs; raw/ synced from botbox by sync.sh):
# 1. `fplopt build all` -> data/ (every season, incl. 2026/27 so far);
# 2. v1 and FPL's ep_next on 2026/27 so far (`models eval`: predictions, regrets, metrics per
#    horizon and band, components) -> results/live/<stamp>/;
# 3. the develop-only copy for ~/fpl-research (autoresearch/develop_data.py).
# The experiment-log rows `models eval` appends go to results/live/experiments-live.csv, and
# the tracked results/experiments.csv is restored (this checkout is a mirror of main).
#
#   cd ~/fpl-live && bash ~/fpl-harness/autoresearch/weekly.sh
set -euo pipefail
harness=~/fpl-harness
test -d raw && test -f pyproject.toml || { echo "run it in ~/fpl-live" >&2; exit 2; }
uv=~/.local/bin/uv
stamp=$(date -u +%Y-%m-%dT%H%MZ)
out="results/live/$stamp"
mkdir -p "$out"
$uv run fplopt build all > "$out/build.log" 2>&1
$uv run fplopt models eval --models v1,ep_next --seasons 2026 --jobs 8 --out "$out" \
  --family "autoresearch live monitor" > "$out/eval.log" 2>&1
git diff --no-color -U0 -- results/experiments.csv | grep '^+[^+]' | cut -c2- \
  >> results/live/experiments-live.csv || true
git checkout -q -- results/experiments.csv
$uv run python "$harness/autoresearch/develop_data.py" data ~/fpl-research/data > "$out/develop_data.log"
echo "$out"
