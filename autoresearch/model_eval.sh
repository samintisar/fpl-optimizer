#!/usr/bin/env bash
# The frozen model harness (autoresearch/program.md): v1's walk-forward xP on develop for the
# code checked out in the current directory (~/fpl-research on fplopt-server, develop-only
# data/), scored by the PINNED harness: this script, dev/calibrate_v1.py and judge.py run
# from ~/fpl-harness (the user updates it), so a candidate can't change how it is scored;
# only its fplopt package (imported from the current checkout's venv) is under test.
# Writes results/research/<sha>-<data hash>/metrics.json; reuses it if already scored.
#
#   cd ~/fpl-research && bash ~/fpl-harness/autoresearch/model_eval.sh
set -euo pipefail
harness=~/fpl-harness
test -f data/DEVELOP_ONLY || { echo "data/ is not the develop-only copy" >&2; exit 2; }
test -z "$(git status --porcelain -- src)" || { echo "uncommitted code" >&2; exit 2; }
sha=$(git rev-parse --short=10 HEAD)
# Keyed by code and data: a rebuilt develop copy (weekly.sh) makes every baseline stale.
data=$(cat data/*.parquet | sha256sum | cut -c1-8)
out="results/research/$sha-$data"
if [ -f "$out/metrics.json" ]; then
  echo "already scored: $out"
  exit 0
fi
mkdir -p "$out"
uv=~/.local/bin/uv
start=$(date +%s)
$uv run python "$harness/dev/calibrate_v1.py" --data-dir data walk --seasons 2016-2022 \
  --jobs 7 --out "$out/walk" --ban-residual > "$out/walk.log" 2>&1
$uv run python "$harness/dev/calibrate_v1.py" fit --walk "$out/walk" --parts p_start:1+ \
  > "$out/fit.log" 2>&1
$uv run python "$harness/autoresearch/judge.py" summarize \
  "$out/walk/calibration-2016-0.json" "$sha" "$(( $(date +%s) - start ))" > "$out/metrics.json"
rm -f "$out"/walk/*.parquet  # ~100 MB per commit; metrics.json and the logs stay
echo "scored: $out"
cat "$out/metrics.json"
