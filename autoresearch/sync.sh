#!/usr/bin/env bash
# Run on the dev PC from the research clone (autoresearch/README.md). Puts code and data on
# fplopt-server, which can't reach GitHub itself:
#   code   push `research` (this clone) and `main` (origin/main) to the server's hub repo
#          ~/fpl-optimizer, then check them out detached in ~/fpl-research and ~/fpl-live
#          and `uv sync` both;
#   raw    copy the files the live archiver's ~/fpl-optimizer/raw has and ~/fpl-live/raw
#          lacks (raw/ is append-only, so missing paths are the whole difference). The bot
#          and archiver run in ~/fpl-optimizer on the same box (botbox was wiped Oct 17):
#          this only reads it, niced, and never writes there.
#
#   try    push a candidate branch try/<id> and check it out in ~/fpl-research (`code` puts
#          ~/fpl-research back on `research`).
#
#   bash autoresearch/sync.sh code|raw|all|try try/<id>
set -euo pipefail
what=${1:-all}
server=fplopt-server
uv='~/.local/bin/uv'

sync_code() {
  git fetch -q origin main
  git push -q -f "$server:fpl-optimizer" research:refs/heads/research \
    origin/main:refs/heads/live-main
  ssh -o BatchMode=yes "$server" "set -e
    for pair in fpl-research:research fpl-live:live-main; do
      dir=\${pair%%:*}; ref=\${pair#*:}
      cd ~/\$dir && git checkout -q --detach \$ref && git log --oneline -1 && $uv sync -q --locked --all-extras
    done"
}

sync_raw() {
  ssh -o BatchMode=yes "$server" 'set -e; export LC_ALL=C; cd ~
    comm -23 <(cd fpl-optimizer && find raw -type f ! -name ".*" | sort) \
      <(cd fpl-live && find raw -type f | sort) > /tmp/fplopt-research-missing
    echo "raw: $(wc -l < /tmp/fplopt-research-missing) new file(s) in the live archive"
    if [ -s /tmp/fplopt-research-missing ]; then
      (cd fpl-optimizer && nice -n 15 tar cf - -T /tmp/fplopt-research-missing) \
        | (cd fpl-live && tar xf -)
    fi'
}

sync_try() {
  branch=$1
  case "$branch" in try/*) ;; *) echo "try needs a try/<id> branch" >&2; exit 2 ;; esac
  git push -q -f "$server:fpl-optimizer" "$branch:refs/heads/$branch"
  ssh -o BatchMode=yes "$server" "set -e
    cd ~/fpl-research && git checkout -q --detach $branch && git log --oneline -1 && $uv sync -q --locked --all-extras"
}

case "$what" in
  code) sync_code ;;
  raw) sync_raw ;;
  all) sync_code; sync_raw ;;
  try) sync_try "${2:?usage: sync.sh try try/<id>}" ;;
  *) echo "usage: sync.sh code|raw|all|try try/<id>" >&2; exit 2 ;;
esac
