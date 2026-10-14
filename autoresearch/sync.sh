#!/usr/bin/env bash
# Run on the dev PC from the research clone (autoresearch/README.md). Puts code and data on
# fplopt-server, which can't reach GitHub or botbox itself:
#   code   push `research` (this clone) and `main` (origin/main) to the server's hub repo
#          ~/fpl-optimizer, then check them out detached in ~/fpl-research and ~/fpl-live
#          and `uv sync` both;
#   raw    `raw` only: copy the files botbox's archive has and ~/fpl-live/raw lacks (raw/
#          is append-only, so missing paths are the whole difference). Read-only on botbox.
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
  tmp=$(mktemp -d)
  ssh -o BatchMode=yes botbox 'cd ~/fpl-optimizer && find raw -type f ! -name ".*" | LC_ALL=C sort' > "$tmp/botbox"
  ssh -o BatchMode=yes "$server" 'cd ~/fpl-live && find raw -type f | LC_ALL=C sort' > "$tmp/server"
  LC_ALL=C comm -23 "$tmp/botbox" "$tmp/server" > "$tmp/missing"
  n=$(wc -l < "$tmp/missing")
  echo "raw: $n new file(s) on botbox"
  if [ "$n" -gt 0 ]; then
    ssh -o BatchMode=yes botbox 'cd ~/fpl-optimizer && tar cf - -T -' < "$tmp/missing" \
      | ssh -o BatchMode=yes "$server" 'cd ~/fpl-live && tar xf -'
  fi
  rm -rf "$tmp"
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
