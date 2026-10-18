"""PreToolUse hook for the research clone (autoresearch/README.md): refuses what the loop
must never do (autoresearch/program.md, *Hard rules*). Exit 2 blocks the call and tells
the agent why. Frozen: the loop never edits this file.

Configured in the research clone's .claude/settings.local.json for Bash, PowerShell, Edit,
Write and NotebookEdit; reads the hook's JSON on stdin.
"""

from __future__ import annotations

import json
import re
import sys

ALLOWED_SEASONS = set(range(2016, 2023))  # develop only
CLONE = "documents/github/fpl-autoresearch"  # the research clone, lower case
FROZEN = (
    r"autoresearch/(program\.md|judge\.py|model_eval\.sh|guard\.py|develop_data\.py|sync\.sh"
    r"|weekly\.sh|README\.md)$",
    r"(^|/)\.claude/",
    r"(^|/)src/fplopt/bot/",
    r"(^|/)src/fplopt/backtest/holdout\.py$",
    r"(^|/)src/fplopt/models/calibration_table\.py$",
    # How outcomes are scored: a candidate must not change what it is judged against.
    r"(^|/)src/fplopt/backtest/(scoring|rules|gw_score|xg_points|evaluate)\.py$",
    r"(^|/)dev/calibrate_v1\.py$",
    r"(^|/)deploy/",
    r"(^|/)config/(scoring|chips)/",
    r"(^|/)(raw|data)/",
    r"(^|/)\.env",
)
# The pinned harness and hook (README *Setup*): only these calls may name them.
PINNED = r"fpl-harness|\.fpl-harness"
PINNED_CALLS = (
    r"bash\s+~/fpl-harness/autoresearch/model_eval\.sh",
    r"python\s+~/fpl-harness/autoresearch/judge\.py\s+compare",
    r"bash\s+~/fpl-harness/autoresearch/weekly\.sh",
    r"python\s+~/fpl-harness/autoresearch/status\.py",
)
# A shell command that writes a frozen file: a redirect into it (`>`, `>>`, `2>`; not
# `2>&1`), or a command that writes the files it names (in place, copy/move/delete,
# git restore, PowerShell and Python file writes).
REDIRECT_TARGET = r"\d?>>?\s*(?!&)([^\s|;&<>]+)"
WRITES = (
    r"\b(sed|perl)\s+-\w*i|\btee\b|\b(mv|cp|rm|truncate|install)\b|\bgit\s+(checkout|restore)\s"
    r"|Set-Content|Out-File|Add-Content|Remove-Item|Move-Item|Copy-Item|write_text|open\("
)
COMMANDS = (
    (r"botbox", "botbox is gone (wiped Oct 17); the bot runs in ~/fpl-optimizer"),
    (
        r"(~|/home/fplopt|\$HOME)/fpl-optimizer\b",
        "~/fpl-optimizer is the live bot and archiver: only autoresearch/sync.sh reads it",
    ),
    (r"\bgit\s+push\b", "pushes go through autoresearch/sync.sh only"),
    (
        r"\bgit\s+(checkout|switch|merge|rebase|reset)\b[^|;&]*\b(main|origin/main)\b",
        "the loop works on `research` only, never main",
    ),
    (
        r"\bgh\s+(?!issue\s+(view|list)\b)",
        "only `gh issue view|list`: no PRs, issues or API writes",
    ),
    (r"\bholdout\b\s+(run|check)|phase6", "the holdout is spent; never run it"),
    (r"(^|[\s/])\.env\b", "secrets are the user's"),
    (r"\b(sudo|systemctl|crontab)\b", "no system changes"),
    (r"\brm\s+-\w*r\w*\s+[^|;&]*\b(raw|data)\b", "raw/ and data/ are not the loop's to delete"),
    (r"--data-dir\s+\S*(fpl-live|fpl-optimizer/data)", "experiments read the develop-only data/"),
    (
        r"FPLOPT_DATA_DIR=\S*(fpl-live|fpl-optimizer/data)",
        "experiments read the develop-only data/",
    ),
)
# Commands that read seasons: they must name develop seasons explicitly (several default to
# ranges that include validate).
SEASON_COMMANDS = r"\b(backtest\s+(run|compare)|models\s+eval|calibrate_v1\.py|minutes_eval\.py)\b"


def seasons(text: str) -> set[int]:
    out: set[int] = set()
    for part in re.split(r"[,\s]+", text.strip("'\"")):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        try:
            a = int(lo)
            b = int(hi) if hi else a
        except ValueError:
            out.add(-1)
            continue
        if b < 100:  # "2021-22" means the 2021/22 season
            b = a
        out.update(range(a, b + 1))
    return out


def check_command(command: str) -> str | None:
    flat = command.replace("\\", "/")
    for pattern, why in COMMANDS:
        if re.search(pattern, flat):
            return why
    rest = flat
    for call in PINNED_CALLS:
        rest = re.sub(call, " ", rest)
    if re.search(PINNED, rest):
        return (
            "the pinned harness is the user's: only its model_eval, judge compare, weekly, status"
        )
    targets = re.findall(REDIRECT_TARGET, flat)
    if re.search(WRITES, flat):
        targets += re.findall(r"[\w./~-]+", flat)
    for path in targets:
        if check_path(path):
            return f"{path} is frozen for the loop (autoresearch/program.md)"
    if re.search(SEASON_COMMANDS, flat):
        found = re.findall(r"--seasons[=\s]+(\S+)", flat)
        if not found:
            return "name the seasons: --seasons within 2016-2022 (develop only)"
        for text in found:
            bad = seasons(text) - ALLOWED_SEASONS
            if bad:
                return (
                    f"seasons {sorted(bad)} are not develop (2016-2022): validate, holdout "
                    "and live are off limits"
                )
    return None


def check_path(path: str) -> str | None:
    flat = path.replace("\\", "/")
    for pattern in FROZEN:
        if re.search(pattern, flat):
            return f"{path} is frozen for the loop (autoresearch/program.md)"
    return None


def in_clone(path: str) -> bool:
    """`path` is inside the research clone (case- and slash-insensitive)."""
    flat = path.replace("\\", "/").lower()
    return CLONE in flat


def main() -> None:
    event = json.load(sys.stdin)
    tool = event.get("tool_name", "")
    data = event.get("tool_input") or {}
    cwd = str(event.get("cwd") or "")
    if tool in ("Bash", "PowerShell"):
        command = str(data.get("command", ""))
        # The loop's sessions start in the main checkout (scheduled tasks can't pick a
        # folder), so the hook is installed there too and acts only on the loop's work:
        # commands run in, or naming, the research clone.
        active = in_clone(cwd) or in_clone(command)
        why = check_command(command) if active else None
    elif tool in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        path = str(data.get("file_path") or data.get("notebook_path") or "")
        why = check_path(path) if in_clone(path) else None
    else:
        why = None
    if why:
        print(f"autoresearch guard: {why}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
