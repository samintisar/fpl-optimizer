"""Where the live season stands, from the newest archived bootstrap (autoresearch/program.md,
step 1): the last GW whose data FPL has checked (bonus final), the next deadline and when
the newest bootstrap was taken. Prints JSON.

    uv run python autoresearch/status.py [raw dir]
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


def main() -> None:
    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    newest = max((raw / "fpl" / "bootstrap-static").glob("*.json.gz"))
    events = json.loads(gzip.decompress(newest.read_bytes()))["events"]
    checked = [e["id"] for e in events if e.get("finished") and e.get("data_checked")]
    upcoming = [e for e in events if not e.get("finished")]
    print(
        json.dumps(
            {
                "bootstrap": newest.name,
                "last_checked_gw": max(checked) if checked else None,
                "next_gw": upcoming[0]["id"] if upcoming else None,
                "next_deadline": upcoming[0]["deadline_time"] if upcoming else None,
            }
        )
    )


if __name__ == "__main__":
    main()
