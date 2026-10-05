"""Runtime settings from environment variables (the CLI loads .env first)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    raw_dir: Path
    odds_api_key: str | None
    telegram_bot_token: str | None
    telegram_admin_chat_id: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            raw_dir=Path(env.get("FPLOPT_RAW_DIR") or "raw"),
            odds_api_key=env.get("ODDS_API_KEY") or None,
            telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN") or None,
            telegram_admin_chat_id=env.get("TELEGRAM_ADMIN_CHAT_ID") or None,
        )
