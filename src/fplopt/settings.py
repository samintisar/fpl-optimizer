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
            raw_dir=Path(_get(env, "FPLOPT_RAW_DIR") or "raw").expanduser().resolve(),
            odds_api_key=_get(env, "ODDS_API_KEY"),
            telegram_bot_token=_get(env, "TELEGRAM_BOT_TOKEN"),
            telegram_admin_chat_id=_get(env, "TELEGRAM_ADMIN_CHAT_ID"),
        )


def _get(env: Mapping[str, str], key: str) -> str | None:
    value = (env.get(key) or "").strip()
    return value or None
