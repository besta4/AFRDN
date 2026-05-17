"""Shared runtime configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def load_environment() -> None:
    """
    Load simple KEY=VALUE pairs from .env files without adding a dependency.

    Existing process environment variables win. This lets EC2/systemd/PM2
    inject secrets while local development can keep a gitignored root .env.
    """
    for env_path in (PROJECT_ROOT / ".env", BACKEND_DIR / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def backend_path_from_env(var_name: str, default_name: str) -> Path:
    """Resolve a path env var relative to backend/ when it is not absolute."""
    raw_value = os.environ.get(var_name)
    path = Path(raw_value) if raw_value else BACKEND_DIR / default_name
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return path.resolve()
