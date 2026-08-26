"""Configuration for the secure Excel path."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class SecureExcelConfig:
    remote_ai_enabled: bool = os.getenv("SECURE_EXCEL_REMOTE_AI", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    remote_ai_provider: str = os.getenv("SECURE_EXCEL_REMOTE_AI_PROVIDER", "gemini")
    max_preview_rows: int = int(os.getenv("SECURE_EXCEL_MAX_PREVIEW_ROWS", "25"))


CONFIG = SecureExcelConfig()

