"""Domain routing and durable adapter discovery for the learner bridge."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Domain = Literal["myntra", "finance", "restaurant", "unknown"]


@dataclass(frozen=True)
class RouteDecision:
    domain: Domain
    adapter_path: Path
    confidence: float
    fallback: bool
    reason: str


_KEYWORDS: dict[str, set[str]] = {
    "myntra": {"myntra", "fashion", "brand", "product", "rating", "discount", "mrp", "size"},
    "finance": {"finance", "income", "expense", "spending", "transaction", "amount", "category"},
    "restaurant": {"restaurant", "hotel", "cuisine", "city", "rating", "cost", "dining", "food"},
}


def _default_root() -> Path:
    return Path(os.getenv("LEARNER_ADAPTER_ROOT", Path(__file__).resolve().parents[3] / "models" / "adapters"))


class DomainRouter:
    """Select one adapter from request text plus the active dataset schema."""

    def __init__(self, adapter_root: str | Path | None = None, min_confidence: float | None = None):
        self.adapter_root = Path(adapter_root or _default_root())
        self.min_confidence = float(os.getenv("LEARNER_ROUTER_MIN_CONFIDENCE", min_confidence if min_confidence is not None else 0.72))

    def route(self, text: str | None, fields: list[str] | None, intent: str | None = None) -> RouteDecision:
        corpus = " ".join([text or "", intent or "", *(fields or [])]).lower()
        scores = {domain: sum(1 for keyword in words if keyword in corpus) for domain, words in _KEYWORDS.items()}
        winner = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = min(0.99, scores[winner] / max(2, total)) if total else 0.0
        schema = set(field.lower() for field in (fields or []))
        if {"brand_name", "rating_count", "marked_price"} <= schema or "myntra" in corpus:
            winner, confidence = "myntra", max(confidence, 0.96)
        elif {"transaction type", "sub category", "amount"} <= schema or "finance" in corpus:
            winner, confidence = "finance", max(confidence, 0.96)
        elif {"restaurant name", "city", "rating"} <= schema or "restaurant" in corpus:
            winner, confidence = "restaurant", max(confidence, 0.96)
        path = self.adapter_root / winner
        available = winner != "unknown" and (path / "adapter_config.json").exists()
        fallback = confidence < self.min_confidence or not available
        reason = "adapter_available" if not fallback else ("low_confidence" if confidence < self.min_confidence else "adapter_unavailable")
        return RouteDecision(winner, path, round(confidence, 4), fallback, reason)
