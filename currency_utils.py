"""Small, dependency-free currency detection/conversion helper for InsightFlow."""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any

TARGET_CURRENCY_REGISTRY: dict[str, dict[str, Any]] = {
    "INR": {
        "aliases": ("INR", "inr", "rupee", "rupees", "Indian rupee", "Indian rupees", "₹", "rs", "rs."),
        "symbol": "₹",
        "decimals": 2,
    },
    "USD": {
        "aliases": ("USD", "usd", "dollar", "dollars", "US dollar", "US dollars", "$", "us$", "us dollar", "us dollars"),
        "symbol": "$",
        "decimals": 2,
    },
    "EUR": {
        "aliases": ("EUR", "eur", "euro", "euros", "€"),
        "symbol": "€",
        "decimals": 2,
    },
    "GBP": {
        "aliases": ("GBP", "gbp", "pound", "pounds", "British pound", "British pounds", "£"),
        "symbol": "£",
        "decimals": 2,
    },
    "JPY": {
        "aliases": ("JPY", "jpy", "yen", "Japanese yen", "¥"),
        "symbol": "¥",
        "decimals": 0,
    },
    "AED": {
        "aliases": ("AED", "aed", "dirham", "dirhams", "د.إ"),
        "symbol": "د.إ",
        "decimals": 2,
    },
    "SGD": {
        "aliases": ("SGD", "sgd", "Singapore dollar", "Singapore dollars", "S$"),
        "symbol": "S$",
        "decimals": 2,
    },
    "CAD": {
        "aliases": ("CAD", "cad", "Canadian dollar", "Canadian dollars", "C$"),
        "symbol": "C$",
        "decimals": 2,
    },
    "RUB": {
        "aliases": ("RUB", "rub", "ruble", "rubles", "rubel", "rubels", "rouble", "roubles", "₽"),
        "symbol": "₽",
        "decimals": 2,
    },
    "BDT": {
        "aliases": ("BDT", "bdt", "taka", "৳"),
        "symbol": "৳",
        "decimals": 2,
    },
    "AUD": {
        "aliases": ("AUD", "aud", "australian dollar", "australian dollars", "A$"),
        "symbol": "A$",
        "decimals": 2,
    },
    "CNY": {
        "aliases": ("CNY", "cny", "yuan", "renminbi", "元"),
        "symbol": "¥",
        "decimals": 2,
    },
    "HKD": {
        "aliases": ("HKD", "hkd", "hong kong dollar", "hong kong dollars", "HK$"),
        "symbol": "HK$",
        "decimals": 2,
    },
}

CURRENCY_ALIASES = {
    alias.strip().lower(): code
    for code, config in TARGET_CURRENCY_REGISTRY.items()
    for alias in config["aliases"]
}

SYMBOL_BY_CODE = {
    code: str(config["symbol"])
    for code, config in TARGET_CURRENCY_REGISTRY.items()
}

FORMAT_BY_CODE = {
    code: (f"{config['symbol']}#,##0" if int(config["decimals"]) == 0 else f"{config['symbol']}#,##0.00")
    for code, config in TARGET_CURRENCY_REGISTRY.items()
}
_CACHE: dict[str, tuple[float, dict[str, float]]] = {}


def _normalize_currency_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"[\u00A0\u202F\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_currency(value: str | None) -> str | None:
    if not value:
        return None
    s = _normalize_currency_text(value).lower()
    if s in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[s]
    # Prefer explicit codes embedded in natural language.
    for code, config in TARGET_CURRENCY_REGISTRY.items():
        for alias in sorted(config["aliases"], key=lambda x: len(str(x)), reverse=True):
            alias_norm = str(alias).strip().lower()
            if not alias_norm:
                continue
            if re.search(rf"(?<!\w){re.escape(alias_norm)}(?!\w)", s):
                return code
    return None


def extract_target_currency(text: str) -> str | None:
    text = str(text or "")
    # Explicit target language is deliberately required; merely mentioning
    # "currency" must not trigger conversion.
    lowered = _normalize_currency_text(text).lower()
    if not re.search(r"\b(?:convert|change|exchange|transform|express|denominate)\b", lowered):
        return None
    patterns = [
        r"\b(?:in|to|into|as)\s+(.+?)(?=$|\b(?:and|then|for|with|from|on|by|please)\b|[.,;:])",
        r"\bcurrency\s+(?:as|in|to|into)\s+(.+?)(?=$|\b(?:and|then|for|with|from|on|by|please)\b|[.,;:])",
    ]
    for pattern in patterns:
        m = re.search(pattern, lowered, re.I)
        if not m:
            continue
        candidate = normalize_currency(m.group(1))
        if candidate:
            return candidate
    return None


def has_currency_conversion_intent(text: str) -> bool:
    """Return True only when the user explicitly asks for conversion.

    Simply mentioning a target currency code should not be treated as a
    conversion request; the request must include a conversion verb/phrase.
    """
    text = str(text or "")
    return bool(re.search(
        r"\b(?:convert|change|exchange|transform|express|denominate)\b.*\bcurrenc(?:y|ies)\b|"
        r"\bcurrenc(?:y|ies)\b.*\b(?:convert|change|exchange|transform|express|denominate)\b|"
        r"\bconvert\b.*\b(?:to|into)\b",
        text,
        re.I,
    ))


def detect_currency_from_value(value: Any) -> str | None:
    if value is None:
        return None
    s = _normalize_currency_text(value)
    low = s.lower()
    ordered_patterns: list[tuple[str, str, bool]] = [
        (r"us\$", "USD", True),
        (r"\bus dollar(?:s)?\b", "USD", True),
        (r"\busd\b", "USD", True),
        (r"s\$", "SGD", True),
        (r"\bsingapore dollar(?:s)?\b", "SGD", True),
        (r"\bsgd\b", "SGD", True),
        (r"c\$", "CAD", True),
        (r"\bcanadian dollar(?:s)?\b", "CAD", True),
        (r"\bcad\b", "CAD", True),
        (r"a\$", "AUD", True),
        (r"\baustralian dollar(?:s)?\b", "AUD", True),
        (r"\baud\b", "AUD", True),
        (r"hk\$", "HKD", True),
        (r"\bhong kong dollar(?:s)?\b", "HKD", True),
        (r"\bhkd\b", "HKD", True),
        (re.escape("د.إ"), "AED", True),
        (r"\baed\b", "AED", True),
        (r"\bdirham(?:s)?\b", "AED", True),
        (r"₹", "INR", False),
        (r"\binr\b", "INR", True),
        (r"\brs\.?\b", "INR", True),
        (r"\brupee(?:s)?\b", "INR", True),
        (r"৳", "BDT", False),
        (r"\bbdt\b", "BDT", True),
        (r"\btaka\b", "BDT", True),
        (r"₽", "RUB", False),
        (r"\brub(?:le|les)?\b", "RUB", True),
        (r"\brubel(?:s)?\b", "RUB", True),
        (r"\brouble(?:s)?\b", "RUB", True),
        (r"€", "EUR", False),
        (r"\beur\b", "EUR", True),
        (r"\beuro(?:s)?\b", "EUR", True),
        (r"£", "GBP", False),
        (r"\bgbp\b", "GBP", True),
        (r"\bpound(?:s)?\b", "GBP", True),
        (r"¥", "JPY", False),
        (r"\bjpy\b", "JPY", True),
        (r"\byen\b", "JPY", True),
        (r"\bcny\b", "CNY", True),
        (r"\byuan\b", "CNY", True),
        (r"\brenminbi\b", "CNY", True),
        (r"元", "CNY", False),
        (r"\$", "USD", True),
    ]
    for pattern, code, is_regex in ordered_patterns:
        if is_regex:
            if re.search(pattern, low):
                return code
        elif pattern in s:
            return code
    return None


def standardize_currency_value(value: Any) -> str | Any:
    """Canonicalize a currency-like value without converting its amount.

    Examples:
      "$1250" -> "USD 1250.00"
      "Aed 150" -> "AED 150.00"
      "Rubel 2500" -> "RUB 2500.00"
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # A bare number has no detectable source currency, so leave it alone.
        return value
    source = detect_currency_from_value(value)
    amount = parse_amount(value)
    if source is None and amount is None:
        return value
    if source is None:
        # Preserve ambiguous values instead of inventing a currency code.
        return value
    if amount is None:
        return source
    if source == "INR":
        symbol = SYMBOL_BY_CODE.get(source, "₹")
        return f"{symbol}{amount:,.2f}"
    return f"{source} {amount:.2f}"


def parse_amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = _normalize_currency_text(value).replace(',', '')
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _fetch_rates(base: str) -> dict[str, float]:
    now = time.time()
    cached = _CACHE.get(base)
    if cached and now - cached[0] < 3600:
        return cached[1]
    url = "https://open.er-api.com/v6/latest/" + urllib.parse.quote(base)
    with urllib.request.urlopen(url, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rates = payload.get("rates") or {}
    if not isinstance(rates, dict):
        raise RuntimeError(f"No exchange rates returned for {base}.")
    rates = {str(k).upper(): float(v) for k, v in rates.items() if isinstance(v, (int, float))}
    _CACHE[base] = (now, rates)
    return rates


def convert_amount(amount: float, source: str, target: str) -> float:
    source = normalize_currency(source) or source.upper()
    target = normalize_currency(target) or target.upper()
    if source == target:
        return amount
    rates = _fetch_rates(source)
    if target not in rates:
        raise RuntimeError(f"Exchange rate {source}->{target} is unavailable.")
    return amount * rates[target]


def currency_format(code: str) -> str:
    return FORMAT_BY_CODE.get(code.upper(), '#,##0.00')


def currency_symbol(code: str) -> str | None:
    normalized = normalize_currency(code) or str(code or "").upper()
    return SYMBOL_BY_CODE.get(normalized)
