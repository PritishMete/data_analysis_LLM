"""Agentic value categorization.

The command agent only decides that a categorization operation was requested.
This module is the second agentic step: it sees the real distinct values in the
selected column, proposes a compact value -> category mapping, validates the
mapping, and applies it with pandas. This prevents the command parser from
hallucinating categories before it has seen the data.
"""
from __future__ import annotations

import json
import re
import traceback
import uuid
from typing import Any

import pandas as pd
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
except Exception:  # pragma: no cover - import fallback for local tests
    class _UnavailableGemini:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("google.adk is unavailable in this environment.")

    class _UnavailableSessionService:
        async def create_session(self, *args, **kwargs):
            return None

    class _TypesNamespace:
        class Content:
            def __init__(self, *args, **kwargs):
                self.parts = kwargs.get("parts", [])

        class Part:
            def __init__(self, *args, **kwargs):
                self.text = kwargs.get("text", "")

    LlmAgent = _UnavailableGemini
    Runner = _UnavailableGemini
    InMemorySessionService = _UnavailableSessionService
    types = _TypesNamespace()
from privacy_context import strict_enabled
MODEL = "gemini-3.5-flash"

INSTRUCTION = """You are the Categorization Agent in an Excel data-analysis product.
Your job is to map EVERY distinct input value shown in the supplied value list to one
meaningful normalization label when, and only when, the column is appropriate for
categorical normalization.

Rules:
- Follow the user's requested categories exactly when they supplied them.
- If categories were not supplied, infer a small, useful set from the column name,
  user request, data type, and actual values. Prefer normalization of obvious
  variants, not business binning.
- Normalize obvious variants into the same category when the column is categorical
  (e.g. Yes/yes/Y/Ye -> Yes, No/no/N -> No; India/india -> India; Asia/asia -> Asia).
- Distinguish normalization from grouping/binning. Never invent Low/Medium/High for
  generic numeric columns.
- Do not perform sentiment analysis or review-theme extraction unless the user has
  explicitly asked for sentiment.
- Currency values are protected during generic categorization. Do not rewrite,
  standardize, or normalize currency unless a separate explicit conversion step
  is being executed elsewhere.
- Do not modify the source values. The output is only a mapping to a new label or a
  standardized representation.
- Every supplied value MUST appear exactly once in the mapping. Use unmatchedLabel only
  for values that cannot be confidently assigned.
- Return ONLY valid JSON with this shape:
{
  "categories": ["label", ...],
  "unmatchedLabel": "Other",
  "mapping": {"original value": "category"},
  "explanation": "short sentence"
}
""" 


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _compact_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _sample_series_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.tolist():
        key = "" if pd.isna(value) else str(value)
        if key not in seen:
            seen.add(key)
            values.append(key)
    return values


def _looks_boolean_series(series: pd.Series) -> bool:
    values = {str(v).strip().lower() for v in series.dropna().tolist()}
    if not values:
        return False
    null_tokens = {"", "none", "null", "na", "n/a", "nan", "unknown"}
    boolean_tokens = {"yes", "no", "y", "n", "ye", "true", "false", "t", "f", "1", "0"}
    non_null = values - null_tokens
    return bool(non_null) and non_null.issubset(boolean_tokens)


def classify_column_operation(source_column: str, series: pd.Series, user_request: str = "") -> str:
    """Classify a column before applying categorization.

    Returns one of:
      categorical_normalization, protected_currency, numeric_measure,
      geographic_coordinate, identifier, datetime, free_text,
      sentiment_text, protected_numeric, unknown
    """
    name = _normalize_name(source_column)
    compact = _compact_name(source_column)
    request = (user_request or "").lower()

    if any(token in name for token in {"latitude", "longitude", "coordinates", "coordinate"}):
        return "geographic_coordinate"
    if name in {"lat", "lng", "long"} or compact in {"lat", "lng", "long"}:
        return "geographic_coordinate"
    if any(token in name for token in {"review", "comment", "feedback"}) or (
        "sentiment" in request and any(token in name for token in {"text", "review", "comment", "feedback"})
    ):
        return "sentiment_text" if "sentiment" in request else "free_text"
    if any(token in name for token in {"url", "link", "email", "phone", "mobile"}):
        return "identifier"
    if compact.endswith("id") or compact.endswith("key") or compact.endswith("uuid") or compact.endswith("guid"):
        return "identifier"
    if "name" in name and not any(token in name for token in {"country", "region", "city", "gender", "bool", "boolean", "flag", "review", "comment", "feedback"}):
        return "identifier"
    if any(token in name for token in {"date", "time", "timestamp", "datetime", "createdat", "updatedat"}):
        return "datetime"
    if any(token in name for token in {"currency", "price", "amount", "cost", "fare", "salary", "revenue", "sales", "income", "budget", "fee", "charge", "value", "balance", "payment", "paid"}) or _compact_name(source_column) == "currency":
        return "protected_currency"

    non_null = series.dropna()
    if non_null.empty:
        return "categorical_normalization"

    numeric_series = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric_series.notna().mean()) if len(non_null) else 0.0
    if numeric_ratio >= 0.9:
        if any(token in name for token in {"rating", "ratings", "reviewrating", "aggregaterating", "ratingcount", "votes", "count", "quantity", "age", "sales", "revenue", "score"}):
            return "numeric_measure"
        return "numeric_measure"

    unique_ratio = non_null.astype(str).nunique(dropna=True) / max(len(non_null), 1)
    avg_len = float(non_null.astype(str).map(len).mean()) if len(non_null) else 0.0
    if unique_ratio > 0.8 and avg_len > 25:
        return "free_text"

    if any(token in name for token in {"country", "region", "city", "gender", "bool", "boolean", "flag", "status", "type", "category", "class", "segment", "zone", "area", "territory"}):
        return "categorical_normalization"
    if _looks_boolean_series(series):
        return "categorical_normalization"
    return "categorical_normalization"


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    first, last = text.find("{"), text.rfind("}")
    return text[first:last + 1] if first >= 0 and last > first else text


async def _ask_agent(user_request: str, source_column: str, values: list[str], categories: list[str], unmatched: str) -> dict[str, Any]:
    agent = LlmAgent(name="categorization_agent", model=MODEL, instruction=INSTRUCTION,
                     description="Maps real spreadsheet values into useful categories.")
    session_service = InMemorySessionService()
    app_name = "categorization_agent_app"
    user_id = "api_user"
    session_id = str(uuid.uuid4())
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    prompt = (
        f"Source column: {source_column}\n"
        f"User request: {user_request}\n"
        f"Column role: {classify_column_operation(source_column, pd.Series(values, name=source_column), user_request)}\n"
        f"Requested categories: {json.dumps(categories, ensure_ascii=False)}\n"
        f"Fallback/unmatched label: {unmatched}\n"
        f"Distinct values ({len(values)}): {json.dumps(values, ensure_ascii=False)}"
    )
    final_text = None
    async for event in runner.run_async(user_id=user_id, session_id=session_id,
                                        new_message=types.Content(role="user", parts=[types.Part(text=prompt)])):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text
    if not final_text:
        raise ValueError("The Categorization Agent returned no response.")
    parsed = json.loads(_extract_json(final_text))
    return parsed



def _deterministic_special_mapping(values: list[str], source_column: str) -> dict[str, str] | None:
    """Canonicalize high-confidence categorical variants without relying on the LLM.

    Boolean-like columns are handled *before* generic text/LLM categorization. This is
    important for real spreadsheets where a boolean column commonly contains mixed
    representations such as Y/N, Yes/No, 1/0, true/false, and Ye. The mapping is
    based on both the column name and the observed values, so a column named ``Bool``
    containing ``Y, N, 0, 1`` is always treated as boolean and never falls through to
    the generic text-normalization fallback.
    """
    name = _normalize_name(source_column).replace(" ", "")
    low = {v: v.strip().lower() for v in values}
    boolean_tokens = {"yes", "no", "y", "n", "ye", "true", "false", "t", "f", "1", "0"}
    null_tokens = {"", "none", "null", "na", "n/a", "nan", "unknown"}

    def _norm_token(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.strip().lower())

    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        curr = [0] * (len(b) + 1)
        for i, ca in enumerate(a, start=1):
            curr[0] = i
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev, curr = curr, prev
        return prev[-1]

    def _fuzzy_lookup(raw: str, aliases: dict[str, str], canonicals: list[str], max_distance: int = 2) -> str | None:
        token = _norm_token(raw)
        if not token:
            return None
        if token in aliases:
            return aliases[token]
        best = None
        best_distance = 10**9
        for canonical in canonicals:
            canonical_token = _norm_token(canonical)
            distance = _levenshtein(token, canonical_token)
            if distance < best_distance:
                best_distance = distance
                best = canonical
            elif distance == best_distance:
                best = None
        return best if best is not None and best_distance <= max_distance else None

    def _collapse_token(value: str) -> str:
        return re.sub(r"(.)\1+$", r"\1", value.strip().lower())

    def _looks_boolean_series_like() -> bool:
        observed = set(low.values())
        non_null = observed - null_tokens
        return bool(non_null) and non_null.issubset(boolean_tokens)

    # Explicit boolean/flag column names get the boolean mapping even when the
    # column has a small amount of missing data or mixed numeric/string storage.
    explicit_bool_name = (
        name in {"bool", "boolean", "flag", "binary", "binaryflag", "boolcolumn"}
        or name.startswith("is")
        or name.startswith("has")
        or name.endswith("flag")
    )
    boolean_like = _looks_boolean_series_like()
    if explicit_bool_name or boolean_like:
        # Do not accidentally turn arbitrary empty text into Yes/No. Unknown/missing
        # values remain explicit so every row is still handled.
        return {
            original: ("Yes" if v in {"yes", "y", "ye", "true", "t", "1"} or _collapse_token(v) in {"yes", "y", "ye", "true", "t", "1"}
                       else "No" if v in {"no", "n", "false", "f", "0"} or _collapse_token(v) in {"no", "n", "false", "f", "0"}
                       else "Unknown")
            for original, v in low.items()
        }

    # Country normalization: common casing, spelling, abbreviation and typo variants.
    if "country" in name or name in {"nation", "countryname"}:
        country_aliases = {
            "india": "India", "ind": "India", "in": "India", "idnia": "India", "indai": "India",
            "uae": "United Arab Emirates", "unitedarabemirates": "United Arab Emirates",
            "arab": "United Arab Emirates",
            "singapore": "Singapore", "singapor": "Singapore",
            "uk": "United Kingdom", "united kingdom": "United Kingdom",
            "us": "United States", "usa": "United States", "united states": "United States",
            "bangladesh": "Bangladesh", "bangladsh": "Bangladesh",
            "bd": "Bangladesh",
            "russia": "Russia", "russina": "Russia",
            "canada": "Canada", "canad": "Canada",
            "china": "China", "japan": "Japan", "germany": "Germany",
            "france": "France", "australia": "Australia",
        }
        canonicals = [
            "India",
            "United Arab Emirates",
            "Singapore",
            "United Kingdom",
            "United States",
            "Bangladesh",
            "Russia",
            "Canada",
        ]
        out = {}
        for original, v in low.items():
            matched = _fuzzy_lookup(v, country_aliases, canonicals, max_distance=2)
            if matched is not None:
                out[original] = matched
            elif v in {"", "none", "null", "na", "n/a", "unknown"}:
                out[original] = "Unknown"
            else:
                # Preserve an unknown country rather than inventing a country.
                out[original] = re.sub(r"\s+", " ", v).strip().title()
        return out

    # Region normalization: case and common spelling variants.
    # City normalization: canonicalize common abbreviations/typos and casing.
    if "city" in name or name in {"town", "cityname"}:
        city_aliases = {
            "new delhi": "New Delhi", "delhi": "Delhi", "newdelhi": "New Delhi",
            "mumbai": "Mumbai", "bombay": "Mumbai",
            "kolkata": "Kolkata", "calcutta": "Kolkata",
            "gurgaon": "Gurgaon", "gurugram": "Gurgaon",
            "bangalore": "Bangalore", "bengaluru": "Bangalore",
            "hyderabad": "Hyderabad", "chennai": "Chennai", "madras": "Chennai",
            "pune": "Pune", "noida": "Noida", "faridabad": "Faridabad",
            "jaipur": "Jaipur", "ahmedabad": "Ahmedabad",
            "dubai": "Dubai", "abu dhabi": "Abu Dhabi", "abudhabi": "Abu Dhabi",
            "london": "London", "singapore": "Singapore", "dhaka": "Dhaka",
            "moscow": "Moscow", "new york": "New York", "newyork": "New York", "nyc": "New York",
            "toronto": "Toronto",
        }
        out = {}
        for original, v in low.items():
            key = re.sub(r"\s+", " ", v).strip()
            out[original] = city_aliases.get(key, "Unknown" if key in {"", "none", "null", "na", "n/a", "unknown"} else key.title())
        return out

    if "region" in name or name in {"area", "zone", "territory"}:
        region_aliases = {
            "asia": "Asia", "asia pacific": "Asia", "apac": "Asia", "ncr": "NCR", "middle east": "Middle East",
            "middleeast": "Middle East", "eu": "Europe", "europe": "Europe",
            "north america": "North America", "northamerica": "North America",
            "south america": "South America", "southamerica": "South America",
            "africa": "Africa", "oceania": "Oceania",
        }
        out = {}
        for original, v in low.items():
            key = re.sub(r"\s+", " ", v).strip()
            out[original] = region_aliases.get(key, "Unknown" if key in {"", "none", "null", "na", "n/a", "unknown"} else key.title())
        return out

    if "gender" in name or name in {"sex", "gendercode"}:
        gender_aliases = {
            "m": "Male",
            "male": "Male",
            "malee": "Male",
            "mlae": "Male",
            "mal": "Male",
            "f": "Female",
            "female": "Female",
            "femalee": "Female",
            "femle": "Female",
            "femaile": "Female",
            "t": "Transgender",
            "transgender": "Transgender",
            "trans": "Transgender",
        }
        canonicals = ["Male", "Female", "Transgender", "Non-binary", "Other", "Unknown"]
        out = {}
        for original, v in low.items():
            matched = _fuzzy_lookup(v, gender_aliases, canonicals, max_distance=2)
            if matched is not None:
                out[original] = matched
            elif v in {"nb", "nonbinary", "non-binary", "non binary"}:
                out[original] = "Non-binary"
            elif v in {"", "none", "null", "na", "n/a", "unknown"}:
                out[original] = "Unknown"
            else:
                out[original] = "Unknown"
        return out
    return None


def _deterministic_fallback_mapping(series: pd.Series) -> tuple[dict[str, str], list[str], str]:
    """Last-resort mapping: every distinct value gets a deterministic category.

    This guarantees multi-column categorization never reports a column as skipped.
    """
    values = []
    seen = set()
    for value in series.tolist():
        key = "" if pd.isna(value) else str(value)
        if key not in seen:
            seen.add(key); values.append(key)
    special = _deterministic_special_mapping(values, str(series.name))
    if special is not None:
        cats = sorted(set(special.values()))
        return special, cats, "Applied deterministic categorical normalization fallback."
    mapping = {}
    for v in values:
        normalized = re.sub(r"\s+", " ", v.strip())
        mapping[v] = normalized.title() if normalized else "Unknown"
    return mapping, sorted(set(mapping.values())), "Normalized categorical values deterministically."

async def categorize_dataframe(df: pd.DataFrame, source_column: str, new_column: str,
                         user_request: str, requested_categories: list[str] | None = None,
                         unmatched_label: str = "Other") -> tuple[pd.DataFrame, dict[str, Any]]:
    # Resolve column names case-insensitively and tolerate the common
    # "bool/boolean column" shorthand. The LLM may return "country" while
    # Excel's real header is "Country"; that should never be a failure.
    def _norm_col(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

    actual_source = next((c for c in df.columns if _norm_col(c) == _norm_col(source_column)), None)
    if actual_source is None and str(source_column).strip() == "__BOOLEAN_COLUMN__":
        def _looks_boolean(series: pd.Series) -> bool:
            vals = [str(v).strip().lower() for v in series.dropna().tolist()]
            if not vals:
                return False
            allowed = {"yes", "no", "y", "n", "true", "false", "t", "f", "1", "0", "ye"}
            return len(vals) > 0 and (len(set(vals)) <= 8) and all(v in allowed for v in vals)
        bool_candidates = [c for c in df.columns if _looks_boolean(df[c])]
        if bool_candidates:
            actual_source = bool_candidates[0]
    if actual_source is None:
        raise ValueError(f"Column '{source_column}' was not found.")
    source_column = str(actual_source)
    if not new_column.strip() or _norm_col(new_column) in {"__booleancolumn__", _norm_col("bool"), _norm_col("boolean")} :
        # Categorization is an in-place transformation. The source column is
        # the destination unless an explicit existing destination was supplied.
        new_column = source_column
    # In-place categorization is intentional: the categorized values replace
    # the original source column in the current worksheet. Never create a
    # *_Category companion column for this operation.

    series = df[source_column]
    original_series = series.copy()
    values = _sample_series_values(series)
    requested_categories = [str(x).strip() for x in (requested_categories or []) if str(x).strip()]
    unmatched_label = str(unmatched_label or "Other").strip() or "Other"
    column_role = classify_column_operation(source_column, series, user_request)
    execution = {
        "ai_used": False,
        "gemini_attempted": False,
        "gemini_request_success": False,
        "gemini_response_parsed": False,
        "gemini_mapping_used": False,
        "privacy_mode": "local_only" if strict_enabled() else "remote_allowed",
        "raw_data_sent_to_ai": False,
        "metadata_sent_to_ai": False,
        "unique_values_sent_to_ai": False,
        "local_fallback_used": False,
        "categorization_engine": "deterministic",
        "engine_used": "deterministic",
        "semantic_type": column_role,
        "currency_engine": "unused",
        "column_role": column_role,
        "fallback_used": False,
        "fallback_reason": None,
        "values_changed_count": 0,
    }

    if column_role in {"numeric_measure", "geographic_coordinate", "identifier", "datetime", "free_text", "sentiment_text", "protected_numeric", "protected_currency"}:
        out = df.copy()
        if column_role == "protected_currency":
            explanation = f"Left '{source_column}' unchanged because currency conversion was not requested."
        else:
            explanation = f"Left '{source_column}' unchanged because it is a {column_role.replace('_', ' ')}."
        metadata = {
            "source_column": source_column,
            "new_column": source_column,
            "write_mode": "unchanged",
            "categories": [],
            "unmatched_label": unmatched_label,
            "mapping": {v: v for v in values},
            "distinct_values": len(values),
            "rows_affected": 0,
            "explanation": explanation,
            "execution": execution,
        }
        return out, metadata

    if column_role == "protected_currency":
        out = df.copy()
        mapping = {v: v for v in values}
        execution.update({
            "categorization_engine": "protected_currency",
            "engine_used": "protected_currency",
            "currency_engine": "unused",
            "local_fallback_used": True,
            "fallback_used": True,
            "fallback_reason": "currency changes require explicit conversion intent",
        })
        plan = {
            "mapping": mapping,
            "categories": [],
            "unmatchedLabel": unmatched_label,
            "explanation": f"Left currency-like values in '{source_column}' unchanged because currency conversion was not requested.",
            "execution": execution.copy(),
        }
    else:
        if len(values) > 1500:
            raise ValueError(
                f"'{source_column}' has {len(values)} distinct values. Categorization is limited to 1,500 distinct values at a time; filter the data or choose a lower-cardinality column."
            )
        try:
            execution.update({
                "gemini_attempted": not strict_enabled(),
                "raw_data_sent_to_ai": not strict_enabled(),
                "metadata_sent_to_ai": not strict_enabled(),
                "unique_values_sent_to_ai": not strict_enabled(),
            })
            if strict_enabled():
                raise RuntimeError("Local processing mode: real worksheet values are not sent to the external AI provider.")
            plan = await _ask_agent(user_request, source_column, values, requested_categories, unmatched_label)
            mapping_raw = plan.get("mapping") if isinstance(plan, dict) else None
            if not isinstance(mapping_raw, dict):
                raise ValueError("The Categorization Agent did not return a valid value mapping.")
            mapping = {str(k): str(v) for k, v in mapping_raw.items()}
            categories = [str(x) for x in (plan.get("categories") or requested_categories) if str(x).strip()]
            fallback = str(plan.get("unmatchedLabel") or unmatched_label)
            execution.update({
                "ai_used": True,
                "categorization_engine": "gemini_assisted",
                "engine_used": "gemini_assisted",
                "gemini_request_success": True,
                "gemini_response_parsed": True,
                "gemini_mapping_used": True,
                "fallback_used": False,
                "fallback_reason": None,
            })
        except Exception as agent_exc:
            print(f"[categorization_agent] LLM failed for {source_column}; using deterministic fallback: {agent_exc}")
            mapping, categories, fallback_explanation = _deterministic_fallback_mapping(series)
            fallback = unmatched_label
            execution.update({
                "local_fallback_used": True,
                "categorization_engine": "deterministic_fallback",
                "engine_used": "deterministic_fallback",
                "fallback_used": True,
                "fallback_reason": str(agent_exc),
            })
            plan = {"mapping": mapping, "categories": categories, "unmatchedLabel": fallback,
                    "explanation": fallback_explanation, "execution": execution.copy()}
    missing = [v for v in values if v not in mapping]
    if missing:
        # Never leave rows silently uncategorized.
        for v in missing:
            mapping[v] = fallback
    allowed = set(categories) | {fallback}
    invalid = sorted({v for v in mapping.values() if v not in allowed}) if allowed else []
    if invalid:
        # Keep the output deterministic even if the model added an unlisted label.
        categories.extend([v for v in invalid if v not in categories and v != fallback])

    out = df.copy()
    out[new_column] = out[source_column].map(lambda v: mapping.get("" if pd.isna(v) else str(v), fallback))
    changed_count = 0
    for before, after in zip(original_series.tolist(), out[new_column].tolist()):
        if pd.isna(before) and pd.isna(after):
            continue
        if pd.isna(before) != pd.isna(after) or str(before) != str(after):
            changed_count += 1
    execution["values_changed_count"] = changed_count
    metadata = {
        "source_column": source_column,
        "new_column": new_column,
        "write_mode": "replace_source",
        "categories": categories,
        "unmatched_label": fallback,
        "mapping": mapping,
        "distinct_values": len(values),
        "rows_affected": int(len(out)),
        "explanation": str(plan.get("explanation") or f"Categorized '{source_column}' into {len(categories)} categories."),
        "execution": (plan.get("execution") if isinstance(plan, dict) else None) or execution,
    }
    if isinstance(metadata["execution"], dict):
        metadata["execution"].setdefault("semantic_type", column_role)
        metadata["execution"].setdefault("engine_used", metadata["execution"].get("categorization_engine", "deterministic"))
        metadata["execution"].setdefault("values_changed_count", changed_count)
    return out, metadata
