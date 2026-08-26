"""Low-memory, batched restaurant-review sentiment analysis.

Important production rule: the endpoint returns restaurant-level performance by
 default. It does NOT echo every original row back through HTTP. This keeps
 Render memory and response size bounded for large workbooks.
"""
import asyncio
import json
import os
import re
from typing import Any

import pandas as pd
try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - import fallback for local tests
    class _UnavailableClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("google.genai is unavailable in this environment.")

    class _TypesNamespace:
        class GenerateContentConfig:
            def __init__(self, *args, **kwargs):
                pass

    class _GenAI:
        Client = _UnavailableClient

    genai = _GenAI()
    types = _TypesNamespace()
from privacy_context import strict_enabled

MODEL = os.getenv("SENTIMENT_AGENT_MODEL", "gemini-2.5-flash")
DEFAULT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "150"))
MAX_CONCURRENT_BATCHES = int(os.getenv("SENTIMENT_MAX_CONCURRENT_BATCHES", "2"))

INSTRUCTION = """You are a restaurant customer-review sentiment analyst.
Classify each review by its meaning, not just keywords. Return ONLY valid JSON.
For every input item return exactly: {\"index\": integer, \"sentiment\": \"Positive|Neutral|Negative|Mixed\", \"score\": number}.
score is from -1.0 to 1.0. Mixed is appropriate when meaningful positive and negative aspects coexist.
Examples: \"noisy place\" and \"disappointing experience\" are Negative.
Do not infer facts not present in the review.
"""


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("Model did not return JSON")


def _fallback_label(text: str) -> tuple[str, float]:
    t = text.lower()
    negative = {"bad", "awful", "terrible", "disappointing", "noisy", "dirty", "rude", "slow", "worst", "poor", "hate", "horrible", "unhappy"}
    positive = {"great", "excellent", "amazing", "good", "friendly", "clean", "fast", "delicious", "love", "wonderful", "perfect"}
    n = sum(w in t for w in negative)
    p = sum(w in t for w in positive)
    if n > p and n:
        return "Negative", max(-1.0, -0.35 - .1 * n)
    if p > n and p:
        return "Positive", min(1.0, .35 + .1 * p)
    return "Neutral", 0.0


_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY is not configured")
        _client = genai.Client(api_key=api_key)
    return _client


async def _run_batch(items: list[dict]) -> list[dict]:
    prompt = INSTRUCTION + "\n\nClassify these reviews. Preserve every index exactly once:\n" + json.dumps(items, ensure_ascii=False)
    client = _get_client()
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return _extract_json(response.text or "")


async def analyze_sentiment(
    df: pd.DataFrame,
    review_column: str | None = None,
    restaurant_column: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_details: bool = False,
) -> dict:
    columns = list(df.columns)
    if not review_column:
        candidates = [c for c in columns if "review" in c.lower() and ("text" in c.lower() or "comment" in c.lower() or "content" in c.lower())]
        if not candidates:
            candidates = [c for c in columns if "review" in c.lower()]
        review_column = candidates[0] if candidates else None
    if not review_column or review_column not in df.columns:
        raise ValueError("Could not find a review-text column. Please use a column such as ReviewText.")

    if not restaurant_column:
        candidates = [c for c in columns if "restaurant" in c.lower() and "name" in c.lower()]
        restaurant_column = candidates[0] if candidates else None
    if restaurant_column and restaurant_column not in df.columns:
        restaurant_column = None

    # Only keep the two fields required for sentiment. Avoid df.copy(), extra
    # columns, and a second full-size DataFrame for large workbooks.
    review_series = df[review_column].fillna("").astype(str).str.strip()
    restaurant_series = df[restaurant_column] if restaurant_column else None
    nonempty = [(int(i), txt) for i, txt in review_series.items() if txt]
    batch_size = max(50, min(int(batch_size or DEFAULT_BATCH_SIZE), 250))
    batches = [nonempty[start:start + batch_size] for start in range(0, len(nonempty), batch_size)]

    results: dict[int, tuple[str, float]] = {}
    semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_BATCHES))

    async def process_batch(batch: list[tuple[int, str]]) -> dict[int, tuple[str, float]]:
        async with semaphore:
            items = [{"index": i, "review": txt} for i, txt in batch]
            try:
                if strict_enabled():
                    return {idx: _fallback_label(txt) for idx, txt in batch}
                parsed = await _run_batch(items)
                out: dict[int, tuple[str, float]] = {}
                for r in parsed:
                    idx = int(r["index"])
                    sentiment = str(r.get("sentiment", "Neutral"))
                    if sentiment not in {"Positive", "Neutral", "Negative", "Mixed"}:
                        sentiment = "Neutral"
                    out[idx] = (sentiment, float(r.get("score", 0)))
                for idx, txt in batch:
                    if idx not in out:
                        out[idx] = _fallback_label(txt)
                return out
            except Exception:
                return {idx: _fallback_label(txt) for idx, txt in batch}

    # Do not retain a huge list of task results. Process bounded chunks of
    # batches so only a small number of LLM responses live in memory at once.
    window = max(1, MAX_CONCURRENT_BATCHES)
    for start in range(0, len(batches), window):
        window_results = await asyncio.gather(*(process_batch(b) for b in batches[start:start + window]))
        for batch_result in window_results:
            results.update(batch_result)
        del window_results

    counts = {"Positive": 0, "Neutral": 0, "Negative": 0, "Mixed": 0}
    score_sum = 0.0
    restaurant_acc: dict[str, dict[str, Any]] = {}

    # Aggregate without creating a sentiment DataFrame or a full detail copy.
    for i, txt in nonempty:
        sentiment, score = results.get(i, _fallback_label(txt))
        counts[sentiment] += 1
        score_sum += score
        if restaurant_series is not None:
            raw_name = restaurant_series.loc[i]
            name = "" if pd.isna(raw_name) else str(raw_name).strip()
            if not name:
                name = "Unknown Restaurant"
            acc = restaurant_acc.setdefault(name, {"Reviews": 0, "Positive": 0, "Neutral": 0, "Negative": 0, "Mixed": 0, "score_sum": 0.0})
            acc["Reviews"] += 1
            acc[sentiment] += 1
            acc["score_sum"] += score

    total = len(nonempty)
    overall = {
        "reviews_analyzed": total,
        "positive": counts["Positive"],
        "neutral": counts["Neutral"],
        "negative": counts["Negative"],
        "mixed": counts["Mixed"],
        "average_score": round(score_sum / total, 3) if total else 0.0,
    }
    overall["satisfaction_rate"] = round((counts["Positive"] + 0.5 * counts["Mixed"]) / total * 100, 1) if total else 0.0

    summary = []
    for name, acc in restaurant_acc.items():
        n = acc["Reviews"]
        summary.append({
            "Restaurant Name": name,
            "Reviews": n,
            "Positive": acc["Positive"],
            "Neutral": acc["Neutral"],
            "Negative": acc["Negative"],
            "Mixed": acc["Mixed"],
            "Satisfaction %": round((acc["Positive"] + 0.5 * acc["Mixed"]) / n * 100, 1),
            "Average Sentiment": round(acc["score_sum"] / n, 3),
        })
    summary.sort(key=lambda x: (-x["Satisfaction %"], -x["Reviews"], x["Restaurant Name"]))

    # Return one compact label per original data row so the client can append
    # a single `Sentiment` column to the CURRENT sheet. Blank reviews remain
    # blank; labels stay aligned with the original row order.
    sentiment_values = []
    for row_index in range(len(df)):
        if row_index in results:
            sentiment_values.append(results[row_index][0])
        else:
            sentiment_values.append("")

    result = {
        "review_column": review_column,
        "restaurant_column": restaurant_column,
        "batch_size": batch_size,
        "overall": overall,
        "sentiment_column": "Sentiment",
        "sentiment_values": sentiment_values,
        # Kept for API compatibility, but no longer used for the normal
        # customer-satisfaction workflow. The client should not create a
        # summary worksheet merely to answer a sentiment-column request.
        "summary_columns": [],
        "summary_rows": [],
        "detail_columns": [],
        "detail_rows": [],
        "details_included": False,
    }

    # Explicit opt-in only. This is intentionally off for the client-facing
    # satisfaction query because echoing 10k+ rows can exhaust Render memory.
    if include_details:
        detail_rows = []
        for i, txt in nonempty:
            sentiment, score = results.get(i, _fallback_label(txt))
            row = {review_column: txt, "Sentiment": sentiment, "SentimentScore": score}
            if restaurant_series is not None:
                raw_name = restaurant_series.loc[i]
                row[restaurant_column] = None if pd.isna(raw_name) else raw_name
            detail_rows.append(row)
        result["detail_columns"] = list(detail_rows[0].keys()) if detail_rows else [review_column, "Sentiment", "SentimentScore"]
        result["detail_rows"] = detail_rows
        result["details_included"] = True
    return result
