# ai_engine.py
import json
import re
import os
import pickle

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common.insights.service import InsightsService
from common.insights.recommendation_engine import RecommendationEngine
from common.insights.chart_recommender import ChartRecommenderEngine
from common.insights.kpi_detector import KpiDetectorEngine
from common.insights.executive_summary import generate_executive_summary
from common.statistics.service import calculate_data_quality_score

# Use /tmp on Render (always writable)
TRAINING_DATA_PATH = "training_data.json"
MODEL_PATH         = "/tmp/intent_model.pkl"


def load_training_data(path=TRAINING_DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = re.sub(r"//.*", "", raw)
    data = json.loads(cleaned)
    texts   = [d["text"]   for d in data]
    intents = [d["intent"] for d in data]
    return texts, intents, data


def train_model(path=TRAINING_DATA_PATH):
    texts, intents, _ = load_training_data(path)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), analyzer="word",
                                  sublinear_tf=True, min_df=1)),
        ("clf",   LogisticRegression(max_iter=500, C=5.0, solver="lbfgs")),
    ])
    model.fit(texts, intents)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model trained on {len(texts)} examples, {len(set(intents))} intents.")
    return model


def load_model():
    # Always retrain on Render since /tmp is cleared on restart
    if not os.path.exists(MODEL_PATH):
        print("⚡ Training model from scratch...")
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def extract_slots(text: str) -> dict:
    t = text.lower().strip()
    slots = {}

    name_match = re.search(
        r"(?:called|named|as)\s+['\"]?([a-z][a-z0-9_ ]*?)['\"]?(?:\s|$)", t)
    if name_match:
        slots["sheet_name"] = name_match.group(1).strip().title()

    col_patterns = [
        r"(?:of the|of|for|in the|in|by|on)\s+['\"]?(\w+)['\"]?\s*(?:column|col)\b",
        r"(?:column|col)\s+['\"]?(\w+)['\"]?",
        r"(?:average|avg|sum|total|min|max|count|mean|minimum|maximum)\s+(?:of\s+)?(?:the\s+)?['\"]?(\w+)['\"]?",
    ]
    for pat in col_patterns:
        col_match = re.search(pat, t)
        if col_match:
            candidate = col_match.group(1).strip()
            if candidate not in {"the", "a", "an", "this", "all", "row", "rows", "data"}:
                slots["column"] = candidate
                break

    op_map = {"average": "average", "avg": "average", "mean": "average",
               "sum": "sum", "total": "sum", "min": "min", "minimum": "min",
               "max": "max", "maximum": "max", "count": "count"}
    for keyword, op in op_map.items():
        if keyword in t:
            slots["operation"] = op
            break

    ft_map = [
        (r"\btop\s*\d+\b", "top_n"),
        (r"above average", "above_average"),
        (r"below average", "below_average"),
        (r"between", "between"),
        (r"greater than or equal|>=", "greater_than_equal"),
        (r"less than or equal|<=", "less_than_equal"),
        (r"greater than|>", "greater_than"),
        (r"less than|<", "less_than"),
        (r"does not equal|not equal|!=", "not_equals"),
        (r"contains", "contains"),
        (r"\bequals?\b", "equals"),
    ]
    for pat, ft in ft_map:
        if re.search(pat, t):
            slots["filter_type"] = ft
            break

    nums = re.findall(r"\b\d+(?:\.\d+)?\b", t)
    if nums:
        slots["value"] = nums[0]
        if len(nums) > 1:
            slots["value2"] = nums[1]

    return slots


_model = None

def parse_command(text: str) -> dict:
    global _model
    if _model is None:
        _model = load_model()
    intent     = _model.predict([text])[0]
    confidence = float(_model.predict_proba([text]).max())
    slots      = extract_slots(text)
    return {"intent": intent, "confidence": round(confidence, 3), "slots": slots}


def add_training_example(text, intent, slots=None, path=TRAINING_DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(re.sub(r"//.*", "", f.read()))
    data.append({"text": text, "intent": intent, "slots": slots or {}})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    global _model
    _model = train_model(path)
    return len(data)


# ── DataFrame insights (Python + pandas only — no AI/LLM) ──────────────────
# Purely additive: nothing above this line was changed. This is the only
# place in ai_engine.py that touches a DataFrame; parse_command()'s NLP
# pipeline above is unaffected either way.

_insights_service = InsightsService()
_recommendation_engine = RecommendationEngine()
_chart_recommender_engine = ChartRecommenderEngine()
_kpi_detector_engine = KpiDetectorEngine()


def generate_statistics(df: pd.DataFrame, value_column: str) -> dict:
    """The "statistics generation" step trend detection runs after. Plain
    pandas .describe() output as a dict — count/mean/std/min/quartiles/max —
    nothing ML-derived, just descriptive statistics."""
    if value_column not in df.columns:
        raise ValueError(f"value_column {value_column!r} is not a column in the given DataFrame")
    described = df[value_column].astype(float).describe()
    return {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in described.to_dict().items()}


def generate_missing_value_report(df: pd.DataFrame) -> dict:
    """% of missing values per column (0-100) — the "statistics" signal
    common.insights.recommendation_engine.HighMissingValuesRule looks for.
    Read-only; never mutates `df`."""
    return {str(col): round(float(pct), 2) for col, pct in (df.isna().mean() * 100).items()}


def generate_outlier_report(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    """Structured outlier findings — IQR AND Z-score, one entry per
    (column, method) pair, via common/insights/outlier_detector.py (through
    InsightsService, not the detector module directly — see service.py).
    This replaces the earlier IQR-only ad-hoc version that used to live
    here: same integration point, now backed by the shared, reusable
    detector module instead of one-off logic duplicated in this file."""
    return _insights_service.detect_outliers(df, columns=columns)


def _outliers_by_column_for_recommendations(outlier_findings: list[dict]) -> dict:
    """Reduces the detector's per-(column, method) findings down to the
    one-entry-per-column shape
    common.insights.recommendation_engine.OutlierQualityRule expects —
    keeping, per column, whichever method flagged the HIGHER percentage
    (the more cautious of the two signals), since the recommendation
    engine only needs "is this column a problem", not a full method
    breakdown."""
    worst_by_column: dict = {}
    for finding in outlier_findings:
        column = finding["column"]
        current = worst_by_column.get(column)
        if current is None or finding["percentage"] > current["outlier_percentage"]:
            worst_by_column[column] = {
                "outlier_count": finding["outlier_count"],
                "outlier_percentage": finding["percentage"],
            }
    return worst_by_column


def generate_dataframe_insights(
    df: pd.DataFrame,
    value_column: str,
    period_column: str | None = None,
    *,
    label: str | None = None,
    kpis: dict | None = None,
    question: str | None = None,
) -> dict:
    """Statistics generation, followed by trend detection over the same
    column, followed by structured outlier detection (IQR + Z-score) across
    every numeric column, followed by automatic KPI detection across the
    whole DataFrame, followed by rule-based recommendations built from all
    of the above plus a lightweight missing-value report and any
    caller-supplied KPIs, followed by a chart-type recommendation — the
    integration point requested for common/insights/trend_detector.py,
    common/insights/outlier_detector.py, common/insights/kpi_detector.py,
    common/insights/recommendation_engine.py, and
    common/insights/chart_recommender.py. This function IS ai_engine.py's
    "AI Report": one call assembles every structured signal a report
    needs, with zero AI/LLM involved anywhere in this file — see each
    module's own docstring for that guarantee.

    `kpis` (a caller-supplied dict, e.g. {"profit_margin_change_percent":
    -3.1}) and `detected_kpis` (this function's own auto-detected KPI list
    in the return value, e.g. [{"name": "Revenue", "value": 1930.0, ...}])
    are deliberately two different things: `kpis` feeds
    recommendation_engine.py's specific expected keys (business metrics
    this function has no way to compute on its own), while `detected_kpis`
    is kpi_detector.py's general-purpose automatic findings from the
    DataFrame's own column names. Neither replaces the other.

    `question` is the user's original natural-language question, if any —
    it sharpens the chart recommendation (e.g. "...over time" vs
    "...compare regions") but every other signal still works without it.
    """
    statistics = generate_statistics(df, value_column)
    trend_insight = _insights_service.detect_trend(df, value_column, period_column, label=label)
    missing_value_report = generate_missing_value_report(df)
    outlier_findings = generate_outlier_report(df)
    detected_kpis = _kpi_detector_engine.detect(df, statistics=statistics)

    recommendations = _recommendation_engine.generate(
        statistics={"missing_percentage": missing_value_report},
        trend={value_column: trend_insight.model_dump()},
        kpis=kpis,
        outliers=_outliers_by_column_for_recommendations(outlier_findings),
    )

    chart_recommendation = _chart_recommender_engine.recommend(
        question=question,
        df=df,
        statistics=statistics,
        trend=trend_insight.model_dump(),
    )

    data_quality = calculate_data_quality_score(df)
    executive_summary = generate_executive_summary(
        statistics=statistics,
        kpis=detected_kpis,
        trend=trend_insight.model_dump(),
        recommendations=recommendations,
        outliers=outlier_findings,
        data_quality=data_quality,
    )

    return {
        "statistics": statistics,
        "trend_insight": trend_insight.model_dump(),
        "outliers": outlier_findings,
        "detected_kpis": detected_kpis,
        "recommendations": recommendations,
        "chart_recommendation": chart_recommendation,
        "data_quality": data_quality,
        "executive_summary": executive_summary,
    }
