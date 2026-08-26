import os
import uuid
import math
import json
import numpy as np
import pandas as pd
try:
    from google.adk.agents import LlmAgent, SequentialAgent
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
    SequentialAgent = _UnavailableGemini
    Runner = _UnavailableGemini
    InMemorySessionService = _UnavailableSessionService
    types = _TypesNamespace()

from command_agent import _extract_json
from common.report.orchestrator import generate_structured_report_data
from privacy_context import dataframe_profile, strict_enabled, sanitize_user_text, safe_columns

MODEL = "gemini-3.5-flash"



def _json_safe(obj):
    """Recursively converts a value into something json.dumps can handle."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj
def _make_tools(df: pd.DataFrame):
    """Builds tool functions bound to this specific request's dataframe,
    via a mutable holder dict (so clean_data can update it in place)."""
    state = {"df": df}

    def inspect_data() -> dict:
        """Returns a structural summary: shape, dtypes, nulls, duplicates."""
        d = state["df"]
        if strict_enabled():
            safe_cols, _, _ = safe_columns(list(d.columns))
            return _json_safe({
                "shape": list(d.shape),
                "dtypes": {safe_cols[i]: str(d.dtypes.iloc[i]) for i in range(len(safe_cols))},
                "null_counts": {safe_cols[i]: int(d.isnull().sum().iloc[i]) for i in range(len(safe_cols))},
                "duplicate_rows": int(d.duplicated().sum()),
            })
        return _json_safe({
            "shape": list(d.shape),
            "dtypes": d.dtypes.astype(str).to_dict(),
            "nulls_per_column": d.isnull().sum().to_dict(),
            "duplicate_rows": int(d.duplicated().sum()),
        })

    def clean_data(drop_duplicates: bool = True, fill_numeric_na_with: str = "median") -> dict:
        """Drops duplicate rows and fills missing values. Returns a summary."""
        d = state["df"]
        before_rows = len(d)
        before_nulls = int(d.isnull().sum().sum())

        if drop_duplicates:
            d = d.drop_duplicates().copy()

        for col in d.select_dtypes(include=[np.number]).columns:
            if d[col].isnull().any():
                fill_val = d[col].median() if fill_numeric_na_with == "median" else d[col].mean()
                d.loc[:, col] = d[col].fillna(fill_val)

        for col in d.select_dtypes(exclude=[np.number]).columns:
            d.loc[:, col] = d[col].fillna("Unknown")

        state["df"] = d
        duplicates_removed = before_rows - len(d)
        # Tracks whether clean_data actually changed anything — the cleaning
        # agent is instructed to always call this tool, even on data that
        # needed no cleaning at all, so before_nulls==0 and
        # duplicates_removed==0 in that case and this correctly stays False.
        state["data_was_modified"] = bool(before_nulls > 0 or duplicates_removed > 0)
        return _json_safe({
            "rows_before": before_rows,
            "rows_after": len(d),
            "duplicates_removed": duplicates_removed,
            "nulls_before": before_nulls,
            "nulls_after": int(d.isnull().sum().sum()),
        })

    def analyze_data() -> dict:
        """Computes descriptive stats, correlations, and top categories."""
        d = state["df"]
        numeric_df = d.select_dtypes(include=[np.number])
        non_numeric_df = d.select_dtypes(exclude=[np.number])

        if strict_enabled():
            safe_cols, _, _ = safe_columns(list(d.columns))
            numeric_summary = {}
            numeric_cols = list(numeric_df.columns)
            for idx, col in enumerate(numeric_cols):
                numeric_summary[safe_cols[list(d.columns).index(col)]] = {
                    k: (None if pd.isna(v) else round(float(v), 4))
                    for k, v in numeric_df[col].describe().to_dict().items()
                }
            corr = {}
            if numeric_df.shape[1] > 1:
                for c1 in numeric_cols:
                    corr[safe_cols[list(d.columns).index(c1)]] = {
                        safe_cols[list(d.columns).index(c2)]: round(float(v), 3)
                        for c2, v in numeric_df.corr().loc[c1].to_dict().items()
                    }
            return _json_safe({"numeric_summary": numeric_summary, "correlations": corr,
                               "categorical_columns": [safe_cols[list(d.columns).index(c)] for c in non_numeric_df.columns]})
        result = {
            "numeric_summary": numeric_df.describe().to_dict(),
            "correlations": numeric_df.corr().round(3).to_dict() if numeric_df.shape[1] > 1 else {},
            "top_categories": {
                str(col): {str(k): int(v) for k, v in non_numeric_df[col].value_counts().head(5).items()}
                for col in non_numeric_df.columns
            },
        }
        return _json_safe(result)

    return inspect_data, clean_data, analyze_data, state


async def generate_report(df: pd.DataFrame, focus_analysis_types: list | None = None) -> dict:
    """Runs the 3-agent pipeline (clean -> analyze -> report) on the given
    dataframe and returns the final text report plus a cleaned data preview.

    focus_analysis_types: optional list of {"id","title"} dicts (from
    suggest_analysis_types) the user selected — when provided, the report
    agent is asked to specifically emphasize those analysis types rather
    than writing a fully generic report.
    """
    inspect_data, clean_data, analyze_data, state = _make_tools(df)

    cleaning_agent = LlmAgent(
        name="cleaning_agent",
        model=MODEL,
        tools=[inspect_data, clean_data],
        instruction=(
            "You are a data cleaning agent. First call inspect_data. Then "
            "call clean_data. Finally, write a short plain-English summary "
            "of what was wrong with the data and what you fixed."
        ),
        description="Inspects and cleans the uploaded dataset.",
        output_key="cleaning_summary",
    )

    analysis_agent = LlmAgent(
        name="analysis_agent",
        model=MODEL,
        tools=[analyze_data],
        instruction=(
            "You are a data analysis agent. Call analyze_data. Then summarize "
            "notable averages/ranges, strong correlations (above 0.5 or below "
            "-0.5), and the most common categories. Plain English, no jargon."
        ),
        description="Analyzes the cleaned dataset and surfaces key patterns.",
        output_key="analysis_summary",
    )

    focus_line = ""
    if focus_analysis_types:
        titles = ", ".join(t.get("title", t.get("id", "")) for t in focus_analysis_types)
        focus_line = (
            f"\n\nThe user specifically asked to focus on these analysis types: {titles}. "
            "Weight 'Key Findings' toward what matters for those specific analyses, "
            "rather than giving equal space to unrelated angles."
        )

    report_agent = LlmAgent(
        name="report_agent",
        model=MODEL,
        instruction=(
            "Using the cleaning summary:\n{cleaning_summary}\n\n"
            "and the analysis summary:\n{analysis_summary}\n\n"
            "Write a short business-style report with three sections: "
            "'Data Quality', 'Key Findings', and 'Recommended Next Steps'. "
            "Concise and non-technical." + focus_line
        ),
        description="Writes the final report.",
        output_key="final_report",
    )

    pipeline = SequentialAgent(
        name="data_analyst_pipeline",
        sub_agents=[cleaning_agent, analysis_agent, report_agent],
    )

    app_name = "data_analyst_app"
    user_id = "api_user"
    session_id = str(uuid.uuid4())  # unique per request, avoids cross-request bleed

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=pipeline, app_name=app_name, session_service=session_service)

    content = types.Content(
        role="user",
        parts=[types.Part(text="Clean and analyze the uploaded dataset, then write the report.")],
    )

    final_text = None
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text

    cleaned_df = state["df"]
    data_was_modified = bool(state.get("data_was_modified", False))

    result = {
        "report": final_text,
        "cleaned_preview": cleaned_df.head(15).fillna("").to_dict(orient="records"),
        # Tells the client whether clean_data actually changed anything
        # (filled a null, dropped a duplicate row) — if False, there's
        # nothing new worth writing to a sheet.
        "data_was_modified": data_was_modified,
    }
    if data_was_modified:
        # Full cleaned dataset (not just the 15-row preview) so the client
        # can write the actual result into a new Excel sheet, instead of
        # only describing the cleaning in the report text with nothing to
        # show for it in the workbook itself.
        result["cleaned_data"] = _json_safe({
            "columns": list(cleaned_df.columns),
            "rows": cleaned_df.to_dict(orient="records"),
        })
    if final_text is None:
        # Never let the caller mistake this for success — without this,
        # the response looks identical in shape to a normal success (same
        # keys present), just with report=null, and nothing tells the
        # client anything went wrong.
        result["error"] = (
            "The report agent did not return a response. This can happen due "
            "to a rate limit or an empty model output — try again."
        )
    return result


async def _run_single_agent(agent: LlmAgent, app_name: str, prompt: str) -> str | None:
    """Runs one LlmAgent in its own fresh session and returns its final
    text response (or None if it produced nothing). Shared by the cleaning
    step and the narrative step in generate_structured_report() below, so
    neither needs to duplicate the session/runner boilerplate."""
    user_id = "api_user"
    session_id = str(uuid.uuid4())
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_text = None
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text
    return final_text


async def generate_structured_report(
    df: pd.DataFrame,
    selected_analysis_ids: list[str] | None = None,
    focus_analysis_types: list | None = None,
    value_column: str | None = None,
    period_column: str | None = None,
    question: str | None = None,
) -> dict:
    """Backs /analyze-report-focused.

    Flow (per the fix requirements):
        DataFrame -> clean -> Python analytics (existing detectors, via
        common/report/orchestrator.generate_structured_report_data) ->
        structured results -> optional Gemini narrative that is given
        those structured results as context and explicitly told not to
        invent its own numbers.

    `selected_analysis_ids` drives which structured sections are computed
    AND exposed (see orchestrator docstring for the dependency handling).
    `focus_analysis_types` (the {"id","title"} dicts the client sends) is
    only used to word the narrative's emphasis — identical to the role it
    played in generate_report() above.
    """
    inspect_data, clean_data, _analyze_data, state = _make_tools(df)

    cleaning_agent = LlmAgent(
        name="cleaning_agent",
        model=MODEL,
        tools=[inspect_data, clean_data],
        instruction=(
            "You are a data cleaning agent. First call inspect_data. Then "
            "call clean_data. Finally, write a short plain-English summary "
            "of what was wrong with the data and what you fixed."
        ),
        description="Inspects and cleans the uploaded dataset.",
    )
    cleaning_summary = await _run_single_agent(
        cleaning_agent,
        app_name="structured_report_cleaning_app",
        prompt="Clean the uploaded dataset.",
    )

    cleaned_df = state["df"]
    data_was_modified = bool(state.get("data_was_modified", False))

    # ── Python analytics only — no Gemini call computes any of this. ───────
    structured_data = generate_structured_report_data(
        cleaned_df,
        selected_analysis_ids or [],
        value_column=value_column,
        period_column=period_column,
        question=question,
    )

    focus_line = ""
    if focus_analysis_types:
        titles = ", ".join(t.get("title", t.get("id", "")) for t in focus_analysis_types)
        focus_line = (
            f"\n\nThe user specifically asked to focus on these analysis types: {titles}. "
            "Weight the narrative toward what matters for those specific analyses, "
            "rather than giving equal space to unrelated angles."
        )

    report_text = None
    if strict_enabled():
        report_text = (
            "Data Quality: Analysis was computed locally. Workbook contents were not sent to the external AI provider.\n\n"
            "Key Findings: The requested analysis was calculated locally from the current worksheet. "
            "Exact worksheet values and headers were kept on the application side."
        )
    elif structured_data or cleaning_summary:
        report_agent = LlmAgent(
            name="structured_report_agent",
            model=MODEL,
            instruction=(
                "You write a short, business-style, non-technical report using ONLY the "
                "already-computed analytical results given to you below. Do NOT calculate, "
                "estimate, or invent any statistic, trend, outlier, KPI, or score of your "
                "own — every number you mention must come directly from the JSON provided. "
                "If a section is empty or missing, simply don't discuss it.\n\n"
                "Write with two short sections: 'Data Quality' and 'Key Findings'." + focus_line
            ),
            description="Writes the final narrative report from precomputed structured results.",
        )
        prompt = (
            "cleaning_summary: " + (cleaning_summary or "No cleaning was necessary.") + "\n\n"
            "structured_json: " + json.dumps(_json_safe(structured_data))
        )
        report_text = await _run_single_agent(
            report_agent,
            app_name="structured_report_narrative_app",
            prompt=prompt,
        )

    result = {
        "report": report_text,
        **structured_data,
        "cleaned_preview": cleaned_df.head(15).fillna("").to_dict(orient="records"),
        "data_was_modified": data_was_modified,
    }
    if data_was_modified:
        result["cleaned_data"] = _json_safe({
            "columns": list(cleaned_df.columns),
            "rows": cleaned_df.to_dict(orient="records"),
        })
    if report_text is None and not structured_data:
        result["error"] = (
            "The report agent did not return a response. This can happen due "
            "to a rate limit or an empty model output — try again."
        )
    return result


# ── Analysis-type suggestion + business-problem explanation ────────────────
# Two small, fast, single-LLM-call helpers (no multi-agent pipeline) used to
# let the user pick which kind of analysis they want BEFORE the heavier
# generate_report() pipeline runs, so the eventual report can be focused
# instead of generic.

def _profile_dataframe(df: pd.DataFrame) -> dict:
    """structural profile; never exposes real headers or cell values in strict mode."""
    return dataframe_profile(df, include_samples=True)


async def suggest_analysis_types(df: pd.DataFrame) -> dict:
    """Given a dataset, proposes 3-6 analysis types genuinely supported by
    the ACTUAL columns present — e.g. only suggests a pricing analysis if
    there's a price/cost-like column, only suggests a growth/trend analysis
    if there's a date/time column.

    Returns: {"analysis_types": [{"id","title","description"}, ...],
              "profile": {...}}  — the profile is echoed back so the client
    can pass it straight into explain_business_problems() below without a
    second file upload.
    """
    profile = _profile_dataframe(df)

    agent = LlmAgent(
        name="analysis_type_suggester",
        model=MODEL,
        instruction=(
            "You are given a dataset's column names, data types, and a few "
            "sample values (JSON below). Propose 3 to 6 types of analysis "
            "that are GENUINELY supported by these exact columns — do not "
            "suggest an analysis type that needs a column this dataset "
            "doesn't have (e.g. don't suggest 'Growth Analysis' if there is "
            "no date/time column; don't suggest 'Pricing Analysis' if there "
            "is no price/cost/revenue-like column). Base every suggestion "
            "on columns that are actually present.\n\n"
            "Respond with ONLY a JSON object — no markdown fences, no "
            "preamble, nothing but the object — matching EXACTLY this shape:\n"
            '{"analysis_types": [{"id": "snake_case_id", "title": "Short '
            'Title", "description": "One sentence, grounded in the actual '
            'column names, explaining what this analysis would compute."}]}'
        ),
        description="Suggests dataset-appropriate analysis types.",
    )

    app_name = "analysis_type_app"
    user_id = "api_user"
    session_id = str(uuid.uuid4())
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    prompt = f"Dataset profile:\n{json.dumps(profile)}"
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_text = None
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text

    if not final_text:
        return {"analysis_types": [], "profile": profile, "error": "No response from the agent."}

    try:
        cleaned = _extract_json(final_text)
        parsed = json.loads(cleaned)
        parsed["profile"] = profile
        return parsed
    except json.JSONDecodeError as e:
        print(f"[suggest_analysis_types] JSON parse failed: {e}. Raw: {final_text!r}")
        return {"analysis_types": [], "profile": profile, "error": "Could not parse suggestions."}


async def explain_business_problems(
    profile: dict,
    selected_ids: list,
    analysis_titles: dict | None = None,
) -> dict:
    """Given the dataset profile (from suggest_analysis_types) and the
    analysis type(s) the user picked, returns RESULT GUIDANCE for each —
    i.e. what running that analysis on THIS dataset would actually reveal,
    how to interpret it, and which concrete business problems/decisions it
    helps address. This is meant to help the user nail down their problem
    statement BEFORE running the full report, not just list generic uses.

    Returns:
      {"results": [
          {
            "id": "...", "title": "...",
            "what_it_reveals": "1-2 sentences on what this analysis computes
                on THIS dataset's actual columns.",
            "how_to_interpret": "1-2 sentences on what a high/low/changing
                result would mean in practice.",
            "business_problems": ["...", "..."]
          }
      ]}
    """
    analysis_titles = analysis_titles or {}
    agent = LlmAgent(
        name="business_problem_explainer",
        model=MODEL,
        instruction=(
            "You are given a dataset's column profile and a list of analysis "
            "types the user is considering running. For EACH selected "
            "analysis type, provide RESULT GUIDANCE to help the user define "
            "their problem statement before they commit to running it — "
            "grounded in the actual column names present, never generic "
            "boilerplate. Specifically, for each one give:\n"
            "  - what_it_reveals: 1-2 sentences on what this analysis would "
            "actually compute using THIS dataset's real columns.\n"
            "  - how_to_interpret: 1-2 sentences on what a high/low value or "
            "a rising/falling trend in the result would practically mean.\n"
            "  - business_problems: 2-4 short bullet points, concrete "
            "decisions or problems this analysis helps address.\n\n"
            "Plain business language, no jargon, no filler.\n\n"
            "Respond with ONLY a JSON object — no markdown fences, no "
            "preamble, nothing but the object — matching EXACTLY this shape:\n"
            '{"results": [{"id": "snake_case_id", "title": "Short Title", '
            '"what_it_reveals": "...", "how_to_interpret": "...", '
            '"business_problems": ["...", "..."]}]}'
        ),
        description="Explains result guidance for the selected analysis type(s).",
    )

    app_name = "business_problem_app"
    user_id = "api_user"
    session_id = str(uuid.uuid4())
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    prompt = (
        f"Dataset profile:\n{json.dumps(profile)}\n\n"
        f"Selected analysis types:\n"
        f"{json.dumps([{'id': i, 'title': analysis_titles.get(i, i)} for i in selected_ids])}"
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_text = None
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text

    if not final_text:
        return {"results": [], "error": "No response from the agent."}

    try:
        cleaned = _extract_json(final_text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[explain_business_problems] JSON parse failed: {e}. Raw: {final_text!r}")
        return {"results": [], "error": "Could not parse result guidance."}
