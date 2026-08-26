# colab_routes.py
# ─────────────────────────────────────────────────────────────────────────────
# Drop this file into your repo root alongside app.py, next to ai_routes.py
# and agentic_cleaning_routes.py. Add these 2 lines to the bottom of main.py:
#
#   from colab_routes import colab_router
#   app.include_router(colab_router)
#
# Exposes:
#   POST /generate_colab_code — text + available_columns (+ optional
#     dataframe_name, available_sheets) -> literal Python source, meant to
#     be inserted into a Colab cell and run there against the user's own
#     already-loaded DataFrame.
#
# This is the Colab equivalent of what /agentic_command and /smart_query
# already do for Excel — EXCEPT Excel has no Python kernel to hand code to,
# so those routes execute the plan/action server-side and return a
# result/JSON for Flutter to apply. Colab already has the DataFrame loaded
# locally, so the right move is to hand back runnable code instead of
# re-uploading the whole dataset on every question.
#
# Reuses the exact same planning agents Excel already depends on:
#   - query_router._run_router_agent()  -> decides "sql" vs "operation",
#     and for "sql" produces the same structured plan query_router.py
#     already turns into DuckDB SQL via build_sql_from_plan() (untouched).
#   - command_agent.parse_agentic_command() -> the same operation JSON
#     /agentic_command already returns for Excel.
# Neither module is modified. This route only converts their OUTPUT into
# Python text via colab_codegen.py, instead of executing it.
#
# Falls back to colab_code_agent.generate_general_code() whenever BOTH of
# those come back empty — i.e. the request is neither an analytical SQL
# question nor one of command_agent's six fixed spreadsheet operations
# (e.g. "write code to read a csv file", "plot this as a bar chart").
# That's a real, expected gap in the Excel-oriented planners (they SHOULD
# say "unknown" for those — Excel has no equivalent action), not something
# to route around; Colab just has a general Python kernel Excel doesn't,
# so it gets a genuinely general code-writing agent as a last resort.
# ─────────────────────────────────────────────────────────────────────────────

import traceback

from fastapi import APIRouter
from pydantic import BaseModel

from query_router import _run_router_agent, build_sql_from_plan, PlanError
from command_agent import parse_agentic_command
from colab_codegen import gen_sql_code, gen_operation_code
from colab_code_agent import generate_general_code

colab_router = APIRouter()


class ColabCodeRequest(BaseModel):
    text: str
    available_columns: list[str] = []
    available_sheets: list[str] = []
    dataframe_name: str = "df"


async def _fallback_to_general_code(req: "ColabCodeRequest", reason: str) -> dict:
    """Last resort: hand the request to the general-purpose code agent
    instead of returning a bare failure. `reason` is logged, not shown to
    the user — the agent's own message (or a generic one) is what's returned.
    """
    print(f"[/generate_colab_code] falling back to general code agent: {reason}")
    result = await generate_general_code(req.text, req.dataframe_name or "df", req.available_columns)
    if not result.get("code"):
        return {
            "success": False,
            "route": "general_code",
            "message": result.get("message") or "Could not generate code for that request.",
        }
    return {
        "success": True,
        "route": "general_code",
        "message": result.get("message", ""),
        "code": result["code"],
    }


@colab_router.post("/generate_colab_code")
async def generate_colab_code(req: ColabCodeRequest):
    """Turns a natural-language request into Python source for Colab.

    Tries, in order:
      1. query_router's SQL planner (analytical questions)
      2. command_agent's operation parser (the 6 fixed spreadsheet actions)
      3. a general-purpose code-writing agent (everything else — reading
         files, plotting, arbitrary scripting)
    """
    df_name = req.dataframe_name or "df"

    try:
        decision = await _run_router_agent(req.text, req.available_columns)
    except Exception:
        print("[/generate_colab_code] EXCEPTION during routing:")
        traceback.print_exc()
        return await _fallback_to_general_code(req, "router agent raised an exception")

    route = decision.get("route")
    confidence = decision.get("confidence", 0.0)
    message = decision.get("message", "")

    if route == "sql":
        plan = decision.get("plan")
        try:
            sql = build_sql_from_plan(plan, req.available_columns)
        except PlanError as e:
            return await _fallback_to_general_code(req, f"PlanError: {e}")
        code = gen_sql_code(sql, df_name)
        return {
            "success": True,
            "route": "sql",
            "confidence": confidence,
            "plan": plan,
            "sql": sql,
            "message": message,
            "code": code,
        }

    # route == "operation" (also the fallback for anything unexpected)
    try:
        action = await parse_agentic_command(req.text, req.available_columns, req.available_sheets)
    except Exception as e:
        print("[/generate_colab_code] EXCEPTION during operation parsing:")
        traceback.print_exc()
        return await _fallback_to_general_code(req, f"parse_agentic_command raised: {e}")

    if action.get("action") in (None, "unknown"):
        return await _fallback_to_general_code(
            req, f"command_agent returned unknown: {action.get('message')}"
        )

    code = gen_operation_code(action, df_name, req.available_columns)
    return {
        "success": True,
        "route": "operation",
        "confidence": action.get("confidence", confidence),
        "action": action,
        "message": action.get("message") or message,
        "code": code,
    }
