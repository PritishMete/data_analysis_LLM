# colab_code_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# command_agent.py and query_router.py both deliberately only understand a
# fixed taxonomy: six spreadsheet operations, or a SQL-shaped analytical
# plan. That's correct for Excel, where every action has to map onto a real
# Office.js call — there's no "just write me some code" option there.
#
# Colab has no such constraint: it's a real Python kernel, so "write code
# to read a csv file", "plot this as a bar chart", "make a function that
# does X" are all completely reasonable requests that the structured
# planners will correctly refuse (they SHOULD say "unknown" for these —
# that's not a bug in them). This module is the fallback for exactly that
# case: a general-purpose code-writing agent, same google-adk pattern as
# command_agent.py, but with no fixed output schema — it just writes code.
# ─────────────────────────────────────────────────────────────────────────────

import re
import traceback
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = """You are a Python code-writing assistant embedded in a Google Colab
notebook, via a Chrome extension. The user describes what they want in plain English; you
write clean, directly-runnable Python code for a Colab cell.

Context you're given on every request:
- The variable name of the user's current DataFrame (may not exist yet, e.g. if they're asking
  to read a file for the first time — in that case, write code that CREATES it under that name).
- The list of column names in that DataFrame, if known (may be empty — the user may not have
  loaded data yet, or may be asking for something unrelated to their current DataFrame).

Guidelines:
- Write idiomatic pandas / numpy / matplotlib / standard-library code as appropriate to the
  request. Prefer pandas for tabular data, matplotlib for plots, standard library for general
  scripting.
- Reference the user's actual DataFrame variable name when the request concerns their data.
  Never invent a different variable name for it.
- If the request needs a file path you don't know (e.g. "read a csv file"), use a clearly
  labeled placeholder like 'your_file.csv' and add a short comment saying to update the path —
  don't invent a specific-sounding fake filename.
- Add brief '#' comments only where they genuinely aid understanding — don't over-comment.
- Do NOT wrap the output in markdown code fences (no ```python). Output ONLY the raw code —
  no leading/trailing prose, no "Here's the code:" preamble, no explanation after.
- If the request is genuinely impossible to turn into code (e.g. it's not a coding request at
  all, or is asking about something with no code-shaped answer), output a single Python comment
  line starting with '#' that briefly explains why, instead of code.
"""


def _strip_code_fences(text: str) -> str:
    """Defensive cleanup in case the model wraps the response in ```python
    fences despite being told not to — mirrors the _extract_json pattern
    already used in command_agent.py / query_router.py for the same reason.
    """
    text = text.strip()
    fence_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


async def generate_general_code(
    user_text: str,
    dataframe_name: str,
    available_columns: list,
) -> dict:
    """Writes arbitrary Python code for a request that doesn't fit the
    structured spreadsheet-operation or SQL-question taxonomies. Returns
    {"code": <str or None>, "message": <str>}.
    """
    try:
        agent = LlmAgent(
            name="colab_code_agent",
            model=MODEL,
            instruction=SYSTEM_INSTRUCTION,
            description="Writes general-purpose Python code for a Colab cell.",
        )

        app_name = "colab_code_agent_app"
        user_id = "api_user"
        session_id = str(uuid.uuid4())

        session_service = InMemorySessionService()
        await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
        runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

        prompt = (
            f"Current DataFrame variable name: {dataframe_name}\n"
            f"Its columns (if known): {available_columns}\n"
            f"User request: {user_text}"
        )
        content = types.Content(role="user", parts=[types.Part(text=prompt)])

        final_text = None
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        final_text = part.text

        print(f"[colab_code_agent] raw model output: {final_text!r}")

        if not final_text:
            return {"code": None, "message": "No response from the code-writing agent."}

        code = _strip_code_fences(final_text)
        return {"code": code, "message": "Generated general-purpose Python code."}

    except Exception:
        print("[colab_code_agent] EXCEPTION during generate_general_code:")
        traceback.print_exc()
        return {"code": None, "message": "Internal error while writing code — check server logs."}
