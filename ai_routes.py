# ai_routes.py
# ─────────────────────────────────────────────────────────────────────────────
# Drop this file into your repo root alongside app.py.
# Then add these 2 lines to the BOTTOM of your app.py:
#
#   from ai_routes import ai_router
#   app.include_router(ai_router)
#
# That's the ONLY change needed in app.py.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter
from pydantic import BaseModel

from ai_engine import parse_command, add_training_example, train_model
from memory import (
    log_command, get_recent_commands, get_command_stats,
    save_alias, resolve_alias, get_all_aliases, log_correction,
)

ai_router = APIRouter()


# ── Pydantic Models ───────────────────────────────────────────────────────────

class CommandRequest(BaseModel):
    text: str
    available_columns: list[str] = []

class LogResultRequest(BaseModel):
    text: str
    intent: str
    slots: dict = {}
    result: dict = {}
    success: bool = False
    confidence: float = 0.0

class AddTrainingRequest(BaseModel):
    text: str
    correct_intent: str
    slots: dict = {}
    wrong_intent: str = ""

class AliasRequest(BaseModel):
    alias: str
    real_name: str


# ── Routes ────────────────────────────────────────────────────────────────────

@ai_router.post("/parse_command")
async def parse_command_endpoint(req: CommandRequest):
    """
    Takes a natural language command, returns intent + slots.
    Flutter calls this, then executes the Excel action locally.
    """
    result = parse_command(req.text)

    # Resolve any learned column aliases from memory
    if "column" in result["slots"]:
        result["slots"]["column"] = resolve_alias(result["slots"]["column"])

    # Fuzzy-match column against columns currently in the workbook
    if "column" in result["slots"] and req.available_columns:
        col = result["slots"]["column"].lower()
        matches = [
            c for c in req.available_columns
            if col in c.lower() or c.lower() in col
        ]
        if matches:
            result["slots"]["column"] = matches[0]

    return result


@ai_router.post("/log_result")
async def log_result(req: LogResultRequest):
    """Flutter calls this after executing each action to build memory."""
    log_command(
        raw_text=req.text,
        intent=req.intent,
        slots=req.slots,
        result=req.result,
        success=req.success,
        confidence=req.confidence,
    )
    return {"logged": True}


@ai_router.get("/memory")
async def get_memory(limit: int = 30):
    """Returns recent command history + stats."""
    return {
        "history": get_recent_commands(limit),
        "stats":   get_command_stats(),
    }


@ai_router.post("/add_training")
async def add_training(req: AddTrainingRequest):
    """
    When the AI gets something wrong, submit the correction here.
    Automatically saves the example and retrains the model.
    """
    if req.wrong_intent:
        log_correction(
            original_text=req.text,
            wrong_intent=req.wrong_intent,
            correct_intent=req.correct_intent,
            slots=req.slots,
        )

    total = add_training_example(
        text=req.text,
        intent=req.correct_intent,
        slots=req.slots,
    )
    return {"added": True, "total_examples": total, "status": "retrained"}


@ai_router.post("/retrain")
async def retrain():
    """Force a full retrain from training_data.json."""
    train_model()
    return {"status": "retrained"}


@ai_router.post("/alias")
async def add_alias(req: AliasRequest):
    """Teach the AI that an alias points to a real column name."""
    save_alias(req.alias, req.real_name)
    return {"saved": True, "alias": req.alias, "real_name": req.real_name}


@ai_router.get("/aliases")
async def list_aliases():
    return {"aliases": get_all_aliases()}
