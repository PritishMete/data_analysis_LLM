from __future__ import annotations

from fastapi import APIRouter

from agent.orchestrator import get_agentic_orchestrator

learning_router = APIRouter(prefix="/v2/learning", tags=["learning"])


@learning_router.get("/skills")
async def list_skills():
    orchestrator = get_agentic_orchestrator()
    registry = orchestrator.registry
    return {
        "skills": [
            {
                **spec.to_dict(),
                "effective_confidence": registry.effective_confidence(spec.id),
                "state": registry.state_for(spec.id).to_dict(),
            }
            for spec in registry.all()
        ]
    }


@learning_router.get("/experiences")
async def list_experiences(limit: int = 50):
    orchestrator = get_agentic_orchestrator()
    return {"experiences": orchestrator.store.load_recent(limit=limit)}


@learning_router.get("/failure-lessons")
async def list_failure_lessons(limit: int = 50):
    orchestrator = get_agentic_orchestrator()
    return {"failure_lessons": orchestrator.store.load_failure_lessons(limit=limit)}


@learning_router.get("/candidate-strategies")
async def list_candidate_strategies(limit: int = 50):
    orchestrator = get_agentic_orchestrator()
    return {"candidate_strategies": orchestrator.store.load_candidate_strategies(limit=limit)}


@learning_router.post("/plan")
async def plan_query(payload: dict):
    orchestrator = get_agentic_orchestrator()
    user_text = str(payload.get("text") or "")
    available_columns = payload.get("available_columns") or []
    decision = orchestrator.plan(user_text, available_columns=available_columns)
    return decision.to_dict()
