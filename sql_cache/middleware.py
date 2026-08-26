# sql_cache/middleware.py
# ─────────────────────────────────────────────────────────────────────────────
# Wraps SqlCacheService as ASGI middleware, per the requirement. This is
# deliberately NOT a change to any route's body — it sits in front of the
# entire app, matching requests purely by HTTP method + URL path. A cache
# MISS is a complete no-op: `call_next(request)` runs the existing route
# exactly as if this middleware didn't exist, which is what makes it safe to
# add without touching command_agent.py, query_router.py, or any route
# function's own code ("do not change existing planner logic").
#
# On a HIT (similarity >= min_confidence), the downstream route — and
# therefore whatever Gemini call it would have made — is NEVER invoked at
# all. This happens at the ASGI layer, one level above FastAPI's own request
# handling, which is the strongest place to guarantee "before calling
# Gemini": the planner code doesn't just get skipped by an early return
# inside itself, it never runs.
# ─────────────────────────────────────────────────────────────────────────────

import json
import traceback

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from core.db import SessionLocal
from query_history.repository import QueryHistoryRepository

from .contracts import SimilarityStrategy
from .multipart_utils import extract_boundary, extract_text_field
from .service import SqlCacheService


class SqlCacheMiddleware(BaseHTTPMiddleware):
    """Modular/replaceable at the wiring level too: which paths are watched,
    which JSON fields carry the query text/dataset scope, which
    SimilarityStrategy is used, and the confidence threshold are ALL
    constructor arguments — see main.py's `app.add_middleware(...)` call.
    Changing any of them means changing that one call, not this class.

    Handles BOTH request shapes actually used in this project:
      - application/json (e.g. /agentic_command) — parsed directly.
      - multipart/form-data (e.g. /smart_query, which uploads a file
        alongside a `text` form field) — the query text is extracted from
        the RAW body bytes via sql_cache/multipart_utils.py, which never
        calls Starlette's `request.form()`. That distinction matters:
        empirically, calling `.form()` here breaks the SAME request's
        downstream FastAPI File()/Form() parsing entirely (verified while
        building this — both fields came back missing). Reading raw bytes
        via `request.body()` has no such problem and leaves the uploaded
        file's bytes completely unmodified for the real route to read.
    """

    def __init__(
        self,
        app,
        *,
        watched_paths: tuple[str, ...] = ("/agentic_command",),
        text_field: str = "text",
        dataset_id_field: str = "dataset_id",
        organization_id_field: str = "organization_id",
        similarity_strategy: SimilarityStrategy | None = None,
        min_confidence: float = 0.95,
    ):
        super().__init__(app)
        self.watched_paths = set(watched_paths)
        self.text_field = text_field
        self.dataset_id_field = dataset_id_field
        self.organization_id_field = organization_id_field
        self.similarity_strategy = similarity_strategy
        self.min_confidence = min_confidence

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method != "POST" or request.url.path not in self.watched_paths:
            return await call_next(request)

        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            extracted = await self._extract_from_json(request)
        elif "multipart/form-data" in content_type:
            extracted = await self._extract_from_multipart(request, content_type)
        else:
            return await call_next(request)  # unsupported content-type — pass through

        if extracted is None:
            return await call_next(request)
        user_query, dataset_id, organization_id = extracted
        if not user_query:
            return await call_next(request)

        hit = self._lookup(user_query=user_query, dataset_id=dataset_id, organization_id=organization_id)
        if hit is None:
            return await call_next(request)

        return JSONResponse(self._build_response_body(hit))

    async def _extract_from_json(self, request: Request):
        try:
            body_bytes = await request.body()
            payload = json.loads(body_bytes) if body_bytes else {}
        except Exception:
            return None  # malformed body — let the real route reject it normally
        return (
            payload.get(self.text_field),
            payload.get(self.dataset_id_field),
            payload.get(self.organization_id_field),
        )

    async def _extract_from_multipart(self, request: Request, content_type: str):
        boundary = extract_boundary(content_type)
        if boundary is None:
            return None
        try:
            raw_body = await request.body()  # RAW bytes only — see class docstring
        except Exception:
            return None
        user_query = extract_text_field(raw_body, boundary, self.text_field)
        dataset_id = extract_text_field(raw_body, boundary, self.dataset_id_field)
        organization_id = extract_text_field(raw_body, boundary, self.organization_id_field)
        return user_query, dataset_id, organization_id

    def _lookup(self, *, user_query: str, dataset_id, organization_id):
        db = SessionLocal()
        try:
            service = SqlCacheService(
                QueryHistoryRepository(db),
                similarity_strategy=self.similarity_strategy,
                min_confidence=self.min_confidence,
            )
            return service.find_similar_cached_query(
                user_query=user_query, dataset_id=dataset_id, organization_id=organization_id
            )
        except Exception:
            # A broken cache lookup must NEVER be able to block or corrupt
            # the real request — log it and fall through to a normal miss.
            traceback.print_exc()
            return None
        finally:
            db.close()

    @staticmethod
    def _build_response_body(hit) -> dict:
        body: dict = dict(hit.python_pipeline) if isinstance(hit.python_pipeline, dict) else {}
        body["generated_sql"] = hit.generated_sql
        body["intent"] = hit.intent
        body["planner_version"] = hit.planner_version
        existing_message = body.get("message", "") or ""
        body["message"] = (
            f"{existing_message} (reused via SQL Cache — {hit.similarity_score:.0%} similar to a "
            f"previous query — no AI call made)"
        ).strip()
        body["_sql_cache_hit"] = True
        body["_similarity_score"] = hit.similarity_score
        body["_matched_query"] = hit.matched_query
        return body
