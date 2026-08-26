# tests/test_sql_cache_middleware.py
# ─────────────────────────────────────────────────────────────────────────────
# Exercises SqlCacheMiddleware through a REAL FastAPI app + TestClient
# (not just the service class directly), because the interesting risk here
# is entirely about the ASGI/request lifecycle — does reading the body in
# middleware break the downstream route, does a miss really fall through
# unchanged, does a hit really never reach the route at all. A pure unit
# test of SqlCacheService can't exercise any of that.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import sql_cache.middleware as sql_cache_middleware
from query_history.repository import QueryHistoryRepository
from query_history.service import QueryHistoryService
from sql_cache.middleware import SqlCacheMiddleware


def _build_app(**middleware_kwargs) -> FastAPI:
    app = FastAPI()
    app.add_middleware(SqlCacheMiddleware, **middleware_kwargs)

    downstream_calls = {"count": 0}

    # Accepts the raw Request (not a pydantic `dict` body model) so this
    # route never triggers FastAPI's OWN body-validation rejections —
    # keeps every test here isolated to the MIDDLEWARE's behavior, not
    # FastAPI's separate (and irrelevant, for this purpose) request
    # validation layer.
    @app.post("/agentic_command")
    async def agentic_command(request: Request):
        downstream_calls["count"] += 1
        # Stand-in for the real route, which would call Gemini here.
        return {"action": "unknown", "confidence": 0.0, "message": "Called Gemini (simulated)."}

    app.state.downstream_calls = downstream_calls
    return app


def _patch_session(monkeypatch, db_session):
    # middleware.py did `from core.db import SessionLocal` at import time,
    # binding that name into ITS OWN module namespace — patching
    # core.db.SessionLocal afterward would NOT affect that already-bound
    # reference. Patch the name where middleware.py actually looks it up.
    monkeypatch.setattr(sql_cache_middleware, "SessionLocal", lambda: db_session)


def test_miss_falls_through_to_the_real_route_unchanged(db_session, monkeypatch):
    _patch_session(monkeypatch, db_session)

    app = _build_app(min_confidence=0.95)
    client = TestClient(app)

    resp = client.post("/agentic_command", json={"text": "a totally new question never seen before"})
    assert resp.status_code == 200
    assert resp.json()["message"] == "Called Gemini (simulated)."
    assert app.state.downstream_calls["count"] == 1  # the real route DID run


def test_hit_never_reaches_the_downstream_route_at_all(db_session, monkeypatch):
    _patch_session(monkeypatch, db_session)

    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    history_service.log_execution(
        user_query="total revenue by region",
        dataset_id=None,
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        intent="aggregate",
        success=True,
    )

    app = _build_app(min_confidence=0.95)
    client = TestClient(app)

    resp = client.post("/agentic_command", json={"text": "total revenue by region"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["_sql_cache_hit"] is True
    assert body["generated_sql"] == "SELECT region, SUM(revenue) FROM data GROUP BY region"
    assert body["_similarity_score"] == 1.0
    assert "no AI call made" in body["message"]

    # The critical assertion: the downstream route (where Gemini would be
    # called) never ran at all.
    assert app.state.downstream_calls["count"] == 0


def test_non_json_content_type_is_passed_through_without_inspection(db_session, monkeypatch):
    _patch_session(monkeypatch, db_session)

    # Even with a matching dataset/query already cached, a request whose
    # Content-Type isn't application/json must be passed straight through —
    # the middleware should never attempt to parse it as JSON at all.
    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    history_service.log_execution(user_query="irrelevant", success=True, generated_sql="SELECT 1")

    app = _build_app(min_confidence=0.95)
    client = TestClient(app)

    resp = client.post(
        "/agentic_command",
        content=b"text=irrelevant",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200
    assert resp.json()["message"] == "Called Gemini (simulated)."
    assert app.state.downstream_calls["count"] == 1  # fell straight through, untouched


def test_malformed_json_body_falls_through_safely(db_session, monkeypatch):
    _patch_session(monkeypatch, db_session)

    app = _build_app(min_confidence=0.95)
    client = TestClient(app)

    resp = client.post(
        "/agentic_command",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    # A malformed JSON body must not crash the middleware — it falls
    # through to the real route, which (in this test double) happily
    # accepts any raw Request regardless of body content.
    assert resp.status_code == 200
    assert app.state.downstream_calls["count"] == 1


def test_unwatched_path_is_never_touched(db_session, monkeypatch):
    _patch_session(monkeypatch, db_session)

    app = _build_app(watched_paths=("/agentic_command",), min_confidence=0.95)

    @app.post("/other_endpoint")
    async def other_endpoint(request: Request):
        app.state.downstream_calls["count"] += 1
        return {"ok": True}

    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    history_service.log_execution(user_query="some query", success=True, generated_sql="SELECT 1")

    client = TestClient(app)
    resp = client.post("/other_endpoint", json={"text": "some query"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}  # NOT the cache's response shape
    assert app.state.downstream_calls["count"] == 1


# ── Multipart (/smart_query-shaped) requests ────────────────────────────────

def _build_multipart_app(**middleware_kwargs) -> FastAPI:
    from fastapi import File, Form, UploadFile

    app = FastAPI()
    app.add_middleware(SqlCacheMiddleware, **middleware_kwargs)

    downstream_calls = {"count": 0, "received_file_bytes": None}

    @app.post("/smart_query")
    async def smart_query(file: UploadFile = File(...), text: str = Form(...)):
        downstream_calls["count"] += 1
        content = await file.read()
        downstream_calls["received_file_bytes"] = content
        return {"summary": f"Ran Gemini-generated SQL for: {text}", "row_count": 2}

    app.state.downstream_calls = downstream_calls
    return app


def test_multipart_miss_reaches_route_with_file_bytes_completely_intact(db_session, monkeypatch):
    _patch_session(monkeypatch, db_session)

    app = _build_multipart_app(watched_paths=("/smart_query",), min_confidence=0.95)
    client = TestClient(app)

    original_file_bytes = b"Region,Revenue\nNorth,100\nSouth,200\n"
    resp = client.post(
        "/smart_query",
        files={"file": ("sales.csv", original_file_bytes, "text/csv")},
        data={"text": "a brand new question never asked before"},
    )

    assert resp.status_code == 200
    assert app.state.downstream_calls["count"] == 1
    # The critical assertion: the uploaded file's bytes reached the real
    # route COMPLETELY UNCHANGED after passing through the middleware.
    assert app.state.downstream_calls["received_file_bytes"] == original_file_bytes


def test_multipart_hit_returns_cached_result_and_never_reaches_the_route(db_session, monkeypatch):
    _patch_session(monkeypatch, db_session)

    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    history_service.log_execution(
        user_query="total revenue by region",
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        intent="aggregate",
        success=True,
    )

    app = _build_multipart_app(watched_paths=("/smart_query",), min_confidence=0.95)
    client = TestClient(app)

    resp = client.post(
        "/smart_query",
        files={"file": ("sales.csv", b"Region,Revenue\nNorth,100\n", "text/csv")},
        data={"text": "total revenue by region"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["_sql_cache_hit"] is True
    assert body["generated_sql"] == "SELECT region, SUM(revenue) FROM data GROUP BY region"
    assert app.state.downstream_calls["count"] == 0  # route (and any Gemini call) never ran


def test_multipart_similar_but_not_identical_question_still_hits(db_session, monkeypatch):
    _patch_session(monkeypatch, db_session)

    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    history_service.log_execution(
        user_query="total revenue by region",
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        success=True,
    )

    app = _build_multipart_app(watched_paths=("/smart_query",), min_confidence=0.95)
    client = TestClient(app)

    resp = client.post(
        "/smart_query",
        files={"file": ("sales.csv", b"Region,Revenue\nNorth,100\n", "text/csv")},
        data={"text": "total revenue by region  "},  # trailing whitespace only
    )
    body = resp.json()
    assert body["_sql_cache_hit"] is True
    assert body["_similarity_score"] >= 0.95
