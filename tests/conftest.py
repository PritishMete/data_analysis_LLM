# tests/conftest.py
# ─────────────────────────────────────────────────────────────────────────────
# Every test gets a FRESH in-memory SQLite database (StaticPool keeps the
# same connection alive for the whole test so the in-memory DB isn't lost
# between statements) — real SQLAlchemy models, real queries, zero mocking
# of the persistence layer, but with no dependency on Render/Postgres/disk
# state. This is what test_dataset_registry.py etc. depend on.
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure the backend_ext package root is importable when running `pytest`
# from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import Base  # noqa: E402


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Import every feature's models so they register on Base.metadata,
    # mirroring core.db.init_db()'s behavior for production startup.
    import datasets.models  # noqa: F401
    import query_history.models  # noqa: F401
    import schema_intelligence.models  # noqa: F401
    import plan_cache.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
