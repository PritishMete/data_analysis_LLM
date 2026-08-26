# core/db.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared SQLAlchemy engine/session/Base for every NEW backend feature
# (dataset registry, schema intelligence, query history, and anything added
# after them). Nothing in the existing codebase (main.py, cleaning_agent.py,
# query_router.py, data_cleaner.py, ai_engine.py, ...) is touched by this —
# it's a brand-new, independent persistence layer that those modules don't
# need to know exists.
#
# DATABASE_URL:
#   - Not set                -> local SQLite file `enterprise_registry.db`
#     in the working directory. Fine for local dev; do NOT rely on this in
#     production on Render, since its filesystem is ephemeral across deploys
#     (ai_engine.py's own comment about /tmp being cleared already flags
#     this same constraint for the intent model).
#   - Set (e.g. to a managed Postgres URL from Render/Neon/Supabase)
#     -> used as-is. This is what real deployments should do; the registry
#     is meant to persist indefinitely, unlike the /tmp-based intent model.
#
# Every feature package imports `Base` from here and defines its models
# against it, so `init_db()` (called once, at process startup) creates
# every table in one place without any package needing to import another
# package's models directly.
# ─────────────────────────────────────────────────────────────────────────────

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./enterprise_registry.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Shared declarative base. Every new feature's models.py subclasses this
    (never a package-local Base), so init_db() below sees every table."""
    pass


def get_db():
    """FastAPI dependency — yields a request-scoped Session, always closed
    afterward even if the request raises. Use as:

        @router.post("/x")
        def endpoint(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates every table registered on `Base.metadata`. Call ONCE at app
    startup (see INTEGRATION.md) — safe to call repeatedly, it's a no-op for
    tables that already exist. Must run AFTER all feature models.py modules
    have been imported at least once (importing the routers below already
    guarantees this, since each router module imports its own models).
    """
    # Import every feature's models so their tables register on Base.metadata
    # before create_all runs. Deliberately done here rather than at module
    # top-level to avoid import-order/circular-import surprises.
    from datasets import models as _dataset_models          # noqa: F401
    from schema_intelligence import models as _schema_models  # noqa: F401
    from query_history import models as _history_models       # noqa: F401
    from plan_cache import models as _plan_cache_models        # noqa: F401

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """`create_all` (above) only CREATES tables that don't exist yet — it
    never ALTERs a table that's already there, so a column added to a
    models.py after a database has already been deployed (e.g.
    `query_history.planner_version`) would silently never show up on an
    existing Render/Postgres/on-disk-SQLite database. This project has no
    Alembic (or other) migration tooling, so this is the minimal, additive
    substitute: for every table already on Base.metadata, add any column
    that's declared on the model but missing from the live table, as a
    NULLable ADD COLUMN. Nullable-only is deliberate — that's the only kind
    of ALTER that's always safe to run against a table that may already
    have rows, on both SQLite and Postgres, with zero risk to existing data
    or existing queries. Safe to call on every startup: columns that
    already exist are simply skipped.
    """
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue  # brand-new table — create_all already built it in full
        existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue
            if not column.nullable:
                # Not safe to auto-add a NOT NULL column to a table that may
                # already have rows — skip; this should be a real migration.
                continue
            ddl_type = column.type.compile(dialect=engine.dialect)
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {ddl_type}'))
