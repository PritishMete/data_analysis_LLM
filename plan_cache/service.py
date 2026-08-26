# plan_cache/service.py
# ─────────────────────────────────────────────────────────────────────────────
# This is where "the backend should eventually become smarter than the LLM
# by learning from previous executions" stops being an aspiration and
# becomes one concrete, callable method: find_cached_plan(). A caller (e.g.
# a future integration point in query_router.py or command_agent.py's
# Flutter-facing route) checks this BEFORE spending a Gemini call — if a
# prior identical question already succeeded against a dataset with the
# same shape, reuse that plan directly. Nothing in this package calls, nor
# is called by, the planner — see main.py/query_router.py for where a
# future integration would go; that wiring is deliberately NOT part of this
# change, so planner logic stays untouched.
#
# Cached along four dimensions, per candidate query_history row:
#   - intent               -> the broader match tier (find_by_intent)
#   - dataset schema hash  -> both tiers; the cross-dataset generalization
#   - planner_version      -> optional filter on the broader tier, so a
#                              plan produced by one planner version isn't
#                              silently reused under a different one
#   - confidence            -> computed HERE (deterministically, see
#                              _compute_confidence below), gating whether a
#                              matching candidate is trustworthy enough to
#                              actually serve as a HIT
#
# Every lookup resolves to exactly one PlanCacheOutcome:
#   HIT         - a sufficiently-confident, non-expired, non-invalidated
#                 candidate was found; `.hit` is populated.
#   MISS        - nothing matched at all, for any tier that was checked.
#   EXPIRED     - something matched, but every candidate is older than
#                 PLAN_CACHE_TTL_DAYS (see below).
#   INVALIDATED - something matched, wasn't expired, but every candidate has
#                 an active invalidation on file (see plan_cache/models.py).
# EXPIRED/INVALIDATED are checked, and reported, in preference to a bare
# MISS whenever they're the more accurate reason nothing was served — a
# caller (or an engineer looking at /v2/plan-cache/lookup) gets a real
# answer to "why didn't this hit" instead of an opaque no.
#
# Deliberately deterministic and narrow for this first version: EXACT text
# match is still the strict tier, and confidence is plain arithmetic over
# already-stored signals (success, feedback_score, recency) — no model, no
# training, no embeddings. Two honest reasons for that:
#   1. A wrong SQL reuse is worse than a redundant LLM call, so every tier
#      needs to be unambiguous to reason about, test, and trust before this
#      is wired into a real request path.
#   2. schema_hash equality means the CANDIDATE plan really did run
#      successfully against a dataset with identically-named, identically-
#      typed columns — the strongest deterministic signal available that
#      the same generated_sql/python_pipeline will still resolve correctly
#      against the new dataset.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from datasets.repository import DatasetRepository
from query_history.models import QueryHistory

from .models import PlanCacheInvalidation
from .repository import PlanCacheRepository

# How long a cached plan stays eligible for reuse after it was logged,
# regardless of confidence. Env-overridable (mirrors core/db.py's
# DATABASE_URL pattern) since the right value is an operational judgment
# call, not a code change — e.g. a business whose data/schemas change
# often might want this much shorter than a business with slow-moving,
# stable exports. Purely a cache-eligibility cutoff; it never touches or
# deletes the underlying query_history row, which is a permanent log.
PLAN_CACHE_TTL_DAYS = int(os.environ.get("PLAN_CACHE_TTL_DAYS", "30"))

# Below this confidence, a technically-matching candidate is treated as not
# good enough to serve — same rationale as the module docstring's point #1.
DEFAULT_MIN_CONFIDENCE = 0.5


class PlanCacheOutcome(str, Enum):
    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


@dataclass
class PlanCacheHit:
    generated_sql: str | None
    python_pipeline: dict | list | None
    intent: str | None
    planner_version: str | None
    confidence: float
    source_dataset_id: str
    matched_on: str  # "same_dataset" | "same_schema_shape" | "same_intent"
    original_query_history_id: int


@dataclass
class PlanCacheResult:
    """The full answer to "should this plan be reused" — not just the plan
    itself. `outcome` is always set; `hit` is populated only when
    `outcome is PlanCacheOutcome.HIT`. `detail` is a short, human-readable
    explanation for any non-HIT outcome (surfaced over HTTP in
    plan_cache/routes.py) so a MISS/EXPIRED/INVALIDATED response is
    actionable, not just an opaque no."""

    outcome: PlanCacheOutcome
    hit: PlanCacheHit | None = None
    detail: str | None = None

    @property
    def is_hit(self) -> bool:
        return self.outcome is PlanCacheOutcome.HIT


class PlanCacheService:
    def __init__(self, dataset_repository: DatasetRepository, plan_cache_repository: PlanCacheRepository):
        self.dataset_repository = dataset_repository
        self.plan_cache_repository = plan_cache_repository

    # ── Public API ──────────────────────────────────────────────────────

    def find_cached_plan(
        self,
        *,
        dataset_id: str,
        user_query: str,
        intent: str | None = None,
        planner_version: str | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> PlanCacheHit | None:
        """Unchanged shape from the original version of this method — still
        returns just the hit, or None — for any existing caller that only
        ever cared about that. `intent` and `planner_version` are new,
        optional: omit them and behavior is identical to before (exact
        query-text match only). Pass `intent` to also allow the broader,
        same-intent/same-schema/same-planner-version tier described in the
        module docstring. This is a thin convenience wrapper around
        evaluate() — see that method for the full HIT/MISS/EXPIRED/
        INVALIDATED picture.
        """
        return self.evaluate(
            dataset_id=dataset_id,
            user_query=user_query,
            intent=intent,
            planner_version=planner_version,
            min_confidence=min_confidence,
        ).hit

    def evaluate(
        self,
        *,
        dataset_id: str,
        user_query: str,
        intent: str | None = None,
        planner_version: str | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> PlanCacheResult:
        """Full cache decision: which of HIT / MISS / EXPIRED / INVALIDATED
        applies, and why. `dataset_id` must already be registered (via the
        Dataset Registry) so its schema_hash is known — datasets that were
        never registered simply can't participate in reuse yet, and this
        returns a plain MISS for them (there was never anything to be
        invalidated or expired)."""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if dataset is None:
            return PlanCacheResult(PlanCacheOutcome.MISS, detail="dataset is not registered")
        schema_hash = dataset.schema_hash

        now = datetime.now(timezone.utc)
        ttl_cutoff = now - timedelta(days=PLAN_CACHE_TTL_DAYS)
        scope_rules = self.plan_cache_repository.scope_invalidations(schema_hash=schema_hash)

        # Tier 1: exact user_query match (unchanged from the original
        # version of this cache).
        tier1_raw = self.plan_cache_repository.find_reusable_plan(schema_hash=schema_hash, user_query=user_query)
        tier1_status, tier1_best = self._classify(tier1_raw, ttl_cutoff, scope_rules)
        if tier1_best is not None:
            matched_on = "same_dataset" if tier1_best.dataset_id == dataset_id else "same_schema_shape"
            confidence = self._compute_confidence(tier1_best, tier="exact_query", now=now)
            if confidence >= min_confidence:
                return PlanCacheResult(PlanCacheOutcome.HIT, hit=self._to_hit(tier1_best, matched_on, confidence))

        # Tier 2: broader (intent, schema_hash[, planner_version]) match —
        # only attempted when the caller actually supplied an intent, since
        # without one there's no signal to broaden the match on.
        tier2_status, tier2_best = None, None
        if intent is not None:
            tier2_raw = self.plan_cache_repository.find_by_intent(
                schema_hash=schema_hash, intent=intent, planner_version=planner_version
            )
            tier2_status, tier2_best = self._classify(tier2_raw, ttl_cutoff, scope_rules)
            if tier2_best is not None:
                # Always "same_intent", never "same_dataset" — dataset
                # identity is incidental here. Tier 2 exists specifically to
                # match past a different exact wording via intent, so the
                # reported reason should say that, even when the matched
                # row also happens to belong to the same dataset (tier 1's
                # exact-text match already covers that "same_dataset" case;
                # if we're in tier 2 at all, intent is why this hit).
                matched_on = "same_intent"
                confidence = self._compute_confidence(tier2_best, tier="same_intent", now=now)
                if confidence >= min_confidence:
                    return PlanCacheResult(PlanCacheOutcome.HIT, hit=self._to_hit(tier2_best, matched_on, confidence))

        return self._combine_miss(tier1_status, tier2_status)

    def invalidate_plan(self, *, query_history_id: int, reason: str | None = None) -> PlanCacheInvalidation:
        """Mark one specific cached plan (by its source query_history row
        id) as no longer reusable. Does not touch the query_history row
        itself — see plan_cache/models.py's module docstring for why."""
        return self.plan_cache_repository.invalidate_query(query_history_id=query_history_id, reason=reason)

    def invalidate_scope(
        self,
        *,
        dataset_id: str,
        intent: str | None = None,
        planner_version: str | None = None,
        reason: str | None = None,
    ) -> PlanCacheInvalidation:
        """Mark every plan currently cached under this dataset's schema
        (optionally narrowed to one intent and/or planner_version) as no
        longer reusable. A fresh, successful execution logged AFTER this
        call is unaffected — same as any cache repopulating after a flush.
        Raises ValueError for an unregistered dataset_id, since there's no
        schema_hash to scope the invalidation to."""
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if dataset is None:
            raise ValueError(f"dataset {dataset_id!r} is not registered")
        return self.plan_cache_repository.invalidate_scope(
            schema_hash=dataset.schema_hash, intent=intent, planner_version=planner_version, reason=reason
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _classify(
        self,
        raw_candidates: list[QueryHistory],
        ttl_cutoff: datetime,
        scope_rules: list[PlanCacheInvalidation],
    ) -> tuple[str, QueryHistory | None]:
        """Reduces a raw candidate list down to (status, best_usable_entry).
        status is one of "no_candidates" | "invalidated" | "expired" | "ok"
        — the three failure statuses let evaluate() explain a non-hit
        precisely instead of just saying MISS for every reason. Ordering
        matters: `raw_candidates` is already sorted most-recent-first, so
        the first entry surviving both filters is the best available."""
        if not raw_candidates:
            return "no_candidates", None

        invalidated_ids = self.plan_cache_repository.invalidated_ids_among([c.id for c in raw_candidates])
        not_invalidated = [
            c for c in raw_candidates if not self._is_invalidated(c, invalidated_ids, scope_rules)
        ]
        if not not_invalidated:
            return "invalidated", None

        fresh = [c for c in not_invalidated if self._as_aware(c.created_at) >= ttl_cutoff]
        if not fresh:
            return "expired", None

        return "ok", fresh[0]

    @staticmethod
    def _is_invalidated(
        entry: QueryHistory, invalidated_ids: set[int], scope_rules: list[PlanCacheInvalidation]
    ) -> bool:
        if entry.id in invalidated_ids:
            return True
        entry_created_at = PlanCacheService._as_aware(entry.created_at)
        for rule in scope_rules:
            if rule.intent is not None and rule.intent != entry.intent:
                continue
            if rule.planner_version is not None and rule.planner_version != entry.planner_version:
                continue
            if entry_created_at <= PlanCacheService._as_aware(rule.invalidated_at):
                return True
        return False

    @staticmethod
    def _as_aware(value: datetime) -> datetime:
        """SQLite doesn't reliably round-trip timezone-aware datetimes, so a
        value read back from it can come back naive even though it was
        written as UTC (see _utcnow() in the models). Treat any naive
        datetime here as UTC rather than letting the tz-aware/naive
        comparison above raise."""
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    @staticmethod
    def _combine_miss(tier1_status: str | None, tier2_status: str | None) -> PlanCacheResult:
        statuses = {s for s in (tier1_status, tier2_status) if s is not None}
        if "invalidated" in statuses:
            return PlanCacheResult(PlanCacheOutcome.INVALIDATED, detail="matching plan(s) found but invalidated")
        if "expired" in statuses:
            return PlanCacheResult(
                PlanCacheOutcome.EXPIRED, detail=f"matching plan(s) found but older than {PLAN_CACHE_TTL_DAYS}d TTL"
            )
        return PlanCacheResult(PlanCacheOutcome.MISS, detail="no matching plan found")

    @staticmethod
    def _compute_confidence(entry: QueryHistory, *, tier: str, now: datetime) -> float:
        """Deterministic arithmetic over already-stored signals — no model,
        no training (see module docstring). Three inputs:
          - tier: an exact user_query match is inherently stronger evidence
            than an intent-level match with different wording.
          - feedback_score: a human already told us this specific plan was
            good (or bad) — the strongest available signal, when present.
          - recency: a linear decay across the TTL window, so a
            just-logged plan is trusted slightly more than one about to
            expire, without a hard cliff at the TTL boundary.
        """
        base = 0.9 if tier == "exact_query" else 0.6

        if entry.feedback_score is not None:
            if entry.feedback_score > 0:
                base += 0.1
            elif entry.feedback_score < 0:
                # Deliberately well clear of DEFAULT_MIN_CONFIDENCE (0.9 base
                # - 0.45 = 0.45 < 0.5), not just barely under it — a human
                # already flagged this plan as bad, and a penalty that only
                # ties the confidence threshold would let recency-decay
                # rounding noise flip a bad plan back into a HIT.
                base -= 0.45

        created_at = PlanCacheService._as_aware(entry.created_at)
        age_days = max((now - created_at).total_seconds() / 86400, 0.0)
        age_fraction = min(age_days / max(PLAN_CACHE_TTL_DAYS, 1), 1.0)
        recency_factor = 1.0 - (0.15 * age_fraction)

        return round(max(0.0, min(1.0, base * recency_factor)), 4)

    @staticmethod
    def _to_hit(entry: QueryHistory, matched_on: str, confidence: float) -> PlanCacheHit:
        return PlanCacheHit(
            generated_sql=entry.generated_sql,
            python_pipeline=entry.python_pipeline,
            intent=entry.intent,
            planner_version=entry.planner_version,
            confidence=confidence,
            source_dataset_id=entry.dataset_id,
            matched_on=matched_on,
            original_query_history_id=entry.id,
        )
