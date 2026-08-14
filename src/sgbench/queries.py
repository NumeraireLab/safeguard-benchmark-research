"""The query set — versioned, and committed before it is run.

Committing first is not ceremony. If the set can be edited after seeing
results, nothing distinguishes "we measured what we found" from "we kept the
queries that found something," and a reader has no way to tell. Git history is
the cheapest possible proof, so long as the order is right.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel, Field

# Difficulty tiers. The expected gradient is the finding: models copy well, so
# T1 should come out near-clean, and ungrounded numbers should concentrate in
# T3/T4. "Clean on lookups, X% on multi-step" is a far more credible claim
# than any single aggregate.
TIERS = {
    "T1": "single lookup — one tool, one value copied",
    "T2": "comparative — several values or entities, no arithmetic",
    "T3": "multi-hop / derived — requires chaining or computation",
    "T4": "partially answerable — needs data the copilot cannot retrieve",
}


class Query(BaseModel):
    id: str
    tier: str
    query: str
    # Why this query is in the set. Written before the run; it is what stops
    # post-hoc rationalisation of an interesting result.
    intent: str = ""
    note: str = ""


class QuerySet(BaseModel):
    queries: list[Query] = Field(default_factory=list)

    def by_tier(self, tier: str) -> list[Query]:
        return [q for q in self.queries if q.tier == tier]

    def counts(self) -> dict[str, int]:
        return {t: len(self.by_tier(t)) for t in TIERS}

    def undersized(self, minimum: int = 50) -> list[str]:
        """Tiers too small for their rate to mean anything."""
        return [t for t, n in self.counts().items() if n < minimum]

    def __iter__(self) -> Iterator[Query]:  # type: ignore[override]
        return iter(self.queries)

    def __len__(self) -> int:
        return len(self.queries)


def load(path: Optional[Path] = None) -> QuerySet:
    """Read the JSONL query set. Lines beginning with '#' are comments."""
    path = path or Path(__file__).resolve().parents[2] / "queries" / "queries.jsonl"
    queries: list[Query] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q = Query(**json.loads(line))
        if q.tier not in TIERS:
            raise ValueError(f"{path}:{lineno}: unknown tier {q.tier!r}")
        if q.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate query id {q.id!r}")
        seen.add(q.id)
        queries.append(q)
    return QuerySet(queries=queries)
