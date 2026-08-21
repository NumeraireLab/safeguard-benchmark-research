"""Capture and verification, kept as **separate** steps.

This split is the most important design decision in the harness.

*Capture* costs money, needs network and API keys, and is non-deterministic —
the same query re-run gives a different answer, so a capture can never be
reproduced. *Verification* is free, offline and deterministic.

Keeping them separate means the expensive artefact (`triples.jsonl`) is
captured **once** and can be re-verified any number of times: when a §4.1 gap
is fixed, re-verify the same recorded runs and compare rates directly. Fusing
the steps would mean re-querying the copilot after every checker change, which
is both unaffordable and unsound — the comparison would confound the fix with a
fresh sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from safeguard.core import FieldMap, Guard, JSONLStorage

from sgbench.adapters.langgraph import triple_from_messages
from sgbench.capture import Triple
from sgbench.queries import Query, QuerySet
from sgbench.verify import Outcome, target_prompt_text, verify_triple


def load_dotenv(path: Optional[Path] = None) -> list[str]:
    """Read KEY=VALUE lines from .env into the environment.

    Capture needs an API key, and a key exported in one shell does not survive
    into the next. A gitignored .env is the one place it can live that is both
    persistent and not committable. Existing environment variables always win.
    """
    import os

    path = path or Path(__file__).resolve().parents[2] / ".env"
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


# A target turns a query into a message list. Keeping it a plain callable is
# what lets the runner stay agnostic while the OpenBB adapter is unverified.
Target = Callable[[str], Any]


def capture(
    queries: Iterable[Query],
    target: Target,
    out_path: Path,
    model: Optional[str] = None,
    target_name: str = "unknown",
    on_error: str = "record",
) -> list[Triple]:
    """Run each query against the target and append one Triple per line.

    Failures are *recorded*, not dropped. A query that errored is not the same
    as a query that produced a clean answer, and silently discarding them
    would bias the denominator.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    triples: list[Triple] = []
    with out_path.open("a", encoding="utf-8") as fh:
        for q in queries:
            try:
                messages = target(q.query)
                triple = triple_from_messages(
                    messages,
                    query_id=q.id,
                    tier=q.tier,
                    model=model,
                    target=target_name,
                )
                # The adapter reads the query from the transcript; prefer the
                # query set's text, which is what was actually asked for.
                triple.query = q.query or triple.query
            except Exception as exc:  # noqa: BLE001 - capture must not abort
                if on_error == "raise":
                    raise
                triple = Triple(
                    query=q.query, query_id=q.id, tier=q.tier,
                    model=model, target=target_name,
                    answer=f"[CAPTURE ERROR] {type(exc).__name__}: {exc}",
                )
            fh.write(triple.model_dump_json() + "\n")
            triples.append(triple)
    return triples


def load_triples(path: Path) -> list[Triple]:
    return [
        Triple.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_all(
    triples: Iterable[Triple],
    audit_path: Path,
    field_map: Optional[FieldMap] = None,
    derivations: Optional[object] = None,
) -> list[Outcome]:
    """Verify captured triples, persisting every audit record.

    Persistence is not incidental: the records *are* the harvest. Each one is a
    dated, inspectable artefact — the difference between telling a prospect a
    failure happened and showing them the evidence.
    """
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    guard = Guard(storage=JSONLStorage(audit_path), field_map=field_map)
    # Read once: the same prompt and tool descriptions the run was captured
    # under, so a number the model quoted from its own configuration traces.
    prompt_text = target_prompt_text(audit_path.parent / "target-provenance.json")
    return [verify_triple(t, guard=guard, prompt_text=prompt_text,
                          derivations=derivations)
            for t in triples]


def write_outcomes(outcomes: Iterable[Outcome], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(o.model_dump_json() + "\n")


def fixture_target(fixtures_dir: Path) -> Target:
    """Replay recorded turns instead of calling a live copilot.

    Lets the whole pipeline be exercised end to end with no keys and no spend,
    and is also how a captured real failure gets replayed in a pitch.
    """
    def _target(query: str) -> Any:
        for path in sorted(fixtures_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = data.get("messages", data)
            for msg in messages:
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str) and content.strip() == query.strip():
                    return messages
        raise LookupError(f"No fixture for query: {query!r}")
    return _target


def preflight(qs: QuerySet, minimum: int = 50) -> list[str]:
    """Warnings worth seeing before spending money on a capture run."""
    warnings = []
    small = qs.undersized(minimum)
    if small:
        warnings.append(
            f"tiers below n={minimum} (rates will not be meaningful): "
            f"{', '.join(small)} — counts {qs.counts()}"
        )
    if not qs.queries:
        warnings.append("query set is empty")
    return warnings
