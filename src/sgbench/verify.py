"""Run Safeguard over a captured triple.

Deliberately uses only Safeguard's **public** surface (`Guard`, `GuardRequest`,
`Verdict`). Reaching into `walk_numeric` or the matching internals would make
the harness a special case rather than an honest integration example — and if
something needed here is not public, that is feedback about the API, not a
reason to bypass it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from safeguard.core import FieldMap, Guard
from safeguard.core.models import GuardRequest

from sgbench.capture import Triple


class Outcome(BaseModel):
    """One verified triple, flattened for aggregation."""

    query_id: Optional[str] = None
    tier: Optional[str] = None
    query: str = ""
    answer: str = ""
    passed: bool = True
    classification: str = "ok"
    flagged: list[str] = Field(default_factory=list)
    # Non-enforcing findings — currently unit/scale suffixes ("$1.87T"), which
    # the gate deliberately does not judge. The adjudicator needs these: they
    # are precisely the `unit_scale` rubric category, and without them a
    # reviewer would have to re-scan the answer by eye to find them.
    observed: list[str] = Field(default_factory=list)
    audit_record_id: Optional[str] = None
    # Capture-quality signal, not a model finding. A triple whose tool results
    # were all unparsed prose will flag everything; those runs measure the
    # harness, not the copilot, and must be excluded before reporting a rate.
    raw_result_count: int = 0
    tool_call_count: int = 0

    @property
    def measurable(self) -> bool:
        return self.tool_call_count > 0 and self.raw_result_count < self.tool_call_count

    @property
    def declined(self) -> bool:
        """No tool was called at all.

        Distinct from unparsed-result exclusion. On the partially-answerable
        tier this is the *correct* behaviour, and folding it into a single
        "excluded" count hides the best thing the copilot does. Reported as a
        headline in its own right, never as an exclusion footnote.
        """
        return self.tool_call_count == 0


def target_prompt_text(provenance_path: Optional[Path] = None) -> str:
    """System prompt + tool descriptions, as one block of prose.

    These are given to the model, so numbers it quotes back out of them are
    grounded — the "60 sessions" cap in a tool description is the model
    reading its own configuration aloud, not inventing a figure. Read from
    results/target-provenance.json so the text used here is the same text the
    run was captured under.
    """
    path = provenance_path or (Path("results") / "target-provenance.json")
    if not path.exists():
        return ""
    prov = json.loads(path.read_text(encoding="utf-8"))
    parts = [str(prov.get("system_prompt") or "")]
    parts += [str(v) for v in (prov.get("tools") or {}).values()]
    return "\n".join(p for p in parts if p)


def verify_triple(
    triple: Triple,
    guard: Optional[Guard] = None,
    field_map: Optional[FieldMap] = None,
    prompt_text: Optional[str] = None,
    derivations: Optional[object] = None,
) -> Outcome:
    """Verify one triple and return a flat outcome row."""
    guard = guard or Guard(field_map=field_map)
    verdict, record = guard.verify(
        GuardRequest(
            source_values=triple.source_values(),
            output_text=triple.answer,
            # The query is part of what the model was given: "value of 500
            # shares" makes 500 a grounded number in the answer.
            prompt_text="\n".join(
                x for x in (triple.query, prompt_text) if x
            ) or None,
            derivations=derivations,
            context={
                "query_id": triple.query_id,
                "tier": triple.tier,
                "model": triple.model,
                "target": triple.target,
                "harness_version": triple.harness_version,
            },
        )
    )
    return Outcome(
        query_id=triple.query_id,
        tier=triple.tier,
        query=triple.query,
        answer=triple.answer,
        passed=verdict.passed,
        classification=verdict.classification.value,
        flagged=list(verdict.flagged),
        observed=[f"{f.token}({f.reason})" for f in verdict.observed],
        audit_record_id=record.record_id,
        raw_result_count=sum(1 for c in triple.tool_calls if c.result_is_raw_text),
        tool_call_count=len(triple.tool_calls),
    )


def describe(outcome: Outcome) -> str:
    """Human-readable one-liner for console output."""
    status = "PASS" if outcome.passed else "FREEZE"
    detail = ""
    if not outcome.passed:
        detail = f"  ungrounded: {', '.join(outcome.flagged)}"
    warn = "" if outcome.measurable else "  [!] not measurable — tool results unparsed"
    return f"[{status}] {outcome.classification}{detail}{warn}"


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    """Aggregate for the study write-up.

    Rates are computed over **measurable** triples only, and the excluded
    count is reported alongside — a rate that quietly drops unmeasurable runs
    is the kind of thing the precision review exists to prevent.
    """
    measurable = [o for o in outcomes if o.measurable]
    # A soft flag is a third outcome, not a freeze and not a pass. Counted
    # separately under every policy so the rate is comparable across runs.
    # Under the enforcing policy this lands in `classification`; under the
    # advisory one it lands in `observed`. Count both so the third outcome is
    # comparable across policies.
    review = [o for o in measurable
              if o.classification == "derived_undeclared"
              or any("(derived_verified_chain)" in f for f in o.observed)]
    declined = [o for o in outcomes if o.declined]
    # Excluded-but-not-a-decline: tools ran, results arrived as unparsed prose.
    # That measures the harness, and is the only honest exclusion.
    unparsed = [o for o in outcomes
                if not o.measurable and not o.declined]
    by_tier: dict[str, dict[str, int]] = {}
    for o in measurable:
        tier = o.tier or "untiered"
        row = by_tier.setdefault(tier, {"n": 0, "frozen": 0})
        row["n"] += 1
        if not o.passed:
            row["frozen"] += 1
    for d in declined:
        by_tier.setdefault(d.tier or "untiered",
                           {"n": 0, "frozen": 0, "declined": 0})
    for d in declined:
        by_tier[d.tier or "untiered"]["declined"] = \
            by_tier[d.tier or "untiered"].get("declined", 0) + 1
    return {
        "total_captured": len(outcomes),
        "measurable": len(measurable),
        "declined_no_tool_call": len(declined),
        "excluded_unparsed": len(unparsed),
        "frozen": sum(1 for o in measurable if not o.passed
                      and o.classification != "derived_undeclared"),
        "review_verified_derivation": len(review),
        "by_tier": by_tier,
    }
