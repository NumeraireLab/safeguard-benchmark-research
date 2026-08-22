"""Export runs for **manual adjudication** — the step that makes the study real.

Without a human deciding, independently of the gate, whether each answer
contains an ungrounded number, only precision is measurable. With it you get
the full matrix:

                    human: ungrounded   human: grounded
    gate: FREEZE           TP                 FP
    gate: PASS           **FN**               TN

    copilot error rate = (TP+FN)/N     recall = TP/(TP+FN)
    precision          = TP/(TP+FP)

FN is the cell that matters internally: every entry is a gap in the product,
found before a customer finds it. It is not hypothetical — the unit-suffix
truncation bug (source 1.8 + narrated "$1.87T" -> PASS) sat in the shipped
checker until a fixture caught it, and would have landed squarely here.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable, Optional

from sgbench.capture import Triple
from sgbench.verify import Outcome

# Decide these BEFORE reviewing anything. A binary grounded/ungrounded call
# collapses on contact with real answers, and a rubric invented halfway through
# means the second half is judged differently from the first.
RUBRIC = {
    "fabricated": "appears nowhere in retrieved data, and is wrong — the headline finding",
    "derived_correct": "arithmetically right but computed, not retrieved (gap #3)",
    "unit_scale": "right value at a different scale, e.g. $1.87T vs 1.87e12 (gap #2)",
    "rounding_error": "derivation right, operands trace, RESULT misrounded by one "
                      "unit in the last place -- a copilot error, not our gap "
                      "(added during adjudication 2026-08-21)",
    "parametric_harmless": "ungrounded but not a value claim, e.g. 'founded in 1976'",
    "scaffolding": "table/section/page reference or echoed year (gap #4)",
    "harness_artifact": "capture failed or tool result arrived unparsed — not a copilot error",
    "clean": "no ungrounded number in the answer",
}

COLUMNS = [
    "query_id", "tier", "query", "retrieved", "answer",
    "gate_verdict", "gate_flagged", "gate_observed",
    # Filled in by hand. `human_ungrounded` drives the matrix; `category` uses
    # RUBRIC; `numbers` records which figures were judged, so a second reviewer
    # can check the call rather than re-derive it.
    "human_ungrounded", "category", "numbers", "notes",
]


def _compact(value: object, limit: int = 600) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def export_for_adjudication(
    triples: Iterable[Triple],
    outcomes: Iterable[Outcome],
    path: Path,
    sample_passed: Optional[int] = 100,
    seed: int = 20260814,
) -> dict[str, int]:
    """Write the adjudication sheet.

    **Every flagged run is included** — precision needs all of them. Passed
    runs are *sampled*, because reviewing all of them is the difference
    between a day's work and a week's. State the sampling scheme when
    publishing: the sample rate is what lets the true incidence rate be
    recovered from a partial review.
    """
    by_id = {t.query_id: t for t in triples if t.query_id}
    all_outcomes = list(outcomes)

    # Unmeasurable runs are kept OUT of the sheet. A capture error or a run
    # whose tool results arrived as unparsed prose has no ground truth to
    # judge against, and it reaches the reviewer showing `PASS` — which invites
    # scoring it "clean" and silently inflates true negatives. They are
    # counted and reported instead, because their number is itself a finding
    # about the target (a copilot that renders tables rather than returning
    # structured data changes the study design).
    outcomes = [o for o in all_outcomes if o.measurable]
    unmeasurable = len(all_outcomes) - len(outcomes)

    flagged = [o for o in outcomes if not o.passed]
    passed = [o for o in outcomes if o.passed]

    if sample_passed is not None and len(passed) > sample_passed:
        rng = random.Random(seed)  # seeded: the sample must be reproducible
        chosen = rng.sample(passed, sample_passed)
    else:
        chosen = passed

    rows = flagged + chosen
    rows.sort(key=lambda o: (o.tier or "", o.query_id or ""))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for o in rows:
            triple = by_id.get(o.query_id)
            writer.writerow(
                {
                    "query_id": o.query_id or "",
                    "tier": o.tier or "",
                    "query": o.query,
                    "retrieved": _compact(triple.source_values()) if triple else "",
                    "answer": o.answer,
                    "gate_verdict": "FREEZE" if not o.passed else "PASS",
                    "gate_flagged": "; ".join(o.flagged),
                    "gate_observed": "; ".join(o.observed),
                    "human_ungrounded": "",
                    "category": "",
                    "numbers": "",
                    "notes": "",
                }
            )

    return {
        "total_runs": len(all_outcomes),
        "excluded_unmeasurable": unmeasurable,
        "measurable": len(outcomes),
        "flagged_all_included": len(flagged),
        "passed_total": len(passed),
        "passed_sampled": len(chosen),
        "rows_written": len(rows),
        "passed_sample_rate": (
            round(len(chosen) / len(passed), 4) if passed else None
        ),
    }


def write_rubric(path: Path) -> None:
    """Drop the rubric next to the sheet, so the categories are fixed in
    writing before the first judgement is made."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Adjudication rubric — fix these BEFORE reviewing.",
        "",
        "human_ungrounded: yes | no   (does the answer state a number that cannot",
        "                              be traced to the retrieved data?)",
        "",
        "category:",
    ]
    lines += [f"  {k:22} {v}" for k, v in RUBRIC.items()]
    lines += [
        "",
        "Only `fabricated` is the headline finding. `derived_correct` and",
        "`unit_scale` are our own known gaps measured on real traffic and must",
        "be reported separately — folding them into the rate would be dishonest",
        "and is the easiest way to discredit the study.",
        "",
        "Record edge-case decisions here as you go, so the second half of the",
        "review is judged the same way as the first:",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
