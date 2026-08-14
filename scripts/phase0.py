"""Phase 0 — the gate on the whole study.

The benchmark's claim is *"of N answers, X contained a number that could not be
traced to the data the copilot itself retrieved."* That is only measurable if
the retrieved data can be captured. This script proves it end to end on ONE
turn before any effort goes into a query set.

    python3 scripts/phase0.py                      # fixture — no deps, no keys
    python3 scripts/phase0.py --messages run.json  # a captured live run

Exit code 0 means the capture path works. Non-zero means stop and fix capture
before writing 250 queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sgbench.adapters.langgraph import triple_from_messages  # noqa: E402
from sgbench.verify import describe, verify_triple  # noqa: E402


def load_messages(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("messages", "state", "output"):
            if key in data:
                inner = data[key]
                return inner.get("messages", inner) if isinstance(inner, dict) else inner
        raise SystemExit(f"No 'messages' key found in {path}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--messages",
        type=Path,
        default=REPO / "fixtures" / "sample_turn.json",
        help="JSON file containing a LangGraph message list (or a state dict).",
    )
    args = ap.parse_args()

    print(f"source: {args.messages}\n")
    triple = triple_from_messages(load_messages(args.messages), target="phase0")

    checks = [
        ("query captured", bool(triple.query)),
        ("answer captured", bool(triple.answer)),
        ("tool calls captured", bool(triple.tool_calls)),
        (
            "tool results parsed as JSON",
            bool(triple.tool_calls)
            and all(not c.result_is_raw_text for c in triple.tool_calls),
        ),
    ]
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")

    print(f"\n  query : {triple.query}")
    print(f"  tools : {', '.join(c.name for c in triple.tool_calls) or '(none)'}")
    print(f"  answer: {triple.answer}")
    print(f"\n  grounding payload: {json.dumps(triple.source_values())}")

    if not triple.is_complete():
        print(
            "\nFAILED — incomplete triple. Without retrieved data the study's "
            "ground truth does not exist; every number would flag and the "
            "result would measure the harness. See adapters/openbb.py.",
        )
        return 1

    outcome = verify_triple(triple)
    print(f"\n  safeguard: {describe(outcome)}")
    print(f"  audit record: {outcome.audit_record_id}")

    # The fixture is built to freeze. A pass there means extraction silently
    # widened the allowed set — the dangerous direction, per Safeguard's
    # asymmetry rule (too little extraction is a visible false freeze; too
    # much is a silent hole).
    if args.messages.name == "sample_turn.json" and outcome.passed:
        print(
            "\nFAILED — the fixture contains ungrounded numbers and should "
            "have frozen. Extraction is admitting values it should not.",
        )
        return 1

    print("\nPhase 0 gate: PASSED — capture path works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
