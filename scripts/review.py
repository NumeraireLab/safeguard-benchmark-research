"""Render one captured row in full, for hand adjudication.

The adjudication CSV clips `retrieved` at 600 chars so the sheet stays
openable, which makes a wide `get_fundamentals` payload unreadable — you
cannot confirm a field:value pair you cannot see. Nothing is lost: the full
payload is in results/triples.jsonl. This prints it, and for every flagged
token lists the source paths it could plausibly have come from (exact,
rescaled, or rounded).

    python3 scripts/review.py t1-005
    python3 scripts/review.py --list            # flagged rows, ids only
    python3 scripts/review.py --list --passed   # passed rows (the FN hunt)

Candidates are a search aid, NOT a verdict. A rescaled or rounded hit still
needs your call on whether the answer's claim is the right one; an empty
candidate list is the strongest signal the sheet can give you, not a ruling.

`walk_numeric` is not part of Safeguard's public API, so the flattener below
is local rather than imported. That is API feedback, not a reason to reach
inside — see the harness README.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sgbench.paths import add_run_arg, run_dir  # noqa: E402

RESULTS = run_dir()   # rebound in main() if --run is given

# Only the readings the CHECKER tries for a bare token. The inverse percent
# reading (x100) is admitted solely inside a verified chain, so offering it
# here invents "rescaled" explanations the gate never used -- eight different
# rows truncate to the same two decimals once divided by 100.
SCALES = [
    (1e12, "trillions"), (1e9, "billions"), (1e6, "millions"),
    (1e3, "thousands"), (1e-2, "percent<->decimal"),
]


def flatten(obj, prefix=""):
    """(path, float) for every number reachable in the payload."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)


def _precision_match(source: float, narrated: float, decimals: int) -> bool:
    """The checker's own rule, imported rather than approximated.

    A looser tolerance here does not merely mis-explain a verdict — it invents
    matches the gate never made and pushes the adjudicator toward
    `derived_correct` on numbers that are not derived at all. The review tool
    must never be more permissive than the thing it explains.
    """
    from safeguard.core.verification import _matches_precision
    return _matches_precision(source, narrated, decimals)


def candidates(token: str, values: list[tuple[str, float]]) -> list[str]:
    clean = token.replace(",", "")
    try:
        want = float(clean)
    except ValueError:
        return []
    dec = len(clean.split(".")[1]) if "." in clean else 0
    hits: list[str] = []
    for path, val in values:
        if val == want:
            hits.append(f"EXACT      {path} = {val!r}")
            continue
        if _precision_match(val, want, dec):
            hits.append(f"rounded    {path} = {val!r}  (to {dec}dp)")
            continue
        for mult, label in SCALES:
            scaled = val / mult
            # Scale the band with the value, exactly as the checker does --
            # and by ROUNDING only. Truncation is accepted for the direct
            # reading; applying it to a rescaled one manufactures matches.
            if abs(scaled - want) <= 0.5 * (10.0 ** -dec) + 1e-9 * max(
                abs(scaled), abs(want), 1.0
            ):
                hits.append(f"rescaled   {path} = {val!r}  (/{mult:g}, {label})")
                break
    return hits


def why_flagged(answer: str, start: int, end: int) -> str:
    """What the checker actually attempted for this token.

    "no source value matches at any scale or rounding" describes only the
    bare-token search. When a number is part of arithmetic the model wrote,
    the real question was whether that chain verified -- and saying otherwise
    pushes the adjudicator toward `unit_scale` when the answer is
    `derived_correct`.
    """
    from safeguard.core.verification import (
        _all_expression_spans, _maskable, _parenthetical_derivations,
    )
    masked = _maskable(answer)
    for a, b in _all_expression_spans(masked):
        if a <= start < b:
            return (f"inside arithmetic the model wrote: {answer[a:b].strip()[:90]!r}"
                    f"\n      the chain did NOT verify -- treat as a derivation, "
                    f"not a scale or rounding issue")
    for p0, p1, body in _parenthetical_derivations(answer):
        if p0 <= start < p1 or (0 <= p0 - end <= 4):
            return (f"working stated alongside it: {body.strip()[:90]!r}"
                    f"\n      the chain did NOT verify -- treat as a derivation")
    return ""


def load(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows[d["query_id"]] = d
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query_ids", nargs="*")
    ap.add_argument("--list", action="store_true", help="list ids and exit")
    ap.add_argument("--passed", action="store_true",
                    help="with --list, show passed rows instead of flagged")
    add_run_arg(ap)
    args = ap.parse_args()
    global RESULTS
    RESULTS = run_dir(args.run)

    triples = load(RESULTS / "triples.jsonl")
    outcomes = load(RESULTS / "outcomes.jsonl")

    if args.list:
        for qid, o in outcomes.items():
            if o.get("tool_call_count", 0) == 0:
                continue
            if bool(o.get("passed")) != bool(args.passed):
                continue
            flags = ",".join(o.get("flagged") or []) or "-"
            print(f"  {qid:10s} {o.get('tier','?'):3s} {flags[:60]}")
        return 0

    for qid in args.query_ids:
        t, o = triples.get(qid), outcomes.get(qid)
        if not t:
            print(f"[{qid}] not found"); continue
        src = {}
        for i, call in enumerate(t.get("tool_calls") or []):
            name = call.get("name", f"tool{i}")
            key = name if name not in src else f"{name}_{i}"
            src[key] = call.get("result")
        values = list(flatten(src))

        print("=" * 78)
        print(f"{qid}   tier={t.get('tier')}   verdict={(o or {}).get('classification','?')}")
        print("=" * 78)
        print(f"\nQUERY\n  {t.get('query')}")
        print(f"\nANSWER\n{t.get('answer')}")
        print(f"\nRETRIEVED  ({len(values)} numeric values across {len(src)} call(s))")
        print(json.dumps(src, indent=2, default=str))

        flags = (o or {}).get("flagged") or []
        print(f"\nFLAGGED ({len(flags)})")
        for f in flags:
            hits = candidates(f, values)
            print(f"\n  {f!r}")
            if not hits:
                why = why_flagged(t.get("answer", ""), 0, 0)
                print("      no source value matches at any scale or rounding")
            for h in hits[:6]:
                print(f"      {h}")
            if len(hits) > 6:
                print(f"      ... {len(hits)-6} more")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
