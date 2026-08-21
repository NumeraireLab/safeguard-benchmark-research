"""Mutation test: perturb a number the gate accepts, and see if it notices.

Adjudication measures precision — how many flags were real. It cannot measure
recall in the places nobody looked. The allowed set is a UNION of readings
(literal, rounded, truncated, rescaled, percent, declared derivations,
same-field shapes, series endpoints, sign polarity, verified chains) and a
number passes if ANY of them matches. Every reading added is another
independent path to a false negative, so precision improves visibly while
recall degrades invisibly.

This measures the invisible half without needing a human to adjudicate
anything. Take an answer the gate currently passes, corrupt one number in it,
and require the gate to freeze. Anything that survives is a false negative,
found by construction.

    python3 scripts/mutation_test.py --derivations config/derivations.json
    python3 scripts/mutation_test.py --limit 40 --verbose

An ESCAPE is a corrupted number the gate accepted. A COLLISION is a corrupted
number that happens to match some other source value — still a miss in
practice (it is the misattribution case: right number, wrong field), but
counted apart because closing it needs label alignment rather than a tighter
reading.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
RESULTS = REPO / "results"

from safeguard import Guard, GuardRequest  # noqa: E402
from safeguard.core.verification import (  # noqa: E402
    _candidates, _candidate_matches, _maskable, walk_numeric,
)
from sgbench.capture import Triple  # noqa: E402
from sgbench.verify import target_prompt_text  # noqa: E402

_NUM = re.compile(r"(?<![\w.])(-?\d+(?:,\d{3})*(?:\.\d+)?)(?![\w])")


def _mutations(tok: str, rng: random.Random) -> list[tuple[str, str]]:
    """(operator, replacement) — each must change the VALUE, not the spelling."""
    plain = tok.replace(",", "")
    try:
        val = float(plain)
    except ValueError:
        return []
    out: list[tuple[str, str]] = []
    digits = [i for i, c in enumerate(plain) if c.isdigit()]

    # a single wrong digit: the typo that survives proofreading
    if digits:
        i = rng.choice(digits)
        for _ in range(4):
            d = rng.choice("0123456789")
            if d != plain[i]:
                out.append(("digit", plain[:i] + d + plain[i + 1:]))
                break

    # two adjacent digits transposed
    for i in digits:
        j = i + 1
        if j < len(plain) and plain[j].isdigit() and plain[i] != plain[j]:
            out.append(("transpose", plain[:i] + plain[j] + plain[i] + plain[j + 1:]))
            break

    # off by a factor of ten, in each direction
    out.append(("scale_up", _fmt(val * 10, plain)))
    out.append(("scale_down", _fmt(val / 10, plain)))
    # sign inverted
    if val != 0:
        out.append(("sign", _fmt(-val, plain)))
    # a small relative drift — inside a loose tolerance, outside a tight one
    if val != 0:
        out.append(("drift_1pct", _fmt(val * 1.01, plain)))
    return [(op, rep) for op, rep in out if rep != plain]


def _fmt(value: float, like: str) -> str:
    dec = len(like.split(".")[1]) if "." in like else 0
    return f"{value:.{dec}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derivations", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="rows to mutate")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--verbose", action="store_true", help="print every escape")
    args = ap.parse_args()

    derivations = None
    if args.derivations:
        from safeguard import DerivationSet
        derivations = DerivationSet.from_json(args.derivations)

    triples = {}
    for line in (RESULTS / "triples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = Triple.model_validate_json(line)
            triples[t.query_id] = t
    outcomes = {}
    for line in (RESULTS / "outcomes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            outcomes[o["query_id"]] = o

    prompt = target_prompt_text(RESULTS / "target-provenance.json")
    guard = Guard()
    rng = random.Random(args.seed)

    # Only rows the gate currently accepts: mutating an already-frozen answer
    # proves nothing.
    clean = [q for q, o in outcomes.items()
             if o.get("passed") and o.get("tool_call_count", 0) > 0]
    clean.sort()
    if args.limit:
        clean = clean[: args.limit]

    stats: dict[str, list[int]] = {}
    escapes: list[tuple[str, str, str, str]] = []
    for qid in clean:
        t = triples[qid]
        sources = t.source_values()
        source_nums = [v for _, v in walk_numeric(sources)]
        masked = _maskable(t.answer)
        for m in _NUM.finditer(t.answer):
            tok = m.group(1)
            # Only mutate numbers the gate actually CHECKS. A digit changed
            # inside "2026-08-21" or "365 days" produces a pass that proves
            # nothing: those are masked as non-claims, so a "survival" there
            # measures the test, not the gate.
            if masked[m.start(1):m.end(1)].strip() == "":
                continue
            # And only grounded claims: a token that traces to nothing was
            # already going to freeze, or was never a claim about the data.
            try:
                cands = _candidates(tok.replace(",", ""),
                                    masked[m.end(1):m.end(1) + 1] == "%")
            except ValueError:
                continue
            if not _candidate_matches(cands, source_nums, None):
                continue
            for op, rep in _mutations(tok, rng):
                mutated = t.answer[:m.start(1)] + rep + t.answer[m.end(1):]
                verdict, _ = guard.verify(GuardRequest(
                    source_values=sources, output_text=mutated,
                    prompt_text="\n".join(x for x in (t.query, prompt) if x),
                    derivations=derivations,
                ))
                row = stats.setdefault(op, [0, 0, 0])   # tried, escaped, collided
                row[0] += 1
                if verdict.passed:
                    try:
                        mv = float(rep)
                    except ValueError:
                        mv = None
                    collided = mv is not None and any(
                        abs(sv - mv) < 1e-9 for sv in source_nums)
                    row[2 if collided else 1] += 1
                    escapes.append((qid, tok, f"{op}->{rep}",
                                    "collision" if collided else "escape"))

    tried = sum(v[0] for v in stats.values())
    esc = sum(v[1] for v in stats.values())
    col = sum(v[2] for v in stats.values())
    print(f"\nrows mutated: {len(clean)}   mutations: {tried}\n")
    print(f"{'operator':14s}{'tried':>8s}{'escaped':>9s}{'collided':>10s}{'caught':>9s}")
    for op in sorted(stats):
        n, e, c = stats[op]
        print(f"  {op:12s}{n:8d}{e:9d}{c:10d}{(n - e - c) / n:8.1%}")
    print(f"  {'TOTAL':12s}{tried:8d}{esc:9d}{col:10d}"
          f"{(tried - esc - col) / max(tried, 1):8.1%}")
    print(f"\n  false-negative rate: {esc / max(tried, 1):.2%}"
          f"  (+{col / max(tried, 1):.2%} collisions)")
    if args.verbose and escapes:
        print("\nescapes:")
        for qid, tok, how, kind in escapes[:60]:
            print(f"  [{kind:9s}] {qid:8s} {tok:>14s}  {how}")
        if len(escapes) > 60:
            print(f"  ... {len(escapes) - 60} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
