"""Render captured audit records into the demo.

    python3 scripts/showcase.py                 # terminal, for driving live
    python3 scripts/showcase.py --html out.html # leave-behind
    python3 scripts/showcase.py --pick 3,7,12   # curate for a specific audience

Curating is legitimate *here* and illegitimate in the study: a demo may show
the vivid cases, a published rate may not. Same capture, different rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sgbench.showcase import load_cases, render_terminal, write_html  # noqa: E402

RESULTS = REPO / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, default=RESULTS / "audit.jsonl")
    ap.add_argument("--triples", type=Path, default=RESULTS / "triples.jsonl")
    ap.add_argument("--html", type=Path, default=None)
    ap.add_argument("--title", default="Safeguard — audit trail")
    ap.add_argument("--pick", default=None, help="1-based indices, e.g. 3,7,12")
    ap.add_argument("--frozen-only", action="store_true",
                    help="not the default: a demo of only freezes sells the "
                         "check, and the check is the clonable half")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.audit.exists():
        print(f"  no audit trail at {args.audit} — run a capture first")
        return 1

    cases = load_cases(args.audit, args.triples)
    if args.frozen_only:
        cases = [c for c in cases if not c.passed]
    if args.pick:
        idx = [int(i) - 1 for i in args.pick.split(",") if i.strip()]
        cases = [cases[i] for i in idx if 0 <= i < len(cases)]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("  nothing to show")
        return 1

    if args.html:
        path = write_html(cases, args.html, title=args.title)
        print(f"  {len(cases)} cases -> {path}")
    else:
        print(render_terminal(cases))
        frozen = sum(1 for c in cases if not c.passed)
        print(f"\n  {len(cases)} verified · {frozen} withheld · "
              f"{len(cases)} audit records\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
