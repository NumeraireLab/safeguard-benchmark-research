"""Study pipeline — capture, verify, export.

Three separate commands on purpose. Capture costs money and cannot be
reproduced; verify is free and deterministic. Re-verifying a recorded capture
after fixing a checker gap is the whole point of the split.

    python3 scripts/run_study.py capture --target fixture
    python3 scripts/run_study.py verify
    python3 scripts/run_study.py export --sample-passed 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sgbench import queries as qmod  # noqa: E402
from sgbench.export import export_for_adjudication, write_rubric  # noqa: E402
from sgbench.run import (  # noqa: E402
    capture,
    fixture_target,
    load_triples,
    preflight,
    verify_all,
    write_outcomes,
)
from sgbench.verify import summarize  # noqa: E402

RESULTS = REPO / "results"
TRIPLES = RESULTS / "triples.jsonl"
OUTCOMES = RESULTS / "outcomes.jsonl"
AUDIT = RESULTS / "audit.jsonl"
SHEET = RESULTS / "adjudication.csv"
RUBRIC = RESULTS / "adjudication-rubric.txt"


def cmd_capture(args) -> int:
    qs = qmod.load()
    for w in preflight(qs, minimum=args.min_per_tier):
        print(f"  [warn] {w}")

    if args.target == "fixture":
        target, name = fixture_target(REPO / "fixtures"), "fixture"
    elif args.target == "reference":
        from sgbench.targets.reference import make_target, prompt_fingerprint
        target, name = make_target(model=args.model or "claude-sonnet-5"), "reference"
        # The prompt and the window caps are part of the result. Recording them
        # next to the capture is what makes the run auditable later.
        (RESULTS).mkdir(parents=True, exist_ok=True)
        (RESULTS / "target-provenance.json").write_text(
            json.dumps(
                {"target": name, "model": args.model or "claude-sonnet-5",
                 **prompt_fingerprint()}, indent=2
            ),
            encoding="utf-8",
        )
    elif args.target == "openbb":
        print(
            "  [stop] The OpenBB adapter is unverified — see "
            "src/sgbench/adapters/openbb.py. Clear Phase 0 first:\n"
            "         python3 scripts/phase0.py --messages results/one_run.json"
        )
        return 2
    else:
        return 2

    selected = [q for q in qs if not args.tier or q.tier == args.tier]
    print(f"\ncapturing {len(selected)} queries against '{name}' -> {TRIPLES}")

    captured = capture(
        selected, target, TRIPLES, model=args.model, target_name=name,
    )
    ok = sum(1 for t in captured if t.is_complete())
    print(f"  captured {len(captured)}  complete {ok}  incomplete {len(captured) - ok}")
    if ok == 0:
        print(
            "\n  [!] Nothing usable. A triple without retrieved data cannot be\n"
            "      measured — every number would flag and the result would\n"
            "      describe the harness, not the copilot."
        )
        return 1
    return 0


def cmd_verify(args) -> int:
    if not TRIPLES.exists():
        print(f"  no capture found at {TRIPLES} — run `capture` first")
        return 1
    triples = load_triples(TRIPLES)
    outcomes = verify_all(triples, AUDIT)
    write_outcomes(outcomes, OUTCOMES)

    stats = summarize(outcomes)
    print(json.dumps(stats, indent=2))
    print(f"\n  audit records -> {AUDIT}")
    print(f"  outcomes      -> {OUTCOMES}")
    print(
        "\n  Note: `frozen` is the GATE's count, not the finding. The copilot's"
        "\n  error rate and our recall both need the manual pass — run `export`."
    )
    return 0


def cmd_export(args) -> int:
    if not (TRIPLES.exists() and OUTCOMES.exists()):
        print("  run `capture` and `verify` first")
        return 1
    from sgbench.verify import Outcome

    triples = load_triples(TRIPLES)
    outcomes = [
        Outcome.model_validate_json(line)
        for line in OUTCOMES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = export_for_adjudication(
        triples, outcomes, SHEET,
        sample_passed=args.sample_passed, seed=args.seed,
    )
    write_rubric(RUBRIC)
    print(json.dumps(stats, indent=2))
    print(f"\n  sheet  -> {SHEET}")
    print(f"  rubric -> {RUBRIC}")
    print(
        "\n  Every flagged run is included; passed runs are sampled with seed"
        f" {args.seed}.\n  State the sampling scheme when publishing — it is what"
        " lets the true\n  incidence rate be recovered from a partial review."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="run queries against a target")
    c.add_argument(
        "--target", default="fixture",
        choices=["fixture", "reference", "openbb"],
    )
    c.add_argument("--tier", default=None, help="limit to one tier")
    c.add_argument("--model", default=None, help="model id, recorded for provenance")
    c.add_argument("--min-per-tier", type=int, default=50)
    c.set_defaults(func=cmd_capture)

    v = sub.add_parser("verify", help="verify a recorded capture (free, repeatable)")
    v.set_defaults(func=cmd_verify)

    e = sub.add_parser("export", help="write the manual adjudication sheet")
    e.add_argument("--sample-passed", type=int, default=100)
    e.add_argument("--seed", type=int, default=20260814)
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
