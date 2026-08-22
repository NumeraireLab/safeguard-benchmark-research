"""Where a run's output lives.

One capture is one directory. The checker is tuned by reading answers, so an
in-sample run and a held-out run must never share a path: the whole claim of
the second is that it was produced by a verifier that never saw it, and that
is only auditable if the two sets of files sit apart and stay apart.

Resolution order, first hit wins:

    --run NAME          explicit, on any script that takes it
    $SGBENCH_RUN        for the scripts that take no arguments
    results/current     a symlink the active run points at
    results/            the flat pre-run-directory layout

The fallback is deliberate: a checkout with no `current` symlink still works
exactly as it did before, so nothing that reads `results/` breaks.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"


def run_dir(name: str | None = None, *, create: bool = False) -> Path:
    """The directory this invocation reads from and writes to."""
    name = name or os.environ.get("SGBENCH_RUN") or None
    if name:
        path = Path(name)
        if not path.is_absolute():
            path = RESULTS / name
    elif (RESULTS / "current").exists():
        path = (RESULTS / "current").resolve()
    else:
        path = RESULTS
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def add_run_arg(parser) -> None:
    """`--run` on any script that has an argparse parser."""
    parser.add_argument(
        "--run", default=None, metavar="NAME",
        help="run directory under results/ (default: results/current, "
             "else results/). Also settable as $SGBENCH_RUN.",
    )
