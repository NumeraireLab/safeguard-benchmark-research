# sgbench

Measuring how often AI copilots over market data state numbers that cannot be
traced to the data they themselves retrieved.

> **The claim this harness is built to support:** *of N answers, X contained a
> number that could not be traced to the data the copilot itself retrieved —
> without a gate, all X reached the user.*

Not a before/after study. [Safeguard](https://github.com/numerairelab/safeguard)
does not change what a model generates, so there is nothing to compare against.
This measures **the problem**, not the product.

## Why the ground truth is free

The tool results *are* the truth by definition. A number in the answer either
appears in what the agent retrieved, or it does not — so no human labelling is
required and one person can run the study. That property is load-bearing, and
it collapses if the retrieved data cannot be captured. Hence Phase 0.

## Phase 0 — run this first

```bash
python3 -m pip install -e ../Safeguard        # local Safeguard
python3 scripts/phase0.py                      # fixture: no deps, no API keys
```

Proves the full path — extract a turn, assemble the grounding payload, verify,
write an audit record. The bundled fixture contains ungrounded numbers, so a
correct run reports `FREEZE`. A `PASS` there means extraction is silently
widening the allowed set, which is the dangerous direction.

Against a live target:

```bash
python3 scripts/phase0.py --messages results/one_run.json
```

**If retrieved data cannot be captured from the target, stop.** No number of
queries fixes a missing ground truth. See `src/sgbench/adapters/openbb.py` for
the investigation checklist and Plan B.

## The study pipeline

```bash
export ANTHROPIC_API_KEY=...                                    # capture calls a live model
.venv/bin/python scripts/run_study.py capture --target reference # expensive, once
.venv/bin/python scripts/run_study.py verify                     # free, repeatable
.venv/bin/python scripts/run_study.py export --sample-passed 100 # adjudication sheet
```

### The target

**OpenBB Copilot is not the target, and cannot be.** It ships inside OpenBB
Workspace — hosted and commercial — and this study does not test commercial
products. What is on PyPI is the `openbb` *data platform* (real market data, no
agent) and `openbb-ai` (an SDK for building agents that plug *into* Workspace).

So the target is a **reference copilot**: a stock LangGraph `create_react_agent`
with six tools backed by the OpenBB data platform, in
`src/sgbench/targets/reference.py`. Everything about it is published — the code,
the tool window caps, and the system prompt verbatim — because the only defence
for measuring a self-built agent is that every choice is inspectable.

The prompt is deliberately **careful**, not naive: it instructs the model to
state only figures the tools returned and to decline when they do not cover the
question. A finding under a lax prompt is dismissible; a finding produced
*despite* an explicit instruction is the claim worth publishing.

Tool windows are capped for a reason that is Safeguard's own rule: the raw
Treasury endpoint returns ~250 rows x ~12 tenors, which would admit ~3,000
numbers as "grounded" and let almost any figure match by accident. Widening the
allowed set is the silent direction. Caps are recorded in
`results/target-provenance.json` alongside the prompt hash.

**Capture and verify are separate on purpose.** Capture costs money, needs
keys, and is non-deterministic — the same query re-run gives a different
answer, so a capture can never be reproduced. Verification is free and
deterministic. Keeping them apart means `results/triples.jsonl` is captured
once and re-verified any number of times: when a checker gap is fixed,
re-verify the *same* recorded runs and compare rates directly. Fusing them
would confound the fix with a fresh sample.

### What the numbers require

`verify` reports what the **gate** did. That is not the finding. The copilot's
error rate and Safeguard's recall both need a human deciding, independently of
the gate, whether each answer contains an ungrounded number:

|  | human: ungrounded | human: grounded |
|---|---|---|
| **gate: FREEZE** | TP | FP |
| **gate: PASS** | **FN** | TN |

```
copilot error rate = (TP+FN)/N      recall = TP/(TP+FN)      precision = TP/(TP+FP)
```

FN is the cell that matters internally — every entry is a product gap found
before a customer finds it, and it is not hypothetical: the unit-suffix
truncation bug (source `1.8` + narrated `"$1.87T"` → PASS) sat in the shipped
checker until a fixture caught it, and would have landed exactly there.

`export` writes `results/adjudication.csv` — every flagged run plus a seeded
random sample of passed runs — and `adjudication-rubric.txt`, whose categories
must be fixed **before** the first judgement. Unmeasurable runs (capture
errors, unparsed tool results) are excluded from the sheet and counted
separately: they show `PASS`, which would invite scoring them "clean" and
silently inflate true negatives.

Only `fabricated` is the headline finding. `derived_correct` and `unit_scale`
are our own known gaps measured on real traffic, and folding them into the rate
would be dishonest — and is the easiest way to discredit the study.

## Layout

```
src/sgbench/
  capture.py            Triple = (query, tool calls + results, answer) + provenance
  queries.py            tiered query set loader; enforces unique ids and tiers
  run.py                capture / verify orchestration, audit persistence
  verify.py             runs Safeguard's public API over a Triple; aggregation
  export.py             adjudication sheet + rubric
  adapters/
    langgraph.py        message-list extraction — works on live objects or JSON
    openbb.py           UNVERIFIED. Phase 0 checklist, not an integration.
queries/queries.jsonl   the query set — COMMIT BEFORE RUNNING
demo/                   per-segment pitch cases
fixtures/               recorded turns; replay source for demos
scripts/phase0.py       the capture gate
scripts/run_study.py    the pipeline
results/                gitignored until publication
```

## Study and demo are not the same thing

They share the capture path and nothing else.

- The **demo** may be curated. You *want* a case that freezes.
- The **study** may not. Commit the query set **before** running it, so the
  ordering in git history shows the queries were not tuned toward the finding.

Same discipline as publishing our own false-positive rate: the credibility
comes from the method being checkable, not from the number being impressive.

## Method commitments

Made in advance, and kept regardless of outcome.

- **Publish whatever it finds.** A near-zero rate is reliability by luck, not
  evidence of a control — it shifts the emphasis, it does not bury the result.
- **Publish our own false-positive rate.** Hand-review ~100 flags and classify:
  true ungrounded / unit mismatch / derived quantity / document noise / other.
- **Severity over frequency.** Three concrete cases (query → retrieved data →
  the sentence → the number that doesn't match) beat any aggregate.
- **Exclude unmeasurable runs and say how many.** A triple whose tool results
  arrived as unparsed prose flags everything; those measure the harness, not
  the copilot. `Outcome.measurable` marks them and `summarize()` reports the
  exclusion count alongside the rate.
- **Open-source targets only.** Testing a commercial product carries ToS risk
  and burns the firms we want as customers. Results are a category finding,
  never a callout of a named product.

## Relationship to Safeguard

One-way dependency, public API only — `Guard`, `GuardRequest`, `Verdict`.
Never internals. If something needed here is not public, that is feedback about
Safeguard's API, not a reason to reach inside.

The Safeguard version is pinned in `pyproject.toml`: verdicts are only
reproducible against a known verifier.

## Licence

MIT. Safeguard itself is proprietary and separately licensed; this harness is
published so the study is reproducible.
