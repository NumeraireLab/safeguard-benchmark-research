"""Render captured audit records into a demo.

**The demo is a rendered audit trail, not a live run.** Three reasons, and the
first is the one that decides it: you cannot make a model hallucinate on cue,
so a live demo may simply produce a correct answer and show nothing. Replaying
real records is also free per pitch, and it is *inspectable* — a prospect can
read the retrieved data, the narrated sentence, the verdict, and the content
hash, and check the arithmetic themselves.

What this renders is deliberately not a highlight reel of failures. A demo made
only of freezes sells the check, and the check is the clonable half. Every view
here shows the **record** — including for answers that passed, because "here is
what you hand an examiner for the 999 answers that were fine" is the product.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class Case:
    """One verified turn, assembled from the audit record plus its capture."""

    record_id: str
    timestamp: str
    passed: bool
    classification: str
    output_text: str
    source_values: Any
    content_sha256: str
    config_sha256: Optional[str] = None
    schema_version: str = ""
    query: str = ""
    tier: str = ""
    target: str = ""
    # (start, end, token, reason) — offsets index output_text, which is what
    # Finding.start/end exist for. Rendering the span is the difference between
    # "it flagged 3.2" and showing the sentence with the number lit up.
    spans: list[tuple[int, int, str, str]] = field(default_factory=list)

    @property
    def enforcing_spans(self) -> list[tuple[int, int, str, str]]:
        return [s for s in self.spans if s[3] != "unhandled_unit"]

    @property
    def advisory_spans(self) -> list[tuple[int, int, str, str]]:
        return [s for s in self.spans if s[3] == "unhandled_unit"]


def _spans_from_verdict(verdict: dict) -> list[tuple[int, int, str, str]]:
    out: list[tuple[int, int, str, str]] = []
    for check in verdict.get("checks", []):
        for f in check.get("findings", []):
            if f.get("start") is None or f.get("end") is None:
                continue
            out.append((f["start"], f["end"], f.get("token", ""),
                        f.get("reason", "")))
    # Sort by position and drop overlaps, so rendering can walk the text once.
    out.sort(key=lambda s: (s[0], s[1]))
    deduped: list[tuple[int, int, str, str]] = []
    for span in out:
        if deduped and span[0] < deduped[-1][1]:
            continue
        deduped.append(span)
    return deduped


def load_cases(
    audit_path: Path, triples_path: Optional[Path] = None
) -> list[Case]:
    """Read audit records, enriching them with query/tier from the capture."""
    meta: dict[str, dict] = {}
    if triples_path and triples_path.exists():
        for line in triples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            # Keyed on the answer text: the audit record stores output_text,
            # and that is the only field the two files reliably share.
            meta[t.get("answer", "")] = t

    cases: list[Case] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        v = r.get("verdict", {})
        m = meta.get(r.get("output_text", ""), {})
        cases.append(
            Case(
                record_id=r["record_id"],
                timestamp=r["timestamp"],
                passed=bool(v.get("passed")),
                classification=v.get("classification", ""),
                output_text=r.get("output_text", ""),
                source_values=r.get("source_values"),
                content_sha256=r.get("content_sha256", ""),
                config_sha256=r.get("config_sha256"),
                schema_version=r.get("schema_version", ""),
                query=m.get("query", "") or (r.get("context") or {}).get("query", ""),
                tier=m.get("tier", "") or (r.get("context") or {}).get("tier", ""),
                target=m.get("target", "") or (r.get("context") or {}).get("target", ""),
                spans=_spans_from_verdict(v),
            )
        )
    return cases


# --------------------------------------------------------------------------
# Terminal view — for driving live in a call
# --------------------------------------------------------------------------

_RED = "\033[41m\033[97m"
_YEL = "\033[43m\033[30m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GRN = "\033[32m"
_OFF = "\033[0m"


def _mark_terminal(case: Case) -> str:
    text, out, cursor = case.output_text, [], 0
    for start, end, _tok, reason in case.spans:
        out.append(text[cursor:start])
        colour = _YEL if reason == "unhandled_unit" else _RED
        out.append(f"{colour}{text[start:end]}{_OFF}")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def render_terminal(cases: Iterable[Case], show_data: bool = True) -> str:
    lines: list[str] = []
    for i, c in enumerate(cases, 1):
        badge = f"{_GRN}PASS{_OFF}" if c.passed else f"{_RED} FREEZE {_OFF}"
        lines.append(f"\n{_BOLD}[{i}] {badge}  {c.classification}{_OFF}")
        if c.query:
            lines.append(f"  question   {c.query}")
        if show_data:
            payload = json.dumps(c.source_values, default=str)
            if len(payload) > 300:
                payload = payload[:299] + "…"
            lines.append(f"  {_DIM}retrieved  {payload}{_OFF}")
        lines.append(f"  answer     {_mark_terminal(c)}")
        if c.enforcing_spans:
            toks = ", ".join(s[2] for s in c.enforcing_spans)
            lines.append(f"  {_RED}ungrounded{_OFF} {toks}")
        if c.advisory_spans:
            toks = ", ".join(s[2] for s in c.advisory_spans)
            lines.append(f"  {_YEL}unverified{_OFF} {toks} (unit/scale — advisory)")
        # The record prints on every case, passing ones included. That is the
        # product; the verdict is how it gets generated.
        lines.append(
            f"  {_DIM}record     {c.record_id}  {c.timestamp}\n"
            f"             content {c.content_sha256[:16]}…"
            + (f"  config {c.config_sha256[:12]}…" if c.config_sha256 else "")
            + f"{_OFF}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# HTML view — the leave-behind
# --------------------------------------------------------------------------

def _mark_html(case: Case) -> str:
    text, out, cursor = case.output_text, [], 0
    for start, end, _tok, reason in case.spans:
        out.append(html.escape(text[cursor:start]))
        cls = "adv" if reason == "unhandled_unit" else "bad"
        out.append(
            f'<mark class="{cls}" title="{html.escape(reason)}">'
            f"{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def render_html(cases: list[Case], title: str = "Safeguard — audit trail") -> str:
    frozen = sum(1 for c in cases if not c.passed)
    blocks: list[str] = []
    for c in cases:
        payload = json.dumps(c.source_values, indent=2, default=str)
        verdict_cls = "pass" if c.passed else "freeze"
        label = "PASS" if c.passed else "FROZEN"
        flags = "".join(
            f'<li><code>{html.escape(s[2])}</code> — {html.escape(s[3])}</li>'
            for s in c.spans
        )
        blocks.append(f"""
<article class="case {verdict_cls}">
  <header><span class="badge">{label}</span>
    <span class="cls">{html.escape(c.classification)}</span>
    {f'<span class="tier">{html.escape(c.tier)}</span>' if c.tier else ''}</header>
  {f'<p class="q">{html.escape(c.query)}</p>' if c.query else ''}
  <h4>Retrieved by the agent</h4>
  <pre class="data">{html.escape(payload)}</pre>
  <h4>What it said</h4>
  <p class="answer">{_mark_html(c)}</p>
  {f'<h4>Findings</h4><ul class="flags">{flags}</ul>' if flags else ''}
  <h4>Audit record</h4>
  <dl class="rec">
    <dt>record</dt><dd><code>{html.escape(c.record_id)}</code></dd>
    <dt>time</dt><dd>{html.escape(c.timestamp)}</dd>
    <dt>content sha256</dt><dd><code>{html.escape(c.content_sha256)}</code></dd>
    {f'<dt>config sha256</dt><dd><code>{html.escape(c.config_sha256)}</code></dd>' if c.config_sha256 else ''}
    <dt>schema</dt><dd>v{html.escape(c.schema_version)}</dd>
  </dl>
</article>""")

    return f"""<title>{html.escape(title)}</title>
<style>
:root {{
  --bg:#fff; --fg:#16181d; --muted:#5c6370; --line:#e3e6ea; --card:#fafbfc;
  --bad:#c0392b; --bad-bg:#fde8e6; --adv:#8a6d1f; --adv-bg:#fdf3d6;
  --ok:#1e7d44; --code:#f2f4f7;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#14161a; --fg:#e6e8ec; --muted:#98a0ad; --line:#2b2f36; --card:#191c21;
    --bad:#ff7a6b; --bad-bg:#3a1f1c; --adv:#e0c168; --adv-bg:#332b12;
    --ok:#5ed992; --code:#20242a;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#14161a; --fg:#e6e8ec; --muted:#98a0ad; --line:#2b2f36; --card:#191c21;
  --bad:#ff7a6b; --bad-bg:#3a1f1c; --adv:#e0c168; --adv-bg:#332b12;
  --ok:#5ed992; --code:#20242a;
}}
body {{ background:var(--bg); color:var(--fg); margin:0; padding:2rem 1.25rem 4rem;
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
.wrap {{ max-width:52rem; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
.sub {{ color:var(--muted); margin:0 0 2rem; }}
.case {{ border:1px solid var(--line); border-radius:10px; background:var(--card);
  padding:1.1rem 1.25rem; margin:0 0 1.5rem; }}
.case header {{ display:flex; gap:.6rem; align-items:center; margin-bottom:.5rem; flex-wrap:wrap; }}
.badge {{ font:600 .72rem/1 ui-monospace,monospace; letter-spacing:.06em;
  padding:.35rem .55rem; border-radius:5px; }}
.freeze .badge {{ background:var(--bad-bg); color:var(--bad); }}
.pass .badge {{ background:transparent; color:var(--ok); border:1px solid currentColor; }}
.cls, .tier {{ color:var(--muted); font:.78rem ui-monospace,monospace; }}
.q {{ font-weight:600; margin:.2rem 0 1rem; }}
h4 {{ font:600 .7rem/1 ui-monospace,monospace; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); margin:1.1rem 0 .4rem; }}
pre.data {{ background:var(--code); border-radius:6px; padding:.7rem .8rem; margin:0;
  overflow-x:auto; font-size:.8rem; }}
.answer {{ margin:0; }}
mark.bad {{ background:var(--bad-bg); color:var(--bad); font-weight:600;
  padding:.05em .2em; border-radius:3px; }}
mark.adv {{ background:var(--adv-bg); color:var(--adv); padding:.05em .2em; border-radius:3px; }}
ul.flags {{ margin:.2rem 0; padding-left:1.2rem; }}
ul.flags code {{ background:var(--code); padding:.1em .3em; border-radius:3px; }}
dl.rec {{ display:grid; grid-template-columns:auto 1fr; gap:.25rem .9rem;
  margin:0; font-size:.8rem; }}
dl.rec dt {{ color:var(--muted); font-family:ui-monospace,monospace; }}
dl.rec dd {{ margin:0; overflow-wrap:anywhere; }}
dl.rec code {{ background:var(--code); padding:.1em .3em; border-radius:3px; font-size:.95em; }}
.note {{ border-left:3px solid var(--line); padding-left:1rem; color:var(--muted);
  font-size:.9rem; margin:2rem 0 0; }}
</style>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">{len(cases)} verified outputs · {frozen} withheld ·
every decision carries a record, pass or freeze.</p>
{''.join(blocks)}
<p class="note">Each record is self-contained: what was said, the data it was
checked against, the verdict, and a content fingerprint. The config hash
identifies which control produced the verdict, so replaying the same text and
data against the same configuration yields the same answer — which is what
makes this evidence rather than a log line.</p>
</div>"""


def write_html(cases: list[Case], path: Path, title: str = "Safeguard — audit trail") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(cases, title), encoding="utf-8")
    return path
