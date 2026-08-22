"""Local adjudication UI. Reads the capture, writes your judgements to JSONL.

    python3 scripts/adjudicate.py            # http://127.0.0.1:8901

Serves one row at a time with the query, the answer (flagged tokens
highlighted), the FULL retrieved payload, and the candidate source paths for
every flagged token. You record human_ungrounded / category / numbers / notes;
each change is saved immediately to results/adjudication.jsonl, keyed by
query_id, so the session is resumable and nothing is lost to a closed tab.

Binds to loopback only. No dependencies beyond the stdlib — nothing leaves the
machine, which is the same posture the product ships with.

BLIND MODE is on by default: the gate's verdict is hidden until you have
recorded yours. The confusion matrix only means something if the human column
was produced independently of the gate column, and `blind` is recorded per row
so the write-up can say so honestly.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from sgbench.paths import add_run_arg, run_dir  # noqa: E402

# Judgements are keyed by query_id and query_ids repeat across runs, so a
# shared adjudication file would silently show one run's verdict against
# another run's answer. One directory per run keeps them apart.
RESULTS = run_dir()
OUT = RESULTS / "adjudication.jsonl"

from review import candidates, flatten, load, why_flagged  # noqa: E402

CATEGORIES = [
    ("fabricated", "appears nowhere in retrieved data, and is wrong — THE headline"),
    ("derived_correct", "arithmetically right but computed, not retrieved (gap #3)"),
    ("unit_scale", "right value at a different scale, $1.87T vs 1.87e12 (gap #2)"),
    ("rounding_error", "derivation right, RESULT misrounded by one ulp -- THEIR error"),
    ("parametric_harmless", "ungrounded but not a value claim — index names, tool meta"),
    ("scaffolding", "table/section/page reference or echoed year (gap #4)"),
    ("harness_artifact", "capture failed or tool result unparsed — not a copilot error"),
    ("clean", "no ungrounded number in the answer"),
]


def load_findings() -> dict[str, list[dict]]:
    """query_id -> findings with char offsets, from the audit records.

    `flagged` is a flat list of token *strings*, so one flagged "3" makes every
    "3" in the answer look flagged. The audit record carries start/end per
    finding; highlighting the exact spans is the difference between "the gate
    flagged eight things" and "the gate flagged these eight characters".
    """
    path = RESULTS / "audit.jsonl"
    out: dict[str, list[dict]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = (rec.get("context") or {}).get("query_id")
        if not qid:
            continue
        seen, items = set(), []
        for chk in (rec.get("verdict") or {}).get("checks") or []:
            for f in chk.get("findings") or []:
                if f.get("start") is None:
                    continue
                key = (f["start"], f["end"], f.get("reason"))
                if key in seen:
                    continue
                seen.add(key)
                items.append({"start": f["start"], "end": f["end"],
                              "token": f.get("token"), "reason": f.get("reason"),
                              "label": f.get("label"), "detail": f.get("detail"),
                              "enforcing": bool(chk.get("enforcing", True))})
        out[qid] = sorted(items, key=lambda d: d["start"])
    return out


def build_rows() -> list[dict]:
    triples, outcomes = load(RESULTS / "triples.jsonl"), load(RESULTS / "outcomes.jsonl")
    findings = load_findings()
    saved = load(OUT) if OUT.exists() else {}
    rows = []
    for qid, t in triples.items():
        o = outcomes.get(qid, {})
        if o.get("tool_call_count", 0) == 0:
            continue  # declined: no retrieved data, nothing to adjudicate
        src, seen = {}, {}
        for call in t.get("tool_calls") or []:
            n = call.get("name", "tool")
            seen[n] = seen.get(n, 0) + 1
            src[n if seen[n] == 1 else f"{n}_{seen[n]}"] = call.get("result")
        values = list(flatten(src))
        flags = o.get("flagged") or []
        fnd = findings.get(qid, [])
        # Keyed by OFFSET, never by token: the same figure appears twice in
        # one answer with different working each time, and a token-keyed map
        # silently shows the last one for both.
        for f in fnd:
            f["why"] = why_flagged(t.get("answer", ""), f["start"], f["end"])
        if not o.get("passed"):
            status = "FREEZE"
        elif any(f.get("reason") == "derived_verified_chain" for f in fnd):
            status = "REVIEW"
        else:
            status = "PASS"
        rows.append({
            "query_id": qid, "tier": t.get("tier"), "query": t.get("query"),
            "answer": t.get("answer"), "retrieved": src,
            "value_count": len(values),
            "flagged": flags,
            "findings": findings.get(qid, []),
            "candidates": {f: candidates(f, values) for f in flags},

            "gate_verdict": status,
            "gate_class": o.get("classification"),
            "saved": saved.get(qid),
        })
    order = {"FREEZE": 0, "REVIEW": 1, "PASS": 2}
    rows.sort(key=lambda r: (order[r["gate_verdict"]], r["query_id"]))
    return rows


def upsert(record: dict) -> int:
    existing = load(OUT) if OUT.exists() else {}
    record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    existing[record["query_id"]] = record
    with OUT.open("w", encoding="utf-8") as fh:
        for qid in sorted(existing):
            fh.write(json.dumps(existing[qid]) + "\n")
    return len(existing)


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []

    def log_message(self, *a):  # keep the terminal quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/rows":
            payload = {"rows": self.rows, "categories": CATEGORIES}
            self._send(json.dumps(payload).encode(), "application/json")
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(b"not found", "text/plain", 404)
        n = int(self.headers.get("Content-Length", 0))
        rec = json.loads(self.rfile.read(n) or b"{}")
        total = upsert(rec)
        self._send(json.dumps({"ok": True, "saved": total}).encode(), "application/json")


PAGE = r"""<!doctype html><meta charset="utf-8"><title>Adjudication</title>
<style>
:root{--bg:#0f1115;--fg:#e6e6e6;--dim:#8b93a1;--card:#171a21;--line:#262b36;
      --accent:#7aa2f7;--warn:#e0af68;--bad:#f7768e;--good:#9ece6a}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;
     background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
       padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
#prog{color:var(--dim)}
.wrap{max-width:1180px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
      padding:14px 16px;margin-bottom:14px}
h3{margin:0 0 8px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,Menlo,monospace}
#retrieved{max-height:320px;overflow:auto;background:#0c0e13;padding:10px;border-radius:6px}
mark{background:#3b2f13;color:var(--warn);padding:0 2px;border-radius:3px}
mark.obs{background:#1d2a3a;color:var(--accent)}
.obs2{color:var(--accent)}
mark[title*="derived_verified_chain"]{background:#16301f;color:var(--good)}
button{background:#222835;color:var(--fg);border:1px solid var(--line);border-radius:6px;
       padding:6px 12px;cursor:pointer;font:inherit}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent);color:#0b0d12;border-color:var(--accent);font-weight:600}
button.no.on{background:var(--good);border-color:var(--good)}
button.yes.on{background:var(--bad);border-color:var(--bad)}
.cats{display:grid;grid-template-columns:1fr;gap:5px}
.cats button{text-align:left;display:flex;gap:10px;align-items:baseline}
.cats .k{font-weight:600;min-width:170px}
.cats .d{color:var(--dim);font-size:12px}
.cats button.on .d{color:#0b0d12;opacity:.8}
input,textarea{width:100%;background:#0c0e13;color:var(--fg);border:1px solid var(--line);
               border-radius:6px;padding:8px;font:inherit}
textarea{min-height:70px;resize:vertical}
.cand{font:12px/1.6 ui-monospace,monospace;color:var(--dim)}
.cand b{color:var(--fg)}
.none{color:var(--bad);font-weight:600}
.tok{color:var(--warn);font-weight:600}
.gate{color:var(--dim)}
.pill{border:1px solid var(--line);border-radius:99px;padding:2px 9px;font-size:12px;color:var(--dim)}
kbd{background:#222835;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px}
</style>
<header>
  <button onclick="go(-1)">← <kbd>k</kbd></button>
  <button onclick="go(1)"><kbd>j</kbd> →</button>
  <button onclick="nextUnreviewed()">next unreviewed <kbd>n</kbd></button>
  <span id="prog"></span>
  <span style="flex:1"></span>
  <label class="pill"><input type="checkbox" id="blind" checked style="width:auto"> blind mode</label>
  <span id="gate" class="gate"></span>
</header>
<span id="rendered" hidden></span>
<div class="wrap">
  <div class="card"><h3>Query <span id="meta" class="gate"></span></h3><pre id="query"></pre></div>
  <div class="card"><h3>Answer</h3><pre id="answer"></pre></div>
  <div class="card"><h3>Flagged tokens &amp; candidate sources</h3><div id="cands"></div></div>
  <div class="card"><h3>Retrieved <span id="vcount" class="gate"></span></h3><pre id="retrieved"></pre></div>
  <div class="card">
    <h3>Your judgement</h3>
    <div style="margin-bottom:10px">
      <span class="gate" style="margin-right:8px">human_ungrounded</span>
      <button class="yes" id="uy" onclick="setU('yes')">yes <kbd>y</kbd></button>
      <button class="no"  id="un" onclick="setU('no')">no <kbd>n</kbd>o</button>
    </div>
    <div class="cats" id="cats"></div>
    <div style="margin-top:10px"><span class="gate">numbers</span><input id="numbers" placeholder="which figures drove the call"></div>
    <div style="margin-top:10px"><span class="gate">notes</span><textarea id="notes" placeholder="reasoning, especially edge cases"></textarea></div>
    <div style="margin-top:10px" class="gate">saved automatically · <span id="status">—</span></div>
  </div>
</div>
<script>
let ROWS=[],CATS=[],i=0,cur={};
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

fetch('/api/rows').then(r=>r.json()).then(d=>{
  ROWS=d.rows;CATS=d.categories;
  $('cats').innerHTML=CATS.map(([k,desc],n)=>
    `<button onclick="setC('${k}')" id="c-${k}"><span class="k">${k} <kbd>${n+1}</kbd></span><span class="d">${esc(desc)}</span></button>`).join('');
  render();
});

function highlight(text,findings){
  // Exact char spans from the audit record. Never string matching: one
  // flagged "3" must not light up every "3" in the answer.
  if(!findings||!findings.length) return esc(text);
  const spans=[...findings].sort((a,b)=>a.start-b.start);
  let out='',at=0;
  for(const f of spans){
    if(f.start==null||f.end==null) continue;
    if(f.start<at) continue;            // overlapping finding: keep the first
    if(f.end>text.length) continue;     // defensive: never slice past the end
    out+=esc(text.slice(at,f.start));
    const cls=f.enforcing===false?'obs':'';
    const why=[f.reason,f.label?('label '+f.label):null,f.detail].filter(Boolean).join(' — ');
    out+='<mark class="'+cls+'" title="'+esc(why)+'">'+esc(text.slice(f.start,f.end))+'</mark>';
    at=f.end;
  }
  return out+esc(text.slice(at));
}

function render(){
  cur=ROWS[i]; if(!cur) return;
  try{ renderRow(); }catch(err){
    // A throw here previously left ANSWER/FLAGGED/RETRIEVED blank with no
    // clue why. Surface it instead of failing silently.
    $('answer').textContent=cur.answer||'';
    $('cands').innerHTML='<span class="none">render error: '+esc(err.message)+'</span>';
    console.error(err);
  }
}
function renderRow(){
  // Reset the form FIRST. Everything below can throw; if it does, a stale
  // numbers/notes left in the DOM would be written onto the NEXT row by the
  // next save(). That corrupted six rows before it was caught.
  cur._u=null; cur._c=[];
  setU(null,true); setC(null,true);
  $('numbers').value=''; $('notes').value='';
  $('rendered')._ok=false;
  const done=ROWS.filter(r=>r.saved&&r.saved.human_ungrounded).length;
  $('prog').textContent=`${i+1} / ${ROWS.length}  ·  ${done} reviewed  ·  ${cur.query_id}`;
  $('meta').textContent=`${cur.tier} · ${cur.gate_verdict}`;
  $('query').textContent=cur.query;
  $('answer').innerHTML=highlight(cur.answer,cur.findings);
  $('vcount').textContent=`${cur.value_count} numeric values`;
  $('retrieved').textContent=JSON.stringify(cur.retrieved,null,2);
  const enf=(cur.findings||[]).filter(f=>f.reason!=='unhandled_unit');
  $('cands').innerHTML = enf.length===0
    ? '<span class="gate">no tokens flagged — read the answer against Retrieved and look for a number the gate missed</span>'
    : enf.map(fd=>{
        const f=fd.token, hits=cur.candidates[f]||[];
        return `<div style="margin-bottom:9px"><span class="tok">${esc(f)}</span>`+
          `<span class="gate"> @${fd.start}</span><div class="cand">`+
          (hits.length? hits.map(h=>'&nbsp;&nbsp;'+esc(h)).join('<br>')
                      : (fd.why
                          ? '&nbsp;&nbsp;<span class="obs2">'+esc(fd.why).replace(/\n/g,'<br>&nbsp;&nbsp;')+'</span>'
                          : '&nbsp;&nbsp;<span class="none">no source value traces this figure</span>'))+
          '</div></div>';
      }).join('');
  const s=cur.saved||{};
  setU(s.human_ungrounded||null,true); setC(s.category||null,true);
  $('numbers').value=s.numbers||''; $('notes').value=s.notes||'';
  $('rendered')._ok=true;
  $('status').textContent=s.reviewed_at?('last saved '+s.reviewed_at.slice(0,19).replace('T',' ')+'Z'):'—';
  showGate();
}
function showGate(){
  const s=cur.saved||{};
  $('gate').textContent = ($('blind').checked && !s.human_ungrounded)
    ? 'gate: hidden (blind)' : `gate: ${cur.gate_verdict} ${cur.gate_class||''}`;
}
$('blind').onchange=showGate;

function setU(v,quiet){cur._u=v;$('uy').classList.toggle('on',v==='yes');$('un').classList.toggle('on',v==='no');if(!quiet)save();}
function setC(v,quiet){
  // Multi-select: a complex answer can be derived_correct AND unit_scale at
  // once. Stored comma-separated so existing single-value rows still load.
  if(v===null){cur._c=[];}
  else if(quiet){cur._c=Array.isArray(v)?v:(v?String(v).split(',').map(x=>x.trim()).filter(Boolean):[]);}
  else{cur._c=cur._c||[];const i=cur._c.indexOf(v);i>=0?cur._c.splice(i,1):cur._c.push(v);}
  CATS.forEach(([k])=>$('c-'+k).classList.toggle('on',(cur._c||[]).includes(k)));
  if(!quiet)save();
}

function save(){
  if(!$('rendered')._ok){
    $('status').textContent='NOT SAVED — this row failed to render';
    return;
  }
  if(!cur._u&&!(cur._c||[]).length&&!$('numbers').value&&!$('notes').value) return;
  const rec={query_id:cur.query_id,tier:cur.tier,human_ungrounded:cur._u||'',
    category:(cur._c||[]).join(','),numbers:$('numbers').value,notes:$('notes').value,
    gate_verdict:cur.gate_verdict,gate_class:cur.gate_class,
    blind:$('blind').checked};
  fetch('/api/save',{method:'POST',body:JSON.stringify(rec)})
    .then(r=>r.json()).then(d=>{cur.saved={...rec,reviewed_at:new Date().toISOString()};
      $('status').textContent=`saved · ${d.saved} rows on disk`;
      const done=ROWS.filter(r=>r.saved&&r.saved.human_ungrounded).length;
      $('prog').textContent=`${i+1} / ${ROWS.length}  ·  ${done} reviewed  ·  ${cur.query_id}`;
      showGate();});
}
$('numbers').onchange=save; $('notes').onchange=save;

function go(d){save();i=Math.max(0,Math.min(ROWS.length-1,i+d));render();window.scrollTo(0,0);}
function nextUnreviewed(){save();const j=ROWS.findIndex((r,n)=>n>i&&!(r.saved&&r.saved.human_ungrounded));
  if(j>=0){i=j;render();window.scrollTo(0,0);}else{$('status').textContent='no unreviewed rows after this one';}}

addEventListener('keydown',e=>{
  if(/INPUT|TEXTAREA/.test(e.target.tagName)) return;
  if(e.key==='j')go(1); else if(e.key==='k')go(-1);
  else if(e.key==='y')setU('yes'); else if(e.key==='n')setU('no');
  else if(/^[1-7]$/.test(e.key))setC(CATS[+e.key-1][0]);
  else if(e.key==='N')nextUnreviewed();
});
</script>
"""

if __name__ == "__main__":
    import argparse
    import review

    ap = argparse.ArgumentParser(description=__doc__)
    add_run_arg(ap)
    ap.add_argument("--port", type=int, default=8901)
    args = ap.parse_args()
    RESULTS = run_dir(args.run)
    OUT = RESULTS / "adjudication.jsonl"
    review.RESULTS = RESULTS          # `candidates`/`load` read it too

    Handler.rows = build_rows()
    port = args.port
    print(f"  run: {RESULTS}")
    print(f"  {len(Handler.rows)} rows to adjudicate")
    print(f"  judgements -> {OUT}")
    print(f"  open http://127.0.0.1:{port}   (ctrl-c to stop)\n")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
