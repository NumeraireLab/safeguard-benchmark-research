"""OpenBB copilot adapter — **UNVERIFIED. Phase 0 must confirm this.**

Nothing in this module has been run against a live OpenBB copilot. It is
written as an investigation checklist rather than a working integration,
because guessing at an API and shipping code that looks confident is worse
than shipping nothing: a wrong adapter produces empty `tool_calls`, every
number in every answer is then flagged, and the study reports a spectacular
finding that is entirely an artefact of the harness.

The working assumption (from the user, to be confirmed) is that OpenBB's
copilot is built on LangGraph — which would make the state's ``messages``
channel the capture point and ``adapters.langgraph`` the whole adapter.

## Phase 0 checklist

1. Install OpenBB and locate the copilot entry point. Note the Python
   version constraint: OpenBB's dependency tree may not support 3.14 yet.
2. Determine whether it exposes a LangGraph graph object or only a chat
   endpoint. If a graph: `triple_from_state(graph.invoke(...))` and this
   module is unnecessary.
3. If only an endpoint, find where tool results live — a run/step API, a
   trace, a callback hook, or a debug flag. Anything that returns the
   *retrieved data* alongside the answer.
4. Capture ONE triple and assert `Triple.is_complete()`.
5. Confirm the tool results parse as JSON (`result_is_raw_text` False). Prose
   results contribute no numbers to the allowed set, so a copilot that hands
   back rendered tables rather than structured data changes the study design.

**If step 3 fails, stop.** The "ground truth is free" property does not hold
without retrieved data, and no number of queries fixes that. Fall back to the
reference copilot (see README, Plan B): a minimal LangGraph agent over a free
data source, where instrumentation is fully controlled and the harness is
still the publishable artefact.
"""

from __future__ import annotations

from typing import Any

from sgbench.capture import Triple
from sgbench.adapters.langgraph import triple_from_state


def capture(graph_state: Any, **meta: Any) -> Triple:
    """Placeholder pending Phase 0.

    If OpenBB does expose LangGraph state, this is the entire adapter and the
    indirection can be deleted.
    """
    return triple_from_state(graph_state, target="openbb", **meta)
