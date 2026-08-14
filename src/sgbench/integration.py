"""Drop-in Safeguard integration for an **existing** LangGraph application.

This is the artefact that matters to someone evaluating Safeguard. A benchmark
against an agent we wrote ourselves answers a question nobody asked; what a
listener wants to know is whether this goes into the stack they already run,
and what it catches there.

Three shapes, in increasing order of coupling:

    verify_state(state)        # zero coupling — read a finished run
    guard(app)                 # wrap any compiled graph; no source changes
    SafeguardNode              # a node, for when you own the graph

**No langgraph import.** These read `state["messages"]` by shape, exactly as
`adapters.langgraph` does, so they work against any framework whose state
carries a message list — and they add no dependency to the host application.

The monitor/enforce split is deliberate and mirrors how security tooling is
actually adopted: run it out-of-band first, return with findings from the
customer's own traffic, and the inline gate sells itself. `guard(app,
mode="observe")` changes nothing about the application's behaviour.
"""

from __future__ import annotations

from typing import Any, Optional

from safeguard.core import FieldMap, Guard

from sgbench.adapters.langgraph import triple_from_messages
from sgbench.capture import Triple


class SafeguardResult:
    """What the gate concluded about one completed run."""

    def __init__(self, triple: Triple, verdict: Any, record: Any):
        self.triple = triple
        self.verdict = verdict
        self.record = record

    @property
    def passed(self) -> bool:
        return bool(self.verdict.passed)

    @property
    def flagged(self) -> list[str]:
        return list(self.verdict.flagged)

    @property
    def observed(self) -> list[str]:
        return [f"{f.token}({f.reason})" for f in self.verdict.observed]

    @property
    def record_id(self) -> str:
        return self.record.record_id

    @property
    def measurable(self) -> bool:
        """False when the run retrieved nothing, or every tool result arrived
        as unparsed prose. Such a run has no ground truth: *everything* would
        flag, and reading that as a finding measures the integration rather
        than the agent."""
        calls = self.triple.tool_calls
        return bool(calls) and sum(c.result_is_raw_text for c in calls) < len(calls)

    def __repr__(self) -> str:
        state = "PASS" if self.passed else "FREEZE"
        return f"<Safeguard {state} flagged={self.flagged} record={self.record_id}>"


class SafeguardFrozen(Exception):
    """Raised by `guard(app, mode="enforce")` when an output is not clean."""

    def __init__(self, result: SafeguardResult):
        self.result = result
        super().__init__(
            f"Output withheld — ungrounded numbers: {', '.join(result.flagged)}. "
            f"Audit record {result.record_id}."
        )


def verify_state(
    state: Any,
    guard_instance: Optional[Guard] = None,
    field_map: Optional[FieldMap] = None,
    **context: Any,
) -> SafeguardResult:
    """Verify a finished LangGraph run. **Zero coupling** — the application
    does not know this happened, which is what makes it approvable in one
    meeting rather than after a security review of the request path.
    """
    messages = state.get("messages") if isinstance(state, dict) else getattr(
        state, "messages", None
    )
    if messages is None:
        raise ValueError(
            "No 'messages' in state. If this graph keeps tool results in "
            "another channel, extract them there — never re-read them from "
            "the answer text, which makes the check circular."
        )
    triple = triple_from_messages(messages, **context)
    g = guard_instance or Guard(field_map=field_map)
    from safeguard.core.models import GuardRequest

    verdict, record = g.verify(
        GuardRequest(
            source_values=triple.source_values(),
            output_text=triple.answer,
            context={"integration": "langgraph", **context},
        )
    )
    return SafeguardResult(triple, verdict, record)


def guard(
    app: Any,
    mode: str = "observe",
    guard_instance: Optional[Guard] = None,
    field_map: Optional[FieldMap] = None,
    on_result: Optional[Any] = None,
    **context: Any,
):
    """Wrap a compiled LangGraph app so every `.invoke()` is verified.

        app = guard(app)                    # monitor: records, never blocks
        app = guard(app, mode="enforce")    # gate: raises SafeguardFrozen

    The host application needs no source change. In observe mode the returned
    state is the original plus a `safeguard` key, so existing callers keep
    working untouched.
    """
    if mode not in ("observe", "enforce"):
        raise ValueError("mode must be 'observe' or 'enforce'")

    class _Guarded:
        def __init__(self, inner: Any):
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            # Everything not intercepted passes straight through, so the wrapper
            # stays a drop-in rather than a re-implementation of the graph API.
            return getattr(self._inner, name)

        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            state = self._inner.invoke(*args, **kwargs)
            result = verify_state(
                state, guard_instance=guard_instance,
                field_map=field_map, **context,
            )
            if on_result is not None:
                on_result(result)
            if mode == "enforce" and not result.passed:
                raise SafeguardFrozen(result)
            if isinstance(state, dict):
                state = {**state, "safeguard": result}
            return state

    return _Guarded(app)


class SafeguardNode:
    """A graph node, for when you own the graph and want a real gate.

        builder.add_node("safeguard", SafeguardNode())
        builder.add_edge("analyst", "safeguard")
        builder.add_conditional_edges(
            "safeguard",
            lambda s: "clean" if s["safeguard"].passed else "frozen",
            {"clean": END, "frozen": "regenerate"},
        )

    This is the shape a LangChain *callback* cannot provide: a callback
    observes and cannot withhold, which is monitor mode by construction. A
    conditional edge on the verdict is an actual gate.
    """

    def __init__(
        self,
        guard_instance: Optional[Guard] = None,
        field_map: Optional[FieldMap] = None,
        **context: Any,
    ):
        self.guard = guard_instance or Guard(field_map=field_map)
        self.context = context

    def __call__(self, state: Any) -> dict[str, Any]:
        return {"safeguard": verify_state(
            state, guard_instance=self.guard, **self.context
        )}
