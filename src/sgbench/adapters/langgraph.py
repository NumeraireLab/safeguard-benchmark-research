"""Extract a `Triple` from a LangGraph / LangChain message list.

**Why messages and not callbacks.** A callback handler observes the run and
would need `run_id` correlation to stitch tool calls to their results; the
signatures have also moved between langchain-core versions. LangGraph's final
state already contains the whole turn in order, so reading messages is both
more stable and simpler. (It is also why LangGraph is the better integration
target for Safeguard itself: state is explicit, and a conditional edge on the
verdict is a real gate, whereas a LangChain callback can only observe — which
is monitor mode, not enforcement.)

**No langchain import.** This module understands the message *shape*, not the
classes, so it works on live message objects and on JSON fixtures alike. That
is what lets Phase 0's fixture check run with no agent framework installed.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from sgbench.capture import ToolCall, Triple, coerce_result


def _get(msg: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a message object or a plain dict."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _msg_type(msg: Any) -> str:
    """Normalise the message kind across objects and serialised dicts.

    Objects expose ``type`` ("human"/"ai"/"tool"); serialised forms vary, so
    fall back to the class name.
    """
    explicit = _get(msg, "type")
    if isinstance(explicit, str) and explicit:
        return explicit.lower()
    role = _get(msg, "role")
    if isinstance(role, str) and role:
        return {"user": "human", "assistant": "ai"}.get(role.lower(), role.lower())
    return type(msg).__name__.replace("Message", "").lower()


def _text(content: Any) -> str:
    """Flatten message content to text.

    Content is a string for most providers but a list of typed blocks for
    others; only text blocks carry narrated numbers, so the rest is dropped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return "" if content is None else str(content)


def triple_from_messages(messages: Iterable[Any], **meta: Any) -> Triple:
    """Build a `Triple` from an ordered LangGraph message list.

    The query is the first human turn; the answer is the last assistant turn
    that carries text and requests no further tools — i.e. the text the user
    actually sees, which is the only text the output-boundary gate applies to.
    """
    messages = list(messages)
    query = ""
    answer = ""
    calls: dict[str, ToolCall] = {}
    ordered: list[ToolCall] = []
    unmatched: list[ToolCall] = []

    for msg in messages:
        kind = _msg_type(msg)

        if kind == "human" and not query:
            query = _text(_get(msg, "content"))

        elif kind == "ai":
            requested = _get(msg, "tool_calls") or []
            for tc in requested:
                call = ToolCall(
                    id=_get(tc, "id"),
                    name=_get(tc, "name") or "unknown",
                    args=_get(tc, "args") or _get(tc, "arguments") or {},
                )
                ordered.append(call)
                if call.id:
                    calls[call.id] = call
            text = _text(_get(msg, "content"))
            # A message that both narrates and requests tools is not the final
            # answer — the run continues, and a later message supersedes it.
            if text and not requested:
                answer = text

        elif kind == "tool":
            result, is_raw = coerce_result(_get(msg, "content"))
            call_id = _get(msg, "tool_call_id")
            target = calls.get(call_id) if call_id else None
            if target is None:
                # Result with no matching request (or an id-less framework):
                # keep it, since it still grounded the answer.
                target = ToolCall(id=call_id, name=_get(msg, "name") or "unknown")
                unmatched.append(target)
            target.result = result
            target.result_is_raw_text = is_raw

    return Triple(
        query=query,
        answer=answer,
        tool_calls=ordered + unmatched,
        **meta,
    )


def triple_from_state(state: Any, **meta: Any) -> Triple:
    """Convenience for ``graph.invoke(...)`` output, which is usually a dict
    with a ``messages`` channel."""
    messages: Optional[Iterable[Any]]
    if isinstance(state, dict):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)
    if messages is None:
        raise ValueError(
            "No 'messages' channel in the graph state. If the target agent "
            "keeps tool results elsewhere, write an adapter for that channel "
            "— do not fall back to parsing the answer text, which would make "
            "the ground truth circular."
        )
    return triple_from_messages(messages, **meta)
