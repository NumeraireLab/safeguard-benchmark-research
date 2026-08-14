"""The captured unit of study: one complete agent turn.

The whole benchmark rests on one property — **ground truth is free**. The tool
results *are* the truth by definition, so no human labelling is needed: a
number in the answer either appears in what the agent retrieved, or it does
not. That property only holds if we can capture what the agent actually
retrieved, which is why `Triple` requires tool calls to be present and why
Phase 0 gates the whole study on extracting one of these successfully.

The shape below is deliberately close to Safeguard's own full-turn record
(query, tool calls with arguments, results, answer, versions) so the study
produces data in the schema the product wants to standardise.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One tool invocation and what it returned.

    ``args`` is recorded even though the output-boundary check does not use
    it: it is what a tool-boundary check would need later, and it costs
    nothing to capture now while the agent is running.
    """

    id: Optional[str] = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    # True when ``result`` could not be parsed as JSON and is being carried as
    # raw text. Prose results contribute no numbers to the allowed set unless
    # the caller opts into string scanning, so a study run where most results
    # are raw is a study measuring almost nothing — surfaced, not hidden.
    result_is_raw_text: bool = False


class Triple(BaseModel):
    """(question, what was retrieved, what was said) — the unit of analysis."""

    query: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    answer: str = ""

    # Provenance of the run itself. Without these a result is not reproducible
    # and therefore not publishable.
    query_id: Optional[str] = None
    tier: Optional[str] = None
    model: Optional[str] = None
    target: Optional[str] = None
    safeguard_version: Optional[str] = None
    harness_version: str = "0.1.0"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def is_complete(self) -> bool:
        """Phase 0's gate. A triple with no tool calls cannot ground anything:
        every number in the answer would be flagged, which measures the
        capture failure rather than the model."""
        return bool(self.query and self.answer and self.tool_calls)

    def source_values(self) -> dict[str, Any]:
        """The grounding payload for Safeguard.

        Keyed per call rather than merged into one blob, so a flagged number
        can be attributed back to *which* call produced (or failed to produce)
        it. Duplicate tool names are disambiguated by position, keeping the
        JSON paths stable and readable in an audit record.
        """
        out: dict[str, Any] = {}
        seen: dict[str, int] = {}
        for call in self.tool_calls:
            seen[call.name] = seen.get(call.name, 0) + 1
            key = call.name if seen[call.name] == 1 else f"{call.name}_{seen[call.name]}"
            out[key] = call.result
        return out


def coerce_result(raw: Any) -> tuple[Any, bool]:
    """Parse a tool result into JSON where possible.

    Frameworks hand tool output back as a string even when the tool produced
    structured data. Parsing it is what makes the numbers visible to
    ``walk_numeric``; failing to parse is recorded rather than silently
    accepted, because an unparsed result silently shrinks the allowed set and
    turns real answers into false freezes.
    """
    if not isinstance(raw, str):
        return raw, False
    text = raw.strip()
    if not text:
        return raw, True
    if text[0] in "[{" or text[0].isdigit() or text[0] == "-":
        try:
            return json.loads(text), False
        except (json.JSONDecodeError, ValueError):
            pass
    return raw, True
