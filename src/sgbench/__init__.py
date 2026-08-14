"""sgbench — benchmark harness for numeric grounding in finance AI copilots."""

from sgbench.capture import ToolCall, Triple, coerce_result
from sgbench.verify import Outcome, describe, summarize, verify_triple

__version__ = "0.1.0"

__all__ = [
    "ToolCall",
    "Triple",
    "coerce_result",
    "Outcome",
    "verify_triple",
    "describe",
    "summarize",
]
