"""Capture adapters — one per agent framework or target copilot.

An adapter's only job is to turn a completed agent turn into a `Triple`. It
must read the *retrieved data*, never re-derive it from the answer text: the
study's ground truth is what the tools returned, and sourcing it from the
answer would make the measurement circular.
"""
