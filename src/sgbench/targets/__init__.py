"""Study targets — the agents being measured.

A target is a callable `(query) -> messages`. Keeping it that narrow is what
lets the runner stay agnostic while adapters and targets change underneath.
"""
