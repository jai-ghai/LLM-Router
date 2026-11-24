"""Stub only. The team decided (2026-11-16) not to build the LangChain integration yet.

When that phase starts, `get_client` returns a LangChain chat model for `model_name`. Until
then it raises, loudly, rather than returning something that looks like it works.
"""

from __future__ import annotations


def get_client(model_name: str) -> None:
    raise NotImplementedError(
        f"LangChain integration is not built yet (model_name={model_name!r}). "
        "See ARCHITECTURE.md §1 and PLAN.md Task 7."
    )
