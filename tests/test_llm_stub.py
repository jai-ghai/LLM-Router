from __future__ import annotations

import pytest

from tierrouter.llm_stub import get_client


def test_get_client_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="LangChain"):
        get_client("some-model")
