"""Compatibility shim: import this module before anything that imports `ragas`.

`ragas` (every version back to at least 0.2.15, including the current 0.4.3)
eagerly does `from langchain_community.chat_models.vertexai import
ChatVertexAI` at import time inside `ragas/llms/base.py`, purely to register
one optional provider integration. `langchain-community` is being sunset
(https://github.com/langchain-ai/langchain-community/issues/674) and the
installed version (0.4.2) no longer ships that submodule, so `import ragas`
fails with `ModuleNotFoundError` before you ever get a chance to pick a
provider — even though this project never uses Vertex AI. Installing the real
`langchain-google-vertexai` package just to satisfy an unused import isn't
worth the extra dependency weight, so this registers a minimal stand-in
module instead.
"""

from __future__ import annotations

import sys
import types


def _install_vertexai_stub() -> None:
    if "langchain_community.chat_models.vertexai" in sys.modules:
        return  # already installed (or the real thing is present)

    class _UnavailableVertexAI:
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError(
                "Vertex AI is not configured for this project — this is a stub "
                "registered by eval/_ragas_compat.py so `import ragas` succeeds."
            )

    chat_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    chat_stub.ChatVertexAI = _UnavailableVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = chat_stub

    # ragas/llms/base.py also does `from langchain_community.llms import
    # VertexAI`, which the real `langchain_community.llms` module no longer
    # exports — patch just that one missing name onto the real module rather
    # than replacing it, so every other integration it provides still works.
    import langchain_community.llms as real_llms

    if not hasattr(real_llms, "VertexAI"):
        real_llms.VertexAI = _UnavailableVertexAI


_install_vertexai_stub()
