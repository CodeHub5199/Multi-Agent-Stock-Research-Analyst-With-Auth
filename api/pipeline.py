"""
api/pipeline.py
---------------
Async wrapper around the synchronous LangGraph research graph.

The graph's `.invoke()` is blocking, so we run it in a thread-pool
executor to keep FastAPI's event loop free.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from langchain_groq import ChatGroq

from api.config import get_settings

logger = logging.getLogger("stock_research.pipeline")


# ── LLM factory (one instance per model/key, cached) ─────────────────

@lru_cache(maxsize=8)
def _get_llm(model: str, api_key: str) -> ChatGroq:
    return ChatGroq(model=model, api_key=api_key, temperature=0)


def _build_graph():
    """
    Lazily import and compile the LangGraph research graph.
    All five agents share the same Groq LLM instance (different nodes
    may be swapped to different models via settings if needed).
    """
    from graph.research_graph import build_research_graph  # noqa: PLC0415

    settings = get_settings()
    llm = _get_llm(settings.groq_model, settings.groq_api_key)

    return build_research_graph(
        fundamental_llm=llm,
        news_llm=llm,
        technical_llm=llm,
        synthesis_llm=llm,
        critic_llm=llm,
    )


# Cache the compiled graph — LangGraph compilation is expensive
@lru_cache(maxsize=1)
def get_compiled_graph():
    logger.info("Compiling LangGraph research graph…")
    g = _build_graph()
    logger.info("Graph compiled ✓")
    return g


# ── Public coroutine ──────────────────────────────────────────────────

async def run_research_pipeline(
    ticker: str,
    depth: str | None = "standard",
) -> dict[str, Any]:
    """
    Run the full multi-agent pipeline for *ticker* and return the
    completed `final_state` dict.

    Executes the synchronous LangGraph `.invoke()` in a thread-pool
    executor so the FastAPI event loop remains unblocked.
    """
    graph = get_compiled_graph()
    initial_state: dict[str, Any] = {"ticker": ticker, "depth": depth}

    logger.info("Invoking graph for ticker=%s depth=%s", ticker, depth)

    loop = asyncio.get_running_loop()
    final_state: dict[str, Any] = await loop.run_in_executor(
        None,  # default ThreadPoolExecutor
        lambda: graph.invoke(initial_state),
    )

    logger.info("Graph completed for ticker=%s", ticker)
    return final_state
