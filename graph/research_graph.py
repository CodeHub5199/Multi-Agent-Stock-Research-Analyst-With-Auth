"""
graph/research_graph.py
-----------------------
LangGraph multi-agent research pipeline.

Topology (parallel fan-out → synthesis → critic → report):

    START
      ├─► fundamental_research ─┐
      ├─► news_research         ├─► synthesis_research ─► critic_research ─► END
      └─► technical_research   ─┘
"""

from typing import TypedDict, Any
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from agents.fundamental_research import FundamentalsOutput, fundamental_research_agent
from agents.news_research import NewsOutput, news_research_agent
from agents.technical_research import TechnicalOutput, technical_research_agent
from agents.synthesis_research import SynthesisOutput, synthesis_research_agent
from agents.critic_research import CriticOutput, critic_research_agent


# ── Shared state schema ───────────────────────────────────────────────

class ResearchState(TypedDict):
    ticker: str
    depth: str | None
    fundamentals_output: FundamentalsOutput
    news_output: NewsOutput
    technical_output: TechnicalOutput
    synthesis_output: SynthesisOutput
    critic_output: CriticOutput


# ── Node factory functions ─────────────────────────────────────────────

def make_fundamental_node(llm: ChatGroq):
    def node(state: ResearchState):
        ticker = state["ticker"].strip().upper()
        output = fundamental_research_agent(ticker, llm)
        return {"fundamentals_output": output}
    node.__name__ = "fundamental_research"
    return node


def make_news_node(llm: ChatGroq):
    def node(state: ResearchState):
        output = news_research_agent(state, llm)
        return {"news_output": output}
    node.__name__ = "news_research"
    return node


def make_technical_node(llm: ChatGroq):
    def node(state: ResearchState):
        output = technical_research_agent(state, llm)
        return {"technical_output": output}
    node.__name__ = "technical_research"
    return node


def make_synthesis_node(llm: ChatGroq):
    def node(state: ResearchState):
        output = synthesis_research_agent(state, llm)
        return {"synthesis_output": output}
    node.__name__ = "synthesis_research"
    return node


def make_critic_node(llm: ChatGroq):
    def node(state: ResearchState):
        output = critic_research_agent(state, llm)
        return {"critic_output": output}
    node.__name__ = "critic_research"
    return node


# ── Graph builder ─────────────────────────────────────────────────────

def build_research_graph(
    *,
    fundamental_llm: ChatGroq,
    news_llm: ChatGroq,
    technical_llm: ChatGroq,
    synthesis_llm: ChatGroq,
    critic_llm: ChatGroq,
) -> Any:
    """
    Construct and compile the LangGraph state machine.
    Returns a compiled graph ready for `.invoke()` / `.astream()`.
    """
    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("fundamental_research", make_fundamental_node(fundamental_llm))
    builder.add_node("news_research", make_news_node(news_llm))
    builder.add_node("technical_research", make_technical_node(technical_llm))
    builder.add_node("synthesis_research", make_synthesis_node(synthesis_llm))
    builder.add_node("critic_research", make_critic_node(critic_llm))

    # Parallel fan-out from START
    builder.add_edge(START, "fundamental_research")
    builder.add_edge(START, "news_research")
    builder.add_edge(START, "technical_research")

    # Fan-in to synthesis
    builder.add_edge("fundamental_research", "synthesis_research")
    builder.add_edge("news_research", "synthesis_research")
    builder.add_edge("technical_research", "synthesis_research")

    # Linear chain to critic → END
    builder.add_edge("synthesis_research", "critic_research")
    builder.add_edge("critic_research", END)

    return builder.compile()
