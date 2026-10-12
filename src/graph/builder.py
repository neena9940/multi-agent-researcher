# src/graph/builder.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import ResearchState
from .nodes import planner_node, researcher_node, synthesizer_node, critic_node

MAX_REVISIONS = 2


def build_planner_graph():
    """Phase 1: Just the planner (for human review)"""
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", END)
    return g.compile(checkpointer=MemorySaver())


def build_research_graph():
    """Phase 2: The full research loop"""
    g = StateGraph(ResearchState)

    g.add_node("researcher", researcher_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("critic", critic_node)

    g.add_edge(START, "researcher")
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", "critic")

    def route_after_critic(state: ResearchState):
        verdict = state.get("verdict", {})
        if verdict.get("is_supported", False) or state["revision_count"] >= MAX_REVISIONS:
            return END
        return "researcher"

    g.add_conditional_edges("critic", route_after_critic)

    return g.compile(checkpointer=MemorySaver())