# src/graph/state.py
from typing import Annotated, TypedDict, List
from operator import add
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    # Chat history - messages are appended, not overwritten
    messages: Annotated[list, add_messages]

    # Research plan as a plain dictionary (not Pydantic)
    plan: dict | None

    # Accumulated web search results
    findings: Annotated[list, add]

    # The synthesized draft answer
    draft_answer: str | None

    # The critic's evaluation as a plain dictionary
    verdict: dict | None

    # Counter to prevent infinite loops
    revision_count: int