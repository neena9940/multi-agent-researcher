# test_nodes.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from src.graph.nodes import planner_node

# Create a mock "whiteboard" using a real LangChain Message object
mock_state = {
    "messages": [HumanMessage(content="What are the health benefits of intermittent fasting?")],
    "plan": None,
    "findings": [],
    "draft_answer": None,
    "verdict": None,
    "revision_count": 0
}

# --- DEBUG: Let's see exactly what is in the messages list ---
print("DEBUG: Type of last message is:", type(mock_state["messages"][-1]))
print("DEBUG: Last message is:", mock_state["messages"][-1])
# -------------------------------------------------------------

# Run the planner node
updates = planner_node(mock_state)
print("--- PLANNER OUTPUT ---")
print(updates["plan"])