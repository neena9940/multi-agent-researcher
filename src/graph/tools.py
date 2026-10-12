from langchain_tavily import TavilySearch
from langchain_core.tools import tool

search_tool = TavilySearch(max_results=5)

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use for any arithmetic."""
    return str(eval(expression))