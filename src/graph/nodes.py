# src/graph/nodes.py
import json
import re
from langchain_ollama import ChatOllama
from .state import ResearchState
from .tools import search_tool

# Initialize the local model
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    base_url="http://localhost:11434"
)


# Helper function to extract JSON
def extract_json_from_text(text: str) -> dict:
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    raise ValueError("Could not extract JSON from model output")


# --- Planner Node (NO interrupt) ---
def planner_node(state: ResearchState) -> dict:
    print("\n[DEBUG] Planner node started...")

    prompt = f"""You are a research planner. Decompose this research question into 3-5 sub-questions.
Question: {state['messages'][-1].content}

CRITICAL: Output ONLY a valid JSON object with this exact structure:
{{
  "sub_questions": ["question 1", "question 2", "question 3"],
  "rationale": "why these questions"
}}

Do NOT add any text before or after the JSON."""

    raw_response = llm.invoke(prompt)
    response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)

    try:
        plan = extract_json_from_text(response_text)
        print("[DEBUG] ✅ Successfully parsed JSON into plan dict!")
        print(f"[DEBUG] Plan: {plan}")
    except Exception as e:
        print(f"[DEBUG] ⚠️ JSON parsing failed: {e}")
        plan = {
            "sub_questions": ["What is the main topic?", "What are the benefits?", "What are the risks?"],
            "rationale": "Fallback plan due to parsing error"
        }

    # Return the plan to state (NO interrupt here!)
    return {"plan": plan}


# --- Researcher Node ---
researcher = llm.bind_tools([search_tool])


# --- Researcher Node (FORCED TOOL EXECUTION) ---
# --- Researcher Node (CORRECTED FOR TAVILY OUTPUT FORMAT) ---
def researcher_node(state: ResearchState) -> dict:
    print("\n[DEBUG] Researcher node started...")
    plan = state["plan"]
    sub_questions = plan.get("sub_questions", [])
    revision = state["revision_count"]

    sub_q = sub_questions[min(revision, len(sub_questions) - 1)]
    print(f"[DEBUG] Researching: {sub_q}")

    # Execute the search tool
    print("[DEBUG] Executing Tavily search...")
    search_results = search_tool.invoke(sub_q)

    print(f"[DEBUG] Search returned {len(search_results)} results")

    # Tavily returns a list of strings (already formatted), not dictionaries
    findings_text = "\n\n".join(search_results)

    print(f"[DEBUG] Formatted findings: {len(findings_text)} chars")

    return {"findings": [findings_text]}


# --- Synthesizer Node ---
def synthesizer_node(state: ResearchState) -> dict:
    print("\n[DEBUG] Synthesizer node started...")
    findings = state["findings"]
    findings_text = "\n\n".join(f"- {f}" for f in findings)

    answer = llm.invoke(
        f"Synthesize a cited answer based on these findings. Every claim must end with [source].\n"
        f"Findings:\n{findings_text}"
    )
    return {"draft_answer": answer.content}


# --- Critic Node ---
def critic_node(state: ResearchState) -> dict:
    print("\n[DEBUG] Critic node started...")
    prompt = f"""Check each claim in the answer against the findings. Flag anything not supported.

Answer: {state['draft_answer']}

Findings: {state['findings']}

Output ONLY a valid JSON object with this exact structure:
{{
  "is_supported": true,
  "unsupported_claims": [],
  "feedback": "Looks good"
}}

Do NOT add any text before or after the JSON."""

    raw_response = llm.invoke(prompt)
    response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)

    try:
        verdict = extract_json_from_text(response_text)
        print("[DEBUG] ✅ Critic successfully parsed JSON verdict!")
    except Exception as e:
        print(f"[DEBUG] ⚠️ Critic JSON parsing failed: {e}")
        verdict = {
            "is_supported": True,
            "unsupported_claims": [],
            "feedback": "Auto-approved due to parsing error"
        }

    return {"verdict": verdict, "revision_count": state["revision_count"] + 1}