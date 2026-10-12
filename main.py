# main.py
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
from src.graph.builder import build_planner_graph, build_research_graph


def main():
    print("🏗️ Building graphs...")
    planner_graph = build_planner_graph()
    research_graph = build_research_graph()

    initial_state = {
        "messages": [HumanMessage(content="What are the health benefits of intermittent fasting?")],
        "plan": None, "findings": [], "draft_answer": None, "verdict": None, "revision_count": 0
    }

    config = {"configurable": {"thread_id": "research-run-1"}}

    # === PHASE 1: PLANNING (with human review) ===
    print("\n" + "=" * 60)
    print("📋 PHASE 1: PLANNING")
    print("=" * 60)
    print("🚀 Generating research plan...")

    plan_state = planner_graph.invoke(initial_state, config)
    current_plan = plan_state.get("plan")

    if current_plan is None:
        print("⚠️ ERROR: Planner failed to generate a plan.")
        return

    print("\n⏸️ HUMAN REVIEW REQUIRED")
    print("-" * 60)
    print("The AI generated the following plan:")

    sub_questions = current_plan.get("sub_questions", [])
    rationale = current_plan.get("rationale", "No rationale provided.")

    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")
    print(f"\nRationale: {rationale}")

    # Get human input
    user_choice = input("\nType 'approve' to continue, or type a new sub-question to add: ")

    if user_choice.lower().strip() != 'approve':
        print("✏️ Adding your question to the plan...")
        current_plan["sub_questions"].append(user_choice.strip())

    print("\n✅ Plan approved! Moving to Phase 2...")

    # === PHASE 2: RESEARCH LOOP ===
    print("\n" + "=" * 60)
    print("🔍 PHASE 2: RESEARCH & WRITING")
    print("=" * 60)

    # Pass the approved plan to the research graph
    research_state = {
        "messages": plan_state["messages"],
        "plan": current_plan,
        "findings": [],
        "draft_answer": None,
        "verdict": None,
        "revision_count": 0
    }

    final_state = research_graph.invoke(research_state, config)

    # === DISPLAY RESULTS ===
    print("\n" + "=" * 60)
    print("✅ FINAL DRAFT ANSWER:")
    print("=" * 60)
    print(final_state.get("draft_answer", "No answer generated."))

    print("\n" + "=" * 60)
    print("📊 CRITIC VERDICT:")
    print("=" * 60)
    verdict = final_state.get("verdict")
    if verdict:
        print(f"Is Supported: {verdict.get('is_supported', 'Unknown')}")
        print(f"Total Revisions: {final_state.get('revision_count', 0)}")
        if verdict.get("feedback"):
            print(f"Feedback: {verdict.get('feedback')}")


if __name__ == "__main__":
    main()