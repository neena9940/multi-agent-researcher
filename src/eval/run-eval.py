import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load API keys
load_dotenv()

# Import our graph builder
from src.graph.builder import build_graph

# 1. Define the Test Dataset
TEST_SET = [
    {
        "id": 1,
        "question": "What are the health benefits of intermittent fasting?",
        "must_contain_keywords": ["weight", "insulin", "metabolism"],
        "min_citations": 2
    },
    {
        "id": 2,
        "question": "How does photosynthesis work in simple terms?",
        "must_contain_keywords": ["sunlight", "chlorophyll", "glucose"],
        "min_citations": 2
    }
]


def run_evaluation():
    print("🏗️ Building graph for evaluation...")
    graph = build_graph()

    results = []

    # 2. Loop through each test case
    for case in TEST_SET:
        print(f"\n🔍 Evaluating Case {case['id']}: {case['question'][:40]}...")

        initial_state = {
            "messages": [HumanMessage(content=case["question"])],
            "plan": None, "findings": [], "draft_answer": None, "verdict": None, "revision_count": 0
        }

        config = {"configurable": {"thread_id": f"eval-run-{case['id']}"}}

        start_time = time.time()

        # 3. Run the graph
        try:
            final_state = graph.invoke(initial_state, config)
            latency = time.time() - start_time

            draft = final_state.get("draft_answer", "")
            verdict = final_state.get("verdict")

            # 4. Calculate Metrics
            # Metric A: Keyword Presence (Did it cover the required topics?)
            keywords_found = sum(1 for kw in case["must_contain_keywords"] if kw.lower() in draft.lower())
            keyword_score = (keywords_found / len(case["must_contain_keywords"])) * 100

            # Metric B: Citation Count (Did it follow the prompt to add [source]?)
            citation_count = draft.lower().count("[source]")
            citation_pass = citation_count >= case["min_citations"]

            # Metric C: Critic Approval (Did our own agent catch hallucinations?)
            critic_approved = verdict.is_supported if verdict else False

            # 5. Store Results
            results.append({
                "id": case["id"],
                "latency_sec": round(latency, 2),
                "keyword_score_pct": round(keyword_score, 1),
                "citations": citation_count,
                "critic_approved": critic_approved,
                "revisions": final_state.get("revision_count", 0)
            })

        except Exception as e:
            print(f"❌ Case {case['id']} failed with error: {e}")
            results.append({"id": case["id"], "status": "FAILED"})

    # 6. Print Summary Report
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(
        f"{'ID':<4} | {'Latency (s)':<12} | {'Keywords (%)':<12} | {'Citations':<10} | {'Critic OK':<10} | {'Revisions'}")
    print("-" * 70)

    for r in results:
        if "status" in r and r["status"] == "FAILED":
            print(f"{r['id']:<4} | {'FAILED':<52}")
        else:
            print(
                f"{r['id']:<4} | {r['latency_sec']:<12} | {r['keyword_score_pct']:<12} | {r['citations']:<10} | {str(r['critic_approved']):<10} | {r['revisions']}")

    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()