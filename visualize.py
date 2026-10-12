# visualize.py
import os
from dotenv import load_dotenv

# 1. LOAD THE .ENV FILE FIRST!
load_dotenv()

# 2. Now it's safe to import the graph builders
from src.graph.builder import build_planner_graph, build_research_graph


def visualize():
    print("🏗️ Building graphs to extract structure...")

    # 1. Visualize Phase 1 (The Planner)
    print("\n--- PHASE 1: PLANNER GRAPH ---")
    planner_graph = build_planner_graph()
    print(planner_graph.get_graph().draw_ascii())

    # 2. Visualize Phase 2 (The Research Loop)
    print("\n--- PHASE 2: RESEARCH GRAPH ---")
    research_graph = build_research_graph()

    # Print ASCII to terminal
    print(research_graph.get_graph().draw_ascii())

    # Generate a PNG image file
    os.makedirs("graphs", exist_ok=True)
    try:
        png_data = research_graph.get_graph().draw_mermaid_png()
        with open("graphs/research_loop.png", "wb") as f:
            f.write(png_data)
        print("\n✅ SUCCESS! Image saved to: graphs/research_loop.png")
        print("   (Open this file in your Mac Finder to see the diagram!)")
    except Exception as e:
        print(f"\n⚠️ Could not generate PNG (missing system dependency). Error: {e}")
        print("   But the ASCII art above still shows your exact graph structure!")


if __name__ == "__main__":
    visualize()