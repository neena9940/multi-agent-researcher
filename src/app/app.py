import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# 1. Load API keys
load_dotenv()

# 2. Import our graph
from src.graph.builder import build_graph

# --- UI SETUP ---
st.set_page_config(page_title="AI Researcher", layout="wide")
st.title("🕵️ Multi-Agent Research Assistant")
st.markdown("An AI team that plans, researches, writes, and critiques. You control the plan!")

# --- STREAMLIT MEMORY (Session State) ---
# Web apps are "stateless" (they forget everything between button clicks).
# st.session_state is Streamlit's way of remembering things.
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default-thread"

# --- 1. USER INPUT ---
user_question = st.text_input("What would you like to research?", placeholder="e.g., Is intermittent fasting healthy?")

if st.button("🚀 Start Research"):
    if user_question:
        # Generate a unique ID for this specific research run
        st.session_state.thread_id = f"thread-{os.urandom(4).hex()}"
        st.session_state.stage = "planning"  # Track where we are in the UI

        initial_state = {
            "messages": [HumanMessage(content=user_question)],
            "plan": None, "findings": [], "draft_answer": None, "verdict": None, "revision_count": 0
        }
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Run the graph. It will pause at the Planner's interrupt()
        with st.spinner("🧠 Planner is generating a research plan..."):
            st.session_state.graph.invoke(initial_state, config)

        st.rerun()  # Refresh the page to show the paused plan

# --- 2. DISPLAYING RESULTS ---
if "stage" in st.session_state:

    # SCENARIO A: The graph is paused (Human-in-the-Loop)
    if st.session_state.stage == "planning":
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        snapshot = st.session_state.graph.get_state(config)
        current_plan = snapshot.values.get("plan")

        if current_plan:
            st.subheader("⏸️ Review the Research Plan")
            st.info("The AI wants to research the following. You can approve or add your own question.")

            for i, q in enumerate(current_plan.sub_questions, 1):
                st.markdown(f"**{i}.** {q}")
            st.caption(f"*Rationale: {current_plan.rationale}*")

            human_addition = st.text_input("Add a custom sub-question (optional):")

            if st.button("✅ Approve & Continue"):
                if human_addition.strip():
                    current_plan.sub_questions.append(human_addition.strip())

                # Wake up the graph with the (potentially edited) plan
                with st.spinner("🔍 Agents are researching, writing, and critiquing..."):
                    st.session_state.graph.invoke(Command(resume=current_plan), config)

                st.session_state.stage = "finished"
                st.rerun()

    # SCENARIO B: The graph has finished completely
    elif st.session_state.stage == "finished":
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        final_state = st.session_state.graph.get_state(config).values

        st.subheader("✅ Final Research Report")
        st.markdown(final_state.get("draft_answer", "No answer generated."))

        st.divider()

        # Hide the complex metadata behind a clickable dropdown
        with st.expander("📊 View Critic Verdict & Metadata"):
            verdict = final_state.get("verdict")
            if verdict:
                st.markdown(f"**Supported by sources?** {'✅ Yes' if verdict.is_supported else '❌ No'}")
                st.markdown(f"**Total Revisions:** {final_state.get('revision_count', 0)}")
                if not verdict.is_supported:
                    st.markdown(f"**Critic Feedback:** {verdict.feedback}")