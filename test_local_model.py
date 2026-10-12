# test_local_model.py
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

# Test with qwen2.5:7b
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    base_url="http://localhost:11434"
)

prompt = """You are a research planner. Decompose this research question into 3-5 sub-questions.
Question: What are the health benefits of intermittent fasting?

CRITICAL: Output ONLY a valid JSON object with this exact structure:
{
  "sub_questions": ["question 1", "question 2", "question 3"],
  "rationale": "why these questions"
}

Do NOT add any text before or after the JSON. Do NOT use markdown."""

print("🧠 Asking local model to generate a plan...")
print("="*60)

response = llm.invoke(prompt)
response_text = response.content if hasattr(response, 'content') else str(response)

print("RAW MODEL OUTPUT:")
print("="*60)
print(response_text)
print("="*60)