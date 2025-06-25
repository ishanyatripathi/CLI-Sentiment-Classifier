# main.py

from langgraph.graph import StateGraph, END
from nodes.inference_node import InferenceNode
from nodes.check_confidence_node import ConfidenceCheckNode
from nodes.fallback_node import FallbackNode
from dataclasses import dataclass
import datetime
import os

# ✅ Define the state schema using dataclass
@dataclass
class State:
    text: str
    label: str = None
    confidence: float = None
    needs_fallback: bool = False
    corrected_by_user: bool = False

# 📁 Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# 📝 Logging helper
def log_interaction(state) -> dict:
    with open("logs/chatlog.txt", "a") as log:
        log.write(f"\n=== {datetime.datetime.now()} ===\n")

        # ✅ Use __dict__ if it's a dataclass, otherwise use .items()
        if hasattr(state, "__dict__"):
            items = state.__dict__.items()
        else:
            items = state.items()

        for k, v in items:
            log.write(f"{k}: {v}\n")

    return state


# ⚙️ Initialize LangGraph nodes
inference = InferenceNode()
check_conf = ConfidenceCheckNode()
fallback = FallbackNode()

# 🔧 Build LangGraph with state schema
builder = StateGraph(State)

# 🔗 Add nodes
builder.add_node("inference", inference)
builder.add_node("check_confidence", check_conf)
builder.add_node("fallback", fallback)
builder.add_node("log", log_interaction)

# ➡️ Define flow
builder.set_entry_point("inference")
builder.add_edge("inference", "check_confidence")

# 🔁 Conditional routing
def route_fallback(state):
    if state.needs_fallback:
        return "fallback"
    return "log"

builder.add_conditional_edges("check_confidence", route_fallback)
builder.add_edge("fallback", "log")
builder.add_edge("log", END)

# ✅ Compile the graph
graph = builder.compile()

# 🖥️ CLI interface
print("🤖 Sentiment Classifier CLI (type /exit to quit)")
while True:
    user_input = input("\n📝 Enter your review: ").strip()
    if user_input.lower() == "/exit":
        print("👋 Exiting chatbot. Goodbye!")
        break

    final_state = graph.invoke({"text": user_input})

    # ✅ Corrected dict-style access
    print(f"\n✅ Final Label: {final_state['label']} (Confidence: {final_state['confidence']}%)")

    if final_state.get("corrected_by_user", False):
        print("📝 Label was corrected by user.")
