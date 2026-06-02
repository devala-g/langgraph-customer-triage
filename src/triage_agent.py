"""Customer Support Triage Agent — LangGraph + LangSmith demo.

A four-node graph that classifies, retrieves, drafts, and either finalizes
or escalates a customer-support ticket. Every node emits a LangSmith trace.

Run:
    python src/triage_agent.py
    python src/triage_agent.py "My card was charged twice last Tuesday"
"""

from __future__ import annotations

import os
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from knowledge_base import search

load_dotenv()

MODEL = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=512,
)

Category = Literal["billing", "technical", "vague"]


class TriageState(TypedDict):
    ticket: str
    category: Category | None
    kb_hits: list[dict]
    draft_reply: str | None
    confidence: float | None
    final_decision: Literal["done", "human_review"] | None


# ---- Nodes -------------------------------------------------------------------


def classify(state: TriageState) -> dict:
    """Decide whether the ticket is billing, technical, or too vague to handle."""
    system = SystemMessage(
        content=(
            "Classify the customer ticket as exactly one of: billing, technical, vague. "
            "Reply with only the single word, lowercase."
        )
    )
    reply = MODEL.invoke([system, HumanMessage(content=state["ticket"])])
    label = reply.content.strip().lower()
    if label not in ("billing", "technical", "vague"):
        label = "vague"
    return {"category": label}


def retrieve(state: TriageState) -> dict:
    """Pull matching KB entries. Vague tickets skip retrieval."""
    if state["category"] == "vague":
        return {"kb_hits": []}
    hits = search(state["category"], state["ticket"])
    return {"kb_hits": hits}


def draft(state: TriageState) -> dict:
    """Compose a customer-facing reply grounded in the KB hits."""
    if not state["kb_hits"]:
        return {"draft_reply": None}

    sources = "\n\n".join(
        f"[{e['id']}] {e['title']}: {e['body']}" for e in state["kb_hits"]
    )
    system = SystemMessage(
        content=(
            "You are a customer support agent. Draft a short, warm reply to "
            "the ticket below, grounded ONLY in the knowledge-base entries "
            "provided. Cite the entry IDs you used at the end in brackets."
        )
    )
    user = HumanMessage(
        content=f"Ticket:\n{state['ticket']}\n\nKnowledge base:\n{sources}"
    )
    reply = MODEL.invoke([system, user])
    return {"draft_reply": reply.content.strip()}


def confidence_check(state: TriageState) -> dict:
    """Ask the model to grade its own draft on a 0.0-1.0 scale."""
    if state["draft_reply"] is None:
        return {"confidence": 0.0}

    system = SystemMessage(
        content=(
            "Grade the draft reply's confidence on a 0.0-1.0 scale. "
            "1.0 = directly answers the ticket using cited KB. "
            "0.0 = does not answer or is speculative. "
            "Reply with only the number."
        )
    )
    user = HumanMessage(
        content=f"Ticket:\n{state['ticket']}\n\nDraft:\n{state['draft_reply']}"
    )
    reply = MODEL.invoke([system, user])
    try:
        score = float(reply.content.strip())
    except ValueError:
        score = 0.0
    return {"confidence": max(0.0, min(1.0, score))}


def route_on_confidence(state: TriageState) -> Literal["done", "human_review"]:
    """Branching edge: high-confidence → done, otherwise → human_review."""
    if (state["confidence"] or 0.0) >= 0.7 and state["draft_reply"]:
        return "done"
    return "human_review"


def finalize(state: TriageState) -> dict:
    return {"final_decision": "done"}


def escalate(state: TriageState) -> dict:
    return {"final_decision": "human_review"}


# ---- Graph -------------------------------------------------------------------


def build_graph():
    g = StateGraph(TriageState)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_node("confidence_check", confidence_check)
    g.add_node("done", finalize)
    g.add_node("human_review", escalate)

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "confidence_check")
    g.add_conditional_edges(
        "confidence_check",
        route_on_confidence,
        {"done": "done", "human_review": "human_review"},
    )
    g.add_edge("done", END)
    g.add_edge("human_review", END)
    return g.compile()


# ---- CLI ---------------------------------------------------------------------


SAMPLE_TICKETS = [
    "Hi, I think you charged my card twice for last Tuesday's order. Can you check?",
    "I keep getting kicked back to the login page after I sign in. I cleared my cache already.",
    "It doesn't work anymore.",
]


def run(ticket: str) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY. Copy .env.example to .env and fill it in.")

    graph = build_graph()
    result = graph.invoke(
        {
            "ticket": ticket,
            "category": None,
            "kb_hits": [],
            "draft_reply": None,
            "confidence": None,
            "final_decision": None,
        }
    )

    print("\n" + "=" * 70)
    print(f"TICKET     : {ticket}")
    print(f"CATEGORY   : {result['category']}")
    print(f"KB HITS    : {[e['id'] for e in result['kb_hits']]}")
    print(f"CONFIDENCE : {result['confidence']:.2f}" if result["confidence"] else "CONFIDENCE : n/a")
    print(f"DECISION   : {result['final_decision']}")
    if result["draft_reply"]:
        print(f"\nDRAFT REPLY:\n{result['draft_reply']}")
    else:
        print("\nNo draft generated — escalating to human review.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    tickets = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else SAMPLE_TICKETS
    for t in tickets:
        run(t)
