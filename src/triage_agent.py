"""Customer Support Triage Agent — LangGraph + LangSmith demo.

A four-node graph that classifies, retrieves, drafts, and either finalizes
or escalates a customer-support ticket. Every node emits a LangSmith trace.

Run:
    python src/triage_agent.py
    python src/triage_agent.py "My card was charged twice last Tuesday"
"""

from __future__ import annotations

import os
import re
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain.agents import create_agent

from knowledge_base import search
from tools import (
    get_customer_recent_activity,
    get_recent_system_status,
    search_knowledge_base,
)

load_dotenv()

MODEL = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=512,
)

Category = Literal["billing", "technical", "vague"] #hard coded. add more for a more fleshed-out version


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
    """Pull KB entries via semantic search over Chroma, filtered by category.
    Only fires for billing and technical tickets; vague tickets are routed
    to the ReAct sub-agent instead (see route_after_classify)."""
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
    """Ask the model to grade its own draft on a 0.0-1.0 scale.

    Defensive parsing: the model frequently adds prose or markdown around
    the number even when told not to. We extract the first decimal number
    from the response with a regex so a stray newline or comment doesn't
    force us to fall back to 0.
    """
    if state["draft_reply"] is None:
        return {"confidence": 0.0}

    system = SystemMessage(
        content=(
            "Grade the draft reply's confidence as a single decimal number "
            "between 0.0 and 1.0. "
            "1.0 = directly answers the ticket using cited KB. "
            "0.0 = does not answer or is speculative. "
            "Reply with ONLY the number (e.g. 0.85). No prose, no markdown."
        )
    )
    user = HumanMessage(
        content=f"Ticket:\n{state['ticket']}\n\nDraft:\n{state['draft_reply']}"
    )
    reply = MODEL.invoke([system, user])

    match = re.search(r"\d*\.?\d+", reply.content)
    score = float(match.group()) if match else 0.0
    return {"confidence": max(0.0, min(1.0, score))}


def route_on_confidence(state: TriageState) -> Literal["done", "human_review"]:
    """Branching edge: high-confidence → done, otherwise → human_review."""
    if (state["confidence"] or 0.0) >= 0.7 and state["draft_reply"]:
        return "done"
    return "human_review"


def route_after_classify(state: TriageState) -> Literal["retrieve", "react_agent"]:
    """Branch right after classification: well-scoped categories (billing,
    technical) go through the straight-line RAG path; vague tickets go to a
    ReAct sub-agent that gathers context with tools before drafting."""
    if state["category"] == "vague":
        return "react_agent"
    return "retrieve"


# ---- ReAct sub-agent for vague tickets ----------------------------------------

REACT_SYSTEM_PROMPT = """You are a customer support agent handling an \
ambiguous ticket where you do not have enough context to answer directly.

Your job: use the tools available to gather context, then draft a brief, \
warm customer-facing reply that either (1) gives a likely answer if your \
investigation gives you confidence, or (2) asks one focused clarifying \
question, ideally enriched by the context you discovered.

Keep replies to 3-5 sentences. Always close politely. Cite any KB article \
IDs you used in brackets at the end. Do not make up information that the \
tools did not return."""


_REACT_EXECUTOR = create_agent(
    model=MODEL,
    tools=[
        get_recent_system_status,
        get_customer_recent_activity,
        search_knowledge_base,
    ],
    system_prompt=REACT_SYSTEM_PROMPT,
)


def react_agent(state: TriageState) -> dict:
    """Run a ReAct loop on a vague ticket. The agent reasons about what
    context it needs, calls tools (zero, one, or many times) to gather it,
    and then drafts a customer-facing reply. The reply lands in draft_reply
    so the downstream confidence_check and routing nodes work unchanged."""
    result = _REACT_EXECUTOR.invoke(
        {"messages": [HumanMessage(content=state["ticket"])]}
    )
    final_message = result["messages"][-1]
    return {"draft_reply": final_message.content.strip()}


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
    g.add_node("react_agent", react_agent)
    g.add_node("confidence_check", confidence_check)
    g.add_node("done", finalize)
    g.add_node("human_review", escalate)

    g.set_entry_point("classify")

    # Branch right after classify: scoped categories go through the
    # straight-line RAG path, vague tickets go to the ReAct sub-agent.
    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {"retrieve": "retrieve", "react_agent": "react_agent"},
    )

    # RAG branch
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "confidence_check")

    # ReAct branch — converges back at confidence_check
    g.add_edge("react_agent", "confidence_check")

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
    print(f"CONFIDENCE : {result['confidence']:.2f}" if result["confidence"] is not None else "CONFIDENCE : n/a")
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
