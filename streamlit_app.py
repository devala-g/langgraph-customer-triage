"""Streamlit front-end for the Customer Support Triage Agent.

Wraps the LangGraph graph in ``src/triage_agent.py`` with a small web UI so the
agent can be demoed in a browser: paste a ticket, watch it route through the
graph node by node (classify -> retrieve/ReAct -> draft -> confidence ->
done/human_review), and read the drafted reply.

Deploy target: Streamlit Community Cloud. Set ``ANTHROPIC_API_KEY`` (and,
optionally, the ``LANGSMITH_*`` keys) in the app's Secrets; locally it falls
back to a ``.env`` file so the same script runs unchanged on a laptop.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# set_page_config must be the first Streamlit call.
st.set_page_config(
    page_title="Support Triage Agent",
    page_icon="🎧",
    layout="centered",
)

# --- Make src/ importable and load API keys BEFORE importing the graph --------
# src/triage_agent.py constructs the ChatAnthropic model at import time, so the
# key has to be in the environment before that import runs (see get_graph()).

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _hydrate_env() -> None:
    """Copy keys from Streamlit secrets (cloud) or .env (local) into os.environ.

    Streamlit secrets win when present; locally we fall back to python-dotenv so
    the same app runs unchanged on a laptop. Reading st.secrets raises when no
    secrets file exists, so every access is guarded.
    """
    keys = (
        "ANTHROPIC_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_TRACING",
    )
    for key in keys:
        try:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
        except Exception:
            pass  # no secrets.toml — fine, fall through to .env

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


_hydrate_env()

GITHUB_URL = "https://github.com/devala-g/langgraph-customer-triage"
MAX_RUNS_PER_SESSION = 12  # soft guard: each run makes live Claude API calls

# node name -> (emoji, human label) for the live timeline
NODE_META = {
    "classify": ("🏷️", "Classify ticket"),
    "retrieve": ("📚", "Retrieve from KB (Chroma RAG)"),
    "draft": ("✍️", "Draft reply"),
    "react_agent": ("🤖", "ReAct sub-agent (tool use)"),
    "confidence_check": ("🎯", "Self-check confidence"),
    "done": ("✅", "Auto-send"),
    "human_review": ("🙋", "Escalate to human"),
}

SAMPLE_TICKETS = {
    "💳 Billing (RAG)": (
        "Hi, I think you charged my card twice for last Tuesday's order. "
        "Can you check?"
    ),
    "🔧 Technical (RAG)": (
        "I keep getting kicked back to the login page after I sign in. "
        "I cleared my cache already."
    ),
    "❓ Vague (ReAct)": "It doesn't work anymore.",
}


@st.cache_resource(show_spinner="Booting the agent (loading the embedding model)…")
def get_graph():
    """Build the compiled graph once per server process and pre-warm Chroma.

    Cached so the ~80 MB embedding model and the in-memory vector store are
    loaded a single time and shared across reruns and visitors, not rebuilt on
    every interaction.
    """
    from triage_agent import build_graph
    import knowledge_base

    knowledge_base._collection()  # pre-warm: index the KB + load the embedder
    return build_graph()


def _node_detail(node: str, update: dict | None) -> str:
    """One-line, human-readable summary of what a node just did."""
    update = update or {}
    if node == "classify":
        return f"category = `{update.get('category')}`"
    if node == "retrieve":
        ids = [h["id"] for h in update.get("kb_hits", [])]
        return f"KB hits: {', '.join(ids) if ids else 'none'}"
    if node == "draft":
        return (
            "drafted a grounded reply"
            if update.get("draft_reply")
            else "no draft (no KB context)"
        )
    if node == "react_agent":
        return "investigated with tools, then drafted a reply"
    if node == "confidence_check":
        conf = update.get("confidence")
        return f"confidence = {conf:.2f}" if conf is not None else "graded its draft"
    if node == "done":
        return "high confidence → auto-send"
    if node == "human_review":
        return "below threshold → routed to a human"
    return ""


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "This agent triages an inbound support ticket as a small **LangGraph** "
        "state machine:\n\n"
        "1. **Classify** — billing, technical, or vague\n"
        "2. **Route** — scoped tickets take a **Chroma RAG** path; vague "
        "tickets go to a **ReAct sub-agent** that gathers context with tools "
        "first\n"
        "3. **Draft** a reply grounded in retrieved knowledge\n"
        "4. **Self-check** confidence and **escalate to a human** when unsure\n\n"
        "Every node emits a **LangSmith** trace, so the full decision path is "
        "observable."
    )
    st.divider()
    st.markdown(f"**Source:** [github.com/devala-g/…]({GITHUB_URL})")
    st.caption("Built by Devala Griffith")


# --- Main --------------------------------------------------------------------

st.title("🎧 Customer Support Triage Agent")
st.caption("LangGraph · Claude · Chroma RAG · ReAct · LangSmith")
st.write(
    "Paste a support ticket and watch the agent classify it, retrieve "
    "knowledge (or investigate with tools), draft a reply, grade its own "
    "confidence, and **escalate to a human when it isn't sure**."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "⚠️ `ANTHROPIC_API_KEY` isn't configured. On Streamlit Community Cloud, "
        "add it under **Settings → Secrets**. Locally, copy `.env.example` to "
        "`.env` and fill it in."
    )
    st.stop()

if "ticket_input" not in st.session_state:
    st.session_state.ticket_input = ""

st.write("**Try a sample ticket:**")
cols = st.columns(len(SAMPLE_TICKETS))
for col, (label, text) in zip(cols, SAMPLE_TICKETS.items()):
    if col.button(label, use_container_width=True):
        st.session_state.ticket_input = text

ticket = st.text_area(
    "Customer ticket",
    key="ticket_input",
    height=120,
    placeholder="Paste a customer support message…",
)

run = st.button("Triage ticket  ▶", type="primary", disabled=not ticket.strip())

if run and ticket.strip():
    runs = st.session_state.get("runs", 0)
    if runs >= MAX_RUNS_PER_SESSION:
        st.warning(
            "Demo run limit reached for this session (this caps live API spend). "
            "Reload the page to reset."
        )
        st.stop()
    st.session_state.runs = runs + 1

    graph = get_graph()
    initial = {
        "ticket": ticket.strip(),
        "category": None,
        "kb_hits": [],
        "draft_reply": None,
        "confidence": None,
        "final_decision": None,
    }

    st.subheader("Triage run")
    timeline = st.container()
    result: dict = {}
    fired: list[str] = []

    try:
        with st.spinner("Running the graph…"):
            for chunk in graph.stream(initial, stream_mode="updates"):
                for node, update in chunk.items():
                    fired.append(node)
                    result.update({k: v for k, v in (update or {}).items()})
                    icon, label = NODE_META.get(node, ("•", node))
                    timeline.markdown(
                        f"{icon} **{label}** — {_node_detail(node, update)}"
                    )
    except Exception as exc:  # keep a live demo from showing a raw traceback
        st.error(f"The run hit an error: {exc}")
        st.stop()

    st.divider()

    cat = result.get("category")
    conf = result.get("confidence")
    decision = result.get("final_decision")
    path = "ReAct sub-agent (tools)" if "react_agent" in fired else "RAG retrieval"

    c1, c2, c3 = st.columns(3)
    c1.metric("Category", cat or "—")
    c2.metric("Confidence", f"{conf:.2f}" if conf is not None else "—")
    c3.metric("Decision", "Auto-send" if decision == "done" else "Human review")
    st.caption(f"Path taken: **{path}**")

    if conf is not None:
        st.progress(conf, text=f"Self-rated confidence {conf:.0%} (threshold 70%)")

    st.markdown("#### Drafted reply")
    draft_reply = result.get("draft_reply")
    if draft_reply:
        st.info(draft_reply)
    else:
        st.warning("No draft generated — escalated straight to human review.")

    if decision == "human_review":
        st.caption(
            "⚠️ The agent wasn't confident enough to auto-send, so a human "
            "reviews before anything reaches the customer — the safety behavior "
            "this design is built for."
        )

runs_left = max(MAX_RUNS_PER_SESSION - st.session_state.get("runs", 0), 0)
st.caption(f"Each run makes live Claude API calls · demo runs left this session: {runs_left}")
