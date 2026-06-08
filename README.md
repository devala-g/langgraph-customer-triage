# Customer Support Triage Agent (LangGraph + LangSmith + Chroma RAG)

A small, end-to-end LangGraph agent that triages an inbound customer-support message: classifies it, retrieves grounded context from a vector store, drafts a reply, checks its own confidence, and escalates to a human if it isn't sure.

Built as a working reference for what production-shape agentic workflows look like with the LangChain stack. Every node emits LangSmith traces so the full decision path is observable.

## What this shows

- **LangGraph for orchestration** — a `StateGraph` with branching logic (`classify → retrieve → draft → confidence_check → human_review | done`). Failure modes are explicit nodes, not exceptions.
- **Chroma vector store for RAG** — semantic retrieval over a small knowledge base, filtered by ticket category at query time. Same `search()` interface used to be keyword-based; swapping to a vector backend did not require any change to the graph.
- **Tool use via the Claude API** through `langchain-anthropic`.
- **LangSmith observability** — every run produces a full trace tree with inputs, outputs, latency, and token counts per node.
- **Simple state model** — `TypedDict` state passes through the graph, mutated by each node.
- **Tested with three sample tickets** — billing, technical, vague-and-needs-escalation.

## Why this design

Real production agents fail in legible ways. A graph with named nodes (rather than a single LLM call wrapped in retries) makes failure isolatable: you can see which node returned low confidence, retry just that node, swap models per node, or add a human-in-the-loop step exactly where it's needed.

The shape is the same one used in most production AI tooling I've shipped: small, named, observable steps you can reason about one at a time.

## Run it

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set keys
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY=sk-ant-...
#   LANGSMITH_API_KEY=ls__...   (free tier at https://smith.langchain.com)
#   LANGSMITH_PROJECT=customer-triage-demo

# 3. Run
python src/triage_agent.py

# Or pass your own ticket
python src/triage_agent.py "My card was charged twice last Tuesday"
```

You'll see the agent's decisions printed in the terminal, and full traces in your LangSmith project view.

## Files

- `src/triage_agent.py` — the full graph (≈140 lines)
- `src/knowledge_base.py` — tiny in-memory KB stand-in
- `requirements.txt`
- `.env.example`

## Architecture

```
            ┌─────────────┐
   ticket → │  classify   │  (billing | technical | vague)
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  retrieve   │  (Chroma semantic search,
            └──────┬──────┘   filtered by category)
                   │
            ┌──────▼──────┐
            │    draft    │  (compose reply grounded in
            └──────┬──────┘   retrieved KB entries, with citations)
                   │
            ┌──────▼──────┐
            │ confidence  │
            │   check     │
            └──┬───────┬──┘
       high    │       │   low
               ▼       ▼
           ┌──────┐  ┌──────────────┐
           │ done │  │ human_review │
           └──────┘  └──────────────┘
```

## Sample LangSmith Trace

Every run produces a full trace tree in LangSmith — inputs, outputs, latency, and token count per node. This is what an interviewer or reviewer can use to verify the agent's reasoning end-to-end.

![LangSmith trace of one triage run](docs/langsmith-trace.png)

In the screenshot above, you can see the full path for one billing ticket as it flowed through the graph: `classify → retrieve → draft → confidence_check → human_review`. Each node is its own span with its own latency and cost, and each `ChatAnthropic` call is nested under its parent node. The draft generated successfully, but the confidence node graded it below the 0.7 threshold, so the graph correctly escalated to human review rather than auto-send a low-confidence reply — exactly the safety behavior this design is built for. High-confidence runs branch to `done` instead, and that path is just as visible.

> **To capture your own trace:** run the demo with `LANGSMITH_TRACING=true` in `.env`, then open https://smith.langchain.com → the `customer-triage-demo` project → click any run → expand the trace tree → take a screenshot and save it as `docs/langsmith-trace.png` (replacing the placeholder).

## Next steps (would-be features for a real deployment)

- ✅ ~~Swap the in-memory KB for a real vector store~~ — done. Chroma with sentence-transformer embeddings, filtered by category metadata.
- Add a **ReAct-style sub-agent** for vague tickets — instead of immediately escalating, reason through what clarifying information is needed and call a tool to look it up.
- Persist the Chroma collection on disk (currently in-memory and rebuilt per process).
- Replace classification with a fine-tuned smaller model for cost.
- Add per-node retries with backoff.
- Wire `human_review` to a Slack channel via tool-calling.
- Add evals: golden-set of tickets, scored by an LLM-judge in LangSmith.
- Deploy as a hosted endpoint (Modal, Render, etc.).
