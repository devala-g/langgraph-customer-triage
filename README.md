# Customer Support Triage Agent (LangGraph + LangSmith)

A small, end-to-end LangGraph agent that triages an inbound customer-support message: classifies it, looks up relevant info in a tiny in-memory knowledge base, drafts a reply, checks its own confidence, and escalates to a human if it isn't sure.

Built as a working reference for what production-shape agentic workflows look like with the LangChain stack. Every node emits LangSmith traces so the full decision path is observable.

## What this shows

- **LangGraph for orchestration** — a `StateGraph` with branching logic (`classify → retrieve → draft → confidence_check → human_review | done`). Failure modes are explicit nodes, not exceptions.
- **Tool use via the Claude API** through `langchain-anthropic`.
- **LangSmith observability** — every run produces a full trace tree with inputs, outputs, latency, and token counts per node.
- **Simple state model** — `TypedDict` state passes through the graph, mutated by each node.
- **Tested with three sample tickets** — billing, technical, vague-and-needs-escalation.

## Why this design

Real production agents fail in legible ways. A graph with named nodes (rather than a single LLM call wrapped in retries) makes failure isolatable: you can see which node returned low confidence, retry just that node, swap models per node, or add a human-in-the-loop step exactly where it's needed.

This is the same shape as the AI tooling I shipped at Infotech (Claude API automation for client legacy migrations), translated into the LangChain stack.

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
            │  retrieve   │  (look up matching KB entries)
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │    draft    │  (compose reply with citations)
            └──────┬──────┘
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

## Next steps (would-be features for a real deployment)

- Swap the in-memory KB for a real vector store (Postgres + pgvector or Chroma)
- Replace classification with a fine-tuned smaller model for cost
- Add per-node retries with backoff
- Wire `human_review` to a Slack channel via tool-calling
- Add evals: golden-set of tickets, scored by GPT-judge in LangSmith
