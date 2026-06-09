"""Tools available to the ReAct sub-agent for vague tickets.

Each function below is decorated with `@tool` so the model can decide to call
it during its Reason -> Act -> Observe loop. The docstring is what the model
sees when deciding whether to use the tool — keep it descriptive and concrete.

These three tools are stubs that return canned data. In a real deployment the
function bodies would call real APIs (a status page, a customer-data API, the
production knowledge-base service). The signatures and docstrings are what the
agent reasons about; the implementations are swappable.
"""

from __future__ import annotations

from langchain_core.tools import tool

from knowledge_base import KB, _collection


@tool
def get_recent_system_status() -> str:
    """Get the current status of the platform's services, including any
    outages or incidents in the past 24 hours. Use this when a customer's
    complaint might be related to a service-wide problem rather than a
    user-specific issue."""
    return (
        "All services currently operational. "
        "Recent incident: the Login service had a partial outage 30 minutes ago "
        "(5 min duration, fully resolved by 14:35 UTC). "
        "All other services have been stable for the past 24 hours."
    )


@tool
def get_customer_recent_activity(user_id: str) -> str:
    """Look up a customer's recent activity (logins, orders, errors) by their
    user_id. Use this to check whether a customer's report is consistent with
    what is on file for them. Pass the user's id as the argument. If you do
    not know the user_id from the ticket, pass 'unknown' and the tool will
    return a generic recent-activity summary."""
    return (
        f"User '{user_id}': last successful login 5 minutes ago. "
        "2 successful orders in the past week, no payment failures, "
        "no support tickets in the last 30 days."
    )


@tool
def search_knowledge_base(query: str) -> str:
    """Search the full knowledge base (all categories) for help articles
    relevant to a customer's question. Returns up to two matching articles
    with their titles and bodies. Use this when you need product-specific
    guidance to draft a reply or to confirm a hunch about what might be
    happening."""
    results = _collection().query(query_texts=[query], n_results=2)
    hit_ids = (results.get("ids") or [[]])[0]
    if not hit_ids:
        return "No matching knowledge-base articles found."

    by_id = {e["id"]: e for e in KB}
    hits = [by_id[i] for i in hit_ids if i in by_id]
    return "\n\n".join(
        f"[{h['id']}] {h['title']}: {h['body']}" for h in hits
    )
