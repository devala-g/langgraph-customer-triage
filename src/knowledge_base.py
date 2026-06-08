"""Knowledge base + semantic retrieval over Chroma.

Earlier versions of this file used naive keyword filtering for retrieval. This
version uses a real vector store (Chroma) with the default sentence-transformer
embedding function, filtered by category metadata. The public `search()`
signature is unchanged so the rest of the graph (see triage_agent.py) does not
need to know the retrieval backend was swapped.

Embedding model: chromadb's default `all-MiniLM-L6-v2` runs locally with no
API key. First run downloads the model (~80 MB); subsequent runs are instant.
"""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

KB: list[dict] = [
    {
        "id": "kb-billing-001",
        "category": "billing",
        "title": "Duplicate charges",
        "body": (
            "If a customer sees the same charge twice, it is almost always a "
            "pending authorization that will drop off in 3-5 business days. "
            "If both charges have cleared (not pending), open a refund ticket "
            "and refund the duplicate within 24 hours."
        ),
    },
    {
        "id": "kb-billing-002",
        "category": "billing",
        "title": "Refund timing",
        "body": (
            "Refunds post to the customer's original payment method within "
            "5-10 business days depending on the card issuer."
        ),
    },
    {
        "id": "kb-technical-001",
        "category": "technical",
        "title": "Login loop",
        "body": (
            "If a user reports being kicked back to the login page after entering "
            "credentials, ask them to (1) clear cookies for our domain, "
            "(2) try an incognito window, (3) confirm their email is verified."
        ),
    },
    {
        "id": "kb-technical-002",
        "category": "technical",
        "title": "API rate limits",
        "body": (
            "Free tier accounts are limited to 60 requests per minute. Paid tier "
            "is 600 rpm. 429 responses include a Retry-After header."
        ),
    },
]


_COLLECTION = None


def _collection():
    """Lazily build (or fetch) the Chroma collection on first use.

    We instantiate the embedding function explicitly (rather than letting the
    collection build it lazily) so any model-load failure surfaces here at
    setup time instead of being swallowed during a later query.
    """
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    embed_fn = DefaultEmbeddingFunction()
    client = chromadb.Client()  # in-memory; survives only the current process
    col = client.get_or_create_collection(name="knowledge_base", embedding_function=embed_fn)

    # First time only: index every KB entry. Chroma embeds on insert. We index
    # the title + body together so the embedding sees the full semantic
    # context, and we store the category as metadata so we can filter by it
    # at query time.
    if col.count() == 0:
        col.add(
            ids=[e["id"] for e in KB],
            documents=[f"{e['title']}. {e['body']}" for e in KB],
            metadatas=[{"category": e["category"], "title": e["title"]} for e in KB],
        )

    _COLLECTION = col
    return col


def search(category: str, query: str, top_k: int = 2) -> list[dict]:
    """Return the top-k KB entries whose semantic similarity to `query` is
    highest, restricted to entries tagged with `category`.

    Returns dicts with the same shape as the raw KB entries (id, category,
    title, body) so the rest of the agent graph stays unchanged.

    Errors are intentionally NOT caught here — let them propagate so the agent
    runs surface real failures rather than silently escalating every ticket.
    """
    results = _collection().query(
        query_texts=[query],
        n_results=top_k,
        where={"category": category},
    )

    hit_ids = (results.get("ids") or [[]])[0]
    if not hit_ids:
        return []

    by_id = {e["id"]: e for e in KB}
    return [by_id[i] for i in hit_ids if i in by_id]
