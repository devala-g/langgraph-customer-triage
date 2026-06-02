"""Tiny in-memory KB. Stand-in for a real vector store."""

KB = [
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


def search(category: str, query: str) -> list[dict]:
    """Naive keyword filter against the KB. Real version = vector retrieval."""
    query_terms = {t.lower() for t in query.split()}
    matches = []
    for entry in KB:
        if entry["category"] != category:
            continue
        haystack = (entry["title"] + " " + entry["body"]).lower()
        if any(term in haystack for term in query_terms):
            matches.append(entry)
    return matches[:2]
