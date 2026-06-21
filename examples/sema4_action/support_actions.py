"""The triage agent's tools, re-expressed as Sema4.ai AI Actions.

These are the SAME capabilities the LangGraph triage agent calls as LangChain
`@tool` functions (see ../../src/tools.py) — here they are written as Sema4
`@action`s. Dropped into a Sema4 Action Server, any agent (a LangChain agent, a
custom GPT, anything that speaks MCP or OpenAPI) can call them. This is the
"same tool, now on your platform" mapping: a typed Python function with a
docstring is all the Action Server needs to generate the schema and serve it.

Run (no RCC needed, serves from the current environment):
    pip install sema4ai-action-server
    action-server start --actions-sync=false

Then open http://localhost:8080 to see the actions, their auto-generated
schemas, and run them from the UI. The server also exposes an MCP endpoint at
/mcp and an OpenAPI spec.
"""

from sema4ai.actions import action


@action
def get_recent_system_status() -> str:
    """Get the current status of the platform's services, including any outages
    or incidents in the past 24 hours.

    Use this when a customer's complaint might be related to a service-wide
    problem rather than a user-specific issue.

    Returns:
        A human-readable summary of current service status and recent incidents.
    """
    return (
        "All services currently operational. "
        "Recent incident: the Login service had a partial outage 30 minutes ago "
        "(5 min duration, fully resolved by 14:35 UTC). "
        "All other services have been stable for the past 24 hours."
    )


@action
def get_customer_recent_activity(user_id: str) -> str:
    """Look up a customer's recent activity (logins, orders, errors) by user_id.

    Use this to check whether a customer's report is consistent with what is on
    file for them.

    Args:
        user_id: The customer's unique identifier. Pass 'unknown' if the ticket
            does not include one.

    Returns:
        A summary of the customer's recent logins, orders, and payment activity.
    """
    return (
        f"User '{user_id}': last successful login 5 minutes ago. "
        "2 successful orders in the past week, no payment failures, "
        "no support tickets in the last 30 days."
    )
