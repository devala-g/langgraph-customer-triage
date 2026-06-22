# Companion example: the agent's tools as Sema4.ai AI Actions

This folder re-expresses two of the triage agent's tools — `get_recent_system_status`
and `get_customer_recent_activity` (from [`../../src/tools.py`](../../src/tools.py)) — as
**Sema4.ai AI Actions**.

The point: a LangChain `@tool` and a Sema4 `@action` are the *same idea* — a typed
Python function with a docstring that an agent can reason about and call. Moving a
capability from this demo onto the Sema4 platform is essentially a one-decorator
change, after which the Action Server exposes it over an auto-generated **MCP
endpoint** and **OpenAPI** spec for any agent to use.

```python
# LangChain tool (this repo)            # Sema4.ai AI Action (this folder)
from langchain_core.tools import tool   from sema4ai.actions import action

@tool                                   @action
def get_recent_system_status() -> str:  def get_recent_system_status() -> str:
    """...docstring the model reads"""      """...docstring the schema is built from"""
    return "All services operational…"      return "All services operational…"
```

## Run it locally (works on most machines — no RCC needed)

The Action Server's full runtime depends on RCC, which has no build for certain computers like macOS
x86_64, so `action-server start` won't run here. But the `sema4ai-actions` runner
executes actions directly in the current Python environment — perfect for a local
walkthrough:

```bash
python -m venv .venv && source .venv/bin/activate
pip install sema4ai-action-server          # also brings sema4ai-actions

# List the discovered actions + their AUTO-GENERATED input/output schemas
python -m sema4ai.actions list

# Run an action that takes no input
python -m sema4ai.actions run -a get_recent_system_status --print-result

# Run an action with a typed argument (schema built from the type hint + docstring)
python -m sema4ai.actions run -a get_customer_recent_activity \
    --json-input dev-input.json --print-result
```

Notice the input schema for `get_customer_recent_activity` is generated for you
from `user_id: str` and its docstring — you never hand-write a schema.

## Run the full Action Server (on a supported platform)

On Linux / Apple Silicon / in Sema4's Control Room, the same folder serves as a
real Action Server:

```bash
action-server start            # builds the env from package.yaml via RCC
```

That gives a web UI at `http://localhost:8080`, an MCP endpoint at `/mcp`, and an
OpenAPI spec — so a LangChain agent, a custom GPT, or any MCP client can call
these actions against real enterprise systems.

## Files

- `support_actions.py` — the two `@action`s
- `package.yaml` — the Action Package descriptor (name + environment spec)
- `dev-input.json` — sample input for the `run` command above
