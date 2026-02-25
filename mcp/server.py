#!/usr/bin/env python3
"""
AgentPay MCP Server — Model Context Protocol server for AI agent payments.

Wraps the AgentPay Python SDK into MCP-compatible tool calls that any
AI agent framework can use.

Usage:
    AGENTPAY_API_KEY=ap_your_key python server.py

Or with the MCP CLI:
    npx @modelcontextprotocol/server agentpay
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Lazy SDK import — the server can start even if the SDK is not yet installed,
# but will fail at first tool call with a helpful message.
# ---------------------------------------------------------------------------

_sdk_available = True
try:
    from agentpay import AgentPayClient, AgentPayError
except ImportError:
    _sdk_available = False


def _get_client() -> "AgentPayClient":
    if not _sdk_available:
        raise RuntimeError(
            "agentpay SDK not installed. Run: pip install agentpay"
        )
    api_key = os.environ.get("AGENTPAY_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "AGENTPAY_API_KEY environment variable is not set."
        )
    base_url = os.environ.get("AGENTPAY_BASE_URL", "https://leofundmybot.dev")
    return AgentPayClient(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def handle_balance(_params: dict[str, Any]) -> dict[str, Any]:
    """Check agent balance."""
    client = _get_client()
    balance = client.get_balance()
    return balance.model_dump()


def handle_spend(params: dict[str, Any]) -> dict[str, Any]:
    """Spend from agent wallet."""
    client = _get_client()
    result = client.spend(
        amount=params["amount"],
        description=params["description"],
        idempotency_key=params.get("idempotency_key"),
    )
    return result.model_dump()


def handle_transactions(params: dict[str, Any]) -> dict[str, Any]:
    """Get recent transactions."""
    client = _get_client()
    limit = params.get("limit", 10)
    txs = client.get_transactions(limit=limit)
    return {"transactions": [tx.model_dump() for tx in txs]}


def handle_refund(params: dict[str, Any]) -> dict[str, Any]:
    """Refund a transaction."""
    client = _get_client()
    result = client.refund(params["transaction_id"])
    return result.model_dump()


def handle_transfer(params: dict[str, Any]) -> dict[str, Any]:
    """Transfer funds to another agent."""
    client = _get_client()
    result = client.transfer(
        to_agent_id=params["to_agent_id"],
        amount=params["amount"],
        description=params.get("description"),
    )
    return result.model_dump()


def handle_wallet(params: dict[str, Any]) -> dict[str, Any]:
    """Get wallet info."""
    client = _get_client()
    chain = params.get("chain", "base")
    wallet = client.get_wallet(chain=chain)
    return wallet.model_dump()


def handle_chains(_params: dict[str, Any]) -> dict[str, Any]:
    """List supported chains."""
    client = _get_client()
    chains = client.list_chains()
    return {"chains": [c.model_dump() for c in chains]}


def handle_x402_pay(params: dict[str, Any]) -> dict[str, Any]:
    """Pay for an x402-gated resource."""
    client = _get_client()
    result = client.x402_pay(
        url=params["url"],
        max_amount=params.get("max_amount"),
    )
    return result.model_dump()


# Tool name → handler mapping
TOOL_HANDLERS: dict[str, Any] = {
    "agentpay_balance": handle_balance,
    "agentpay_spend": handle_spend,
    "agentpay_transactions": handle_transactions,
    "agentpay_refund": handle_refund,
    "agentpay_transfer": handle_transfer,
    "agentpay_wallet": handle_wallet,
    "agentpay_chains": handle_chains,
    "agentpay_x402_pay": handle_x402_pay,
}

# ---------------------------------------------------------------------------
# MCP JSON-RPC stdio transport
# ---------------------------------------------------------------------------

def _read_message() -> dict[str, Any] | None:
    """Read a JSON-RPC message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


def _write_message(msg: dict[str, Any]) -> None:
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _load_tool_definitions() -> list[dict[str, Any]]:
    """Load tool definitions from the companion JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), "agentpay_tool.json")
    with open(json_path) as f:
        data = json.load(f)
    return data.get("tools", [])


def main() -> None:
    """Run the MCP server on stdio."""
    tools = _load_tool_definitions()

    while True:
        msg = _read_message()
        if msg is None:
            break

        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            _write_message({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "agentpay",
                        "version": "0.1.0",
                    },
                },
            })
        elif method == "tools/list":
            _write_message({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": tools},
            })
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            handler = TOOL_HANDLERS.get(tool_name)

            if handler is None:
                _write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}",
                    },
                })
                continue

            try:
                result = handler(tool_args)
                _write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2)},
                        ],
                    },
                })
            except Exception as exc:
                _write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": f"Error: {exc}"},
                        ],
                        "isError": True,
                    },
                })
        elif method == "notifications/initialized":
            # Acknowledgement, no response needed
            pass
        else:
            if msg_id is not None:
                _write_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                })


if __name__ == "__main__":
    main()
