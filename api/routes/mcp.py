"""
AgentPay MCP Streamable HTTP endpoint.

Mounts at /mcp on the FastAPI app, providing remote MCP access
via Streamable HTTP transport (POST for requests, GET for SSE).

This allows Smithery and other MCP clients to connect to AgentPay
as a remote URL server instead of requiring local stdio installation.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger("agentpay.mcp")

router = APIRouter(tags=["mcp"])

# ---------------------------------------------------------------------------
# Load tool definitions from the companion JSON file
# ---------------------------------------------------------------------------

_mcp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mcp")


def _load_tool_definitions() -> list[dict[str, Any]]:
    json_path = os.path.join(_mcp_dir, "agentpay_tool.json")
    with open(json_path) as f:
        data = json.load(f)
    return data.get("tools", [])


_TOOLS: list[dict[str, Any]] = []


def _get_tools() -> list[dict[str, Any]]:
    global _TOOLS
    if not _TOOLS:
        try:
            _TOOLS = _load_tool_definitions()
        except Exception as e:
            logger.error(f"Failed to load MCP tool definitions: {e}")
            _TOOLS = []
    return _TOOLS


# ---------------------------------------------------------------------------
# Lazy SDK import — reuse from mcp/server.py pattern
# ---------------------------------------------------------------------------

_sdk_available = True
try:
    from agentpay import AgentPayClient
except ImportError:
    _sdk_available = False


def _get_client(api_key: str | None = None) -> "AgentPayClient":
    if not _sdk_available:
        raise RuntimeError("agentpay SDK not installed. Run: pip install agentpay")

    key = api_key or os.environ.get("AGENTPAY_API_KEY", "")
    if not key:
        raise RuntimeError("No API key provided. Pass X-API-Key header or set AGENTPAY_API_KEY.")

    base_url = os.environ.get("AGENTPAY_BASE_URL", "https://leofundmybot.dev")
    return AgentPayClient(api_key=key, base_url=base_url)


# ---------------------------------------------------------------------------
# Tool handlers (mirrors mcp/server.py)
# ---------------------------------------------------------------------------

def _handle_tool(name: str, args: dict, api_key: str | None = None) -> dict[str, Any]:
    """Execute an MCP tool call, returning the result dict."""
    client = _get_client(api_key)

    if name == "agentpay_balance":
        return client.get_balance().model_dump()
    elif name == "agentpay_spend":
        return client.spend(
            amount=args["amount"],
            description=args["description"],
            idempotency_key=args.get("idempotency_key"),
        ).model_dump()
    elif name == "agentpay_transactions":
        txs = client.get_transactions(limit=args.get("limit", 10))
        return {"transactions": [tx.model_dump() for tx in txs]}
    elif name == "agentpay_refund":
        return client.refund(args["transaction_id"]).model_dump()
    elif name == "agentpay_transfer":
        return client.transfer(
            to_agent_id=args["to_agent_id"],
            amount=args["amount"],
            description=args.get("description"),
        ).model_dump()
    elif name == "agentpay_wallet":
        return client.get_wallet(chain=args.get("chain", "base")).model_dump()
    elif name == "agentpay_chains":
        chains = client.list_chains()
        return {"chains": [c.model_dump() for c in chains]}
    elif name == "agentpay_x402_pay":
        return client.x402_pay(
            url=args["url"],
            max_amount=args.get("max_amount"),
        ).model_dump()
    elif name == "agentpay_webhook":
        return client.register_webhook(
            url=args["url"],
            events=args.get("events"),
        ).model_dump()
    elif name == "agentpay_identity":
        return client._request("GET", "/v1/identity")
    else:
        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# JSON-RPC message handling
# ---------------------------------------------------------------------------

def _process_jsonrpc(msg: dict[str, Any], api_key: str | None = None) -> dict[str, Any] | None:
    """Process a single JSON-RPC message, return response or None for notifications."""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        return {
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
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": _get_tools()},
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        try:
            result = _handle_tool(tool_name, tool_args, api_key)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(result, indent=2)},
                    ],
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {"type": "text", "text": f"Error: {exc}"},
                    ],
                    "isError": True,
                },
            }

    elif method.startswith("notifications/"):
        # Notifications have no id and expect no response
        return None

    else:
        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        return None


# ---------------------------------------------------------------------------
# Streamable HTTP transport endpoints
# ---------------------------------------------------------------------------

@router.post("/mcp")
async def mcp_post(request: Request):
    """
    Streamable HTTP MCP endpoint.

    Accepts JSON-RPC messages (single or batch) and returns responses.
    If Accept includes text/event-stream, responds with SSE.
    Otherwise returns plain JSON-RPC.
    """
    api_key = request.headers.get("x-api-key") or request.query_params.get("apiKey")

    try:
        body = await request.json()
    except Exception:
        return Response(
            content=json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }),
            media_type="application/json",
            status_code=400,
        )

    accept = request.headers.get("accept", "")
    use_sse = "text/event-stream" in accept

    # Handle batch
    if isinstance(body, list):
        responses = []
        for msg in body:
            resp = _process_jsonrpc(msg, api_key)
            if resp is not None:
                responses.append(resp)

        if use_sse:
            async def sse_batch():
                for resp in responses:
                    yield f"data: {json.dumps(resp)}\n\n"

            return StreamingResponse(
                sse_batch(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        return Response(
            content=json.dumps(responses),
            media_type="application/json",
        )

    # Single message
    resp = _process_jsonrpc(body, api_key)

    if resp is None:
        # Notification — accepted but no response body
        return Response(status_code=202)

    if use_sse:
        async def sse_single():
            yield f"data: {json.dumps(resp)}\n\n"

        return StreamingResponse(
            sse_single(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return Response(
        content=json.dumps(resp),
        media_type="application/json",
    )


@router.get("/mcp")
async def mcp_get():
    """
    MCP endpoint info (GET).
    Returns server capabilities for discovery.
    """
    return {
        "name": "agentpay",
        "version": "0.1.0",
        "protocol": "mcp",
        "transport": "streamable-http",
        "tools_count": len(_get_tools()),
        "docs": "https://leofundmybot.dev/docs-site/",
    }
