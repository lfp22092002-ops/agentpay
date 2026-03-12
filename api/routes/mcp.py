"""
AgentPay MCP Streamable HTTP endpoint.

Mounts at /mcp on the FastAPI app, providing remote MCP access
via Streamable HTTP transport (POST for requests, GET for SSE).

Implements the MCP Streamable HTTP spec including:
- Mcp-Session-Id header for session management
- Session creation on initialize, validation on subsequent requests
- DELETE /mcp for session termination
- SSE and JSON response modes

This allows Smithery and other MCP clients to connect to AgentPay
as a remote URL server instead of requiring local stdio installation.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger("agentpay.mcp")

router = APIRouter(tags=["mcp"])

# ---------------------------------------------------------------------------
# Session store (in-memory, TTL-based)
# ---------------------------------------------------------------------------

_SESSION_TTL = 3600  # 1 hour
_sessions: dict[str, dict[str, Any]] = {}  # session_id -> {created_at, last_used, api_key}


def _create_session(api_key: str | None = None) -> str:
    """Create a new MCP session and return its ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "created_at": time.time(),
        "last_used": time.time(),
        "api_key": api_key,
    }
    # Prune expired sessions periodically
    _prune_sessions()
    return session_id


def _validate_session(session_id: str) -> bool:
    """Check if a session exists and is not expired."""
    session = _sessions.get(session_id)
    if not session:
        return False
    if time.time() - session["last_used"] > _SESSION_TTL:
        del _sessions[session_id]
        return False
    session["last_used"] = time.time()
    return True


def _delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if it existed."""
    return _sessions.pop(session_id, None) is not None


def _prune_sessions() -> None:
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s["last_used"] > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]


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
                "protocolVersion": "2025-11-25",
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
# Response helpers
# ---------------------------------------------------------------------------

def _make_response(resp: dict | list | None, use_sse: bool, session_id: str | None = None) -> Response:
    """Build HTTP response with optional SSE and session header."""
    extra_headers = {}
    if session_id:
        extra_headers["Mcp-Session-Id"] = session_id

    if resp is None:
        return Response(status_code=202, headers=extra_headers)

    if use_sse:
        items = resp if isinstance(resp, list) else [resp]

        async def sse_stream():
            for item in items:
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", **extra_headers},
        )

    return Response(
        content=json.dumps(resp),
        media_type="application/json",
        headers=extra_headers,
    )


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

    Session management:
    - On `initialize`: creates a new session, returns Mcp-Session-Id header
    - On subsequent requests: validates Mcp-Session-Id, returns 404 if invalid
    """
    api_key = request.headers.get("x-api-key") or request.query_params.get("apiKey")
    session_id = request.headers.get("mcp-session-id")

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

    # Detect if this is an initialize request
    is_init = False
    if isinstance(body, dict) and body.get("method") == "initialize":
        is_init = True
    elif isinstance(body, list):
        is_init = any(m.get("method") == "initialize" for m in body if isinstance(m, dict))

    # Session validation
    if is_init:
        # Create new session on initialize
        session_id = _create_session(api_key)
    elif session_id:
        # Validate existing session; per spec, return 404 if unknown/expired
        if not _validate_session(session_id):
            return Response(
                content=json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid or expired session"},
                }),
                media_type="application/json",
                status_code=404,
            )

    # Handle batch
    if isinstance(body, list):
        responses = []
        for msg in body:
            resp = _process_jsonrpc(msg, api_key)
            if resp is not None:
                responses.append(resp)
        return _make_response(responses if responses else None, use_sse, session_id)

    # Single message
    resp = _process_jsonrpc(body, api_key)
    return _make_response(resp, use_sse, session_id)


@router.get("/mcp")
async def mcp_get(request: Request):
    """
    MCP SSE stream endpoint (GET).

    Per the Streamable HTTP spec, GET opens an SSE stream for server-initiated
    messages. For AgentPay, we return server info since we don't push events.
    Validates Mcp-Session-Id if provided.
    """
    session_id = request.headers.get("mcp-session-id")

    if session_id and not _validate_session(session_id):
        return Response(status_code=404)

    return {
        "name": "agentpay",
        "version": "0.1.0",
        "protocol": "mcp",
        "transport": "streamable-http",
        "tools_count": len(_get_tools()),
        "docs": "https://leofundmybot.dev/docs-site/",
    }


@router.delete("/mcp")
async def mcp_delete(request: Request):
    """
    Terminate an MCP session.

    Per the Streamable HTTP spec, clients send DELETE to end a session.
    Returns 200 if session was terminated, 404 if session not found.
    """
    session_id = request.headers.get("mcp-session-id")

    if not session_id:
        return Response(
            content=json.dumps({"error": "Mcp-Session-Id header required"}),
            media_type="application/json",
            status_code=400,
        )

    if _delete_session(session_id):
        return Response(status_code=200)
    else:
        return Response(status_code=404)
