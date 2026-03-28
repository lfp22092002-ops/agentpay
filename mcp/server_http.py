#!/usr/bin/env python3
"""
AgentPay MCP Server — Streamable HTTP Transport.

Adds HTTP-based MCP transport alongside the stdio server, enabling remote
agent connections without process spawning.

Usage:
    AGENTPAY_API_KEY=ap_your_key python server_http.py --port 8090

Endpoints:
    POST /mcp  — JSON-RPC endpoint (Streamable HTTP transport)
    GET  /mcp  — SSE stream for server-initiated notifications (optional)

Reference: MCP spec 2025-11-25, Streamable HTTP transport
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

# Reuse handlers from the stdio server
sys.path.insert(0, os.path.dirname(__file__))
from server import TOOL_HANDLERS, _load_tool_definitions  # noqa: E402


class MCPHTTPHandler(BaseHTTPRequestHandler):
    """Handle MCP JSON-RPC over HTTP POST."""

    tools: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        if self.path != "/mcp":
            self._send_json(404, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            msg = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            })
            return

        response = self._handle_jsonrpc(msg)
        if response is not None:
            self._send_json(200, response)
        else:
            # Notification — no response needed
            self.send_response(204)
            self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/mcp":
            self._send_json(404, {"error": "Not found"})
            return

        # SSE endpoint for server-initiated messages (keep-alive)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send an initial ping and hold connection open
        try:
            self.wfile.write(b"event: ping\ndata: {}\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def do_OPTIONS(self) -> None:
        """CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _handle_jsonrpc(self, msg: dict[str, Any]) -> dict[str, Any] | None:
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

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.tools},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            handler = TOOL_HANDLERS.get(tool_name)

            if handler is None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

            try:
                result = handler(tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                }

        if method == "notifications/initialized":
            return None  # No response for notifications

        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        return None

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default access logs for cleaner output."""
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentPay MCP HTTP Server")
    parser.add_argument("--port", type=int, default=8090, help="Port (default: 8090)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    MCPHTTPHandler.tools = _load_tool_definitions()

    server = HTTPServer((args.host, args.port), MCPHTTPHandler)
    print(f"AgentPay MCP HTTP server listening on {args.host}:{args.port}/mcp")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
