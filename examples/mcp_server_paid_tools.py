"""
AgentPay MCP Server Example — Payment-Gated Tools
===================================================
Build an MCP server where tools cost money to use.
Agents pay via AgentPay before accessing premium capabilities.

This is the pattern for monetizing AI agent tools:
agents discover your MCP server, see tool prices, and pay per call.

Prerequisites:
    pip install mcp agentpay

Usage:
    python mcp_server_paid_tools.py

Then connect any MCP client (Claude Desktop, Cursor, etc.)
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# -- Replace with your AgentPay API key --
AGENTPAY_BASE_URL = "https://leofundmybot.dev"

# Tool pricing (USD per call)
TOOL_PRICES = {
    "premium_search": 0.05,
    "generate_report": 0.25,
    "translate_document": 0.10,
}

app = Server("agentpay-paid-tools")


def _get_client():
    """Lazy-import AgentPay client so the example is readable top-to-bottom."""
    from agentpay import AgentPayClient  # noqa: E402

    import os

    api_key = os.environ.get("AGENTPAY_API_KEY", "ap_your_key_here")
    return AgentPayClient(api_key, base_url=AGENTPAY_BASE_URL)


# ------------------------------------------------------------------
# Tool definitions — agents see these when they connect
# ------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="premium_search",
            description=f"Deep web search with AI summarization. Cost: ${TOOL_PRICES['premium_search']}/call",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="generate_report",
            description=f"Generate a structured analysis report. Cost: ${TOOL_PRICES['generate_report']}/call",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Report topic"},
                    "format": {"type": "string", "enum": ["brief", "detailed"], "default": "brief"},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="translate_document",
            description=f"Translate text to any language. Cost: ${TOOL_PRICES['translate_document']}/call",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_language": {"type": "string", "description": "Target language code (e.g. 'es', 'fr', 'de')"},
                },
                "required": ["text", "target_language"],
            },
        ),
        Tool(
            name="check_balance",
            description="Check your AgentPay balance (free)",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ------------------------------------------------------------------
# Tool execution — charge first, then deliver
# ------------------------------------------------------------------

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    client = _get_client()

    # Free tool: balance check
    if name == "check_balance":
        balance = client.get_balance()
        return [TextContent(
            type="text",
            text=json.dumps({
                "balance_usd": balance.balance_usd,
                "daily_remaining_usd": balance.daily_remaining_usd,
            }, indent=2),
        )]

    # Paid tools: charge via AgentPay first
    price = TOOL_PRICES.get(name)
    if price is None:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    # Charge the agent
    try:
        tx = client.spend(
            amount=price,
            description=f"MCP tool call: {name}",
            idempotency_key=f"{name}_{hash(json.dumps(arguments, sort_keys=True))}",
        )
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Payment failed: {e}. Check your balance with check_balance.",
        )]

    # Execute the tool (replace with your real logic)
    result = _execute_tool(name, arguments)

    return [TextContent(
        type="text",
        text=json.dumps({
            "result": result,
            "charged_usd": price,
            "transaction_id": tx.transaction_id,
            "remaining_balance": tx.remaining_balance,
        }, indent=2),
    )]


def _execute_tool(name: str, args: dict) -> str:
    """Placeholder implementations — replace with your actual tool logic."""
    if name == "premium_search":
        return f"Search results for '{args['query']}': [Result 1, Result 2, ...]"
    elif name == "generate_report":
        fmt = args.get("format", "brief")
        return f"{'Detailed' if fmt == 'detailed' else 'Brief'} report on '{args['topic']}': ..."
    elif name == "translate_document":
        return f"Translated to {args['target_language']}: [{args['text'][:50]}...]"
    return "Unknown tool"


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
