"""
AgentPay MCP Client Example
============================
Connect to AgentPay's MCP server using the official Python MCP SDK
and call wallet tools (balance, spend, transactions).

Prerequisites:
    pip install mcp

Usage:
    python mcp_client.py
"""

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


AGENTPAY_MCP_URL = "https://leofundmybot.dev/mcp"
API_KEY = "ap_your_api_key_here"


async def main():
    """Connect to AgentPay MCP and demonstrate wallet tools."""

    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with streamablehttp_client(AGENTPAY_MCP_URL, headers=headers) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")

            # Check balance
            result = await session.call_tool("agentpay_balance", {})
            balance = json.loads(result.content[0].text)
            print(f"Balance: ${balance['balance_usd']}")

            # Spend (small test amount)
            result = await session.call_tool(
                "agentpay_spend",
                {"amount": 0.01, "description": "MCP client test"},
            )
            tx = json.loads(result.content[0].text)
            print(f"Spent ${tx['amount']}, tx: {tx['transaction_id']}")

            # Get recent transactions
            result = await session.call_tool(
                "agentpay_transactions",
                {"limit": 5},
            )
            txns = json.loads(result.content[0].text)
            print(f"Recent transactions: {len(txns)} found")
            for t in txns[:3]:
                print(f"  {t['description']}: ${t['amount']}")


if __name__ == "__main__":
    asyncio.run(main())
