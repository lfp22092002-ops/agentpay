"""
AgentPay + Anthropic Claude Example
====================================
Claude agent with tool_use for budget-aware spending.

Requirements:
    pip install anthropic agentpay

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export AGENTPAY_API_KEY=ap_...
    python anthropic_claude_agent.py
"""

import json
import os

from anthropic import Anthropic
from agentpay import AgentPay

client = Anthropic()
pay = AgentPay(api_key=os.environ["AGENTPAY_API_KEY"])

# Define AgentPay tools for Claude
tools = [
    {
        "name": "check_balance",
        "description": "Check current agent balance in USD",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "spend",
        "description": "Spend from agent budget for a task",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount in USD"},
                "description": {"type": "string", "description": "What the spend is for"},
            },
            "required": ["amount", "description"],
        },
    },
    {
        "name": "get_transactions",
        "description": "List recent transactions",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results", "default": 5},
            },
        },
    },
]


def handle_tool(name: str, input: dict) -> str:
    """Execute an AgentPay tool and return the result as a string."""
    if name == "check_balance":
        bal = pay.balance()
        return json.dumps({"balance": bal.available, "currency": bal.currency})
    elif name == "spend":
        tx = pay.spend(amount=input["amount"], description=input["description"])
        return json.dumps({"tx_id": tx.id, "status": tx.status, "remaining": tx.remaining})
    elif name == "get_transactions":
        txs = pay.transactions(limit=input.get("limit", 5))
        return json.dumps([{"id": t.id, "amount": t.amount, "desc": t.description} for t in txs])
    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent(task: str) -> str:
    """Run a Claude agent loop with AgentPay tools."""
    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You are a budget-aware AI agent. Check balance before spending. Be frugal.",
            tools=tools,
            messages=messages,
        )

        # If no tool use, return the final text
        if response.stop_reason == "end_turn":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = handle_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    result = run_agent("Check my balance, then spend $0.50 on a research task. Show remaining budget.")
    print(result)
"""
Expected output:
    Your balance was $10.00. I spent $0.50 on a research task.
    Remaining budget: $9.50.
"""
