"""AgentPay Smolagents Tools — drop-in tools for HuggingFace Smolagents.

Usage:
    from agentpay.integrations.smolagents import get_agentpay_tools

    tools = get_agentpay_tools(api_key="ap_xxxx...")
    agent = CodeAgent(tools=tools, model=model)
"""

from .tools import (
    agentpay_balance,
    agentpay_spend,
    agentpay_transfer,
    agentpay_transactions,
    agentpay_x402_pay,
    get_agentpay_tools,
)

__all__ = [
    "agentpay_balance",
    "agentpay_spend",
    "agentpay_transfer",
    "agentpay_transactions",
    "agentpay_x402_pay",
    "get_agentpay_tools",
]
