"""
AgentPay — Python SDK for AI agent payment infrastructure.

Quick start::

    from agentpay import AgentPayClient

    client = AgentPayClient("ap_your_api_key")
    balance = client.get_balance()
    print(f"Balance: ${balance.balance_usd}")

For async usage::

    from agentpay import AgentPayAsyncClient

    async with AgentPayAsyncClient("ap_your_api_key") as client:
        balance = await client.get_balance()
"""

from .async_client import AgentPayAsyncClient
from .client import AgentPayClient
from .exceptions import (
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
    verify_webhook_signature,
)
from .models import (
    Balance,
    Chain,
    RefundResponse,
    SpendResponse,
    Transaction,
    TransferResponse,
    Wallet,
    Webhook,
    X402Response,
)

__version__ = "0.1.1"

# Convenience alias — many examples use `AgentPay` as shorthand
AgentPay = AgentPayClient

__all__ = [
    "AgentPay",
    "AgentPayClient",
    "AgentPayAsyncClient",
    "AgentPayError",
    "AuthenticationError",
    "InsufficientBalanceError",
    "RateLimitError",
    "verify_webhook_signature",
    "Balance",
    "Chain",
    "RefundResponse",
    "SpendResponse",
    "Transaction",
    "TransferResponse",
    "Wallet",
    "Webhook",
    "X402Response",
    "__version__",
]
