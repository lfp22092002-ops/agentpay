"""AgentPay Python SDK — asynchronous client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .exceptions import (
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
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


class AgentPayAsyncClient:
    """Asynchronous client for the AgentPay API.

    Example::

        import asyncio
        from agentpay import AgentPayAsyncClient

        async def main():
            async with AgentPayAsyncClient("ap_your_api_key") as client:
                balance = await client.get_balance()
                print(f"Balance: ${balance.balance_usd}")

        asyncio.run(main())
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://leofundmybot.dev",
        timeout: float = 30.0,
    ) -> None:
        """Initialise the async AgentPay client.

        Args:
            api_key: Your agent API key (starts with ``ap_``).
            base_url: Base URL of the AgentPay API server.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-Key": self._api_key},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Send an HTTP request and return parsed JSON.

        Raises:
            AuthenticationError: If the API key is invalid (401).
            InsufficientBalanceError: On insufficient funds.
            RateLimitError: If rate limited (429).
            AgentPayError: For all other non-2xx responses.
        """
        response = await self._client.request(method, path, json=json, params=params)
        if response.is_success:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        if response.status_code == 401:
            raise AuthenticationError(detail)
        if response.status_code == 429:
            raise RateLimitError(detail)
        if response.status_code in (402, 400) and "balance" in str(detail).lower():
            raise InsufficientBalanceError(detail)
        raise AgentPayError(response.status_code, detail)

    # ------------------------------------------------------------------
    # Balance & wallet
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """Get the current agent balance.

        Returns:
            A :class:`Balance` object with balance details.
        """
        data = await self._request("GET", "/v1/balance")
        return Balance(**data)

    async def get_wallet(self, chain: str = "base") -> Wallet:
        """Get the agent's on-chain wallet details.

        Args:
            chain: Blockchain network (``base``, ``polygon``, ``bnb``, ``solana``).

        Returns:
            A :class:`Wallet` with address and balance info.
        """
        data = await self._request("GET", "/v1/wallet", params={"chain": chain})
        return Wallet(**data)

    async def list_chains(self) -> List[Chain]:
        """List all supported blockchain networks.

        Returns:
            A list of :class:`Chain` objects.
        """
        data = await self._request("GET", "/v1/chains")
        return [Chain(**c) for c in data.get("chains", [])]

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    async def spend(
        self,
        amount: float,
        description: str,
        idempotency_key: Optional[str] = None,
    ) -> SpendResponse:
        """Spend funds from the agent balance.

        Args:
            amount: Amount to spend in USD.
            description: Human-readable description of the charge.
            idempotency_key: Optional key to prevent duplicate charges.

        Returns:
            A :class:`SpendResponse` with transaction details.
        """
        payload: Dict[str, Any] = {"amount": amount, "description": description}
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        data = await self._request("POST", "/v1/spend", json=payload)
        return SpendResponse(**data)

    async def refund(self, transaction_id: str) -> RefundResponse:
        """Refund a previous transaction.

        Args:
            transaction_id: The ID of the transaction to refund.

        Returns:
            A :class:`RefundResponse` with refund details.
        """
        data = await self._request("POST", "/v1/refund", json={"transaction_id": transaction_id})
        return RefundResponse(**data)

    async def transfer(
        self,
        to_agent_id: str,
        amount: float,
        description: Optional[str] = None,
    ) -> TransferResponse:
        """Transfer funds to another agent.

        Args:
            to_agent_id: The target agent's ID.
            amount: Amount to transfer in USD.
            description: Optional description.

        Returns:
            A :class:`TransferResponse` with transfer details.
        """
        payload: Dict[str, Any] = {"to_agent_id": to_agent_id, "amount": amount}
        if description is not None:
            payload["description"] = description
        data = await self._request("POST", "/v1/transfer", json=payload)
        return TransferResponse(**data)

    async def get_transactions(self, limit: int = 20) -> List[Transaction]:
        """Retrieve recent transactions.

        Args:
            limit: Maximum number of transactions (default 20, max 100).

        Returns:
            A list of :class:`Transaction` objects.
        """
        data = await self._request("GET", "/v1/transactions", params={"limit": limit})
        return [Transaction(**tx) for tx in data]

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------

    async def register_webhook(
        self,
        url: str,
        secret: Optional[str] = None,
        events: Optional[List[str]] = None,
    ) -> Webhook:
        """Register a webhook endpoint.

        Args:
            url: HTTPS URL to receive webhook events.
            secret: Ignored (server generates the secret).
            events: Event types to subscribe to (default: all).

        Returns:
            A :class:`Webhook` with URL and signing secret.
        """
        payload: Dict[str, Any] = {"url": url}
        if events is not None:
            payload["events"] = events
        data = await self._request("POST", "/v1/webhook", json=payload)
        return Webhook(**data)

    # ------------------------------------------------------------------
    # x402 Protocol
    # ------------------------------------------------------------------

    async def x402_pay(
        self,
        url: str,
        max_amount: Optional[float] = None,
    ) -> X402Response:
        """Pay for an x402-gated resource using the agent's wallet.

        Args:
            url: The x402-gated resource URL.
            max_amount: Maximum price willing to pay in USD.

        Returns:
            An :class:`X402Response` with the resource data.
        """
        payload: Dict[str, Any] = {"url": url}
        if max_amount is not None:
            payload["max_price_usd"] = max_amount
        data = await self._request("POST", "/v1/x402/pay", json=payload)
        return X402Response(**data)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> AgentPayAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
