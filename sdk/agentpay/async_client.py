"""AgentPay Python SDK — asynchronous client."""

from __future__ import annotations

import asyncio
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
        max_retries: int = 3,
    ) -> None:
        """Initialise the async AgentPay client.

        Args:
            api_key: Your agent API key (starts with ``ap_``).
            base_url: Base URL of the AgentPay API server.
            timeout: Request timeout in seconds.
            max_retries: Max retries for transient failures (429, 5xx, network).
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
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

        Retries on transient errors (429, 5xx, network) with exponential backoff.

        Raises:
            AuthenticationError: If the API key is invalid (401).
            InsufficientBalanceError: On insufficient funds.
            RateLimitError: If rate limited (429) and retries exhausted.
            AgentPayError: For all other non-2xx responses.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method, path, json=json, params=params)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise AgentPayError(0, f"Network error after {self._max_retries + 1} attempts: {exc}")

            if response.is_success:
                return response.json()

            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text

            # Non-retryable errors
            if response.status_code == 401:
                raise AuthenticationError(detail)
            if response.status_code in (402, 400) and "balance" in str(detail).lower():
                raise InsufficientBalanceError(detail)

            # Retryable errors
            if response.status_code in (429, 500, 502, 503, 504):
                last_exc = AgentPayError(response.status_code, detail)
                if attempt < self._max_retries:
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after else min(2 ** attempt, 8)
                    else:
                        delay = min(2 ** attempt, 8)
                    await asyncio.sleep(delay)
                    continue
                if response.status_code == 429:
                    raise RateLimitError(detail)

            raise AgentPayError(response.status_code, detail)

        raise last_exc or AgentPayError(0, "Unknown error")

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
    # Identity
    # ------------------------------------------------------------------

    async def get_identity(self) -> Dict[str, Any]:
        """Get the agent's identity profile (KYA).

        Returns:
            A dict with display_name, description, trust_score,
            verified status, and directory info.
        """
        return await self._request("GET", "/v1/agent/identity")

    async def get_trust_score(self) -> Dict[str, Any]:
        """Get the agent's trust score breakdown.

        Returns:
            A dict with total score and per-category points.
        """
        return await self._request("GET", "/v1/agent/identity/score")

    # ------------------------------------------------------------------
    # Payee Rules
    # ------------------------------------------------------------------

    async def list_payee_rules(self) -> Dict[str, Any]:
        """List all payee rules for this agent."""
        return await self._request("GET", "/v1/agent/payee-rules")

    async def create_payee_rule(
        self,
        payee_type: str,
        payee_value: str,
        rule_type: str = "allow",
        max_amount_usd: Optional[float] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a payee allow/deny rule.

        Args:
            payee_type: One of 'agent_id', 'domain', 'category', 'address'.
            payee_value: The payee identifier to match.
            rule_type: 'allow' or 'deny' (default: 'allow').
            max_amount_usd: Per-payee transaction cap (optional).
            note: Optional human-readable note.
        """
        payload: Dict[str, Any] = {
            "rule_type": rule_type,
            "payee_type": payee_type,
            "payee_value": payee_value,
        }
        if max_amount_usd is not None:
            payload["max_amount_usd"] = max_amount_usd
        if note is not None:
            payload["note"] = note
        return await self._request("POST", "/v1/agent/payee-rules", json=payload)

    async def delete_payee_rule(self, rule_id: str) -> Dict[str, Any]:
        """Delete (deactivate) a payee rule."""
        return await self._request("DELETE", f"/v1/agent/payee-rules/{rule_id}")

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
