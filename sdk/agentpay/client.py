"""AgentPay Python SDK — synchronous client."""

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


class AgentPayClient:
    """Synchronous client for the AgentPay API.

    Example::

        from agentpay import AgentPayClient

        client = AgentPayClient("ap_your_api_key")
        balance = client.get_balance()
        print(f"Balance: ${balance.balance_usd}")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://leofundmybot.dev",
        timeout: float = 30.0,
    ) -> None:
        """Initialise the AgentPay client.

        Args:
            api_key: Your agent API key (starts with ``ap_``).
            base_url: Base URL of the AgentPay API server.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"X-API-Key": self._api_key},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
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
            InsufficientBalanceError: On insufficient funds (402/400 with balance hint).
            RateLimitError: If rate limited (429).
            AgentPayError: For all other non-2xx responses.
        """
        response = self._client.request(method, path, json=json, params=params)
        if response.is_success:
            return response.json()

        # Parse error detail
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

    def get_balance(self) -> Balance:
        """Get the current agent balance.

        Returns:
            A :class:`Balance` object with balance details.

        Example::

            balance = client.get_balance()
            print(f"${balance.balance_usd} available")
        """
        data = self._request("GET", "/v1/balance")
        return Balance(**data)

    def get_wallet(self, chain: str = "base") -> Wallet:
        """Get the agent's on-chain wallet details.

        Args:
            chain: Blockchain network (``base``, ``polygon``, ``bnb``, ``solana``).

        Returns:
            A :class:`Wallet` with address and balance info.

        Example::

            wallet = client.get_wallet(chain="base")
            print(f"Address: {wallet.address}")
        """
        data = self._request("GET", "/v1/wallet", params={"chain": chain})
        return Wallet(**data)

    def list_chains(self) -> List[Chain]:
        """List all supported blockchain networks.

        Returns:
            A list of :class:`Chain` objects.

        Example::

            chains = client.list_chains()
            for c in chains:
                print(f"{c.name} ({c.id})")
        """
        data = self._request("GET", "/v1/chains")
        return [Chain(**c) for c in data.get("chains", [])]

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def spend(
        self,
        amount: float,
        description: str,
        idempotency_key: Optional[str] = None,
    ) -> SpendResponse:
        """Spend funds from the agent balance.

        Args:
            amount: Amount to spend in USD (positive, max 10 000).
            description: Human-readable description of the charge.
            idempotency_key: Optional key to prevent duplicate charges.

        Returns:
            A :class:`SpendResponse` with transaction details.

        Example::

            tx = client.spend(0.50, "GPT-4 API call")
            print(f"Spent ${tx.amount}, remaining ${tx.remaining_balance}")
        """
        payload: Dict[str, Any] = {"amount": amount, "description": description}
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        data = self._request("POST", "/v1/spend", json=payload)
        return SpendResponse(**data)

    def refund(self, transaction_id: str) -> RefundResponse:
        """Refund a previous transaction.

        Args:
            transaction_id: The ID of the transaction to refund.

        Returns:
            A :class:`RefundResponse` with refund details.

        Example::

            result = client.refund("tx_abc123")
            print(f"Refunded ${result.amount_refunded}")
        """
        data = self._request("POST", "/v1/refund", json={"transaction_id": transaction_id})
        return RefundResponse(**data)

    def transfer(
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

        Example::

            result = client.transfer("agent_xyz", 5.0, "Payment for services")
        """
        payload: Dict[str, Any] = {"to_agent_id": to_agent_id, "amount": amount}
        if description is not None:
            payload["description"] = description
        data = self._request("POST", "/v1/transfer", json=payload)
        return TransferResponse(**data)

    def get_transactions(self, limit: int = 20) -> List[Transaction]:
        """Retrieve recent transactions.

        Args:
            limit: Maximum number of transactions (default 20, max 100).

        Returns:
            A list of :class:`Transaction` objects.

        Example::

            txs = client.get_transactions(limit=10)
            for tx in txs:
                print(f"{tx.type}: ${tx.amount} — {tx.description}")
        """
        data = self._request("GET", "/v1/transactions", params={"limit": limit})
        return [Transaction(**tx) for tx in data]

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------

    def register_webhook(
        self,
        url: str,
        secret: Optional[str] = None,
        events: Optional[List[str]] = None,
    ) -> Webhook:
        """Register a webhook endpoint.

        Args:
            url: HTTPS URL to receive webhook events.
            secret: Not sent to server (server generates the secret).
                    Kept for API compatibility; ignored.
            events: Event types to subscribe to (default: all).

        Returns:
            A :class:`Webhook` with URL and signing secret.

        Example::

            wh = client.register_webhook(
                url="https://my-server.com/webhook",
                events=["spend", "refund"],
            )
            print(f"Secret: {wh.secret}")
        """
        payload: Dict[str, Any] = {"url": url}
        if events is not None:
            payload["events"] = events
        data = self._request("POST", "/v1/webhook", json=payload)
        return Webhook(**data)

    # ------------------------------------------------------------------
    # x402 Protocol
    # ------------------------------------------------------------------

    def x402_pay(
        self,
        url: str,
        max_amount: Optional[float] = None,
    ) -> X402Response:
        """Pay for an x402-gated resource using the agent's wallet.

        Args:
            url: The x402-gated resource URL.
            max_amount: Maximum price willing to pay in USD (default 1.0).

        Returns:
            An :class:`X402Response` with the resource data.

        Example::

            result = client.x402_pay("https://api.example.com/premium")
            print(result.data)
        """
        payload: Dict[str, Any] = {"url": url}
        if max_amount is not None:
            payload["max_price_usd"] = max_amount
        data = self._request("POST", "/v1/x402/pay", json=payload)
        return X402Response(**data)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_identity(self) -> Dict[str, Any]:
        """Get the agent's identity profile (KYA).

        Returns:
            A dict with display_name, description, trust_score,
            verified status, and directory info.

        Example::

            identity = client.get_identity()
            print(f"Trust score: {identity['trust_score']}")
        """
        return self._request("GET", "/v1/agent/identity")

    def get_trust_score(self) -> Dict[str, Any]:
        """Get the agent's trust score breakdown.

        Returns:
            A dict with total score and per-category points.

        Example::

            score = client.get_trust_score()
            print(f"Total: {score['total']}/100")
        """
        return self._request("GET", "/v1/agent/identity/score")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> AgentPayClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
