"""AgentPay Python SDK client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .exceptions import AgentPayError


class AgentPayClient:
    """Synchronous client for the AgentPay API.

    Usage::

        from agentpay import AgentPayClient

        client = AgentPayClient("ap_your_api_key")
        print(client.get_balance())
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://leofundmybot.dev",
    ) -> None:
        """Initialise the AgentPay client.

        Args:
            api_key: Your agent API key (starts with ``ap_``).
            base_url: Base URL of the AgentPay API server.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"X-API-Key": self._api_key},
            timeout=30.0,
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
        raw: bool = False,
    ) -> Any:
        """Send a request and return the parsed JSON (or raw text).

        Raises:
            AgentPayError: If the response status code is not 2xx.
        """
        response = self._client.request(method, path, json=json, params=params)
        if not response.is_success:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise AgentPayError(response.status_code, detail)
        if raw:
            return response.text
        return response.json()

    # ------------------------------------------------------------------
    # Balance & wallet
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        """Get the current agent balance.

        Returns:
            A dict with balance information, e.g.
            ``{"balance": 4.20, "currency": "USD"}``.
        """
        return self._request("GET", "/v1/balance")

    def get_wallet(self) -> dict:
        """Get the agent's on-chain USDC wallet details.

        Returns:
            A dict with wallet address and balance on Base network.
        """
        return self._request("GET", "/v1/wallet")

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def spend(
        self,
        amount: float,
        description: str,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """Spend funds from the agent balance.

        Args:
            amount: Amount to spend (positive float).
            description: Human-readable description of the charge.
            idempotency_key: Optional key to prevent duplicate charges.

        Returns:
            A dict with the created transaction details.
        """
        payload: Dict[str, Any] = {"amount": amount, "description": description}
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        return self._request("POST", "/v1/spend", json=payload)

    def refund(self, transaction_id: str) -> dict:
        """Refund a previous transaction.

        Args:
            transaction_id: The ID of the transaction to refund.

        Returns:
            A dict with the refund transaction details.
        """
        return self._request("POST", f"/v1/transactions/{transaction_id}/refund")

    def transfer(
        self,
        to_agent_id: str,
        amount: float,
        description: Optional[str] = None,
    ) -> dict:
        """Transfer funds to another agent.

        Args:
            to_agent_id: The target agent's ID.
            amount: Amount to transfer (positive float).
            description: Optional description for the transfer.

        Returns:
            A dict with the transfer transaction details.
        """
        payload: Dict[str, Any] = {"to_agent_id": to_agent_id, "amount": amount}
        if description is not None:
            payload["description"] = description
        return self._request("POST", "/v1/transfer", json=payload)

    def get_transactions(self, limit: int = 20) -> list:
        """Retrieve recent transactions.

        Args:
            limit: Maximum number of transactions to return (default 20).

        Returns:
            A list of transaction dicts.
        """
        return self._request("GET", "/v1/transactions", params={"limit": limit})

    # ------------------------------------------------------------------
    # USDC & cards
    # ------------------------------------------------------------------

    def send_usdc(self, to_address: str, amount: float) -> dict:
        """Send USDC on Base network to an external address.

        Args:
            to_address: Destination wallet address.
            amount: Amount of USDC to send.

        Returns:
            A dict with the on-chain transaction details.
        """
        return self._request(
            "POST",
            "/v1/wallet/send",
            json={"to_address": to_address, "amount": amount},
        )

    def get_card(self) -> dict:
        """Get the agent's virtual Visa card details.

        Returns:
            A dict with card number, expiry, and status.
        """
        return self._request("GET", "/v1/card")

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------

    def register_webhook(
        self,
        url: str,
        events: Optional[List[str]] = None,
    ) -> dict:
        """Register a webhook endpoint.

        Args:
            url: The HTTPS URL to receive webhook events.
            events: Optional list of event types to subscribe to.
                    If ``None``, subscribes to all events.

        Returns:
            A dict with the webhook registration details including the
            HMAC signing secret.
        """
        payload: Dict[str, Any] = {"url": url}
        if events is not None:
            payload["events"] = events
        return self._request("POST", "/v1/webhooks", json=payload)

    def delete_webhook(self) -> dict:
        """Delete the registered webhook endpoint.

        Returns:
            A dict confirming deletion.
        """
        return self._request("DELETE", "/v1/webhooks")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self) -> str:
        """Export all transactions as CSV.

        Returns:
            Raw CSV text content.
        """
        return self._request("GET", "/v1/export/csv", raw=True)
