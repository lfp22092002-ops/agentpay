"""Tests for AgentPay Python SDK — sync client."""

import json
from decimal import Decimal
from unittest.mock import patch, MagicMock

import httpx
import pytest

from agentpay import (
    AgentPayClient,
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
)
from agentpay.models import Balance, SpendResponse, Wallet, Transaction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    c = AgentPayClient("ap_test_key_123", base_url="https://test.example.com")
    yield c
    c.close()


def _mock_response(status_code: int = 200, data: dict | None = None, text: str = ""):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.text = text or json.dumps(data or {})
    resp.json.return_value = data or {}
    return resp


# ---------------------------------------------------------------------------
# Authentication errors
# ---------------------------------------------------------------------------


class TestAuth:
    def test_401_raises_auth_error(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(401, {"detail": "Invalid API key"}),
        ):
            with pytest.raises(AuthenticationError) as exc:
                client.get_balance()
            assert exc.value.status_code == 401

    def test_429_raises_rate_limit(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(429, {"detail": "Too many requests"}),
        ):
            with pytest.raises(RateLimitError) as exc:
                client.get_balance()
            assert exc.value.status_code == 429

    def test_402_insufficient_balance(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(402, {"detail": "Insufficient balance"}),
        ):
            with pytest.raises(InsufficientBalanceError):
                client.spend(100.0, "test")

    def test_500_raises_generic_error(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(500, {"detail": "Internal error"}),
        ):
            with pytest.raises(AgentPayError) as exc:
                client.get_balance()
            assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# Balance & wallet
# ---------------------------------------------------------------------------


class TestBalance:
    def test_get_balance(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, {
                "agent_id": "agent_1",
                "balance_usd": 42.50,
                "daily_limit_usd": 100.0,
                "daily_spent_usd": 12.0,
                "daily_remaining_usd": 88.0,
                "tx_limit_usd": 25.0,
                "agent_name": "test-agent",
                "is_active": True,
            }),
        ):
            balance = client.get_balance()
            assert isinstance(balance, Balance)
            assert balance.balance_usd == 42.50
            assert balance.agent_name == "test-agent"

    def test_get_wallet(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, {
                "address": "0xabc123",
                "chain": "base",
                "balance_usdc": "10.0",
                "balance_native": "0.001",
            }),
        ):
            wallet = client.get_wallet(chain="base")
            assert isinstance(wallet, Wallet)
            assert wallet.address == "0xabc123"
            assert wallet.chain == "base"


# ---------------------------------------------------------------------------
# Spend / refund / transfer
# ---------------------------------------------------------------------------


class TestTransactions:
    def test_spend(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, {
                "success": True,
                "transaction_id": "tx_abc",
                "amount": 5.0,
                "fee": 0.10,
                "remaining_balance": 37.40,
            }),
        ):
            result = client.spend(5.0, "API call")
            assert isinstance(result, SpendResponse)
            assert result.success is True
            assert result.transaction_id == "tx_abc"
            assert result.amount == 5.0

    def test_spend_with_idempotency(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, {
                "success": True,
                "transaction_id": "tx_idem",
                "amount": 1.0,
                "fee": 0.02,
                "remaining_balance": 99.0,
            }),
        ) as mock_req:
            client.spend(1.0, "test", idempotency_key="key123")
            call_args = mock_req.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["idempotency_key"] == "key123"

    def test_refund(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, {
                "success": True,
                "refund_transaction_id": "tx_ref",
                "amount_refunded": 5.0,
                "new_balance": 42.40,
            }),
        ):
            result = client.refund("tx_abc")
            assert result.success is True
            assert result.amount_refunded == 5.0

    def test_transfer(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, {
                "success": True,
                "from_balance": 30.0,
                "to_balance": 60.0,
            }),
        ):
            result = client.transfer("agent_xyz", 10.0, "Payment")
            assert result.success is True

    def test_get_transactions(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, [
                {"id": "tx_1", "type": "spend", "amount": 5.0, "fee": 0.10, "status": "completed", "created_at": "2026-03-07T12:00:00Z"},
                {"id": "tx_2", "type": "deposit", "amount": 100.0, "fee": 0.0, "status": "completed", "created_at": "2026-03-07T11:00:00Z"},
            ]),
        ):
            txs = client.get_transactions(limit=10)
            assert len(txs) == 2
            assert all(isinstance(t, Transaction) for t in txs)


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


class TestWebhooks:
    def test_register_webhook(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, {
                "url": "https://my-server.com/webhook",
                "secret": "whsec_abc123",
                "active": True,
            }),
        ):
            wh = client.register_webhook("https://my-server.com/webhook", events=["spend"])
            assert wh.url == "https://my-server.com/webhook"
            assert wh.secret == "whsec_abc123"


# ---------------------------------------------------------------------------
# x402
# ---------------------------------------------------------------------------


class Testx402:
    def test_x402_pay(self, client: AgentPayClient):
        with patch.object(
            client._client, "request",
            return_value=_mock_response(200, {
                "success": True,
                "amount_paid": 0.01,
                "data": "premium content body",
            }),
        ):
            result = client.x402_pay("https://api.example.com/premium", max_amount=0.05)
            assert result.success is True


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager(self):
        with AgentPayClient("ap_key") as client:
            assert client._api_key == "ap_key"
        # After exit, close() was called (no assertion needed — just no error)
