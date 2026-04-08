"""Tests for the AgentPay Python sync SDK client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from sdk.agentpay.client import AgentPayClient
from sdk.agentpay.exceptions import (
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
)


# -- Full model-compatible fixtures --

BALANCE_DATA = {
    "agent_id": "a_1", "agent_name": "TestBot", "balance_usd": 42.50,
    "daily_limit_usd": 100, "daily_spent_usd": 10, "daily_remaining_usd": 90,
    "tx_limit_usd": 50, "is_active": True,
}

WALLET_DATA = {"address": "0xabc", "chain": "base"}

SPEND_DATA = {"success": True, "transaction_id": "tx_1", "amount": 5.0, "fee": 0.1, "remaining_balance": 37.5, "status": "completed"}

REFUND_DATA = {"success": True, "refund_transaction_id": "tx_99_r", "amount_refunded": 5.0, "new_balance": 42.5, "status": "refunded"}

TRANSFER_DATA = {"success": True, "transaction_id": "tx_t1", "amount": 10.0, "from_balance": 32.5}

TX_LIST = [
    {"id": "tx_1", "type": "spend", "amount": 5.0, "fee": 0.1, "status": "completed", "created_at": "2026-01-01T00:00:00Z"},
    {"id": "tx_2", "type": "refund", "amount": 5.0, "fee": 0, "status": "refunded", "created_at": "2026-01-01T01:00:00Z"},
]

WEBHOOK_DATA = {"url": "https://hook.test", "secret": "whsec_123", "events": ["spend", "refund"]}

X402_DATA = {"success": True, "status": 200, "data": "content", "paid_usd": 0.5}

CHAIN_DATA = {"chains": [
    {"id": "base", "name": "Base", "type": "evm", "native_token": "ETH", "usdc_supported": True, "explorer": "https://basescan.org"},
    {"id": "polygon", "name": "Polygon", "type": "evm", "native_token": "MATIC", "usdc_supported": True, "explorer": "https://polygonscan.com"},
]}

IDENTITY_DATA = {"agent_id": "a_1", "agent_name": "TestBot", "description": "Test agent", "capabilities": []}

TRUST_DATA = {"score": 85, "factors": {"age": 50, "tx_volume": 35}}


def _mock_response(data=None, status=200, headers=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.is_success = 200 <= status < 300
    resp.json.return_value = data if data is not None else {}
    resp.text = str(data)
    resp.headers = httpx.Headers(headers or {})
    return resp


@pytest.fixture
def client():
    with patch("sdk.agentpay.client.httpx.Client"):
        c = AgentPayClient("ap_test_key", base_url="https://test.local", max_retries=0)
    return c


class TestBalance:
    def test_get_balance(self, client):
        client._client.request.return_value = _mock_response(BALANCE_DATA)
        balance = client.get_balance()
        assert balance.balance_usd == 42.50
        assert balance.is_active is True


class TestWallet:
    def test_default_chain(self, client):
        client._client.request.return_value = _mock_response(WALLET_DATA)
        wallet = client.get_wallet()
        client._client.request.assert_called_once_with("GET", "/v1/wallet", json=None, params={"chain": "base"})
        assert wallet.chain == "base"

    def test_polygon(self, client):
        client._client.request.return_value = _mock_response({**WALLET_DATA, "chain": "polygon"})
        wallet = client.get_wallet("polygon")
        assert wallet.chain == "polygon"


class TestChains:
    def test_list_chains(self, client):
        client._client.request.return_value = _mock_response(CHAIN_DATA)
        chains = client.list_chains()
        assert len(chains) == 2
        assert chains[0].id == "base"

    def test_empty(self, client):
        client._client.request.return_value = _mock_response({})
        assert client.list_chains() == []


class TestSpend:
    def test_with_idempotency(self, client):
        client._client.request.return_value = _mock_response(SPEND_DATA)
        result = client.spend(5.0, "test", idempotency_key="idem_1")
        body = client._client.request.call_args[1]["json"]
        assert body["idempotency_key"] == "idem_1"
        assert result.transaction_id == "tx_1"

    def test_without_idempotency(self, client):
        client._client.request.return_value = _mock_response(SPEND_DATA)
        client.spend(1.0, "no key")
        body = client._client.request.call_args[1]["json"]
        assert "idempotency_key" not in body


class TestRefund:
    def test_refund(self, client):
        client._client.request.return_value = _mock_response(REFUND_DATA)
        result = client.refund("tx_99")
        body = client._client.request.call_args[1]["json"]
        assert body["transaction_id"] == "tx_99"
        assert result.success is True


class TestTransfer:
    def test_with_description(self, client):
        client._client.request.return_value = _mock_response(TRANSFER_DATA)
        client.transfer("agent_2", 10.0, description="payment")
        body = client._client.request.call_args[1]["json"]
        assert body["to_agent_id"] == "agent_2"
        assert body["description"] == "payment"

    def test_without_description(self, client):
        client._client.request.return_value = _mock_response(TRANSFER_DATA)
        client.transfer("agent_3", 5.0)
        body = client._client.request.call_args[1]["json"]
        assert "description" not in body


class TestTransactions:
    def test_get_transactions(self, client):
        client._client.request.return_value = _mock_response(TX_LIST)
        txs = client.get_transactions(limit=5)
        client._client.request.assert_called_once_with("GET", "/v1/transactions", json=None, params={"limit": 5})
        assert len(txs) == 2


class TestWebhook:
    def test_register(self, client):
        client._client.request.return_value = _mock_response(WEBHOOK_DATA)
        wh = client.register_webhook("https://hook.test", events=["spend", "refund"])
        body = client._client.request.call_args[1]["json"]
        assert body["events"] == ["spend", "refund"]


class TestX402:
    def test_pay(self, client):
        client._client.request.return_value = _mock_response(X402_DATA)
        result = client.x402_pay("https://paywall.test", max_amount=0.5)
        body = client._client.request.call_args[1]["json"]
        assert body["max_price_usd"] == 0.5
        assert result.success is True


class TestIdentity:
    def test_get_identity(self, client):
        client._client.request.return_value = _mock_response(IDENTITY_DATA)
        identity = client.get_identity()
        # Returns raw dict (no Pydantic model for identity)
        assert identity["agent_id"] == "a_1"

    def test_get_trust_score(self, client):
        client._client.request.return_value = _mock_response(TRUST_DATA)
        score = client.get_trust_score()
        assert score["score"] == 85


class TestErrors:
    def test_401(self, client):
        client._client.request.return_value = _mock_response({"detail": "Invalid key"}, 401)
        with pytest.raises(AuthenticationError):
            client.get_balance()

    def test_402_balance(self, client):
        client._client.request.return_value = _mock_response({"detail": "Insufficient balance"}, 402)
        with pytest.raises(InsufficientBalanceError):
            client.spend(9999, "too much")

    def test_429(self, client):
        client._client.request.return_value = _mock_response({"detail": "Too many requests"}, 429)
        with pytest.raises(RateLimitError):
            client.get_balance()

    def test_404(self, client):
        client._client.request.return_value = _mock_response({"detail": "Not found"}, 404)
        with pytest.raises(AgentPayError):
            client.get_balance()

    def test_network_error(self, client):
        client._client.request.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(AgentPayError, match="Network error"):
            client.get_balance()


class TestRetries:
    def test_retry_500_then_success(self):
        with patch("sdk.agentpay.client.httpx.Client"):
            c = AgentPayClient("key", base_url="https://test.local", max_retries=1)

        c._client.request.side_effect = [
            _mock_response({"detail": "Internal"}, 500),
            _mock_response(BALANCE_DATA),
        ]

        with patch("sdk.agentpay.client.time.sleep"):
            balance = c.get_balance()
        assert balance.balance_usd == 42.50
        assert c._client.request.call_count == 2
