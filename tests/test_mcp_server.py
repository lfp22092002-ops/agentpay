"""Tests for the MCP server tool handlers and JSON-RPC transport."""

from __future__ import annotations

import json
import io
import sys
from unittest.mock import MagicMock, patch

import pytest

# We need to mock the SDK before importing the server
sys.modules.setdefault("agentpay", MagicMock())

from mcp.server import (  # noqa: E402
    TOOL_HANDLERS,
    handle_balance,
    handle_spend,
    handle_transactions,
    handle_refund,
    handle_transfer,
    handle_wallet,
    handle_chains,
    handle_x402_pay,
    handle_webhook,
    handle_identity,
    _write_message,
    _read_message,
    _load_tool_definitions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeModel:
    """Mimics a Pydantic model with .model_dump()."""
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self):
        return self._data


@pytest.fixture
def mock_client():
    """Patch _get_client to return a controllable mock."""
    client = MagicMock()
    with patch("mcp.server._get_client", return_value=client):
        yield client


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------

class TestHandleBalance:
    def test_returns_balance(self, mock_client):
        mock_client.get_balance.return_value = FakeModel({"available": 100.0, "pending": 5.0})
        result = handle_balance({})
        assert result == {"available": 100.0, "pending": 5.0}
        mock_client.get_balance.assert_called_once()


class TestHandleSpend:
    def test_spend_basic(self, mock_client):
        mock_client.spend.return_value = FakeModel({"id": "tx_1", "status": "completed"})
        result = handle_spend({"amount": 10.0, "description": "Test"})
        assert result["id"] == "tx_1"
        mock_client.spend.assert_called_once_with(
            amount=10.0, description="Test", idempotency_key=None
        )

    def test_spend_with_idempotency(self, mock_client):
        mock_client.spend.return_value = FakeModel({"id": "tx_2", "status": "completed"})
        result = handle_spend({"amount": 5.0, "description": "Dup", "idempotency_key": "key123"})
        mock_client.spend.assert_called_once_with(
            amount=5.0, description="Dup", idempotency_key="key123"
        )


class TestHandleTransactions:
    def test_default_limit(self, mock_client):
        mock_client.get_transactions.return_value = [FakeModel({"id": "tx_1"})]
        result = handle_transactions({})
        assert len(result["transactions"]) == 1
        mock_client.get_transactions.assert_called_once_with(limit=10)

    def test_custom_limit(self, mock_client):
        mock_client.get_transactions.return_value = []
        handle_transactions({"limit": 5})
        mock_client.get_transactions.assert_called_once_with(limit=5)


class TestHandleRefund:
    def test_refund(self, mock_client):
        mock_client.refund.return_value = FakeModel({"id": "rf_1", "status": "refunded"})
        result = handle_refund({"transaction_id": "tx_1"})
        assert result["status"] == "refunded"
        mock_client.refund.assert_called_once_with("tx_1")


class TestHandleTransfer:
    def test_transfer(self, mock_client):
        mock_client.transfer.return_value = FakeModel({"id": "tr_1"})
        result = handle_transfer({"to_agent_id": "agent_2", "amount": 25.0})
        assert result["id"] == "tr_1"
        mock_client.transfer.assert_called_once_with(
            to_agent_id="agent_2", amount=25.0, description=None
        )

    def test_transfer_with_description(self, mock_client):
        mock_client.transfer.return_value = FakeModel({"id": "tr_2"})
        handle_transfer({"to_agent_id": "agent_3", "amount": 10.0, "description": "Payment"})
        mock_client.transfer.assert_called_once_with(
            to_agent_id="agent_3", amount=10.0, description="Payment"
        )


class TestHandleWallet:
    def test_default_chain(self, mock_client):
        mock_client.get_wallet.return_value = FakeModel({"address": "0xabc", "chain": "base"})
        result = handle_wallet({})
        assert result["chain"] == "base"
        mock_client.get_wallet.assert_called_once_with(chain="base")

    def test_specific_chain(self, mock_client):
        mock_client.get_wallet.return_value = FakeModel({"address": "0xdef", "chain": "polygon"})
        handle_wallet({"chain": "polygon"})
        mock_client.get_wallet.assert_called_once_with(chain="polygon")


class TestHandleChains:
    def test_list_chains(self, mock_client):
        mock_client.list_chains.return_value = [
            FakeModel({"name": "base"}), FakeModel({"name": "polygon"})
        ]
        result = handle_chains({})
        assert len(result["chains"]) == 2


class TestHandleX402Pay:
    def test_pay(self, mock_client):
        mock_client.x402_pay.return_value = FakeModel({"status": "paid", "amount": 0.01})
        result = handle_x402_pay({"url": "https://example.com/resource"})
        assert result["status"] == "paid"
        mock_client.x402_pay.assert_called_once_with(url="https://example.com/resource", max_amount=None)


class TestHandleWebhook:
    def test_register(self, mock_client):
        mock_client.register_webhook.return_value = FakeModel({"id": "wh_1", "url": "https://hook.example.com"})
        result = handle_webhook({"url": "https://hook.example.com"})
        assert result["id"] == "wh_1"


class TestHandleIdentity:
    def test_identity(self, mock_client):
        mock_client._request.return_value = {"agent_id": "agent_1", "name": "TestBot"}
        result = handle_identity({})
        assert result["agent_id"] == "agent_1"
        mock_client._request.assert_called_once_with("GET", "/v1/identity")


# ---------------------------------------------------------------------------
# Tool registry completeness
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_all_handlers_registered(self):
        expected = {
            "agentpay_balance", "agentpay_spend", "agentpay_transactions",
            "agentpay_refund", "agentpay_transfer", "agentpay_wallet",
            "agentpay_chains", "agentpay_x402_pay", "agentpay_webhook",
            "agentpay_identity",
        }
        assert set(TOOL_HANDLERS.keys()) == expected

    def test_all_handlers_callable(self):
        for name, handler in TOOL_HANDLERS.items():
            assert callable(handler), f"{name} handler is not callable"


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------

class TestWriteMessage:
    def test_writes_json_line(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _write_message({"jsonrpc": "2.0", "id": 1, "result": {}})
        line = buf.getvalue()
        assert line.endswith("\n")
        parsed = json.loads(line.strip())
        assert parsed["jsonrpc"] == "2.0"


class TestReadMessage:
    def test_reads_json_line(self):
        buf = io.StringIO('{"jsonrpc":"2.0","method":"initialize","id":1}\n')
        with patch("sys.stdin", buf):
            msg = _read_message()
        assert msg["method"] == "initialize"

    def test_eof_returns_none(self):
        buf = io.StringIO("")
        with patch("sys.stdin", buf):
            assert _read_message() is None
