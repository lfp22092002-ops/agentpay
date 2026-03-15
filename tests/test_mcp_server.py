"""
Tests for MCP Server (mcp/server.py).

Tests the JSON-RPC stdio transport, tool routing, error handling,
and tool handler functions (mocked SDK calls).
"""
import json
import os
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mcp"))


# ═══════════════════════════════════════
# MESSAGE I/O
# ═══════════════════════════════════════

class TestMessageIO:
    def test_read_message_valid(self):
        from mcp.server import _read_message
        msg = {"jsonrpc": "2.0", "method": "test", "id": 1}
        with patch("sys.stdin", StringIO(json.dumps(msg) + "\n")):
            result = _read_message()
            assert result == msg

    def test_read_message_eof(self):
        from mcp.server import _read_message
        with patch("sys.stdin", StringIO("")):
            result = _read_message()
            assert result is None

    def test_write_message(self):
        from mcp.server import _write_message
        buf = StringIO()
        with patch("sys.stdout", buf):
            _write_message({"jsonrpc": "2.0", "id": 1, "result": {}})
        output = buf.getvalue()
        parsed = json.loads(output.strip())
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 1


# ═══════════════════════════════════════
# TOOL DEFINITIONS
# ═══════════════════════════════════════

class TestToolDefinitions:
    def test_load_tool_definitions(self):
        from mcp.server import _load_tool_definitions
        tools = _load_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) > 0
        names = [t["name"] for t in tools]
        assert "agentpay_balance" in names
        assert "agentpay_spend" in names

    def test_all_tools_have_handlers(self):
        from mcp.server import _load_tool_definitions, TOOL_HANDLERS
        tools = _load_tool_definitions()
        for tool in tools:
            assert tool["name"] in TOOL_HANDLERS, f"No handler for tool: {tool['name']}"


# ═══════════════════════════════════════
# _get_client GUARD
# ═══════════════════════════════════════

class TestGetClient:
    def test_no_sdk_raises(self):
        from mcp.server import _get_client
        with patch("mcp.server._sdk_available", False):
            with pytest.raises(RuntimeError, match="agentpay SDK not installed"):
                _get_client()

    def test_no_api_key_raises(self):
        from mcp.server import _get_client
        with patch("mcp.server._sdk_available", True), \
             patch.dict(os.environ, {"AGENTPAY_API_KEY": ""}, clear=False):
            with pytest.raises(RuntimeError, match="AGENTPAY_API_KEY"):
                _get_client()


# ═══════════════════════════════════════
# TOOL HANDLERS
# ═══════════════════════════════════════

def _mock_model(data: dict):
    """Create a mock object with model_dump()."""
    m = MagicMock()
    m.model_dump.return_value = data
    return m


class TestToolHandlers:
    def test_handle_balance(self):
        from mcp.server import handle_balance
        mock_client = MagicMock()
        mock_client.get_balance.return_value = _mock_model({
            "balance_usd": 100.0, "daily_limit_usd": 50.0
        })
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_balance({})
            assert result["balance_usd"] == 100.0
            mock_client.get_balance.assert_called_once()

    def test_handle_spend(self):
        from mcp.server import handle_spend
        mock_client = MagicMock()
        mock_client.spend.return_value = _mock_model({
            "transaction_id": "tx_123", "amount": 5.0
        })
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_spend({"amount": 5.0, "description": "test"})
            assert result["transaction_id"] == "tx_123"
            mock_client.spend.assert_called_once_with(
                amount=5.0, description="test", idempotency_key=None
            )

    def test_handle_spend_with_idempotency(self):
        from mcp.server import handle_spend
        mock_client = MagicMock()
        mock_client.spend.return_value = _mock_model({"transaction_id": "tx_456"})
        with patch("mcp.server._get_client", return_value=mock_client):
            handle_spend({"amount": 10.0, "description": "dup-safe", "idempotency_key": "key1"})
            mock_client.spend.assert_called_once_with(
                amount=10.0, description="dup-safe", idempotency_key="key1"
            )

    def test_handle_transactions(self):
        from mcp.server import handle_transactions
        mock_client = MagicMock()
        mock_client.get_transactions.return_value = [
            _mock_model({"id": "tx_1"}), _mock_model({"id": "tx_2"})
        ]
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_transactions({"limit": 5})
            assert len(result["transactions"]) == 2
            mock_client.get_transactions.assert_called_once_with(limit=5)

    def test_handle_transactions_default_limit(self):
        from mcp.server import handle_transactions
        mock_client = MagicMock()
        mock_client.get_transactions.return_value = []
        with patch("mcp.server._get_client", return_value=mock_client):
            handle_transactions({})
            mock_client.get_transactions.assert_called_once_with(limit=10)

    def test_handle_refund(self):
        from mcp.server import handle_refund
        mock_client = MagicMock()
        mock_client.refund.return_value = _mock_model({"refund_id": "ref_1"})
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_refund({"transaction_id": "tx_1"})
            assert result["refund_id"] == "ref_1"

    def test_handle_transfer(self):
        from mcp.server import handle_transfer
        mock_client = MagicMock()
        mock_client.transfer.return_value = _mock_model({"transfer_id": "tr_1"})
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_transfer({
                "to_agent_id": "agent_2", "amount": 25.0, "description": "payment"
            })
            assert result["transfer_id"] == "tr_1"
            mock_client.transfer.assert_called_once_with(
                to_agent_id="agent_2", amount=25.0, description="payment"
            )

    def test_handle_wallet(self):
        from mcp.server import handle_wallet
        mock_client = MagicMock()
        mock_client.get_wallet.return_value = _mock_model({
            "address": "0xABC", "chain": "base"
        })
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_wallet({"chain": "polygon"})
            assert result["address"] == "0xABC"
            mock_client.get_wallet.assert_called_once_with(chain="polygon")

    def test_handle_wallet_default_chain(self):
        from mcp.server import handle_wallet
        mock_client = MagicMock()
        mock_client.get_wallet.return_value = _mock_model({"chain": "base"})
        with patch("mcp.server._get_client", return_value=mock_client):
            handle_wallet({})
            mock_client.get_wallet.assert_called_once_with(chain="base")

    def test_handle_chains(self):
        from mcp.server import handle_chains
        mock_client = MagicMock()
        mock_client.list_chains.return_value = [
            _mock_model({"chain": "base"}), _mock_model({"chain": "solana"})
        ]
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_chains({})
            assert len(result["chains"]) == 2

    def test_handle_x402_pay(self):
        from mcp.server import handle_x402_pay
        mock_client = MagicMock()
        mock_client.x402_pay.return_value = _mock_model({
            "status": "paid", "url": "https://api.example.com/data"
        })
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_x402_pay({"url": "https://api.example.com/data"})
            assert result["status"] == "paid"

    def test_handle_webhook(self):
        from mcp.server import handle_webhook
        mock_client = MagicMock()
        mock_client.register_webhook.return_value = _mock_model({"id": "wh_1"})
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_webhook({
                "url": "https://example.com/hook", "events": ["spend"]
            })
            assert result["id"] == "wh_1"

    def test_handle_identity(self):
        from mcp.server import handle_identity
        mock_client = MagicMock()
        mock_client._request.return_value = {"display_name": "TestBot", "trust_score": 85}
        with patch("mcp.server._get_client", return_value=mock_client):
            result = handle_identity({})
            assert result["display_name"] == "TestBot"


# ═══════════════════════════════════════
# MAIN LOOP (JSON-RPC PROTOCOL)
# ═══════════════════════════════════════

class TestMainLoop:
    def _run_main(self, messages: list[dict]) -> list[dict]:
        """Feed messages through main() and collect responses."""
        from mcp.server import main
        stdin_data = "\n".join(json.dumps(m) for m in messages) + "\n"
        stdout_buf = StringIO()
        with patch("sys.stdin", StringIO(stdin_data)), \
             patch("sys.stdout", stdout_buf):
            main()
        lines = stdout_buf.getvalue().strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]

    def test_initialize(self):
        responses = self._run_main([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        ])
        assert len(responses) == 1
        r = responses[0]
        assert r["id"] == 1
        assert r["result"]["serverInfo"]["name"] == "agentpay"
        assert "protocolVersion" in r["result"]

    def test_tools_list(self):
        responses = self._run_main([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ])
        tools_resp = responses[1]
        assert tools_resp["id"] == 2
        tools = tools_resp["result"]["tools"]
        assert any(t["name"] == "agentpay_balance" for t in tools)

    def test_tools_call_success(self):
        mock_client = MagicMock()
        mock_client.get_balance.return_value = _mock_model({"balance_usd": 42.0})
        with patch("mcp.server._get_client", return_value=mock_client):
            responses = self._run_main([
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                    "name": "agentpay_balance", "arguments": {}
                }},
            ])
        assert len(responses) == 1
        content = responses[0]["result"]["content"]
        assert content[0]["type"] == "text"
        parsed = json.loads(content[0]["text"])
        assert parsed["balance_usd"] == 42.0

    def test_tools_call_unknown_tool(self):
        responses = self._run_main([
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "nonexistent_tool", "arguments": {}
            }},
        ])
        assert "error" in responses[0]
        assert responses[0]["error"]["code"] == -32601

    def test_tools_call_handler_error(self):
        with patch("mcp.server._get_client", side_effect=RuntimeError("No API key")):
            responses = self._run_main([
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                    "name": "agentpay_balance", "arguments": {}
                }},
            ])
        result = responses[0]["result"]
        assert result["isError"] is True
        assert "No API key" in result["content"][0]["text"]

    def test_notifications_initialized_no_response(self):
        responses = self._run_main([
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        ])
        # notifications don't produce responses
        assert len(responses) == 0

    def test_unknown_method(self):
        responses = self._run_main([
            {"jsonrpc": "2.0", "id": 99, "method": "unknown/method", "params": {}},
        ])
        assert responses[0]["error"]["code"] == -32601
        assert "Method not found" in responses[0]["error"]["message"]
