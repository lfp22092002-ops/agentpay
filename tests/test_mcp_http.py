"""
Tests for the MCP Streamable HTTP endpoint (api/routes/mcp.py).
"""
import json
import os
import sys
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest_asyncio.fixture
async def mcp_client():
    """AsyncClient wired to the app."""
    from api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ═══════════════════════════════════════
# GET /mcp — Discovery
# ═══════════════════════════════════════

class TestMCPGet:
    @pytest.mark.asyncio
    async def test_get_mcp_info(self, mcp_client):
        resp = await mcp_client.get("/mcp")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "agentpay"
        assert data["protocol"] == "mcp"
        assert data["transport"] == "streamable-http"
        assert data["tools_count"] >= 10


# ═══════════════════════════════════════
# POST /mcp — JSON-RPC
# ═══════════════════════════════════════

class TestMCPInitialize:
    @pytest.mark.asyncio
    async def test_initialize(self, mcp_client):
        resp = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 1
        assert data["result"]["serverInfo"]["name"] == "agentpay"
        assert data["result"]["protocolVersion"] == "2025-11-25"
        assert "tools" in data["result"]["capabilities"]


class TestMCPToolsList:
    @pytest.mark.asyncio
    async def test_tools_list(self, mcp_client):
        resp = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        tools = data["result"]["tools"]
        assert len(tools) >= 10
        names = [t["name"] for t in tools]
        assert "agentpay_balance" in names
        assert "agentpay_spend" in names
        assert "agentpay_x402_pay" in names


class TestMCPToolsCall:
    @pytest.mark.asyncio
    async def test_unknown_tool(self, mcp_client):
        resp = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["isError"] is True
        error_text = data["result"]["content"][0]["text"]
        # Depending on test ordering, SDK may or may not be imported
        assert any(msg in error_text for msg in [
            "Unknown tool", "SDK not installed", "API key", "AGENTPAY_API_KEY",
        ])

    @pytest.mark.asyncio
    async def test_tool_call_no_api_key(self, mcp_client):
        """Calling a real tool with no API key should return an error result."""
        resp = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "agentpay_balance", "arguments": {}},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["isError"] is True


class TestMCPNotification:
    @pytest.mark.asyncio
    async def test_notification_returns_202(self, mcp_client):
        resp = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert resp.status_code == 202


class TestMCPUnknownMethod:
    @pytest.mark.asyncio
    async def test_unknown_method(self, mcp_client):
        resp = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 5,
            "method": "foo/bar",
            "params": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"]["code"] == -32601
        assert "Method not found" in data["error"]["message"]


class TestMCPParseError:
    @pytest.mark.asyncio
    async def test_invalid_json(self, mcp_client):
        resp = await mcp_client.post(
            "/mcp",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == -32700


class TestMCPBatch:
    @pytest.mark.asyncio
    async def test_batch_request(self, mcp_client):
        resp = await mcp_client.post("/mcp", json=[
            {"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ])
        assert resp.status_code == 200
        data = resp.json()
        # Batch returns array; notification is excluded (no id)
        assert isinstance(data, list)
        assert len(data) == 2
        ids = [d["id"] for d in data]
        assert 10 in ids
        assert 11 in ids


class TestMCPSSE:
    @pytest.mark.asyncio
    async def test_sse_single(self, mcp_client):
        resp = await mcp_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 20, "method": "initialize", "params": {}},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        text = resp.text
        assert text.startswith("data: ")
        payload = json.loads(text.strip().removeprefix("data: "))
        assert payload["id"] == 20

    @pytest.mark.asyncio
    async def test_sse_batch(self, mcp_client):
        resp = await mcp_client.post(
            "/mcp",
            json=[
                {"jsonrpc": "2.0", "id": 30, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 31, "method": "tools/list", "params": {}},
            ],
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        lines = [line for line in resp.text.strip().split("\n") if line.startswith("data: ")]
        assert len(lines) == 2


class TestMCPToolCallMocked:
    """Test MCP tool dispatch with mocked SDK client."""

    @pytest.mark.asyncio
    async def test_tool_balance_mocked(self, mcp_client):
        """agentpay_balance tool call succeeds with mocked client."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_balance = MagicMock()
        mock_balance.model_dump.return_value = {"balance_usd": 10.0, "agent_id": "test"}
        mock_client.get_balance.return_value = mock_balance

        with patch("api.routes.mcp._get_client", return_value=mock_client):
            resp = await mcp_client.post("/mcp", json={
                "jsonrpc": "2.0",
                "id": 50,
                "method": "tools/call",
                "params": {"name": "agentpay_balance", "arguments": {}},
            }, headers={"X-API-Key": "ap_test_key"})

        assert resp.status_code == 200
        data = resp.json()
        assert "isError" not in data["result"]
        assert "balance_usd" in data["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_spend_mocked(self, mcp_client):
        """agentpay_spend tool call dispatches correctly."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True, "amount": 1.0}
        mock_client.spend.return_value = mock_result

        with patch("api.routes.mcp._get_client", return_value=mock_client):
            resp = await mcp_client.post("/mcp", json={
                "jsonrpc": "2.0",
                "id": 51,
                "method": "tools/call",
                "params": {"name": "agentpay_spend", "arguments": {"amount": 1.0, "description": "test"}},
            }, headers={"X-API-Key": "ap_test_key"})

        assert resp.status_code == 200
        data = resp.json()
        assert "isError" not in data["result"]

    @pytest.mark.asyncio
    async def test_tool_transactions_mocked(self, mcp_client):
        """agentpay_transactions tool returns list."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_tx = MagicMock()
        mock_tx.model_dump.return_value = {"id": "tx1", "amount": 1.0}
        mock_client.get_transactions.return_value = [mock_tx]

        with patch("api.routes.mcp._get_client", return_value=mock_client):
            resp = await mcp_client.post("/mcp", json={
                "jsonrpc": "2.0",
                "id": 52,
                "method": "tools/call",
                "params": {"name": "agentpay_transactions", "arguments": {"limit": 5}},
            }, headers={"X-API-Key": "ap_test_key"})

        assert resp.status_code == 200
        data = resp.json()
        assert "isError" not in data["result"]
        assert "transactions" in data["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_chains_mocked(self, mcp_client):
        """agentpay_chains tool returns chain list."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_chain = MagicMock()
        mock_chain.model_dump.return_value = {"id": "base", "name": "Base"}
        mock_client.list_chains.return_value = [mock_chain]

        with patch("api.routes.mcp._get_client", return_value=mock_client):
            resp = await mcp_client.post("/mcp", json={
                "jsonrpc": "2.0",
                "id": 53,
                "method": "tools/call",
                "params": {"name": "agentpay_chains", "arguments": {}},
            }, headers={"X-API-Key": "ap_test_key"})

        assert resp.status_code == 200
        data = resp.json()
        assert "isError" not in data["result"]
