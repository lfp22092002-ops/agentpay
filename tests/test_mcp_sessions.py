"""
Tests for MCP Streamable HTTP session management (Mcp-Session-Id).

Covers:
- Session creation on initialize
- Session validation on subsequent requests
- Expired/invalid session → 404
- DELETE /mcp session termination
- Session header propagation in responses
"""
import json
import os
import sys
import time
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest_asyncio.fixture
async def mcp_client():
    """Create an HTTPX AsyncClient for MCP endpoint testing."""
    from api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestMCPSessionCreate:
    @pytest.mark.asyncio
    async def test_initialize_returns_session_id(self, mcp_client):
        """POST initialize should create session and return Mcp-Session-Id header."""
        resp = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })
        assert resp.status_code == 200
        assert "mcp-session-id" in resp.headers
        session_id = resp.headers["mcp-session-id"]
        assert len(session_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_batch_with_initialize_returns_session(self, mcp_client):
        """Batch containing initialize should create a session."""
        resp = await mcp_client.post("/mcp", json=[
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ])
        assert resp.status_code == 200
        assert "mcp-session-id" in resp.headers


class TestMCPSessionValidation:
    @pytest.mark.asyncio
    async def test_valid_session_accepted(self, mcp_client):
        """Subsequent requests with valid session ID should succeed."""
        # Initialize to get session
        resp1 = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        session_id = resp1.headers["mcp-session-id"]

        # Use session for tools/list
        resp2 = await mcp_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers={"Mcp-Session-Id": session_id},
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert "result" in data
        assert "tools" in data["result"]

    @pytest.mark.asyncio
    async def test_invalid_session_returns_404(self, mcp_client):
        """Request with unknown session ID should return 404."""
        resp = await mcp_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Mcp-Session-Id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_session_header_still_works(self, mcp_client):
        """Requests without session header should work (stateless fallback)."""
        resp = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_expired_session_returns_404(self, mcp_client):
        """Expired session should return 404."""
        # Initialize
        resp1 = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        session_id = resp1.headers["mcp-session-id"]

        # Manually expire the session
        from api.routes.mcp import _sessions
        _sessions[session_id]["last_used"] = time.time() - 7200  # 2 hours ago

        # Should get 404
        resp2 = await mcp_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers={"Mcp-Session-Id": session_id},
        )
        assert resp2.status_code == 404


class TestMCPSessionDelete:
    @pytest.mark.asyncio
    async def test_delete_existing_session(self, mcp_client):
        """DELETE /mcp with valid session should return 200."""
        # Create session
        resp1 = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        session_id = resp1.headers["mcp-session-id"]

        # Delete it
        resp2 = await mcp_client.request(
            "DELETE", "/mcp",
            headers={"Mcp-Session-Id": session_id},
        )
        assert resp2.status_code == 200

        # Using it again should fail
        resp3 = await mcp_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers={"Mcp-Session-Id": session_id},
        )
        assert resp3.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_unknown_session(self, mcp_client):
        """DELETE /mcp with unknown session should return 404."""
        resp = await mcp_client.request(
            "DELETE", "/mcp",
            headers={"Mcp-Session-Id": "nonexistent-session-id"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_without_header(self, mcp_client):
        """DELETE /mcp without session header should return 400."""
        resp = await mcp_client.request("DELETE", "/mcp")
        assert resp.status_code == 400


class TestMCPGetWithSession:
    @pytest.mark.asyncio
    async def test_get_with_valid_session(self, mcp_client):
        """GET /mcp with valid session should succeed."""
        # Create session
        resp1 = await mcp_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        session_id = resp1.headers["mcp-session-id"]

        resp2 = await mcp_client.get("/mcp", headers={"Mcp-Session-Id": session_id})
        assert resp2.status_code == 200

    @pytest.mark.asyncio
    async def test_get_with_invalid_session(self, mcp_client):
        """GET /mcp with invalid session should return 404."""
        resp = await mcp_client.get(
            "/mcp",
            headers={"Mcp-Session-Id": "invalid-session-id"},
        )
        assert resp.status_code == 404
