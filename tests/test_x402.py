"""
Tests for providers/x402_protocol.py — x402 payment protocol.

Tests the client (pay for resources), server (middleware), and helpers
(cost estimation) using mocked HTTP responses.
"""
import json
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def mock_402_response():
    """Build a mock 402 response with payment requirements."""
    resp = MagicMock()
    resp.status_code = 402
    resp.json.return_value = {
        "accepts": [{
            "maxAmountRequired": 10000,  # $0.01 in atomic USDC
            "asset": "USDC",
            "payTo": "0xRecipient123",
            "network": "base-sepolia",
            "scheme": "exact",
        }]
    }
    return resp


def mock_200_response(data="OK"):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = data
    resp.json.return_value = {"result": data}
    return resp


def mock_wallet_file(tmp_path, agent_id="test-agent"):
    """Create a mock wallet file."""
    wallet_data = {
        "address": "0xTestWallet123",
        "private_key": "0xdeadbeef1234567890",
        "chain": "base",
        "encrypted": False,
    }
    wallet_file = tmp_path / f"{agent_id}.json"
    wallet_file.write_text(json.dumps(wallet_data))
    return tmp_path


# ═══════════════════════════════════════
# CLIENT — pay_x402_resource
# ═══════════════════════════════════════

class TestX402Client:
    """Test x402 client (agent pays for gated resources)."""

    @pytest.mark.asyncio
    async def test_no_wallet_returns_error(self, tmp_path):
        """Agent with no wallet file gets clear error."""
        from providers.x402_protocol import pay_x402_resource

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = await pay_x402_resource("ghost-agent", "https://example.com/resource")

        assert result["success"] is False
        assert "no wallet" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_non_402_passes_through(self, tmp_path):
        """Non-402 response is returned directly without payment."""
        from providers.x402_protocol import pay_x402_resource

        wallets_dir = mock_wallet_file(tmp_path)

        mock_client_instance = AsyncMock()
        mock_resp = mock_200_response("hello world")
        mock_client_instance.get = AsyncMock(return_value=mock_resp)

        with patch("providers.local_wallet.WALLETS_DIR", wallets_dir), \
             patch("httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await pay_x402_resource("test-agent", "https://example.com/free")

        assert result["success"] is True
        assert result["paid_usd"] == 0
        assert result["data"] == "hello world"

    @pytest.mark.asyncio
    async def test_402_price_too_high(self, tmp_path):
        """Rejects payment when price exceeds max_price_usd."""
        from providers.x402_protocol import pay_x402_resource

        wallets_dir = mock_wallet_file(tmp_path)

        expensive_resp = MagicMock()
        expensive_resp.status_code = 402
        expensive_resp.json.return_value = {
            "accepts": [{
                "maxAmountRequired": 5_000_000,  # $5.00
                "payTo": "0xExpensive",
                "network": "base",
                "scheme": "exact",
            }]
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=expensive_resp)

        with patch("providers.local_wallet.WALLETS_DIR", wallets_dir), \
             patch("httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await pay_x402_resource("test-agent", "https://expensive.com", max_price_usd=1.0)

        assert result["success"] is False
        assert "exceeds max" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_402_no_payment_requirements(self, tmp_path):
        """402 with empty accepts list returns error."""
        from providers.x402_protocol import pay_x402_resource

        wallets_dir = mock_wallet_file(tmp_path)

        empty_402 = MagicMock()
        empty_402.status_code = 402
        empty_402.json.return_value = {"accepts": []}

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=empty_402)

        with patch("providers.local_wallet.WALLETS_DIR", wallets_dir), \
             patch("httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await pay_x402_resource("test-agent", "https://empty402.com")

        assert result["success"] is False
        assert "no payment requirements" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_402_successful_payment(self, tmp_path):
        """Successful x402 payment flow — 402 → sign → paid response."""
        from providers.x402_protocol import pay_x402_resource

        wallets_dir = mock_wallet_file(tmp_path)

        resp_402 = mock_402_response()
        paid_resp = MagicMock()
        paid_resp.status_code = 200
        paid_resp.text = '{"result": "paid content"}'

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=resp_402)

        mock_x402_client = AsyncMock()
        mock_x402_client.get = AsyncMock(return_value=paid_resp)

        mock_x402_mod = MagicMock()
        mock_x402_mod.x402HttpxClient = MagicMock(return_value=mock_x402_client)

        with patch("providers.local_wallet.WALLETS_DIR", wallets_dir), \
             patch("httpx.AsyncClient") as MockAsyncClient, \
             patch.dict("sys.modules", {"x402.clients.httpx": mock_x402_mod}):
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await pay_x402_resource("test-agent", "https://gated.com/api", max_price_usd=1.0)

        assert result["success"] is True
        assert result["paid_usd"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_encrypted_wallet(self, tmp_path):
        """Encrypted wallet key gets decrypted before use."""
        from providers.x402_protocol import pay_x402_resource

        wallet_data = {
            "address": "0xEncryptedWallet",
            "private_key": "encrypted:deadbeef",
            "chain": "base",
            "encrypted": True,
        }
        (tmp_path / "enc-agent.json").write_text(json.dumps(wallet_data))

        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_200_response())

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("httpx.AsyncClient") as MockAsyncClient, \
             patch("core.encryption.decrypt", return_value="0xdecrypted_key") as mock_decrypt:
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await pay_x402_resource("enc-agent", "https://example.com/free")

        mock_decrypt.assert_called_once_with("encrypted:deadbeef")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_post_method(self, tmp_path):
        """POST requests are forwarded correctly."""
        from providers.x402_protocol import pay_x402_resource

        wallets_dir = mock_wallet_file(tmp_path)

        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_200_response("created"))

        with patch("providers.local_wallet.WALLETS_DIR", wallets_dir), \
             patch("httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockAsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await pay_x402_resource(
                "test-agent", "https://example.com/create",
                method="POST", body={"key": "value"}
            )

        assert result["success"] is True
        mock_client_instance.request.assert_called_once_with("POST", "https://example.com/create", json={"key": "value"})


# ═══════════════════════════════════════
# SERVER — get_x402_middleware
# ═══════════════════════════════════════

class TestX402Server:
    """Test x402 server middleware creation."""

    def test_create_middleware_success(self):
        """Creates middleware when x402 SDK is available."""
        from providers.x402_protocol import get_x402_middleware

        mock_middleware = MagicMock()
        mock_require_payment = MagicMock(return_value=mock_middleware)

        with patch.dict("sys.modules", {
            "x402": MagicMock(),
            "x402.fastapi": MagicMock(),
            "x402.fastapi.middleware": MagicMock(require_payment=mock_require_payment),
        }):
            result = get_x402_middleware("0xMyAddress", price_usd=0.05)

        assert result is mock_middleware

    def test_create_middleware_no_sdk(self):
        """Raises error when x402 SDK is not installed."""
        from providers.x402_protocol import get_x402_middleware

        with patch.dict("sys.modules", {"x402.fastapi.middleware": None}):
            with pytest.raises(Exception):
                get_x402_middleware("0xMyAddress")


# ═══════════════════════════════════════
# HELPERS — estimate_x402_cost
# ═══════════════════════════════════════

class TestX402CostEstimation:
    """Test x402 cost estimation helper."""

    def test_non_gated_resource(self):
        """Non-402 resource is not gated."""
        from providers.x402_protocol import estimate_x402_cost

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.get", return_value=mock_resp):
            result = estimate_x402_cost("https://free.com/api")

        assert result["gated"] is False
        assert result["status"] == 200

    def test_gated_resource_single_cost(self):
        """402 resource returns cost breakdown."""
        from providers.x402_protocol import estimate_x402_cost

        mock_resp = MagicMock()
        mock_resp.status_code = 402
        mock_resp.json.return_value = {
            "accepts": [{
                "maxAmountRequired": 50000,
                "network": "base",
                "asset": "USDC",
                "scheme": "exact",
            }]
        }

        with patch("httpx.get", return_value=mock_resp):
            result = estimate_x402_cost("https://gated.com/api")

        assert result["gated"] is True
        assert len(result["costs"]) == 1
        assert result["costs"][0]["price_usd"] == pytest.approx(0.05)
        assert result["costs"][0]["network"] == "base"

    def test_gated_resource_multiple_costs(self):
        """402 resource with multiple payment options."""
        from providers.x402_protocol import estimate_x402_cost

        mock_resp = MagicMock()
        mock_resp.status_code = 402
        mock_resp.json.return_value = {
            "accepts": [
                {"maxAmountRequired": 10000, "network": "base", "asset": "USDC", "scheme": "exact"},
                {"maxAmountRequired": 15000, "network": "polygon", "asset": "USDC", "scheme": "exact"},
            ]
        }

        with patch("httpx.get", return_value=mock_resp):
            result = estimate_x402_cost("https://multi.com/api")

        assert result["gated"] is True
        assert len(result["costs"]) == 2
        assert result["costs"][0]["price_usd"] == pytest.approx(0.01)
        assert result["costs"][1]["price_usd"] == pytest.approx(0.015)

    def test_network_error(self):
        """Network error returns error dict."""
        from providers.x402_protocol import estimate_x402_cost

        with patch("httpx.get", side_effect=Exception("Connection refused")):
            result = estimate_x402_cost("https://unreachable.com")

        assert "error" in result
        assert "Connection refused" in result["error"]
