"""
Tests for providers/coinbase_wallet.py — CDP-based on-chain wallet management.

All Coinbase AgentKit calls are mocked — no CDP API keys needed.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class MockWalletProvider:
    """Mock CdpEvmWalletProvider."""
    def __init__(self, address="0xAbCdEf0123456789AbCdEf0123456789AbCdEf01"):
        self._address = address

    def get_address(self):
        return self._address

    def get_balance(self):
        return "0.05"

    def invoke_contract(self, **kwargs):
        return "0xtxhash_mock_abc123"


class TestCreateAgentWallet:
    def test_create_new_wallet(self, tmp_path):
        from providers.coinbase_wallet import create_agent_wallet

        mock_provider = MockWalletProvider()

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.coinbase_wallet._get_cdp_keys", return_value=("key_id", "secret", "wallet_secret")), \
             patch("providers.coinbase_wallet.CdpEvmWalletProvider", return_value=mock_provider):
            result = create_agent_wallet("agent-cb-001")

        assert result["address"] == "0xAbCdEf0123456789AbCdEf0123456789AbCdEf01"
        assert "network" in result
        # Wallet file persisted
        assert (tmp_path / "agent-cb-001.json").exists()
        saved = json.loads((tmp_path / "agent-cb-001.json").read_text())
        assert saved["agent_id"] == "agent-cb-001"
        assert saved["address"] == result["address"]

    def test_create_wallet_no_keys(self, tmp_path):
        from providers.coinbase_wallet import create_agent_wallet

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.coinbase_wallet._get_cdp_keys", side_effect=ValueError("CDP API keys not configured")):
            with pytest.raises(ValueError, match="CDP API keys not configured"):
                create_agent_wallet("agent-cb-002")


class TestGetWalletAddress:
    def test_existing_wallet(self, tmp_path):
        from providers.coinbase_wallet import get_wallet_address

        wallet_data = {"agent_id": "agent-cb-003", "address": "0x1234abcd", "network": "base-sepolia"}
        (tmp_path / "agent-cb-003.json").write_text(json.dumps(wallet_data))

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path):
            result = get_wallet_address("agent-cb-003")

        assert result == "0x1234abcd"

    def test_no_wallet(self, tmp_path):
        from providers.coinbase_wallet import get_wallet_address

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path):
            result = get_wallet_address("nonexistent")

        assert result is None


class TestGetWalletBalance:
    def test_balance_success(self, tmp_path):
        from providers.coinbase_wallet import get_wallet_balance

        wallet_data = {"agent_id": "agent-cb-004", "address": "0xbalance", "network": "base-sepolia"}
        (tmp_path / "agent-cb-004.json").write_text(json.dumps(wallet_data))

        mock_provider = MockWalletProvider(address="0xbalance")

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.coinbase_wallet._get_cdp_keys", return_value=("k", "s", "w")), \
             patch("providers.coinbase_wallet.CdpEvmWalletProvider", return_value=mock_provider):
            result = get_wallet_balance("agent-cb-004")

        assert result["balance_eth"] == "0.05"
        assert result["address"] == "0xbalance"

    def test_balance_no_wallet(self, tmp_path):
        from providers.coinbase_wallet import get_wallet_balance

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path):
            result = get_wallet_balance("ghost")

        assert "error" in result

    def test_balance_api_error(self, tmp_path):
        from providers.coinbase_wallet import get_wallet_balance

        wallet_data = {"agent_id": "agent-cb-005", "address": "0xerr", "network": "base-sepolia"}
        (tmp_path / "agent-cb-005.json").write_text(json.dumps(wallet_data))

        mock_provider = MagicMock()
        mock_provider.get_address.return_value = "0xerr"
        mock_provider.get_balance.side_effect = Exception("API down")

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.coinbase_wallet._get_cdp_keys", return_value=("k", "s", "w")), \
             patch("providers.coinbase_wallet.CdpEvmWalletProvider", return_value=mock_provider):
            result = get_wallet_balance("agent-cb-005")

        assert "error" in result
        assert "API down" in result["error"]


class TestSendUsdc:
    @pytest.mark.asyncio
    async def test_send_success(self, tmp_path):
        from providers.coinbase_wallet import send_usdc

        wallet_data = {"agent_id": "agent-cb-006", "address": "0xsender", "network": "base-sepolia"}
        (tmp_path / "agent-cb-006.json").write_text(json.dumps(wallet_data))

        mock_provider = MockWalletProvider(address="0xsender")

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.coinbase_wallet._get_cdp_keys", return_value=("k", "s", "w")), \
             patch("providers.coinbase_wallet.CdpEvmWalletProvider", return_value=mock_provider):
            result = await send_usdc("agent-cb-006", "0xrecipient", 10.0)

        assert result["success"] is True
        assert result["tx_hash"] == "0xtxhash_mock_abc123"
        assert result["amount"] == 10.0
        assert result["to"] == "0xrecipient"

    @pytest.mark.asyncio
    async def test_send_no_wallet(self, tmp_path):
        from providers.coinbase_wallet import send_usdc

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path):
            result = await send_usdc("ghost", "0xrecipient", 5.0)

        assert result["success"] is False
        assert "No wallet" in result["error"]

    @pytest.mark.asyncio
    async def test_send_transfer_fails(self, tmp_path):
        from providers.coinbase_wallet import send_usdc

        wallet_data = {"agent_id": "agent-cb-007", "address": "0xfail", "network": "base-sepolia"}
        (tmp_path / "agent-cb-007.json").write_text(json.dumps(wallet_data))

        mock_provider = MagicMock()
        mock_provider.get_address.return_value = "0xfail"
        mock_provider.invoke_contract.side_effect = Exception("insufficient gas")

        with patch("providers.coinbase_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.coinbase_wallet._get_cdp_keys", return_value=("k", "s", "w")), \
             patch("providers.coinbase_wallet.CdpEvmWalletProvider", return_value=mock_provider):
            result = await send_usdc("agent-cb-007", "0xrecipient", 100.0)

        assert result["success"] is False
        assert "insufficient gas" in result["error"]


class TestGetCdpKeys:
    def test_keys_from_env(self):
        from providers.coinbase_wallet import _get_cdp_keys

        with patch.dict(os.environ, {"CDP_API_KEY_ID": "my-key", "CDP_API_KEY_SECRET": "my-secret", "CDP_WALLET_SECRET": "ws"}):
            key_id, secret, wallet_secret = _get_cdp_keys()

        assert key_id == "my-key"
        assert secret == "my-secret"
        assert wallet_secret == "ws"

    def test_missing_keys_raises(self):
        from providers.coinbase_wallet import _get_cdp_keys

        with patch.dict(os.environ, {"CDP_API_KEY_ID": "", "CDP_API_KEY_SECRET": ""}, clear=False):
            with pytest.raises(ValueError, match="CDP API keys not configured"):
                _get_cdp_keys()
