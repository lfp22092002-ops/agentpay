"""
Tests for local_wallet.py send functions — USDC and native token transfers.

Uses mocked Web3 to test send_usdc, send_native, send_eth, and get_all_chain_balances
without hitting real RPCs.
"""
import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _write_wallet(tmp_path, agent_id="agent-send-1"):
    """Helper: write a wallet file for testing."""
    wallet = {
        "address": "0x1234567890abcdef1234567890abcdef12345678",
        "private_key": "0xdeadbeef" * 8,
        "encrypted": False,
    }
    wallet_file = tmp_path / f"{agent_id}.json"
    wallet_file.write_text(json.dumps(wallet))
    return wallet


def _mock_w3():
    """Build a mock Web3 instance."""
    w3 = MagicMock()
    w3.eth.get_transaction_count.return_value = 42
    w3.eth.gas_price = 1_000_000_000
    w3.eth.chain_id = 8453
    w3.to_wei.return_value = 1_000_000_000_000_000_000  # 1 ETH in wei

    # Mock contract
    contract = MagicMock()
    transfer_fn = MagicMock()
    transfer_fn.build_transaction.return_value = {
        "from": "0x1234567890abcdef1234567890abcdef12345678",
        "nonce": 42,
        "gas": 100000,
        "gasPrice": 1_000_000_000,
        "to": "0x0000000000000000000000000000000000000000",
        "data": "0x",
    }
    contract.functions.transfer.return_value = transfer_fn
    w3.eth.contract.return_value = contract

    # Mock signing
    signed = MagicMock()
    signed.raw_transaction = b"\x00" * 32
    w3.eth.account.sign_transaction.return_value = signed
    w3.eth.send_raw_transaction.return_value = b"\xaa" * 32

    return w3


class TestSendUSDC:
    @pytest.mark.asyncio
    async def test_send_usdc_success(self, tmp_path):
        from providers.local_wallet import send_usdc

        _write_wallet(tmp_path)
        mock = _mock_w3()

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock):
            result = await send_usdc("agent-send-1", "0x" + "ab" * 20, 10.0, chain="base")

        assert result["success"] is True
        assert result["amount"] == 10.0
        assert result["chain"] == "base"
        assert "tx_hash" in result

    @pytest.mark.asyncio
    async def test_send_usdc_no_wallet(self, tmp_path):
        from providers.local_wallet import send_usdc

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = await send_usdc("ghost", "0x" + "0" * 40, 5.0)

        assert result["success"] is False
        assert "No wallet" in result["error"]

    @pytest.mark.asyncio
    async def test_send_usdc_invalid_chain(self, tmp_path):
        from providers.local_wallet import send_usdc

        _write_wallet(tmp_path)

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             pytest.raises(ValueError, match="Unsupported chain"):
            await send_usdc("agent-send-1", "0x" + "0" * 40, 5.0, chain="avalanche")

    @pytest.mark.asyncio
    async def test_send_usdc_rpc_error(self, tmp_path):
        from providers.local_wallet import send_usdc

        _write_wallet(tmp_path)
        mock = _mock_w3()
        mock.eth.send_raw_transaction.side_effect = Exception("RPC error: insufficient funds")

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock):
            result = await send_usdc("agent-send-1", "0x" + "0" * 40, 10.0)

        assert result["success"] is False
        assert "insufficient funds" in result["error"]

    @pytest.mark.asyncio
    async def test_send_usdc_polygon(self, tmp_path):
        from providers.local_wallet import send_usdc

        _write_wallet(tmp_path)
        mock = _mock_w3()

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock):
            result = await send_usdc("agent-send-1", "0x" + "a" * 40, 25.0, chain="polygon")

        assert result["success"] is True
        assert result["chain"] == "polygon"


class TestSendNative:
    @pytest.mark.asyncio
    async def test_send_native_success(self, tmp_path):
        from providers.local_wallet import send_native

        _write_wallet(tmp_path)
        mock = _mock_w3()

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock):
            result = await send_native("agent-send-1", "0x" + "b" * 40, 0.01, chain="base")

        assert result["success"] is True
        assert result["native_token"] == "ETH"
        assert result["chain"] == "base"

    @pytest.mark.asyncio
    async def test_send_native_bnb(self, tmp_path):
        from providers.local_wallet import send_native

        _write_wallet(tmp_path)
        mock = _mock_w3()

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock):
            result = await send_native("agent-send-1", "0x" + "c" * 40, 0.05, chain="bnb")

        assert result["success"] is True
        assert result["native_token"] == "BNB"

    @pytest.mark.asyncio
    async def test_send_native_no_wallet(self, tmp_path):
        from providers.local_wallet import send_native

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = await send_native("ghost", "0x" + "0" * 40, 0.01)

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_send_native_rpc_error(self, tmp_path):
        from providers.local_wallet import send_native

        _write_wallet(tmp_path)
        mock = _mock_w3()
        mock.eth.send_raw_transaction.side_effect = Exception("nonce too low")

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock):
            result = await send_native("agent-send-1", "0x" + "0" * 40, 0.01)

        assert result["success"] is False
        assert "nonce" in result["error"]


class TestSendEthAlias:
    @pytest.mark.asyncio
    async def test_send_eth_calls_send_native(self, tmp_path):
        from providers.local_wallet import send_eth

        _write_wallet(tmp_path)
        mock = _mock_w3()

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock):
            result = await send_eth("agent-send-1", "0x" + "d" * 40, 0.1)

        assert result["success"] is True
        assert result["chain"] == "base"  # default chain


class TestGetAllChainBalances:
    def test_all_chain_balances_no_wallet(self, tmp_path):
        from providers.local_wallet import get_all_chain_balances

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = get_all_chain_balances("ghost")

        assert result == []

    def test_all_chain_balances_returns_list(self, tmp_path):
        from providers.local_wallet import get_all_chain_balances

        _write_wallet(tmp_path)

        def mock_balance(agent_id, chain):
            return {"chain": chain, "balance_usdc": "0", "balance_native": "0"}

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet.get_wallet_balance", side_effect=mock_balance):
            result = get_all_chain_balances("agent-send-1")

        assert len(result) == 3  # base, polygon, bnb
        chains = [r["chain"] for r in result]
        assert "base" in chains
        assert "polygon" in chains
        assert "bnb" in chains
