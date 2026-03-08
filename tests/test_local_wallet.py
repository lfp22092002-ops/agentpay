"""
Tests for providers/local_wallet.py — EVM wallet creation, address lookup, balance checks.

Uses mocked Web3/Account so no real RPC calls or private keys are needed.
"""
import json
import os
from unittest.mock import patch, MagicMock, PropertyMock

import pytest


class TestChainConfig:
    """Test chain configuration utilities."""

    def test_supported_chains(self):
        from providers.local_wallet import SUPPORTED_EVM_CHAINS, CHAIN_CONFIGS
        assert "base" in SUPPORTED_EVM_CHAINS
        assert "polygon" in SUPPORTED_EVM_CHAINS
        assert "bnb" in SUPPORTED_EVM_CHAINS
        for chain in SUPPORTED_EVM_CHAINS:
            cfg = CHAIN_CONFIGS[chain]
            assert "rpc_mainnet" in cfg
            assert "rpc_testnet" in cfg
            assert "usdc_mainnet" in cfg
            assert "native_token" in cfg

    def test_get_chain_config_valid(self):
        from providers.local_wallet import _get_chain_config
        cfg = _get_chain_config("base")
        assert cfg["native_token"] == "ETH"

    def test_get_chain_config_invalid(self):
        from providers.local_wallet import _get_chain_config
        with pytest.raises(ValueError, match="Unsupported chain"):
            _get_chain_config("avalanche")

    def test_get_rpc_testnet(self):
        from providers.local_wallet import _get_rpc
        with patch("providers.local_wallet._is_testnet", return_value=True):
            rpc = _get_rpc("base")
            assert "sepolia" in rpc

    def test_get_rpc_mainnet(self):
        from providers.local_wallet import _get_rpc
        with patch("providers.local_wallet._is_testnet", return_value=False):
            rpc = _get_rpc("base")
            assert "mainnet" in rpc

    def test_get_usdc_address(self):
        from providers.local_wallet import _get_usdc_address
        with patch("providers.local_wallet._is_testnet", return_value=False):
            addr = _get_usdc_address("base")
            assert addr.startswith("0x")
            assert len(addr) == 42

    def test_get_network_name(self):
        from providers.local_wallet import _get_network_name
        with patch("providers.local_wallet._is_testnet", return_value=True):
            assert _get_network_name("polygon") == "polygon-testnet"
        with patch("providers.local_wallet._is_testnet", return_value=False):
            assert _get_network_name("polygon") == "polygon-mainnet"


class TestWalletCreation:
    """Test EVM wallet creation."""

    def test_create_new_wallet(self, tmp_path):
        from providers.local_wallet import create_agent_wallet

        mock_account = MagicMock()
        mock_account.address = "0x1234567890abcdef1234567890abcdef12345678"
        mock_account.key.hex.return_value = "deadbeef" * 8

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet.Account.create", return_value=mock_account), \
             patch("providers.local_wallet.encrypt", return_value="encrypted_key"):
            result = create_agent_wallet("agent-w1", chain="base")

        assert result["address"] == "0x1234567890abcdef1234567890abcdef12345678"
        assert result["chain"] == "base"
        assert "already_existed" not in result
        # Verify file saved
        assert (tmp_path / "agent-w1.json").exists()
        saved = json.loads((tmp_path / "agent-w1.json").read_text())
        assert saved["private_key"] == "encrypted_key"
        assert saved["encrypted"] is True

    def test_create_wallet_already_exists(self, tmp_path):
        from providers.local_wallet import create_agent_wallet

        existing = {
            "agent_id": "agent-w2",
            "address": "0xexisting",
            "network": "base-testnet",
            "private_key": "enc_key",
            "encrypted": True,
        }
        (tmp_path / "agent-w2.json").write_text(json.dumps(existing))

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = create_agent_wallet("agent-w2", chain="polygon")

        assert result["address"] == "0xexisting"
        assert result["already_existed"] is True

    def test_create_wallet_invalid_chain(self):
        from providers.local_wallet import create_agent_wallet
        with pytest.raises(ValueError, match="Unsupported chain"):
            create_agent_wallet("agent-w3", chain="solana")


class TestWalletAddress:
    """Test wallet address retrieval."""

    def test_get_address_exists(self, tmp_path):
        from providers.local_wallet import get_wallet_address

        data = {"address": "0xMyAddr", "private_key": "pk"}
        (tmp_path / "agent-a1.json").write_text(json.dumps(data))

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            addr = get_wallet_address("agent-a1")

        assert addr == "0xMyAddr"

    def test_get_address_not_found(self, tmp_path):
        from providers.local_wallet import get_wallet_address

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            addr = get_wallet_address("nonexistent")

        assert addr is None


class TestWalletBalance:
    """Test balance retrieval (mocked Web3)."""

    def test_get_balance_no_wallet(self, tmp_path):
        from providers.local_wallet import get_wallet_balance

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = get_wallet_balance("ghost")

        assert "error" in result
        assert "No wallet" in result["error"]

    def test_get_balance_success(self, tmp_path):
        from providers.local_wallet import get_wallet_balance

        addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18"
        data = {"address": addr, "private_key": "pk"}
        (tmp_path / "agent-b1.json").write_text(json.dumps(data))

        mock_w3 = MagicMock()
        mock_w3.eth.get_balance.return_value = 1_000_000_000_000_000_000  # 1 ETH in wei
        mock_w3.from_wei.return_value = 1.0
        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call.return_value = 50_000_000  # 50 USDC
        mock_w3.eth.contract.return_value = mock_contract

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock_w3):
            result = get_wallet_balance("agent-b1", chain="base")

        assert result["address"] == addr
        assert result["chain"] == "base"
        assert result["native_token"] == "ETH"
        assert result["balance_usdc"] == "50.0"

    def test_get_balance_rpc_error(self, tmp_path):
        from providers.local_wallet import get_wallet_balance

        addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18"
        data = {"address": addr, "private_key": "pk"}
        (tmp_path / "agent-b2.json").write_text(json.dumps(data))

        mock_w3 = MagicMock()
        mock_w3.eth.get_balance.side_effect = Exception("RPC timeout")

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock_w3):
            result = get_wallet_balance("agent-b2", chain="polygon")

        assert "error" in result
        assert result["address"] == addr


class TestPrivateKeyLoading:
    """Test private key decryption."""

    def test_load_encrypted_key(self):
        from providers.local_wallet import _load_private_key

        with patch("providers.local_wallet.decrypt", return_value="decrypted_pk"):
            result = _load_private_key({"private_key": "enc_data", "encrypted": True})

        assert result == "decrypted_pk"

    def test_load_unencrypted_key(self):
        from providers.local_wallet import _load_private_key

        result = _load_private_key({"private_key": "raw_pk", "encrypted": False})
        assert result == "raw_pk"

    def test_load_key_no_encrypted_flag(self):
        from providers.local_wallet import _load_private_key

        result = _load_private_key({"private_key": "raw_pk"})
        assert result == "raw_pk"


class TestSendUSDC:
    """Test USDC transfer (mocked)."""

    @pytest.mark.asyncio
    async def test_send_usdc_no_wallet(self, tmp_path):
        from providers.local_wallet import send_usdc

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = await send_usdc("ghost", "0xRecipient", 10.0)

        assert result["success"] is False
        assert "No wallet" in result["error"]

    @pytest.mark.asyncio
    async def test_send_usdc_success(self, tmp_path):
        from providers.local_wallet import send_usdc

        addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18"
        recipient = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
        data = {"address": addr, "private_key": "0x" + "ab" * 32, "encrypted": False}
        (tmp_path / "agent-s1.json").write_text(json.dumps(data))

        mock_w3 = MagicMock()
        mock_w3.eth.get_transaction_count.return_value = 0
        mock_w3.eth.gas_price = 1_000_000_000
        mock_contract = MagicMock()
        mock_contract.functions.transfer.return_value.build_transaction.return_value = {"tx": True}
        mock_w3.eth.contract.return_value = mock_contract
        mock_w3.eth.account.sign_transaction.return_value.raw_transaction = b"\x00"
        mock_w3.eth.send_raw_transaction.return_value = b"\xab" * 32

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock_w3), \
             patch("providers.local_wallet.Web3") as MockWeb3:
            MockWeb3.to_checksum_address = lambda x: x  # pass through
            result = await send_usdc("agent-s1", recipient, 25.0, chain="base")

        assert result["success"] is True
        assert result["amount"] == 25.0
        assert result["chain"] == "base"

    @pytest.mark.asyncio
    async def test_send_usdc_rpc_failure(self, tmp_path):
        from providers.local_wallet import send_usdc

        addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18"
        data = {"address": addr, "private_key": "0x" + "cd" * 32, "encrypted": False}
        (tmp_path / "agent-s2.json").write_text(json.dumps(data))

        mock_w3 = MagicMock()
        mock_w3.eth.get_transaction_count.side_effect = Exception("nonce error")

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock_w3), \
             patch("providers.local_wallet.Web3") as MockWeb3:
            MockWeb3.to_checksum_address = lambda x: x
            result = await send_usdc("agent-s2", "0xRecipient", 5.0)

        assert result["success"] is False
        assert "nonce error" in result["error"]


class TestSendNative:
    """Test native token transfer (mocked)."""

    @pytest.mark.asyncio
    async def test_send_native_no_wallet(self, tmp_path):
        from providers.local_wallet import send_native

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = await send_native("ghost", "0xTo", 0.1)

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_send_native_success(self, tmp_path):
        from providers.local_wallet import send_native

        addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18"
        data = {"address": addr, "private_key": "0x" + "ef" * 32, "encrypted": False}
        (tmp_path / "agent-n1.json").write_text(json.dumps(data))

        mock_w3 = MagicMock()
        mock_w3.eth.get_transaction_count.return_value = 5
        mock_w3.eth.gas_price = 5_000_000_000
        mock_w3.to_wei.return_value = 100_000_000_000_000_000
        mock_w3.eth.chain_id = 56
        mock_w3.eth.account.sign_transaction.return_value.raw_transaction = b"\x01"
        mock_w3.eth.send_raw_transaction.return_value = b"\xcd" * 32

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock_w3), \
             patch("providers.local_wallet.Web3") as MockWeb3:
            MockWeb3.to_checksum_address = lambda x: x
            result = await send_native("agent-n1", "0xRecipient", 0.1, chain="bnb")

        assert result["success"] is True
        assert result["native_token"] == "BNB"
        assert result["chain"] == "bnb"


class TestSendEthAlias:
    """Test backward compat alias."""

    @pytest.mark.asyncio
    async def test_send_eth_calls_send_native(self, tmp_path):
        from providers.local_wallet import send_eth

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = await send_eth("ghost", "0xTo", 0.5)

        assert result["success"] is False  # no wallet, but function works


class TestAllChainBalances:
    """Test multi-chain balance aggregation."""

    def test_get_all_chain_balances_no_wallet(self, tmp_path):
        from providers.local_wallet import get_all_chain_balances

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            results = get_all_chain_balances("ghost")

        assert results == []

    def test_get_all_chain_balances_returns_all_chains(self, tmp_path):
        from providers.local_wallet import get_all_chain_balances, SUPPORTED_EVM_CHAINS

        addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18"
        data = {"address": addr, "private_key": "pk"}
        (tmp_path / "agent-m1.json").write_text(json.dumps(data))

        mock_w3 = MagicMock()
        mock_w3.eth.get_balance.return_value = 0
        mock_w3.from_wei.return_value = 0.0
        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call.return_value = 0
        mock_w3.eth.contract.return_value = mock_contract

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet._get_w3", return_value=mock_w3):
            results = get_all_chain_balances("agent-m1")

        assert len(results) == len(SUPPORTED_EVM_CHAINS)
        chains_returned = {r["chain"] for r in results}
        assert chains_returned == set(SUPPORTED_EVM_CHAINS)
