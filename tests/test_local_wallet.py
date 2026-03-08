"""
Tests for providers/local_wallet.py — EVM multi-chain wallet operations.

Tests wallet creation, address derivation, balance checks, and USDC transfers
using mocked Web3/eth_account. No real RPC calls or private keys.
"""
import json
from unittest.mock import patch, MagicMock



class MockAccount:
    """Mock eth_account.Account.create() result."""
    def __init__(self, address="0x1234567890abcdef1234567890abcdef12345678",
                 key=b'\x01' * 32):
        self.address = address
        self.key = key


class TestWalletCreation:
    """Test EVM wallet creation and key storage."""

    def test_create_wallet_new(self, tmp_path):
        """Create a new wallet — stores encrypted key, returns address."""
        from providers.local_wallet import create_agent_wallet

        mock_acct = MockAccount()
        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet.Account") as mock_account_cls, \
             patch("providers.local_wallet.encrypt", side_effect=lambda x: f"enc:{x}"):
            mock_account_cls.create.return_value = mock_acct
            result = create_agent_wallet("agent-w1")

        assert result["address"] == mock_acct.address
        assert (tmp_path / "agent-w1.json").exists()
        saved = json.loads((tmp_path / "agent-w1.json").read_text())
        assert saved["address"] == mock_acct.address
        assert saved["encrypted"] is True

    def test_create_wallet_already_exists(self, tmp_path):
        """If wallet exists, return existing address."""
        from providers.local_wallet import create_agent_wallet

        existing = {
            "address": "0xAAAABBBBCCCCDDDD",
            "private_key": "enc:pk_existing",
            "encrypted": True,
        }
        (tmp_path / "agent-w2.json").write_text(json.dumps(existing))

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = create_agent_wallet("agent-w2")

        assert result["address"] == "0xAAAABBBBCCCCDDDD"

    def test_same_address_all_chains(self, tmp_path):
        """EVM wallets share one address across all chains."""
        from providers.local_wallet import create_agent_wallet

        mock_acct = MockAccount(address="0xSharedAddr")
        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet.Account") as mock_account_cls, \
             patch("providers.local_wallet.encrypt", side_effect=lambda x: f"enc:{x}"):
            mock_account_cls.create.return_value = mock_acct
            base_result = create_agent_wallet("agent-w3", chain="base")
            poly_result = create_agent_wallet("agent-w3", chain="polygon")
            bnb_result = create_agent_wallet("agent-w3", chain="bnb")

        assert base_result["address"] == poly_result["address"] == bnb_result["address"]


class TestWalletAddress:
    """Test address retrieval."""

    def test_get_wallet_address(self, tmp_path):
        """Get address from stored wallet."""
        from providers.local_wallet import get_wallet_address

        wallet_data = {"address": "0xMyAddr123", "private_key": "enc:pk", "encrypted": True}
        (tmp_path / "agent-w4.json").write_text(json.dumps(wallet_data))

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            addr = get_wallet_address("agent-w4")

        assert addr == "0xMyAddr123"

    def test_get_wallet_address_no_wallet(self, tmp_path):
        """No wallet file returns None."""
        from providers.local_wallet import get_wallet_address

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            addr = get_wallet_address("nonexistent")

        assert addr is None


class TestWalletBalance:
    """Test balance checking across chains."""

    def test_get_wallet_balance(self, tmp_path):
        """Get balance returns native + USDC balances."""
        from providers.local_wallet import get_wallet_balance

        wallet_data = {"address": "0xBalanceTest", "private_key": "enc:pk", "encrypted": True}
        (tmp_path / "agent-w5.json").write_text(json.dumps(wallet_data))

        mock_web3 = MagicMock()
        mock_web3.eth.get_balance.return_value = 1_000_000_000_000_000_000  # 1 ETH in wei
        mock_contract = MagicMock()
        mock_contract.functions.balanceOf.return_value.call.return_value = 50_000_000  # 50 USDC (6 decimals)
        mock_web3.eth.contract.return_value = mock_contract
        mock_web3.from_wei.return_value = 1.0

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.local_wallet.Web3", return_value=mock_web3):
            result = get_wallet_balance("agent-w5", chain="base")

        assert result["address"] == "0xBalanceTest"
        assert "balance_usdc" in result
        assert "balance_native" in result

    def test_get_wallet_balance_no_wallet(self, tmp_path):
        """No wallet returns error dict."""
        from providers.local_wallet import get_wallet_balance

        with patch("providers.local_wallet.WALLETS_DIR", tmp_path):
            result = get_wallet_balance("nonexistent", chain="base")

        assert "error" in result


class TestChainConfig:
    """Test chain configuration."""

    def test_chain_configs_exist(self):
        """All expected chains are configured."""
        from providers.local_wallet import CHAIN_CONFIGS

        assert "base" in CHAIN_CONFIGS
        assert "polygon" in CHAIN_CONFIGS
        assert "bnb" in CHAIN_CONFIGS

    def test_chain_config_structure(self):
        """Each chain config has required fields."""
        from providers.local_wallet import CHAIN_CONFIGS

        required_fields = ["name", "rpc_mainnet", "rpc_testnet", "usdc_mainnet", "native_token"]
        for chain, config in CHAIN_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"{chain} missing {field}"

    def test_chain_native_tokens(self):
        """Correct native tokens per chain."""
        from providers.local_wallet import CHAIN_CONFIGS

        assert CHAIN_CONFIGS["base"]["native_token"] == "ETH"
        assert CHAIN_CONFIGS["polygon"]["native_token"] == "POL"
        assert CHAIN_CONFIGS["bnb"]["native_token"] == "BNB"
