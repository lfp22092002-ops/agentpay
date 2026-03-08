"""
Tests for providers/solana_wallet.py — Solana wallet lifecycle.

Mocks solders/solana imports so tests run without native Solana deps.
Covers wallet creation, address retrieval, balance check, and transfers.
"""
import base64
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

os.environ.setdefault("API_SECRET", "test-secret-key-for-tests")


# ═══════════════════════════════════════
# MOCK SOLANA PRIMITIVES
# ═══════════════════════════════════════

class MockPubkey:
    def __init__(self, addr="SoL1111111111111111111111111111111111111111"):
        self._addr = addr

    def __str__(self):
        return self._addr

    def __bytes__(self):
        return self._addr.encode()[:32].ljust(32, b'\x00')

    @classmethod
    def from_string(cls, s):
        return cls(s)

    @classmethod
    def find_program_address(cls, seeds, program_id):
        return cls("ATA_" + str(seeds[0][:8])), 255


class MockKeypair:
    def __init__(self):
        self._secret = os.urandom(64)
        self._pubkey = MockPubkey("AgentSolWallet" + "1" * 30)

    def pubkey(self):
        return self._pubkey

    def __bytes__(self):
        return self._secret

    @classmethod
    def from_bytes(cls, data):
        kp = cls()
        kp._secret = data
        return kp


class TestSolanaWalletCreation:
    """Test wallet creation and address retrieval."""

    def test_create_new_wallet(self, tmp_path):
        """Create a new Solana wallet."""
        from providers.solana_wallet import create_solana_wallet, _wallet_file

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.solana_wallet.encrypt", side_effect=lambda x: f"enc:{x}"), \
             patch.dict("sys.modules", {"solders.keypair": MagicMock()}):
            # Mock Keypair()
            import sys
            mock_solders = MagicMock()
            mock_kp_instance = MockKeypair()
            mock_solders.Keypair.return_value = mock_kp_instance
            sys.modules["solders.keypair"] = mock_solders

            result = create_solana_wallet("agent-sol-001")

        assert result["chain"] == "solana"
        assert "address" in result
        assert "network" in result
        # Wallet file saved
        wallet_file = tmp_path / "agent-sol-001_solana.json"
        assert wallet_file.exists()
        data = json.loads(wallet_file.read_text())
        assert data["encrypted"] is True
        assert data["chain"] == "solana"

    def test_create_wallet_already_exists(self, tmp_path):
        """If wallet exists, return existing address."""
        from providers.solana_wallet import create_solana_wallet

        existing = {
            "agent_id": "agent-sol-002",
            "address": "ExistingSolAddress123",
            "network": "solana-devnet",
            "chain": "solana",
            "private_key": "enc:secret",
            "encrypted": True,
        }
        wallet_file = tmp_path / "agent-sol-002_solana.json"
        wallet_file.write_text(json.dumps(existing))

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path):
            result = create_solana_wallet("agent-sol-002")

        assert result["address"] == "ExistingSolAddress123"
        assert result["already_existed"] is True


class TestSolanaWalletAddress:
    """Test address retrieval."""

    def test_get_address_exists(self, tmp_path):
        """Get address for existing wallet."""
        from providers.solana_wallet import get_solana_wallet_address

        wallet_data = {"address": "SolAddr999", "chain": "solana"}
        (tmp_path / "agent-sol-003_solana.json").write_text(json.dumps(wallet_data))

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path):
            addr = get_solana_wallet_address("agent-sol-003")

        assert addr == "SolAddr999"

    def test_get_address_not_found(self, tmp_path):
        """No wallet returns None."""
        from providers.solana_wallet import get_solana_wallet_address

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path):
            addr = get_solana_wallet_address("nonexistent")

        assert addr is None


class TestSolanaBalance:
    """Test balance retrieval."""

    def test_balance_no_wallet(self, tmp_path):
        """Balance check on nonexistent wallet returns error."""
        from providers.solana_wallet import get_solana_balance

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path):
            result = get_solana_balance("ghost-agent")

        assert "error" in result

    def test_balance_success(self, tmp_path):
        """Successful balance check with mocked RPC."""
        from providers.solana_wallet import get_solana_balance

        wallet_data = {"address": "SolBalAddr", "chain": "solana"}
        (tmp_path / "agent-bal_solana.json").write_text(json.dumps(wallet_data))

        mock_client = MagicMock()
        # SOL balance: 1.5 SOL = 1_500_000_000 lamports
        mock_balance_resp = MagicMock()
        mock_balance_resp.value = 1_500_000_000
        mock_client.get_balance.return_value = mock_balance_resp

        # USDC balance: empty
        mock_token_resp = MagicMock()
        mock_token_resp.value = []
        mock_client.get_token_accounts_by_owner_json_parsed.return_value = mock_token_resp

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(),
                 "solders.pubkey": MagicMock(),
                 "solana.rpc.types": MagicMock(),
             }):
            import sys
            mock_solana_rpc = MagicMock()
            mock_solana_rpc.Client.return_value = mock_client
            sys.modules["solana.rpc.api"] = mock_solana_rpc

            mock_solders_pubkey = MagicMock()
            mock_solders_pubkey.Pubkey = MockPubkey
            sys.modules["solders.pubkey"] = mock_solders_pubkey

            result = get_solana_balance("agent-bal")

        assert result["chain"] == "solana"
        assert result["native_token"] == "SOL"
        assert result["balance_sol"] == "1.5"
        assert result["balance_usdc"] == "0.0"

    def test_balance_rpc_error(self, tmp_path):
        """RPC failure returns error dict."""
        from providers.solana_wallet import get_solana_balance

        wallet_data = {"address": "SolErrAddr", "chain": "solana"}
        (tmp_path / "agent-err_solana.json").write_text(json.dumps(wallet_data))

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(),
                 "solders.pubkey": MagicMock(),
             }):
            import sys
            mock_rpc = MagicMock()
            mock_rpc.Client.return_value.get_balance.side_effect = Exception("Connection refused")
            sys.modules["solana.rpc.api"] = mock_rpc

            mock_pubkey = MagicMock()
            mock_pubkey.Pubkey = MockPubkey
            sys.modules["solders.pubkey"] = mock_pubkey

            result = get_solana_balance("agent-err")

        assert "error" in result
        assert "Connection refused" in result["error"]


class TestSolanaTransfers:
    """Test SOL and USDC transfers."""

    @pytest.mark.asyncio
    async def test_send_sol_no_wallet(self, tmp_path):
        """Transfer from nonexistent wallet fails."""
        from providers.solana_wallet import send_sol

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.solana_wallet._load_keypair", return_value=None), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(),
                 "solana.transaction": MagicMock(),
                 "solders.pubkey": MagicMock(),
                 "solders.system_program": MagicMock(),
             }):
            result = await send_sol("ghost", "SolRecipient111", 1.0)

        assert result["success"] is False
        assert "No Solana wallet" in result["error"]

    @pytest.mark.asyncio
    async def test_send_usdc_no_wallet(self, tmp_path):
        """USDC transfer from nonexistent wallet fails."""
        from providers.solana_wallet import send_solana_usdc

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.solana_wallet._load_keypair", return_value=None), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(),
                 "solana.transaction": MagicMock(),
                 "solders.pubkey": MagicMock(),
                 "spl.token.instructions": MagicMock(),
                 "spl.token.constants": MagicMock(),
             }):
            result = await send_solana_usdc("ghost", "SolRecipient222", 10.0)

        assert result["success"] is False
        assert "No Solana wallet" in result["error"]

    @pytest.mark.asyncio
    async def test_send_sol_success(self, tmp_path):
        """Successful SOL transfer with mocked RPC."""
        from providers.solana_wallet import send_sol

        mock_kp = MockKeypair()
        mock_client = MagicMock()
        mock_blockhash_resp = MagicMock()
        mock_blockhash_resp.value.blockhash = "FakeBlockhash123"
        mock_client.get_latest_blockhash.return_value = mock_blockhash_resp
        mock_send_resp = MagicMock()
        mock_send_resp.value = "TxSigABC123"
        mock_client.send_transaction.return_value = mock_send_resp

        with patch("providers.solana_wallet._load_keypair", return_value=mock_kp), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(),
                 "solana.transaction": MagicMock(),
                 "solders.pubkey": MagicMock(),
                 "solders.system_program": MagicMock(),
             }):
            import sys
            mock_rpc = MagicMock()
            mock_rpc.Client.return_value = mock_client
            sys.modules["solana.rpc.api"] = mock_rpc

            mock_pubkey = MagicMock()
            mock_pubkey.Pubkey = MockPubkey
            sys.modules["solders.pubkey"] = mock_pubkey

            result = await send_sol("agent-send", "RecipientAddr", 0.5)

        assert result["success"] is True
        assert result["chain"] == "solana"
        assert result["amount"] == 0.5

    @pytest.mark.asyncio
    async def test_send_sol_rpc_failure(self, tmp_path):
        """RPC failure during transfer returns error."""
        from providers.solana_wallet import send_sol

        mock_kp = MockKeypair()

        with patch("providers.solana_wallet._load_keypair", return_value=mock_kp), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(),
                 "solana.transaction": MagicMock(),
                 "solders.pubkey": MagicMock(),
                 "solders.system_program": MagicMock(),
             }):
            import sys
            mock_rpc = MagicMock()
            mock_rpc.Client.return_value.get_latest_blockhash.side_effect = Exception("RPC timeout")
            sys.modules["solana.rpc.api"] = mock_rpc

            mock_pubkey = MagicMock()
            mock_pubkey.Pubkey = MockPubkey
            sys.modules["solders.pubkey"] = mock_pubkey

            result = await send_sol("agent-fail", "RecipientAddr", 1.0)

        assert result["success"] is False
        assert "RPC timeout" in result["error"]
