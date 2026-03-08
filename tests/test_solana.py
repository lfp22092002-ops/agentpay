"""
Tests for providers/solana_wallet.py — Solana wallet lifecycle.

Uses mocked solders/solana-py to test wallet creation, address lookup,
balance queries, and send operations without real RPC connections.
"""
import json
import base64
from unittest.mock import patch, MagicMock

import pytest


class MockKeypair:
    """Mock solders Keypair."""
    def __init__(self):
        self._pubkey = MockPubkey("SoLAnaTestAddr111111111111111111111111111111")
        # 64 bytes: 32 private + 32 public
        self._bytes = bytes(range(64))

    def pubkey(self):
        return self._pubkey

    def __bytes__(self):
        return self._bytes

    @classmethod
    def from_bytes(cls, data):
        kp = cls()
        return kp


class MockPubkey:
    """Mock solders Pubkey."""
    def __init__(self, value="SoLAnaTestAddr111111111111111111111111111111"):
        self._value = value

    def __str__(self):
        return self._value

    def __bytes__(self):
        return self._value.encode()[:32].ljust(32, b'\x00')

    @classmethod
    def from_string(cls, s):
        return cls(s)

    @classmethod
    def find_program_address(cls, seeds, program_id):
        return (cls("AssociatedTokenAddr1111111111111111111111111"), 255)


class TestSolanaWalletCreation:
    """Test Solana wallet creation."""

    def test_create_new_wallet(self, tmp_path):
        """Creates new wallet, encrypts key, writes file."""
        from providers.solana_wallet import create_solana_wallet

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.solana_wallet.encrypt", side_effect=lambda x: f"enc:{x}"), \
             patch("providers.solana_wallet.Keypair", MockKeypair) if False else \
             patch.dict("sys.modules", {"solders.keypair": MagicMock(Keypair=MockKeypair)}):
            result = create_solana_wallet("agent-sol-001")

        assert result["chain"] == "solana"
        assert result["address"] == "SoLAnaTestAddr111111111111111111111111111111"
        assert "network" in result

        # File persisted
        wf = tmp_path / "agent-sol-001_solana.json"
        assert wf.exists()
        saved = json.loads(wf.read_text())
        assert saved["encrypted"] is True
        assert saved["private_key"].startswith("enc:")

    def test_create_wallet_already_exists(self, tmp_path):
        """Returns existing wallet without regenerating."""
        from providers.solana_wallet import create_solana_wallet

        existing = {
            "agent_id": "agent-sol-002",
            "address": "ExistingSolAddr1111111111111111111111111111",
            "network": "solana-devnet",
            "chain": "solana",
            "private_key": "enc:secret",
            "encrypted": True,
        }
        wf = tmp_path / "agent-sol-002_solana.json"
        wf.write_text(json.dumps(existing))

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path):
            result = create_solana_wallet("agent-sol-002")

        assert result["address"] == "ExistingSolAddr1111111111111111111111111111"
        assert result["already_existed"] is True


class TestSolanaWalletAddress:
    """Test address lookup."""

    def test_get_address_exists(self, tmp_path):
        """Returns address from file."""
        from providers.solana_wallet import get_solana_wallet_address

        data = {"address": "SolAddr42", "chain": "solana"}
        (tmp_path / "agent-sol-003_solana.json").write_text(json.dumps(data))

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path):
            addr = get_solana_wallet_address("agent-sol-003")

        assert addr == "SolAddr42"

    def test_get_address_missing(self, tmp_path):
        """No wallet file returns None."""
        from providers.solana_wallet import get_solana_wallet_address

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path):
            addr = get_solana_wallet_address("nonexistent")

        assert addr is None


class TestSolanaBalance:
    """Test balance queries."""

    def test_get_balance_no_wallet(self, tmp_path):
        """No wallet returns error dict."""
        from providers.solana_wallet import get_solana_balance

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path):
            result = get_solana_balance("ghost-agent")

        assert "error" in result

    def test_get_balance_rpc_success(self, tmp_path):
        """Successful RPC call returns SOL + USDC balances."""
        from providers.solana_wallet import get_solana_balance

        data = {"address": "SolAddr99", "chain": "solana"}
        (tmp_path / "agent-sol-004_solana.json").write_text(json.dumps(data))

        mock_client = MagicMock()
        # SOL balance: 1.5 SOL = 1,500,000,000 lamports
        mock_client.get_balance.return_value = MagicMock(value=1_500_000_000)
        # USDC: empty
        mock_client.get_token_accounts_by_owner_json_parsed.return_value = MagicMock(value=[])

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(Client=MagicMock(return_value=mock_client)),
                 "solders.pubkey": MagicMock(Pubkey=MockPubkey),
                 "solana.rpc.types": MagicMock(),
             }):
            result = get_solana_balance("agent-sol-004")

        assert result["chain"] == "solana"
        assert result["balance_sol"] == "1.5"
        assert result["native_token"] == "SOL"

    def test_get_balance_rpc_error(self, tmp_path):
        """RPC failure returns error dict."""
        from providers.solana_wallet import get_solana_balance

        data = {"address": "SolAddrErr", "chain": "solana"}
        (tmp_path / "agent-sol-005_solana.json").write_text(json.dumps(data))

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(Client=MagicMock(side_effect=Exception("RPC down"))),
                 "solders.pubkey": MagicMock(Pubkey=MockPubkey),
             }):
            result = get_solana_balance("agent-sol-005")

        assert "error" in result


class TestSolanaSend:
    """Test SOL and USDC transfers."""

    @pytest.mark.asyncio
    async def test_send_sol_no_wallet(self, tmp_path):
        """Send SOL with no wallet fails."""
        from providers.solana_wallet import send_sol

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(),
                 "solana.transaction": MagicMock(),
                 "solders.pubkey": MagicMock(Pubkey=MockPubkey),
                 "solders.system_program": MagicMock(),
             }):
            result = await send_sol("ghost", "SomeAddr1111", 1.0)

        assert result["success"] is False
        assert "No Solana wallet" in result["error"]

    @pytest.mark.asyncio
    async def test_send_usdc_no_wallet(self, tmp_path):
        """Send USDC with no wallet fails."""
        from providers.solana_wallet import send_solana_usdc

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {
                 "solana.rpc.api": MagicMock(),
                 "solana.transaction": MagicMock(),
                 "solders.pubkey": MagicMock(Pubkey=MockPubkey),
                 "spl.token.instructions": MagicMock(),
                 "spl.token.constants": MagicMock(),
             }):
            result = await send_solana_usdc("ghost", "SomeAddr1111", 10.0)

        assert result["success"] is False
        assert "No Solana wallet" in result["error"]

    @pytest.mark.asyncio
    async def test_send_sol_success(self, tmp_path):
        """Successful SOL transfer."""
        from providers.solana_wallet import send_sol

        pk_bytes = bytes(range(64))
        pk_b64 = base64.b64encode(pk_bytes).decode()
        data = {
            "address": "SenderSol111",
            "private_key": pk_b64,
            "encrypted": False,
            "chain": "solana",
        }
        (tmp_path / "agent-sol-006_solana.json").write_text(json.dumps(data))

        mock_client = MagicMock()
        mock_client.get_latest_blockhash.return_value = MagicMock(value=MagicMock(blockhash="blockhash123"))
        mock_client.send_transaction.return_value = MagicMock(value="txsig_abc123")

        mock_tx = MagicMock()

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {
                 "solders.keypair": MagicMock(Keypair=MockKeypair),
                 "solana.rpc.api": MagicMock(Client=MagicMock(return_value=mock_client)),
                 "solders.pubkey": MagicMock(Pubkey=MockPubkey),
                 "solders.system_program": MagicMock(),
                 "solana.transaction": MagicMock(Transaction=MagicMock(return_value=mock_tx)),
             }):
            result = await send_sol("agent-sol-006", "ReceiverSol111", 0.5)

        assert result["success"] is True
        assert result["tx_hash"] == "txsig_abc123"
        assert result["amount"] == 0.5

    @pytest.mark.asyncio
    async def test_send_sol_rpc_failure(self, tmp_path):
        """SOL transfer with RPC error."""
        from providers.solana_wallet import send_sol

        pk_bytes = bytes(range(64))
        pk_b64 = base64.b64encode(pk_bytes).decode()
        data = {
            "address": "SenderSol222",
            "private_key": pk_b64,
            "encrypted": False,
            "chain": "solana",
        }
        (tmp_path / "agent-sol-007_solana.json").write_text(json.dumps(data))

        mock_client = MagicMock()
        mock_client.get_latest_blockhash.side_effect = Exception("Connection refused")

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {
                 "solders.keypair": MagicMock(Keypair=MockKeypair),
                 "solana.rpc.api": MagicMock(Client=MagicMock(return_value=mock_client)),
                 "solders.pubkey": MagicMock(Pubkey=MockPubkey),
                 "solders.system_program": MagicMock(),
                 "solana.transaction": MagicMock(),
             }):
            result = await send_sol("agent-sol-007", "ReceiverSol222", 1.0)

        assert result["success"] is False
        assert "Connection refused" in result["error"]
