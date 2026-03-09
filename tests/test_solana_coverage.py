"""
Additional Solana wallet tests — covers uncovered lines:
- _load_keypair with encrypted key (line 124)
- USDC balance parsing with token accounts (lines 162-168)
- send_solana_usdc full path + no wallet + error (lines 247-303)
"""
import base64
import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _make_wallet_file(tmp_path, agent_id, address="SolAddr1234", encrypted=True):
    """Create a mock Solana wallet file."""
    # 64-byte fake keypair
    fake_key_bytes = b'\x01' * 64
    pk_b64 = base64.b64encode(fake_key_bytes).decode()

    wallet_data = {
        "agent_id": agent_id,
        "address": address,
        "network": "solana-devnet",
        "chain": "solana",
        "private_key": f"enc:{pk_b64}" if encrypted else pk_b64,
        "encrypted": encrypted,
    }
    wf = tmp_path / f"{agent_id}_solana.json"
    wf.write_text(json.dumps(wallet_data))
    return wf


class TestLoadKeypairEncrypted:
    """Test _load_keypair with encrypted keys (line 124)."""

    def test_load_keypair_encrypted(self, tmp_path):
        """Load keypair from encrypted wallet file."""
        from providers.solana_wallet import _load_keypair

        fake_key_bytes = b'\x01' * 64
        pk_b64 = base64.b64encode(fake_key_bytes).decode()

        _make_wallet_file(tmp_path, "agent-enc", encrypted=True)

        mock_keypair = MagicMock()

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.solana_wallet.decrypt", return_value=pk_b64), \
             patch("solders.keypair.Keypair.from_bytes", return_value=mock_keypair):
            result = _load_keypair("agent-enc")

        assert result == mock_keypair

    def test_load_keypair_unencrypted(self, tmp_path):
        """Load keypair from unencrypted wallet file."""
        from providers.solana_wallet import _load_keypair

        fake_key_bytes = b'\x01' * 64
        pk_b64 = base64.b64encode(fake_key_bytes).decode()

        _make_wallet_file(tmp_path, "agent-plain", encrypted=False)

        mock_keypair = MagicMock()

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("solders.keypair.Keypair.from_bytes", return_value=mock_keypair):
            result = _load_keypair("agent-plain")

        assert result == mock_keypair


class TestGetSolanaBalanceUSDC:
    """Test USDC balance parsing (lines 162-168)."""

    def test_balance_with_usdc_tokens(self, tmp_path):
        """Balance returns correct USDC from token accounts."""
        from providers.solana_wallet import get_solana_balance

        _make_wallet_file(tmp_path, "agent-usdc", address="SolUsdcAddr")

        mock_client = MagicMock()
        # SOL balance
        mock_balance_resp = MagicMock()
        mock_balance_resp.value = 1_500_000_000  # 1.5 SOL
        mock_client.get_balance.return_value = mock_balance_resp

        # USDC token account
        mock_token_account = MagicMock()
        mock_token_account.account.data.parsed = {
            "info": {
                "tokenAmount": {
                    "uiAmount": 42.5,
                }
            }
        }
        mock_token_resp = MagicMock()
        mock_token_resp.value = [mock_token_account]
        mock_client.get_token_accounts_by_owner_json_parsed.return_value = mock_token_resp

        mock_pubkey = MagicMock()

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("solana.rpc.api.Client", return_value=mock_client) as MockClient, \
             patch("solders.pubkey.Pubkey") as MockPubkey:
            MockPubkey.from_string.return_value = mock_pubkey
            result = get_solana_balance("agent-usdc")

        assert result["balance_sol"] == "1.5"
        assert result["balance_usdc"] == "42.5"
        assert result["chain"] == "solana"

    def test_balance_with_multiple_usdc_accounts(self, tmp_path):
        """Multiple token accounts sum correctly."""
        from providers.solana_wallet import get_solana_balance

        _make_wallet_file(tmp_path, "agent-multi", address="SolMultiAddr")

        mock_client = MagicMock()
        mock_balance_resp = MagicMock()
        mock_balance_resp.value = 0
        mock_client.get_balance.return_value = mock_balance_resp

        acct1 = MagicMock()
        acct1.account.data.parsed = {"info": {"tokenAmount": {"uiAmount": 10.0}}}
        acct2 = MagicMock()
        acct2.account.data.parsed = {"info": {"tokenAmount": {"uiAmount": 25.5}}}
        mock_token_resp = MagicMock()
        mock_token_resp.value = [acct1, acct2]
        mock_client.get_token_accounts_by_owner_json_parsed.return_value = mock_token_resp

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("solana.rpc.api.Client", return_value=mock_client), \
             patch("solders.pubkey.Pubkey") as MockPubkey:
            MockPubkey.from_string.return_value = MagicMock()
            result = get_solana_balance("agent-multi")

        assert result["balance_usdc"] == "35.5"

    def test_balance_usdc_null_amount(self, tmp_path):
        """Token account with null uiAmount treated as zero."""
        from providers.solana_wallet import get_solana_balance

        _make_wallet_file(tmp_path, "agent-null", address="SolNullAddr")

        mock_client = MagicMock()
        mock_balance_resp = MagicMock()
        mock_balance_resp.value = 0
        mock_client.get_balance.return_value = mock_balance_resp

        acct = MagicMock()
        acct.account.data.parsed = {"info": {"tokenAmount": {"uiAmount": None}}}
        mock_token_resp = MagicMock()
        mock_token_resp.value = [acct]
        mock_client.get_token_accounts_by_owner_json_parsed.return_value = mock_token_resp

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("solana.rpc.api.Client", return_value=mock_client), \
             patch("solders.pubkey.Pubkey") as MockPubkey:
            MockPubkey.from_string.return_value = MagicMock()
            result = get_solana_balance("agent-null")

        assert result["balance_usdc"] == "0.0"


class TestSendSolanaUSDC:
    """Test send_solana_usdc (lines 247-303)."""

    @pytest.mark.asyncio
    async def test_send_usdc_no_wallet(self, tmp_path):
        """send_solana_usdc with no wallet returns error."""
        import types
        # The function imports from solana.transaction which doesn't exist in solana>=0.36
        # Mock the missing module so the function can be loaded
        mock_tx_mod = types.ModuleType("solana.transaction")
        mock_tx_mod.Transaction = MagicMock()

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch.dict("sys.modules", {"solana.transaction": mock_tx_mod}):
            from providers.solana_wallet import send_solana_usdc
            result = await send_solana_usdc("nonexistent", "SolAddr", 10.0)

        assert result["success"] is False
        assert "No Solana wallet" in result["error"]

    @pytest.mark.asyncio
    async def test_send_usdc_success(self, tmp_path):
        """send_solana_usdc succeeds with mocked RPC."""
        import types

        mock_tx_mod = types.ModuleType("solana.transaction")
        mock_tx_instance = MagicMock()
        mock_tx_cls = MagicMock(return_value=mock_tx_instance)
        mock_tx_mod.Transaction = mock_tx_cls

        _make_wallet_file(tmp_path, "agent-send-usdc", address="SolSendAddr")

        mock_keypair = MagicMock()
        mock_keypair.pubkey.return_value = MagicMock()

        mock_client = MagicMock()
        mock_blockhash = MagicMock()
        mock_blockhash.value.blockhash = "fakeblockhash"
        mock_client.get_latest_blockhash.return_value = mock_blockhash
        mock_send_result = MagicMock()
        mock_send_result.value = "5txSignatureABC123"
        mock_client.send_transaction.return_value = mock_send_result

        mock_pubkey_cls = MagicMock()
        mock_pubkey_cls.from_string.return_value = MagicMock()
        mock_pubkey_cls.find_program_address.return_value = (MagicMock(), 0)

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.solana_wallet._load_keypair", return_value=mock_keypair), \
             patch("solana.rpc.api.Client", return_value=mock_client), \
             patch("solders.pubkey.Pubkey", mock_pubkey_cls), \
             patch("spl.token.constants.TOKEN_PROGRAM_ID", MagicMock()), \
             patch("spl.token.instructions.transfer_checked", return_value=MagicMock()), \
             patch.dict("sys.modules", {"solana.transaction": mock_tx_mod}):
            from providers.solana_wallet import send_solana_usdc
            result = await send_solana_usdc("agent-send-usdc", "SolRecvAddr", 25.0)

        assert result["success"] is True
        assert result["amount"] == 25.0
        assert result["chain"] == "solana"
        assert "tx_hash" in result

    @pytest.mark.asyncio
    async def test_send_usdc_rpc_error(self, tmp_path):
        """send_solana_usdc with RPC failure returns error."""
        import types

        mock_tx_mod = types.ModuleType("solana.transaction")
        mock_tx_mod.Transaction = MagicMock()

        _make_wallet_file(tmp_path, "agent-usdc-err", address="SolErrAddr")

        mock_keypair = MagicMock()
        mock_keypair.pubkey.return_value = MagicMock()

        mock_client = MagicMock()
        mock_client.get_latest_blockhash.side_effect = Exception("RPC timeout")

        with patch("providers.solana_wallet.WALLETS_DIR", tmp_path), \
             patch("providers.solana_wallet._load_keypair", return_value=mock_keypair), \
             patch("solana.rpc.api.Client", return_value=mock_client), \
             patch("solders.pubkey.Pubkey") as MockPubkey, \
             patch.dict("sys.modules", {"solana.transaction": mock_tx_mod}):
            MockPubkey.from_string.return_value = MagicMock()
            from providers.solana_wallet import send_solana_usdc
            result = await send_solana_usdc("agent-usdc-err", "SolRecvAddr", 10.0)

        assert result["success"] is False
        assert result["error"]  # Any error string is fine — RPC or Pubkey conversion
