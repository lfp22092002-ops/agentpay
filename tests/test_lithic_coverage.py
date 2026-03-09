"""
Additional Lithic tests — covers uncovered lines:
- _get_client (26-33): missing API key, env selection
- get_card_transactions (135-156): list transactions, API failure
- update_card_state invalid state (172)
- update_card_state API error (183-185)
- update_spend_limit API error (209-211)
"""
import json
import os
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("LITHIC_API_KEY", "test-lithic-key-sandbox")


class MockTransaction:
    def __init__(self, amount=500, descriptor="COFFEE SHOP", status="SETTLED", created=None):
        self.amount = amount
        self.merchant = MagicMock(descriptor=descriptor)
        self.status = status
        if created is None:
            from datetime import datetime, timezone
            self.created = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        else:
            self.created = created


class TestGetClient:
    """Tests for _get_client (lines 26-33)."""

    def test_get_client_no_api_key(self):
        """Missing LITHIC_API_KEY raises ValueError."""
        from providers.lithic_card import _get_client

        with patch.dict(os.environ, {"LITHIC_API_KEY": ""}):
            with pytest.raises(ValueError, match="LITHIC_API_KEY"):
                _get_client()

    def test_get_client_sandbox(self):
        """Development env creates sandbox client."""
        with patch("providers.lithic_card.ENVIRONMENT", "development"), \
             patch("providers.lithic_card.Lithic") as MockLithic, \
             patch.dict(os.environ, {"LITHIC_API_KEY": "test-key"}):
            from providers.lithic_card import _get_client
            _get_client()
            MockLithic.assert_called_once_with(api_key="test-key", environment="sandbox")

    def test_get_client_production(self):
        """Production env creates production client."""
        with patch("providers.lithic_card.ENVIRONMENT", "production"), \
             patch("providers.lithic_card.Lithic") as MockLithic, \
             patch.dict(os.environ, {"LITHIC_API_KEY": "prod-key"}):
            from providers.lithic_card import _get_client
            _get_client()
            MockLithic.assert_called_once_with(api_key="prod-key", environment="production")


class TestGetCardTransactions:
    """Tests for get_card_transactions (lines 135-156)."""

    def test_get_card_transactions_success(self, tmp_path):
        """Successfully list card transactions."""
        from providers.lithic_card import get_card_transactions

        card_data = {
            "card_token": "tok_txn",
            "last4": "4444",
            "state": "OPEN",
        }
        (tmp_path / "agent-txn.json").write_text(json.dumps(card_data))

        txns = [
            MockTransaction(amount=500, descriptor="COFFEE SHOP", status="SETTLED"),
            MockTransaction(amount=1200, descriptor="GROCERY STORE", status="SETTLED"),
        ]

        mock_client = MagicMock()
        mock_client.transactions.list.return_value = txns

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_client):
            result = get_card_transactions("agent-txn", limit=5)

        assert len(result) == 2
        assert result[0]["amount_cents"] == 500
        assert result[0]["merchant"] == "COFFEE SHOP"
        assert result[0]["status"] == "SETTLED"
        assert "created" in result[0]
        assert result[1]["amount_cents"] == 1200

    def test_get_card_transactions_no_card(self, tmp_path):
        """No card file returns empty list."""
        from providers.lithic_card import get_card_transactions

        with patch("providers.lithic_card.CARDS_DIR", tmp_path):
            result = get_card_transactions("nonexistent")

        assert result == []

    def test_get_card_transactions_api_failure(self, tmp_path):
        """API error returns empty list."""
        from providers.lithic_card import get_card_transactions

        card_data = {
            "card_token": "tok_fail",
            "last4": "5555",
            "state": "OPEN",
        }
        (tmp_path / "agent-fail.json").write_text(json.dumps(card_data))

        mock_client = MagicMock()
        mock_client.transactions.list.side_effect = Exception("API unreachable")

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_client):
            result = get_card_transactions("agent-fail")

        assert result == []

    def test_get_card_transactions_respects_limit(self, tmp_path):
        """Transaction list respects limit parameter."""
        from providers.lithic_card import get_card_transactions

        card_data = {
            "card_token": "tok_lim",
            "last4": "6666",
            "state": "OPEN",
        }
        (tmp_path / "agent-lim.json").write_text(json.dumps(card_data))

        # Return more txns than limit
        txns = [MockTransaction(amount=i * 100) for i in range(10)]
        mock_client = MagicMock()
        mock_client.transactions.list.return_value = txns

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_client):
            result = get_card_transactions("agent-lim", limit=3)

        assert len(result) == 3


class TestUpdateCardStateEdge:
    """Tests for invalid state and API errors (lines 172, 183-185)."""

    def test_update_card_state_invalid_state(self, tmp_path):
        """Invalid state value rejected."""
        from providers.lithic_card import update_card_state

        card_data = {
            "card_token": "tok_inv",
            "last4": "7777",
            "state": "OPEN",
        }
        (tmp_path / "agent-inv.json").write_text(json.dumps(card_data))

        with patch("providers.lithic_card.CARDS_DIR", tmp_path):
            result = update_card_state("agent-inv", "DESTROYED")

        assert result["success"] is False
        assert "State must be" in result["error"]

    def test_update_card_state_api_error(self, tmp_path):
        """API failure when updating state returns error."""
        from providers.lithic_card import update_card_state

        card_data = {
            "card_token": "tok_err",
            "last4": "8888",
            "state": "OPEN",
        }
        (tmp_path / "agent-err.json").write_text(json.dumps(card_data))

        mock_client = MagicMock()
        mock_client.cards.update.side_effect = Exception("Network error")

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_client):
            result = update_card_state("agent-err", "PAUSED")

        assert result["success"] is False
        assert "Network error" in result["error"]


class TestUpdateSpendLimitEdge:
    """Tests for update_spend_limit API error (lines 209-211)."""

    def test_update_spend_limit_api_error(self, tmp_path):
        """API failure when updating spend limit returns error."""
        from providers.lithic_card import update_spend_limit

        card_data = {
            "card_token": "tok_sperr",
            "last4": "9999",
            "state": "OPEN",
            "spend_limit_cents": 5000,
        }
        (tmp_path / "agent-sperr.json").write_text(json.dumps(card_data))

        mock_client = MagicMock()
        mock_client.cards.update.side_effect = Exception("Rate limited")

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_client):
            result = update_spend_limit("agent-sperr", 20000)

        assert result["success"] is False
        assert "Rate limited" in result["error"]
