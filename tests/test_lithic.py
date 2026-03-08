"""
Tests for providers/lithic_card.py — Virtual card lifecycle.

Uses mocked Lithic client to test card creation, state management,
detail retrieval, and spend limit updates without a real API key.
"""
import json
import os
from unittest.mock import patch, MagicMock


# We need to set env before import
os.environ.setdefault("LITHIC_API_KEY", "test-lithic-key-sandbox")


class MockCard:
    """Mock Lithic card object."""
    def __init__(self, token="card_tok_abc123", last_four="4242",
                 exp_month=12, exp_year=2027, state="OPEN",
                 spend_limit=5000, pan="4111111111114242", cvv="123"):
        self.token = token
        self.last_four = last_four
        self.exp_month = exp_month
        self.exp_year = exp_year
        self.state = state
        self.spend_limit = spend_limit
        self.pan = pan
        self.cvv = cvv


def mock_lithic_client():
    """Build a mock Lithic client."""
    client = MagicMock()
    card = MockCard()
    client.cards.create.return_value = card
    client.cards.update.return_value = card
    client.cards.retrieve.return_value = card
    return client


class TestCardCreation:
    """Test virtual card creation."""

    def test_create_card_new(self, tmp_path):
        """Create a new card for an agent."""
        from providers.lithic_card import create_virtual_card

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_lithic_client()), \
             patch("providers.lithic_card.encrypt", side_effect=lambda x: f"enc:{x}"):
            result = create_virtual_card("agent-001", spend_limit=10000, memo="Test card")

        assert result["last4"] == "4242"
        assert result["exp_month"] == 12
        assert result["exp_year"] == 2027
        assert result["state"] == "OPEN"
        # Card file saved
        assert (tmp_path / "agent-001.json").exists()

    def test_create_card_already_exists(self, tmp_path):
        """If card exists, return existing details."""
        from providers.lithic_card import create_virtual_card

        # Pre-create card file
        existing = {
            "card_token": "existing_tok",
            "last4": "9999",
            "exp_month": 6,
            "exp_year": 2028,
            "state": "OPEN",
            "spend_limit_cents": 5000,
        }
        card_file = tmp_path / "agent-002.json"
        card_file.write_text(json.dumps(existing))

        with patch("providers.lithic_card.CARDS_DIR", tmp_path):
            result = create_virtual_card("agent-002")

        assert result["last4"] == "9999"
        assert result["state"] == "OPEN"


class TestCardDetails:
    """Test card detail retrieval."""

    def test_get_card_details(self, tmp_path):
        """Get card details (without full PAN)."""
        from providers.lithic_card import get_card_details

        card_data = {
            "card_token": "tok_xyz",
            "last4": "1234",
            "exp_month": 3,
            "exp_year": 2029,
            "state": "OPEN",
            "spend_limit_cents": 7500,
        }
        (tmp_path / "agent-003.json").write_text(json.dumps(card_data))

        with patch("providers.lithic_card.CARDS_DIR", tmp_path):
            result = get_card_details("agent-003")

        assert result["last4"] == "1234"
        assert result["state"] == "OPEN"
        assert "pan" not in result  # No full PAN in basic view

    def test_get_card_details_with_full(self, tmp_path):
        """Get card details with full PAN and CVV."""
        from providers.lithic_card import get_card_details

        card_data = {
            "card_token": "tok_full",
            "last4": "5678",
            "exp_month": 9,
            "exp_year": 2030,
            "state": "OPEN",
            "spend_limit_cents": 3000,
            "pan": "enc:4111111111115678",
            "cvv": "enc:456",
            "encrypted": True,
        }
        (tmp_path / "agent-004.json").write_text(json.dumps(card_data))

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card.decrypt", side_effect=lambda x: x.replace("enc:", "")):
            result = get_card_details("agent-004", show_full=True)

        assert result["pan"] == "4111111111115678"
        assert result["cvv"] == "456"

    def test_get_card_details_no_card(self, tmp_path):
        """No card file returns None."""
        from providers.lithic_card import get_card_details

        with patch("providers.lithic_card.CARDS_DIR", tmp_path):
            result = get_card_details("nonexistent")

        assert result is None


class TestCardStateManagement:
    """Test card pause/resume/close."""

    def test_update_card_state_pause(self, tmp_path):
        """Pause an active card."""
        from providers.lithic_card import update_card_state

        card_data = {
            "card_token": "tok_pause",
            "last4": "1111",
            "state": "OPEN",
        }
        (tmp_path / "agent-005.json").write_text(json.dumps(card_data))

        mock_client = mock_lithic_client()
        paused_card = MockCard(state="PAUSED")
        mock_client.cards.update.return_value = paused_card

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_client):
            result = update_card_state("agent-005", "PAUSED")

        assert result["success"] is True
        # Verify state saved to file
        saved = json.loads((tmp_path / "agent-005.json").read_text())
        assert saved["state"] == "PAUSED"

    def test_update_card_state_no_card(self, tmp_path):
        """Update state on nonexistent card fails gracefully."""
        from providers.lithic_card import update_card_state

        with patch("providers.lithic_card.CARDS_DIR", tmp_path):
            result = update_card_state("ghost-agent", "PAUSED")

        assert result["success"] is False

    def test_update_card_state_close(self, tmp_path):
        """Close a card permanently."""
        from providers.lithic_card import update_card_state

        card_data = {
            "card_token": "tok_close",
            "last4": "2222",
            "state": "OPEN",
        }
        (tmp_path / "agent-006.json").write_text(json.dumps(card_data))

        mock_client = mock_lithic_client()
        closed_card = MockCard(state="CLOSED")
        mock_client.cards.update.return_value = closed_card

        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_client):
            result = update_card_state("agent-006", "CLOSED")

        assert result["success"] is True
        saved = json.loads((tmp_path / "agent-006.json").read_text())
        assert saved["state"] == "CLOSED"


class TestSpendLimitUpdate:
    """Test card spend limit updates."""

    def test_update_spend_limit(self, tmp_path):
        """Update monthly spend limit."""
        from providers.lithic_card import update_spend_limit

        card_data = {
            "card_token": "tok_limit",
            "last4": "3333",
            "state": "OPEN",
            "spend_limit_cents": 5000,
        }
        (tmp_path / "agent-007.json").write_text(json.dumps(card_data))

        mock_client = mock_lithic_client()
        with patch("providers.lithic_card.CARDS_DIR", tmp_path), \
             patch("providers.lithic_card._get_client", return_value=mock_client):
            result = update_spend_limit("agent-007", 15000)

        assert result["success"] is True
        saved = json.loads((tmp_path / "agent-007.json").read_text())
        assert saved["spend_limit"] == 15000

    def test_update_spend_limit_no_card(self, tmp_path):
        """Update limit on nonexistent card fails."""
        from providers.lithic_card import update_spend_limit

        with patch("providers.lithic_card.CARDS_DIR", tmp_path):
            result = update_spend_limit("ghost-agent", 10000)

        assert result["success"] is False
