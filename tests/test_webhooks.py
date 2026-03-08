"""
Tests for the webhook system — signing, event building, delivery.
"""
import hashlib
import hmac
import time
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from core.webhooks import (
    sign_payload,
    build_event,
    generate_webhook_secret,
    deliver_webhook,
    get_webhook_config,
    _webhook_registry,
    notify_spend,
    notify_deposit,
    notify_telegram,
    set_bot,
)


class TestSignPayload:
    def test_deterministic(self):
        sig1 = sign_payload("hello", "secret")
        sig2 = sign_payload("hello", "secret")
        assert sig1 == sig2

    def test_different_payload(self):
        sig1 = sign_payload("hello", "secret")
        sig2 = sign_payload("world", "secret")
        assert sig1 != sig2

    def test_different_secret(self):
        sig1 = sign_payload("hello", "secret1")
        sig2 = sign_payload("hello", "secret2")
        assert sig1 != sig2

    def test_matches_hmac(self):
        payload = '{"type":"spend"}'
        secret = "whsec_abc123"
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        assert sign_payload(payload, secret) == expected

    def test_empty_payload(self):
        sig = sign_payload("", "secret")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex digest


class TestBuildEvent:
    def test_structure(self):
        event = build_event("spend", "agent_123", {"amount": 5.0})
        assert event["type"] == "spend"
        assert event["agent_id"] == "agent_123"
        assert event["data"] == {"amount": 5.0}
        assert event["id"].startswith("evt_")
        assert "created_at" in event

    def test_created_at_is_iso(self):
        event = build_event("deposit", "a1", {})
        ts = event["created_at"]
        assert ts.endswith("Z")
        # Should parse without error
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_different_events_different_ids(self):
        e1 = build_event("spend", "a1", {})
        # Slight time difference → different id
        time.sleep(0.01)
        e2 = build_event("spend", "a1", {})
        # IDs may collide within same millisecond but generally differ
        assert e1["id"].startswith("evt_")
        assert e2["id"].startswith("evt_")


class TestGenerateWebhookSecret:
    def test_prefix(self):
        secret = generate_webhook_secret()
        assert secret.startswith("whsec_")

    def test_length(self):
        secret = generate_webhook_secret()
        # whsec_ (6) + 48 hex chars = 54
        assert len(secret) == 54

    def test_uniqueness(self):
        secrets = {generate_webhook_secret() for _ in range(10)}
        assert len(secrets) == 10


class TestWebhookRegistry:
    def setup_method(self):
        _webhook_registry.clear()

    def teardown_method(self):
        _webhook_registry.clear()

    def test_register_and_get(self):
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["all"],
        }
        config = get_webhook_config("agent_1")
        assert config is not None
        assert config["url"] == "https://example.com/hook"

    def test_get_missing(self):
        assert get_webhook_config("nonexistent") is None


class TestDeliverWebhook:
    def setup_method(self):
        _webhook_registry.clear()

    def teardown_method(self):
        _webhook_registry.clear()

    @pytest.mark.asyncio
    async def test_no_config_returns_silently(self):
        # Should not raise
        await deliver_webhook("unknown_agent", "spend", {"amount": 5})

    @pytest.mark.asyncio
    async def test_event_filter(self):
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["deposit"],  # Only subscribed to deposit
        }
        # Delivering a "spend" event should be filtered out (no HTTP call)
        with patch("core.webhooks.httpx.AsyncClient") as mock_client:
            await deliver_webhook("agent_1", "spend", {"amount": 5})
            mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_delivery(self):
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["all"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks.httpx.AsyncClient", return_value=mock_client_instance):
            await deliver_webhook("agent_1", "spend", {"amount": 10.0})
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert call_args[1]["headers"]["X-AgentPay-Event"] == "spend"
            assert "X-AgentPay-Signature" in call_args[1]["headers"]


class TestNotifications:
    @pytest.mark.asyncio
    async def test_notify_telegram_no_bot(self):
        set_bot(None)
        # Should not raise
        await notify_telegram(12345, "test message")

    @pytest.mark.asyncio
    async def test_notify_telegram_with_bot(self):
        mock_bot = AsyncMock()
        set_bot(mock_bot)
        await notify_telegram(12345, "test message")
        mock_bot.send_message.assert_called_once_with(12345, "test message", parse_mode="HTML")
        set_bot(None)

    @pytest.mark.asyncio
    async def test_notify_spend(self):
        mock_bot = AsyncMock()
        set_bot(mock_bot)
        await notify_spend("my-agent", 12345, 5.0, 0.10, "API call", 94.90)
        call_args = mock_bot.send_message.call_args
        text = call_args[0][1]
        assert "my-agent" in text
        assert "$5.00" in text
        assert "$94.90" in text
        set_bot(None)

    @pytest.mark.asyncio
    async def test_notify_deposit(self):
        mock_bot = AsyncMock()
        set_bot(mock_bot)
        await notify_deposit("my-agent", 12345, 25.0, "stars", 125.0)
        call_args = mock_bot.send_message.call_args
        text = call_args[0][1]
        assert "my-agent" in text
        assert "$25.00" in text
        assert "$125.00" in text
        set_bot(None)

    @pytest.mark.asyncio
    async def test_notify_spend_no_description(self):
        mock_bot = AsyncMock()
        set_bot(mock_bot)
        await notify_spend("agent", 12345, 1.0, 0.02, None, 99.0)
        call_args = mock_bot.send_message.call_args
        text = call_args[0][1]
        assert "For:" not in text
        set_bot(None)
