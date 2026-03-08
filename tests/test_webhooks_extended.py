"""
Tests for webhook delivery retries, DB load/register/unregister, and approval notifications.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.webhooks import (
    deliver_webhook,
    register_webhook,
    unregister_webhook,
    get_webhook_config,
    _webhook_registry,
    notify_approval_request,
    set_bot,
)


class TestDeliverWebhookRetries:
    """Test webhook delivery retry logic."""

    def setup_method(self):
        _webhook_registry.clear()

    def teardown_method(self):
        _webhook_registry.clear()

    @pytest.mark.asyncio
    async def test_retries_on_server_error(self):
        """Should retry up to 3 times on 5xx status."""
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["all"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks.httpx.AsyncClient", return_value=mock_client), \
             patch("core.webhooks.asyncio.sleep", new_callable=AsyncMock):
            await deliver_webhook("agent_1", "spend", {"amount": 5.0})
            assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        """Should retry on connection exceptions."""
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["all"],
        }

        mock_client = AsyncMock()
        mock_client.post.side_effect = ConnectionError("refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks.httpx.AsyncClient", return_value=mock_client), \
             patch("core.webhooks.asyncio.sleep", new_callable=AsyncMock):
            await deliver_webhook("agent_1", "spend", {"amount": 5.0})
            assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_stops_on_success(self):
        """Should stop retrying after first success."""
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["all"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks.httpx.AsyncClient", return_value=mock_client):
            await deliver_webhook("agent_1", "spend", {"amount": 5.0})
            assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_signature_in_headers(self):
        """Delivery should include HMAC signature."""
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test123",
            "events": ["all"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks.httpx.AsyncClient", return_value=mock_client):
            await deliver_webhook("agent_1", "deposit", {"amount": 50.0})
            call_kwargs = mock_client.post.call_args[1]
            headers = call_kwargs["headers"]
            assert "X-AgentPay-Signature" in headers
            assert headers["X-AgentPay-Event"] == "deposit"
            # Verify payload is valid JSON
            payload = call_kwargs["content"]
            parsed = json.loads(payload)
            assert parsed["type"] == "deposit"
            assert parsed["agent_id"] == "agent_1"

    @pytest.mark.asyncio
    async def test_event_all_matches_any_type(self):
        """Events=['all'] should match any event type."""
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["all"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks.httpx.AsyncClient", return_value=mock_client):
            await deliver_webhook("agent_1", "custom_event", {})
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_specific_event_subscription(self):
        """Only subscribed events should be delivered."""
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["spend", "deposit"],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks.httpx.AsyncClient", return_value=mock_client):
            # spend should work
            await deliver_webhook("agent_1", "spend", {})
            assert mock_client.post.call_count == 1

            # approval should be filtered
            await deliver_webhook("agent_1", "approval", {})
            assert mock_client.post.call_count == 1  # no new call


class TestWebhookDBOperations:
    """Test register/unregister webhook with DB persistence."""

    def setup_method(self):
        _webhook_registry.clear()

    def teardown_method(self):
        _webhook_registry.clear()

    @pytest.mark.asyncio
    async def test_register_updates_memory(self):
        """register_webhook should update in-memory registry."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_agent = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent
        mock_db.execute.return_value = mock_result
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("models.database.async_session", return_value=mock_db):
            await register_webhook("agent_1", "https://example.com/hook", "whsec_123", ["spend"])

        config = get_webhook_config("agent_1")
        assert config is not None
        assert config["url"] == "https://example.com/hook"
        assert config["secret"] == "whsec_123"
        assert config["events"] == ["spend"]

    @pytest.mark.asyncio
    async def test_register_persists_to_db(self):
        """register_webhook should update DB agent record."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("models.database.async_session", return_value=mock_db):
            await register_webhook("agent_1", "https://hook.test/abc", "whsec_abc")

        assert mock_agent.webhook_url == "https://hook.test/abc"
        assert mock_agent.webhook_secret == "whsec_abc"
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_unregister_clears_memory(self):
        """unregister_webhook should remove from in-memory registry."""
        _webhook_registry["agent_1"] = {
            "url": "https://example.com/hook",
            "secret": "whsec_test",
            "events": ["all"],
        }

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("models.database.async_session", return_value=mock_db):
            await unregister_webhook("agent_1")

        assert get_webhook_config("agent_1") is None

    @pytest.mark.asyncio
    async def test_unregister_clears_db(self):
        """unregister_webhook should null out DB fields."""
        _webhook_registry["agent_1"] = {"url": "x", "secret": "y", "events": ["all"]}

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("models.database.async_session", return_value=mock_db):
            await unregister_webhook("agent_1")

        assert mock_agent.webhook_url is None
        assert mock_agent.webhook_secret is None
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_no_error(self):
        """Unregistering unknown agent shouldn't raise."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("models.database.async_session", return_value=mock_db):
            await unregister_webhook("ghost_agent")  # should not raise

    @pytest.mark.asyncio
    async def test_register_default_events_all(self):
        """Default events should be ['all'] when not specified."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("models.database.async_session", return_value=mock_db):
            await register_webhook("agent_1", "https://hook.test", "whsec_x")

        assert get_webhook_config("agent_1")["events"] == ["all"]


class TestApprovalNotification:
    """Test approval request Telegram notification with inline buttons."""

    @pytest.mark.asyncio
    async def test_approval_sends_inline_buttons(self):
        """Approval notification should include approve/deny buttons."""
        mock_bot = AsyncMock()
        set_bot(mock_bot)

        await notify_approval_request("my-agent", 12345, 50.0, "GPU rental", "appr_abc123")

        mock_bot.send_message.assert_called_once()
        call_args = mock_bot.send_message.call_args
        text = call_args[0][1]
        assert "Approval Required" in text
        assert "my-agent" in text
        assert "$50.00" in text
        assert "GPU rental" in text

        # Check inline keyboard
        reply_markup = call_args[1]["reply_markup"]
        buttons = reply_markup.inline_keyboard[0]
        assert any("approve:appr_abc123" in b.callback_data for b in buttons)
        assert any("deny:appr_abc123" in b.callback_data for b in buttons)

        set_bot(None)

    @pytest.mark.asyncio
    async def test_approval_no_description(self):
        """Approval without description omits 'For:' line."""
        mock_bot = AsyncMock()
        set_bot(mock_bot)

        await notify_approval_request("agent", 12345, 10.0, None, "appr_xyz")

        text = mock_bot.send_message.call_args[0][1]
        assert "For:" not in text
        set_bot(None)

    @pytest.mark.asyncio
    async def test_approval_no_bot_silently_returns(self):
        """No bot instance → return silently."""
        set_bot(None)
        await notify_approval_request("agent", 12345, 10.0, "test", "appr_1")
        # Should not raise

    @pytest.mark.asyncio
    async def test_approval_bot_error_handled(self):
        """Bot send error is caught and logged."""
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = Exception("Telegram timeout")
        set_bot(mock_bot)

        # Should not raise
        await notify_approval_request("agent", 12345, 10.0, "test", "appr_1")
        set_bot(None)
