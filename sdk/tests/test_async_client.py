"""Tests for AgentPayAsyncClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentpay.async_client import AgentPayAsyncClient
from agentpay.exceptions import (
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
)

# Full mock data matching Pydantic models
BALANCE_DATA = {
    "agent_id": "a_1", "agent_name": "TestBot", "balance_usd": 42.50,
    "daily_limit_usd": 100, "daily_spent_usd": 10, "daily_remaining_usd": 90,
    "tx_limit_usd": 50, "is_active": True,
}
WALLET_DATA = {"address": "0xabc", "chain": "polygon"}
CHAIN_DATA = {"id": "base", "name": "Base", "type": "evm", "native_token": "ETH", "usdc_supported": True, "explorer": "https://basescan.org"}
SPEND_DATA = {"success": True, "transaction_id": "tx_1", "amount": 5.0, "fee": 0.1, "remaining_balance": 37.4, "status": "completed"}
REFUND_DATA = {"success": True, "refund_transaction_id": "rtx_1", "amount_refunded": 5.0, "new_balance": 42.5}
TRANSFER_DATA = {"success": True, "transaction_id": "tx_2", "amount": 10.0, "from_balance": 32.5}
TX_DATA = {"id": "tx_1", "type": "spend", "amount": 5.0, "fee": 0.1, "status": "completed", "created_at": "2026-04-08T12:00:00Z"}
WEBHOOK_DATA = {"url": "https://hook.test", "secret": "s", "events": ["spend"]}
X402_DATA = {"success": True, "status": 200, "data": "content", "paid_usd": 0.5}


def mock_response(data=None, status=200, headers=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.is_success = 200 <= status < 300
    resp.json.return_value = data or {}
    resp.text = str(data)
    resp.headers = headers or {}
    return resp


@pytest.fixture
def client():
    return AgentPayAsyncClient("ap_test_key", base_url="https://test.example.com", max_retries=0)


@pytest.mark.asyncio
async def test_get_balance(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response(BALANCE_DATA)
        balance = await client.get_balance()
        assert balance.balance_usd == 42.50
        assert balance.agent_id == "a_1"


@pytest.mark.asyncio
async def test_get_wallet(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response(WALLET_DATA)
        wallet = await client.get_wallet("polygon")
        assert wallet.chain == "polygon"


@pytest.mark.asyncio
async def test_list_chains(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"chains": [CHAIN_DATA, {**CHAIN_DATA, "id": "polygon", "name": "Polygon"}]})
        chains = await client.list_chains()
        assert len(chains) == 2


@pytest.mark.asyncio
async def test_spend(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response(SPEND_DATA)
        result = await client.spend(5.0, "test", "idem_1")
        assert result.success is True
        body = m.call_args.kwargs.get("json") or m.call_args[1].get("json")
        assert body == {"amount": 5.0, "description": "test", "idempotency_key": "idem_1"}


@pytest.mark.asyncio
async def test_spend_no_idempotency(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response(SPEND_DATA)
        await client.spend(1.0, "no key")
        body = m.call_args.kwargs.get("json") or m.call_args[1].get("json")
        assert "idempotency_key" not in body


@pytest.mark.asyncio
async def test_refund(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response(REFUND_DATA)
        result = await client.refund("tx_99")
        assert result.success is True


@pytest.mark.asyncio
async def test_transfer(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response(TRANSFER_DATA)
        result = await client.transfer("agent_2", 10.0, "payment")
        assert result.success is True


@pytest.mark.asyncio
async def test_get_transactions(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response([TX_DATA, {**TX_DATA, "id": "tx_2"}])
        txs = await client.get_transactions(5)
        assert len(txs) == 2


@pytest.mark.asyncio
async def test_register_webhook(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response(WEBHOOK_DATA)
        wh = await client.register_webhook("https://hook.test", events=["spend"])
        assert wh.url == "https://hook.test"


@pytest.mark.asyncio
async def test_x402_pay(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response(X402_DATA)
        result = await client.x402_pay("https://paywall.test", 0.5)
        assert result.success is True


@pytest.mark.asyncio
async def test_get_identity(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"agent_id": "a_1", "name": "Bot"})
        identity = await client.get_identity()
        assert identity["agent_id"] == "a_1"


@pytest.mark.asyncio
async def test_get_trust_score(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"score": 85})
        score = await client.get_trust_score()
        assert score["score"] == 85


@pytest.mark.asyncio
async def test_auth_error(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"detail": "Invalid key"}, 401)
        with pytest.raises(AuthenticationError):
            await client.get_balance()


@pytest.mark.asyncio
async def test_insufficient_balance(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"detail": "Insufficient balance"}, 402)
        with pytest.raises(InsufficientBalanceError):
            await client.spend(999, "too much")


@pytest.mark.asyncio
async def test_rate_limit(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"detail": "Too many requests"}, 429)
        with pytest.raises(RateLimitError):
            await client.get_balance()


@pytest.mark.asyncio
async def test_generic_error(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"detail": "Not found"}, 404)
        with pytest.raises(AgentPayError):
            await client.get_balance()


@pytest.mark.asyncio
async def test_network_error(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(AgentPayError, match="Network error"):
            await client.get_balance()


@pytest.mark.asyncio
async def test_retry_on_500():
    c = AgentPayAsyncClient("key", base_url="https://test.example.com", max_retries=1)
    with patch.object(c._client, "request", new_callable=AsyncMock) as m:
        m.side_effect = [
            mock_response({"detail": "error"}, 500),
            mock_response(BALANCE_DATA),
        ]
        with patch("agentpay.async_client.asyncio.sleep", new_callable=AsyncMock):
            balance = await c.get_balance()
            assert balance.balance_usd == 42.50
            assert m.call_count == 2


@pytest.mark.asyncio
async def test_context_manager():
    async with AgentPayAsyncClient("key", base_url="https://test.example.com") as c:
        assert c._api_key == "key"


@pytest.mark.asyncio
async def test_payee_rules(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"rules": []})
        result = await client.list_payee_rules()
        assert "rules" in result


@pytest.mark.asyncio
async def test_create_payee_rule(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"id": "rule_1"})
        await client.create_payee_rule("domain", "example.com", max_amount_usd=100.0)
        body = m.call_args.kwargs.get("json") or m.call_args[1].get("json")
        assert body["payee_type"] == "domain"
        assert body["max_amount_usd"] == 100.0


@pytest.mark.asyncio
async def test_delete_payee_rule(client):
    with patch.object(client._client, "request", new_callable=AsyncMock) as m:
        m.return_value = mock_response({"status": "deleted"})
        await client.delete_payee_rule("rule_1")
