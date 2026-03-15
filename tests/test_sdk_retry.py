"""
Tests for SDK retry / exponential backoff (sync + async).

Covers: transient 429/5xx retries, Retry-After header, non-retryable fast-fail,
        max_retries=0 disabling retries, network error retries.
"""
import pytest
import httpx

import sys
import os
SDK_DIR = os.path.join(os.path.dirname(__file__), "..", "sdk")
sys.path.insert(0, SDK_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
if "agentpay" in sys.modules:
    mod = sys.modules["agentpay"]
    mod_file = getattr(mod, "__file__", "") or ""
    if "sdk" not in mod_file:
        del sys.modules["agentpay"]

from agentpay import (
    AgentPayClient,
    AgentPayAsyncClient,
    AgentPayError,
    AuthenticationError,
    RateLimitError,
)

BALANCE_OK = {
    "agent_id": "agent_123",
    "agent_name": "test-agent",
    "balance_usd": 100.0,
    "daily_limit_usd": 50.0,
    "daily_spent_usd": 5.0,
    "daily_remaining_usd": 45.0,
    "tx_limit_usd": 25.0,
    "is_active": True,
}


# ---------------------------------------------------------------------------
# Stateful mock transports that fail N times then succeed
# ---------------------------------------------------------------------------

class RetryTransport(httpx.BaseTransport):
    """Fails with given status for `fail_count` requests, then returns success."""

    def __init__(self, fail_count: int, fail_status: int, fail_headers: dict | None = None):
        self._fail_count = fail_count
        self._fail_status = fail_status
        self._fail_headers = fail_headers or {}
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            return httpx.Response(
                self._fail_status,
                json={"detail": f"Error {self._fail_status}"},
                headers=self._fail_headers,
            )
        return httpx.Response(200, json=BALANCE_OK)


class AsyncRetryTransport(httpx.AsyncBaseTransport):
    """Async version — fails N times then succeeds."""

    def __init__(self, fail_count: int, fail_status: int, fail_headers: dict | None = None):
        self._fail_count = fail_count
        self._fail_status = fail_status
        self._fail_headers = fail_headers or {}
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            return httpx.Response(
                self._fail_status,
                json={"detail": f"Error {self._fail_status}"},
                headers=self._fail_headers,
            )
        return httpx.Response(200, json=BALANCE_OK)


class AlwaysFailTransport(httpx.BaseTransport):
    """Always returns the given status code."""

    def __init__(self, status: int, headers: dict | None = None):
        self._status = status
        self._headers = headers or {}
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._attempts += 1
        return httpx.Response(
            self._status,
            json={"detail": f"Error {self._status}"},
            headers=self._headers,
        )


class AsyncAlwaysFailTransport(httpx.AsyncBaseTransport):
    """Async version — always fails."""

    def __init__(self, status: int, headers: dict | None = None):
        self._status = status
        self._headers = headers or {}
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._attempts += 1
        return httpx.Response(
            self._status,
            json={"detail": f"Error {self._status}"},
            headers=self._headers,
        )


class NetworkErrorTransport(httpx.BaseTransport):
    """Raises ConnectError for `fail_count` requests, then returns success."""

    def __init__(self, fail_count: int):
        self._fail_count = fail_count
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise httpx.ConnectError("Connection refused")
        return httpx.Response(200, json=BALANCE_OK)


def _sync_with_transport(transport, max_retries: int = 3) -> AgentPayClient:
    client = AgentPayClient("ap_test", base_url="https://test.local", max_retries=max_retries)
    client._client = httpx.Client(
        base_url="https://test.local",
        headers={"X-API-Key": "ap_test"},
        transport=transport,
    )
    return client


def _async_with_transport(transport, max_retries: int = 3) -> AgentPayAsyncClient:
    client = AgentPayAsyncClient.__new__(AgentPayAsyncClient)
    client._api_key = "ap_test"
    client._base_url = "https://test.local"
    client._max_retries = max_retries
    client._client = httpx.AsyncClient(
        base_url="https://test.local",
        headers={"X-API-Key": "ap_test"},
        transport=transport,
    )
    return client


# ===========================================================================
# Sync retry tests
# ===========================================================================


class TestSyncRetrySuccess:
    """Transient errors that eventually succeed within max_retries."""

    def test_retry_429_then_success(self):
        transport = RetryTransport(fail_count=2, fail_status=429)
        client = _sync_with_transport(transport, max_retries=3)
        balance = client.get_balance()
        assert balance.balance_usd == 100.0
        assert transport.attempts == 3  # 2 fails + 1 success
        client.close()

    def test_retry_500_then_success(self):
        transport = RetryTransport(fail_count=1, fail_status=500)
        client = _sync_with_transport(transport, max_retries=3)
        balance = client.get_balance()
        assert balance.balance_usd == 100.0
        assert transport.attempts == 2
        client.close()

    def test_retry_502_then_success(self):
        transport = RetryTransport(fail_count=2, fail_status=502)
        client = _sync_with_transport(transport, max_retries=3)
        balance = client.get_balance()
        assert balance.balance_usd == 100.0
        assert transport.attempts == 3
        client.close()

    def test_retry_503_then_success(self):
        transport = RetryTransport(fail_count=1, fail_status=503)
        client = _sync_with_transport(transport, max_retries=2)
        balance = client.get_balance()
        assert balance.balance_usd == 100.0
        client.close()

    def test_retry_504_then_success(self):
        transport = RetryTransport(fail_count=1, fail_status=504)
        client = _sync_with_transport(transport, max_retries=2)
        balance = client.get_balance()
        assert balance.balance_usd == 100.0
        client.close()

    def test_network_error_then_success(self):
        transport = NetworkErrorTransport(fail_count=1)
        client = _sync_with_transport(transport, max_retries=2)
        balance = client.get_balance()
        assert balance.balance_usd == 100.0
        assert transport.attempts == 2
        client.close()


class TestSyncRetryExhausted:
    """All retries exhausted — should raise appropriate error."""

    def test_429_exhausted_raises_rate_limit(self):
        transport = AlwaysFailTransport(status=429)
        client = _sync_with_transport(transport, max_retries=2)
        with pytest.raises(RateLimitError):
            client.get_balance()
        assert transport.attempts == 3  # initial + 2 retries
        client.close()

    def test_500_exhausted_raises_agentpay_error(self):
        transport = AlwaysFailTransport(status=500)
        client = _sync_with_transport(transport, max_retries=1)
        with pytest.raises(AgentPayError):
            client.get_balance()
        assert transport.attempts == 2
        client.close()


class TestSyncNoRetryForNonTransient:
    """Non-retryable errors should fail immediately (no retries)."""

    def test_401_no_retry(self):
        transport = AlwaysFailTransport(status=401)
        client = _sync_with_transport(transport, max_retries=3)
        with pytest.raises(AuthenticationError):
            client.get_balance()
        assert transport.attempts == 1  # no retries
        client.close()

    def test_402_insufficient_balance_no_retry(self):
        transport = AlwaysFailTransport(status=402)
        # 402 with "balance" in detail triggers InsufficientBalanceError
        # AlwaysFailTransport says "Error 402" which doesn't contain "balance"
        # so it falls through to retryable check — 402 is not in [429,500,502,503,504]
        # so it raises AgentPayError immediately
        client = _sync_with_transport(transport, max_retries=3)
        with pytest.raises(AgentPayError):
            client.get_balance()
        assert transport.attempts == 1
        client.close()

    def test_404_no_retry(self):
        transport = AlwaysFailTransport(status=404)
        client = _sync_with_transport(transport, max_retries=3)
        with pytest.raises(AgentPayError):
            client.get_balance()
        assert transport.attempts == 1
        client.close()


class TestSyncRetryZero:
    """max_retries=0 should disable retries entirely."""

    def test_no_retries_on_500(self):
        transport = AlwaysFailTransport(status=500)
        client = _sync_with_transport(transport, max_retries=0)
        with pytest.raises(AgentPayError):
            client.get_balance()
        assert transport.attempts == 1
        client.close()


# ===========================================================================
# Async retry tests
# ===========================================================================


class TestAsyncRetrySuccess:
    @pytest.mark.asyncio
    async def test_retry_429_then_success(self):
        transport = AsyncRetryTransport(fail_count=2, fail_status=429)
        client = _async_with_transport(transport, max_retries=3)
        balance = await client.get_balance()
        assert balance.balance_usd == 100.0
        assert transport.attempts == 3
        await client.close()

    @pytest.mark.asyncio
    async def test_retry_500_then_success(self):
        transport = AsyncRetryTransport(fail_count=1, fail_status=500)
        client = _async_with_transport(transport, max_retries=2)
        balance = await client.get_balance()
        assert balance.balance_usd == 100.0
        assert transport.attempts == 2
        await client.close()


class TestAsyncRetryExhausted:
    @pytest.mark.asyncio
    async def test_429_exhausted(self):
        transport = AsyncAlwaysFailTransport(status=429)
        client = _async_with_transport(transport, max_retries=1)
        with pytest.raises(RateLimitError):
            await client.get_balance()
        assert transport.attempts == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_500_exhausted(self):
        transport = AsyncAlwaysFailTransport(status=500)
        client = _async_with_transport(transport, max_retries=1)
        with pytest.raises(AgentPayError):
            await client.get_balance()
        assert transport.attempts == 2
        await client.close()


class TestAsyncNoRetryForNonTransient:
    @pytest.mark.asyncio
    async def test_401_no_retry(self):
        transport = AsyncAlwaysFailTransport(status=401)
        client = _async_with_transport(transport, max_retries=3)
        with pytest.raises(AuthenticationError):
            await client.get_balance()
        assert transport.attempts == 1
        await client.close()
