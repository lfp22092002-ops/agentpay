"""
Tests for API middleware: security headers, rate limiting, exception handler.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.middleware import (
    check_api_key_rate_limit,
    log_requests_middleware,
    rate_limit_handler,
    global_exception_handler,
    API_KEY_RATE_LIMIT,
    _api_key_requests,
)


class TestApiKeyRateLimit:
    """Per-API-key rate limiter tests."""

    def setup_method(self):
        _api_key_requests.clear()

    def test_allows_under_limit(self):
        for _ in range(API_KEY_RATE_LIMIT - 1):
            assert check_api_key_rate_limit("test-key") is True

    def test_blocks_over_limit(self):
        for _ in range(API_KEY_RATE_LIMIT):
            check_api_key_rate_limit("over-key")
        assert check_api_key_rate_limit("over-key") is False

    def test_separate_keys_independent(self):
        for _ in range(API_KEY_RATE_LIMIT):
            check_api_key_rate_limit("key-a")
        # key-b should still be allowed
        assert check_api_key_rate_limit("key-b") is True

    def test_uses_hashed_key(self):
        """Raw API key should NOT appear in the internal dict."""
        check_api_key_rate_limit("my-secret-api-key")
        assert "my-secret-api-key" not in _api_key_requests


@pytest.mark.asyncio
class TestLogRequestsMiddleware:
    async def test_adds_security_headers(self):
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/v1/agents"

        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        call_next = AsyncMock(return_value=response)

        result = await log_requests_middleware(request, call_next)

        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["X-Frame-Options"] == "DENY"
        assert "max-age=" in result.headers["Strict-Transport-Security"]
        assert result.headers["X-XSS-Protection"] == "1; mode=block"
        assert "strict-origin" in result.headers["Referrer-Policy"]
        assert "camera=()" in result.headers["Permissions-Policy"]
        assert "default-src" in result.headers["Content-Security-Policy"]

    async def test_strips_server_header(self):
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/test"

        response = MagicMock()
        response.status_code = 200
        response.headers = {"server": "uvicorn"}
        call_next = AsyncMock(return_value=response)

        result = await log_requests_middleware(request, call_next)
        assert "server" not in result.headers


@pytest.mark.asyncio
class TestRateLimitHandler:
    async def test_returns_429(self):
        request = MagicMock()
        exc = MagicMock()
        response = await rate_limit_handler(request, exc)
        assert response.status_code == 429


@pytest.mark.asyncio
class TestGlobalExceptionHandler:
    async def test_returns_500_without_stack_trace(self):
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/boom"
        exc = ValueError("something broke")
        response = await global_exception_handler(request, exc)
        assert response.status_code == 500
        assert b"something broke" not in response.body
