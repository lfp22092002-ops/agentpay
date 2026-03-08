"""
Tests for the rate limiter and input validation/sanitization.
"""
import pytest
import time

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════
# Per-API-Key Rate Limiter
# ═══════════════════════════════════════

class TestAPIKeyRateLimiter:
    def test_allows_under_limit(self):
        from api.middleware import _check_api_key_rate_limit
        # Use a unique key per test to avoid cross-contamination
        test_key = f"test_rate_under_{time.time()}"
        for _ in range(10):
            assert _check_api_key_rate_limit(test_key) is True

    def test_blocks_at_limit(self):
        from api.middleware import _check_api_key_rate_limit, API_KEY_RATE_LIMIT
        test_key = f"test_rate_block_{time.time()}"
        # Fill up to the limit
        for _ in range(API_KEY_RATE_LIMIT):
            assert _check_api_key_rate_limit(test_key) is True
        # Next request should be blocked
        assert _check_api_key_rate_limit(test_key) is False

    def test_different_keys_independent(self):
        from api.middleware import _check_api_key_rate_limit, API_KEY_RATE_LIMIT
        key_a = f"test_rate_a_{time.time()}"
        key_b = f"test_rate_b_{time.time()}"
        # Fill key_a to the limit
        for _ in range(API_KEY_RATE_LIMIT):
            _check_api_key_rate_limit(key_a)
        # key_b should still be allowed
        assert _check_api_key_rate_limit(key_b) is True
        # key_a should be blocked
        assert _check_api_key_rate_limit(key_a) is False

    def test_uses_hashed_key_id(self):
        """Raw API key should NOT be stored in the dict; only a hash prefix."""
        from api.middleware import _check_api_key_rate_limit, _api_key_requests
        import hashlib
        raw_key = f"ap_sensitive_key_{time.time()}"
        _check_api_key_rate_limit(raw_key)
        expected_id = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
        assert expected_id in _api_key_requests
        assert raw_key not in _api_key_requests


# ═══════════════════════════════════════
# Input Validation (Pydantic Models)
# ═══════════════════════════════════════

class TestInputValidation:
    def test_spend_request_positive_amount(self):
        from api.models import SpendRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SpendRequest(amount=-1, description="negative")

    def test_spend_request_zero_amount(self):
        from api.models import SpendRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SpendRequest(amount=0, description="zero")

    def test_spend_request_exceeds_max(self):
        from api.models import SpendRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SpendRequest(amount=10001, description="too much")

    def test_spend_request_valid(self):
        from api.models import SpendRequest
        req = SpendRequest(amount=5.0, description="valid purchase")
        assert req.amount == 5.0
        assert req.description == "valid purchase"

    def test_spend_request_description_max_length(self):
        from api.models import SpendRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SpendRequest(amount=1.0, description="x" * 501)

    def test_refund_request_requires_tx_id(self):
        from api.models import RefundRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RefundRequest()

    def test_transfer_request_validation(self):
        from api.models import TransferRequest
        from pydantic import ValidationError
        # Zero amount
        with pytest.raises(ValidationError):
            TransferRequest(to_agent_id="test", amount=0)
        # Negative amount
        with pytest.raises(ValidationError):
            TransferRequest(to_agent_id="test", amount=-5)
        # Over max
        with pytest.raises(ValidationError):
            TransferRequest(to_agent_id="test", amount=10001)

    def test_webhook_url_max_length(self):
        from api.models import WebhookSetRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            WebhookSetRequest(url="https://x.com/" + "a" * 512)

    def test_agent_identity_update_validation(self):
        from api.models import AgentIdentityUpdate
        from pydantic import ValidationError
        # display_name too long
        with pytest.raises(ValidationError):
            AgentIdentityUpdate(display_name="x" * 256)
        # description too long
        with pytest.raises(ValidationError):
            AgentIdentityUpdate(display_name="ok", description="x" * 2001)

    def test_send_usdc_request_valid(self):
        from api.models import SendUsdcRequest
        req = SendUsdcRequest(to_address="0xABC123", amount=10.0, chain="base")
        assert req.chain == "base"
        assert req.amount == 10.0

    def test_send_native_request_max_amount(self):
        from api.models import SendNativeRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SendNativeRequest(to_address="0xABC", amount=101, chain="base")


# ═══════════════════════════════════════
# Trust Score Calculation
# ═══════════════════════════════════════

class _MockAgent:
    """Lightweight mock for Agent (avoids SQLAlchemy instrumentation)."""
    def __init__(self, created_at):
        self.created_at = created_at


class _MockIdentity:
    """Lightweight mock for AgentIdentity."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestTrustScore:
    def test_trust_score_empty_profile(self):
        from api.routes.identity import _calculate_trust_score
        from datetime import datetime, timezone
        from decimal import Decimal

        agent = _MockAgent(created_at=datetime.now(timezone.utc))
        identity = _MockIdentity(
            display_name="",
            description=None,
            homepage_url=None,
            logo_url=None,
            verified=False,
            total_transactions=0,
            total_volume_usd=Decimal("0"),
        )

        score = _calculate_trust_score(identity, agent)
        assert score.total == 0
        assert score.verified_pts == 0

    def test_trust_score_verified_agent(self):
        from api.routes.identity import _calculate_trust_score
        from datetime import datetime, timedelta, timezone
        from decimal import Decimal

        agent = _MockAgent(created_at=datetime.now(timezone.utc) - timedelta(weeks=20))
        identity = _MockIdentity(
            display_name="TradingBot",
            description="A verified trading bot",
            homepage_url="https://example.com",
            logo_url="https://example.com/logo.png",
            verified=True,
            total_transactions=500,
            total_volume_usd=Decimal("5000.00"),
        )

        score = _calculate_trust_score(identity, agent)
        assert score.total > 50  # Should be high
        assert score.verified_pts == 20
        assert score.account_age_pts == 15  # capped at 15
        assert score.profile_completeness_pts == 15  # name + desc + url + logo
        assert score.volume_pts == 25  # capped at 25 ($5000/$100=50 → cap 25)
        assert score.transaction_count_pts == 25  # capped at 25 (500/10=50 → cap 25)

    def test_trust_score_max_is_100(self):
        from api.routes.identity import _calculate_trust_score
        from datetime import datetime, timedelta, timezone
        from decimal import Decimal

        agent = _MockAgent(created_at=datetime.now(timezone.utc) - timedelta(weeks=100))
        identity = _MockIdentity(
            display_name="MaxBot",
            description="Maximum trust",
            homepage_url="https://maxbot.com",
            logo_url="https://maxbot.com/logo.png",
            verified=True,
            total_transactions=10000,
            total_volume_usd=Decimal("1000000.00"),
        )

        score = _calculate_trust_score(identity, agent)
        assert score.total == 100


# ═══════════════════════════════════════
# Category Validation
# ═══════════════════════════════════════

class TestCategoryValidation:
    def test_valid_categories(self):
        from api.models import VALID_CATEGORIES
        assert "trading" in VALID_CATEGORIES
        assert "research" in VALID_CATEGORIES
        assert "defi" in VALID_CATEGORIES
        assert "other" in VALID_CATEGORIES

    def test_invalid_category_not_in_set(self):
        from api.models import VALID_CATEGORIES
        assert "hacking" not in VALID_CATEGORIES
        assert "" not in VALID_CATEGORIES
