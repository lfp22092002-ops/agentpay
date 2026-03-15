"""Tests for verify_webhook_signature utility."""

import json
import sys
import os

# Use the same SDK path trick as test_sdk.py
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

from agentpay import verify_webhook_signature


class TestVerifyWebhookSignature:
    def test_valid_signature(self):
        import hashlib
        import hmac

        secret = "whsec_abc123"
        payload = json.dumps({"type": "spend", "agent_id": "a1"})
        sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

        assert verify_webhook_signature(payload, sig, secret) is True

    def test_invalid_signature(self):
        payload = '{"type": "spend"}'
        assert verify_webhook_signature(payload, "badsig", "whsec_abc123") is False

    def test_bytes_payload(self):
        import hashlib
        import hmac

        secret = "whsec_test"
        payload = b'{"event":"deposit"}'
        sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        assert verify_webhook_signature(payload, sig, secret) is True

    def test_wrong_secret(self):
        import hashlib
        import hmac

        payload = '{"data": 1}'
        sig = hmac.new(b"correct_secret", payload.encode(), hashlib.sha256).hexdigest()

        assert verify_webhook_signature(payload, sig, "wrong_secret") is False

    def test_empty_payload(self):
        import hashlib
        import hmac

        secret = "whsec_empty"
        payload = ""
        sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

        assert verify_webhook_signature(payload, sig, secret) is True
