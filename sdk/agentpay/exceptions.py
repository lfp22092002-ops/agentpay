"""AgentPay SDK exceptions and utilities."""

from __future__ import annotations


class AgentPayError(Exception):
    """Base exception for AgentPay API errors.

    Attributes:
        status_code: HTTP status code from the API.
        detail: Human-readable error message.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"AgentPay API error {status_code}: {detail}")


class AuthenticationError(AgentPayError):
    """Raised when the API key is invalid or missing (401)."""

    def __init__(self, detail: str = "Invalid or missing API key") -> None:
        super().__init__(401, detail)


class InsufficientBalanceError(AgentPayError):
    """Raised when the agent balance is too low for the requested operation."""

    def __init__(self, detail: str = "Insufficient balance") -> None:
        super().__init__(402, detail)


class RateLimitError(AgentPayError):
    """Raised when the API rate limit is exceeded (429)."""

    def __init__(self, detail: str = "Rate limit exceeded") -> None:
        super().__init__(429, detail)


# ------------------------------------------------------------------
# Webhook signature verification
# ------------------------------------------------------------------


def verify_webhook_signature(
    payload: str | bytes,
    signature: str,
    secret: str,
) -> bool:
    """Verify that a webhook payload was signed by AgentPay.

    Use this in your webhook receiver to ensure the request is authentic.

    Args:
        payload: The raw request body (string or bytes).
        signature: Value of the ``X-AgentPay-Signature`` header.
        secret: Your webhook signing secret (``whsec_…``).

    Returns:
        ``True`` if the signature is valid.

    Example::

        from agentpay import verify_webhook_signature

        @app.post("/webhook")
        async def handle(request: Request):
            body = await request.body()
            sig = request.headers["X-AgentPay-Signature"]
            if not verify_webhook_signature(body, sig, WEBHOOK_SECRET):
                raise HTTPException(401, "Bad signature")
            ...
    """
    import hashlib
    import hmac as _hmac

    if isinstance(payload, str):
        payload = payload.encode()
    expected = _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return _hmac.compare_digest(expected, signature)
