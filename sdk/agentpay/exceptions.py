"""AgentPay SDK exceptions."""

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
