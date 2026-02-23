class AgentPayError(Exception):
    """Raised when the AgentPay API returns a non-2xx response."""

    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"AgentPay API error {status_code}: {detail}")
