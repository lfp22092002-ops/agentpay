// AgentPay TypeScript SDK — Error classes

/** Base error for all AgentPay SDK errors. */
export class AgentPayError extends Error {
  readonly statusCode: number;
  readonly detail: string;

  constructor(statusCode: number, detail: string) {
    super(`AgentPay API error ${statusCode}: ${detail}`);
    this.name = "AgentPayError";
    this.statusCode = statusCode;
    this.detail = detail;
  }
}

/** Thrown when the API key is invalid or missing (HTTP 401). */
export class AuthenticationError extends AgentPayError {
  constructor(detail: string) {
    super(401, detail);
    this.name = "AuthenticationError";
  }
}

/** Thrown when the agent has insufficient balance (HTTP 402 / 400 with balance hint). */
export class InsufficientBalanceError extends AgentPayError {
  constructor(detail: string) {
    super(402, detail);
    this.name = "InsufficientBalanceError";
  }
}

/** Thrown when rate limited (HTTP 429) and retries exhausted. */
export class RateLimitError extends AgentPayError {
  constructor(detail: string) {
    super(429, detail);
    this.name = "RateLimitError";
  }
}
