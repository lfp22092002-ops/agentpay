// src/errors.ts
var AgentPayError = class extends Error {
  constructor(statusCode, detail) {
    super(`AgentPay API error ${statusCode}: ${detail}`);
    this.name = "AgentPayError";
    this.statusCode = statusCode;
    this.detail = detail;
  }
};
var AuthenticationError = class extends AgentPayError {
  constructor(detail) {
    super(401, detail);
    this.name = "AuthenticationError";
  }
};
var InsufficientBalanceError = class extends AgentPayError {
  constructor(detail) {
    super(402, detail);
    this.name = "InsufficientBalanceError";
  }
};
var RateLimitError = class extends AgentPayError {
  constructor(detail) {
    super(429, detail);
    this.name = "RateLimitError";
  }
};

// src/client.ts
var DEFAULT_BASE_URL = "https://leofundmybot.dev";
var DEFAULT_TIMEOUT_MS = 3e4;
var DEFAULT_MAX_RETRIES = 3;
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
var AgentPayClient = class {
  constructor(apiKey, config) {
    this.apiKey = apiKey;
    this.baseUrl = (config?.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.timeoutMs = config?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = config?.maxRetries ?? DEFAULT_MAX_RETRIES;
  }
  // -----------------------------------------------------------------
  // Internal
  // -----------------------------------------------------------------
  async request(method, path, options) {
    let url = `${this.baseUrl}${path}`;
    if (options?.params) {
      const qs = new URLSearchParams();
      for (const [k, v] of Object.entries(options.params)) {
        qs.set(k, String(v));
      }
      url += `?${qs.toString()}`;
    }
    let lastError;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      let response;
      try {
        response = await fetch(url, {
          method,
          headers: {
            "X-API-Key": this.apiKey,
            "Content-Type": "application/json"
          },
          body: options?.body ? JSON.stringify(options.body) : void 0,
          signal: controller.signal
        });
      } catch (err) {
        clearTimeout(timer);
        lastError = err instanceof Error ? err : new Error(String(err));
        if (attempt < this.maxRetries) {
          await sleep(Math.min(2 ** attempt * 1e3, 8e3));
          continue;
        }
        throw new AgentPayError(
          0,
          `Network error after ${this.maxRetries + 1} attempts: ${lastError.message}`
        );
      } finally {
        clearTimeout(timer);
      }
      if (response.ok) {
        return await response.json();
      }
      let detail;
      try {
        const body = await response.json();
        detail = body.detail ?? response.statusText;
      } catch {
        detail = response.statusText;
      }
      if (response.status === 401) {
        throw new AuthenticationError(detail);
      }
      if ((response.status === 402 || response.status === 400) && detail.toLowerCase().includes("balance")) {
        throw new InsufficientBalanceError(detail);
      }
      if ([429, 500, 502, 503, 504].includes(response.status)) {
        lastError = new AgentPayError(response.status, detail);
        if (attempt < this.maxRetries) {
          let delay;
          if (response.status === 429) {
            const retryAfter = response.headers.get("Retry-After");
            delay = retryAfter ? parseFloat(retryAfter) * 1e3 : Math.min(2 ** attempt * 1e3, 8e3);
          } else {
            delay = Math.min(2 ** attempt * 1e3, 8e3);
          }
          await sleep(delay);
          continue;
        }
        if (response.status === 429) {
          throw new RateLimitError(detail);
        }
      }
      throw new AgentPayError(response.status, detail);
    }
    throw lastError ?? new AgentPayError(0, "Unknown error");
  }
  // -----------------------------------------------------------------
  // Balance & Wallet
  // -----------------------------------------------------------------
  /** Get the current agent balance. */
  async getBalance() {
    return this.request("GET", "/v1/balance");
  }
  /** Get the agent's on-chain wallet details. */
  async getWallet(chain = "base") {
    return this.request("GET", "/v1/wallet", {
      params: { chain }
    });
  }
  /** List all supported blockchain networks. */
  async listChains() {
    const data = await this.request("GET", "/v1/chains");
    return data.chains ?? [];
  }
  // -----------------------------------------------------------------
  // Transactions
  // -----------------------------------------------------------------
  /** Spend funds from the agent balance. */
  async spend(amount, description, idempotencyKey) {
    const body = { amount, description };
    if (idempotencyKey !== void 0) body.idempotency_key = idempotencyKey;
    return this.request("POST", "/v1/spend", { body });
  }
  /** Refund a previous transaction. */
  async refund(transactionId) {
    return this.request("POST", "/v1/refund", {
      body: { transaction_id: transactionId }
    });
  }
  /** Transfer funds to another agent. */
  async transfer(toAgentId, amount, description) {
    const body = {
      to_agent_id: toAgentId,
      amount
    };
    if (description !== void 0) body.description = description;
    return this.request("POST", "/v1/transfer", { body });
  }
  /** Retrieve recent transactions. */
  async getTransactions(limit = 20) {
    return this.request("GET", "/v1/transactions", {
      params: { limit }
    });
  }
  // -----------------------------------------------------------------
  // Webhooks
  // -----------------------------------------------------------------
  /** Register a webhook endpoint. */
  async registerWebhook(url, events) {
    const body = { url };
    if (events !== void 0) body.events = events;
    return this.request("POST", "/v1/webhook", { body });
  }
  // -----------------------------------------------------------------
  // x402 Protocol
  // -----------------------------------------------------------------
  /** Pay for an x402-gated resource using the agent's wallet. */
  async x402Pay(url, maxAmount) {
    const body = { url };
    if (maxAmount !== void 0) body.max_price_usd = maxAmount;
    return this.request("POST", "/v1/x402/pay", { body });
  }
  // -----------------------------------------------------------------
  // Identity
  // -----------------------------------------------------------------
  /** Get the agent's identity profile (KYA). */
  async getIdentity() {
    return this.request("GET", "/v1/agent/identity");
  }
  /** Get the agent's trust score breakdown. */
  async getTrustScore() {
    return this.request("GET", "/v1/agent/identity/score");
  }
};

export { AgentPayClient, AgentPayError, AuthenticationError, InsufficientBalanceError, RateLimitError };
//# sourceMappingURL=index.mjs.map
//# sourceMappingURL=index.mjs.map