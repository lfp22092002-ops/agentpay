// AgentPay TypeScript SDK — client

import type {
  AgentPayOptions,
  AgentIdentity,
  Balance,
  Chain,
  PayeeRule,
  PayeeRulesResponse,
  RefundResponse,
  SpendResponse,
  Transaction,
  TransferResponse,
  TrustScoreBreakdown,
  Wallet,
  Webhook,
  X402Response,
} from "./types.js";

import {
  AgentPayError,
  AuthenticationError,
  InsufficientBalanceError,
  RateLimitError,
} from "./errors.js";

const DEFAULT_BASE_URL = "https://leofundmybot.dev";
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_RETRIES = 3;
const RETRYABLE_STATUS_CODES = new Set([429, 500, 502, 503, 504]);

/**
 * AgentPay client for managing AI agent wallets and payments.
 *
 * @example
 * ```ts
 * import { AgentPayClient } from "agentpay";
 *
 * const client = new AgentPayClient("ap_your_api_key");
 * const balance = await client.getBalance();
 * console.log(`Balance: $${balance.balance_usd}`);
 * ```
 */
export class AgentPayClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  constructor(apiKey: string, options?: AgentPayOptions) {
    this.apiKey = apiKey;
    this.baseUrl = (options?.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = options?.maxRetries ?? DEFAULT_MAX_RETRIES;
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  private async request<T>(
    method: string,
    path: string,
    options?: { json?: Record<string, unknown>; params?: Record<string, string> },
  ): Promise<T> {
    let url = `${this.baseUrl}${path}`;
    if (options?.params) {
      const sp = new URLSearchParams(options.params);
      url += `?${sp.toString()}`;
    }

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        const res = await fetch(url, {
          method,
          headers: {
            "X-API-Key": this.apiKey,
            ...(options?.json ? { "Content-Type": "application/json" } : {}),
          },
          body: options?.json ? JSON.stringify(options.json) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timer);

        if (res.ok) {
          return (await res.json()) as T;
        }

        let detail: string;
        try {
          const body = await res.json();
          detail = (body as Record<string, unknown>).detail as string ?? res.statusText;
        } catch {
          detail = res.statusText;
        }

        // Non-retryable errors — throw immediately
        if (res.status === 401) throw new AuthenticationError(detail);
        if ((res.status === 402 || res.status === 400) && detail.toLowerCase().includes("balance")) {
          throw new InsufficientBalanceError(detail);
        }

        // Retryable errors — retry with exponential backoff
        if (RETRYABLE_STATUS_CODES.has(res.status) && attempt < this.maxRetries) {
          const retryAfter = res.headers.get("Retry-After");
          const delay = retryAfter ? parseFloat(retryAfter) * 1000 : Math.min(2 ** attempt * 1000, 8000);
          await new Promise((r) => setTimeout(r, delay));
          continue;
        }

        if (res.status === 429) throw new RateLimitError(detail);
        throw new AgentPayError(res.status, detail);
      } catch (err) {
        clearTimeout(timer);
        if (err instanceof AgentPayError) throw err;
        // Network errors — retry if attempts remain
        if (attempt < this.maxRetries) {
          await new Promise((r) => setTimeout(r, Math.min(2 ** attempt * 1000, 8000)));
          continue;
        }
        throw err;
      }
    }

    // Unreachable, but TypeScript needs it
    throw new AgentPayError(0, "Max retries exceeded");
  }

  // ------------------------------------------------------------------
  // Balance & wallet
  // ------------------------------------------------------------------

  /** Get the current agent balance. */
  async getBalance(): Promise<Balance> {
    return this.request<Balance>("GET", "/v1/balance");
  }

  /**
   * Get the agent's on-chain wallet details.
   * @param chain Blockchain network: `base`, `polygon`, `bnb`, `solana`.
   */
  async getWallet(chain: string = "base"): Promise<Wallet> {
    return this.request<Wallet>("GET", "/v1/wallet", { params: { chain } });
  }

  /** List all supported blockchain networks. */
  async listChains(): Promise<Chain[]> {
    const data = await this.request<{ chains: Chain[] }>("GET", "/v1/chains");
    return data.chains;
  }

  // ------------------------------------------------------------------
  // Transactions
  // ------------------------------------------------------------------

  /**
   * Spend funds from the agent balance.
   * @param amount Amount in USD.
   * @param description Human-readable description.
   * @param idempotencyKey Optional key to prevent duplicate charges.
   */
  async spend(
    amount: number,
    description: string,
    idempotencyKey?: string,
  ): Promise<SpendResponse> {
    const json: Record<string, unknown> = { amount, description };
    if (idempotencyKey !== undefined) json.idempotency_key = idempotencyKey;
    return this.request<SpendResponse>("POST", "/v1/spend", { json });
  }

  /**
   * Refund a previous transaction.
   * @param transactionId The transaction ID to refund.
   */
  async refund(transactionId: string): Promise<RefundResponse> {
    return this.request<RefundResponse>("POST", "/v1/refund", {
      json: { transaction_id: transactionId },
    });
  }

  /**
   * Transfer funds to another agent.
   * @param toAgentId Target agent ID.
   * @param amount Amount in USD.
   * @param description Optional description.
   */
  async transfer(
    toAgentId: string,
    amount: number,
    description?: string,
  ): Promise<TransferResponse> {
    const json: Record<string, unknown> = { to_agent_id: toAgentId, amount };
    if (description !== undefined) json.description = description;
    return this.request<TransferResponse>("POST", "/v1/transfer", { json });
  }

  /**
   * Retrieve recent transactions.
   * @param limit Maximum results (default 20, max 100).
   */
  async getTransactions(limit: number = 20): Promise<Transaction[]> {
    return this.request<Transaction[]>("GET", "/v1/transactions", {
      params: { limit: String(limit) },
    });
  }

  // ------------------------------------------------------------------
  // Webhooks
  // ------------------------------------------------------------------

  /**
   * Register a webhook endpoint.
   * @param url HTTPS URL to receive events.
   * @param events Event types to subscribe to (default: all).
   */
  async registerWebhook(url: string, events?: string[]): Promise<Webhook> {
    const json: Record<string, unknown> = { url };
    if (events !== undefined) json.events = events;
    return this.request<Webhook>("POST", "/v1/webhook", { json });
  }

  // ------------------------------------------------------------------
  // x402 Protocol
  // ------------------------------------------------------------------

  /**
   * Pay for an x402-gated resource using the agent's wallet.
   * @param url The x402-gated resource URL.
   * @param maxAmount Maximum price willing to pay in USD.
   */
  async x402Pay(url: string, maxAmount?: number): Promise<X402Response> {
    const json: Record<string, unknown> = { url };
    if (maxAmount !== undefined) json.max_price_usd = maxAmount;
    return this.request<X402Response>("POST", "/v1/x402/pay", { json });
  }

  // ------------------------------------------------------------------
  // Identity (KYA)
  // ------------------------------------------------------------------

  /** Get the agent's identity profile. */
  async getIdentity(): Promise<AgentIdentity> {
    return this.request<AgentIdentity>("GET", "/v1/agent/identity");
  }

  /** Get the agent's trust score breakdown. */
  async getTrustScore(): Promise<TrustScoreBreakdown> {
    return this.request<TrustScoreBreakdown>("GET", "/v1/agent/identity/score");
  }

  // ------------------------------------------------------------------
  // Payee Rules
  // ------------------------------------------------------------------

  /** List all payee rules for this agent. */
  async listPayeeRules(): Promise<PayeeRulesResponse> {
    return this.request<PayeeRulesResponse>("GET", "/v1/agent/payee-rules");
  }

  /**
   * Create a payee allow/deny rule.
   * @param payeeType One of 'agent_id', 'domain', 'category', 'address'.
   * @param payeeValue The payee identifier to match.
   * @param ruleType 'allow' or 'deny' (default: 'allow').
   * @param maxAmountUsd Per-payee transaction cap (optional).
   * @param note Optional human-readable note.
   */
  async createPayeeRule(
    payeeType: string,
    payeeValue: string,
    ruleType: string = "allow",
    maxAmountUsd?: number,
    note?: string,
  ): Promise<{ success: boolean; rule: PayeeRule }> {
    const json: Record<string, unknown> = {
      rule_type: ruleType,
      payee_type: payeeType,
      payee_value: payeeValue,
    };
    if (maxAmountUsd !== undefined) json.max_amount_usd = maxAmountUsd;
    if (note !== undefined) json.note = note;
    return this.request("POST", "/v1/agent/payee-rules", { json });
  }

  /** Delete (deactivate) a payee rule. */
  async deletePayeeRule(ruleId: string): Promise<{ success: boolean; message: string }> {
    return this.request("DELETE", `/v1/agent/payee-rules/${ruleId}`);
  }
}
