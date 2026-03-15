/**
 * AgentPay TypeScript SDK — Client
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

import {
  AgentPayError,
  AuthenticationError,
  InsufficientBalanceError,
  RateLimitError,
} from "./errors.js";
import type {
  AgentIdentity,
  AgentPayConfig,
  Balance,
  Chain,
  RefundResponse,
  SpendResponse,
  Transaction,
  TransferResponse,
  TrustScore,
  Wallet,
  WebhookInfo,
  X402Response,
} from "./types.js";

const DEFAULT_BASE_URL = "https://leofundmybot.dev";
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_RETRIES = 3;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export class AgentPayClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  constructor(apiKey: string, config?: AgentPayConfig) {
    this.apiKey = apiKey;
    this.baseUrl = (config?.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.timeoutMs = config?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = config?.maxRetries ?? DEFAULT_MAX_RETRIES;
  }

  // -----------------------------------------------------------------
  // Internal
  // -----------------------------------------------------------------

  private async request<T = unknown>(
    method: string,
    path: string,
    options?: {
      body?: Record<string, unknown>;
      params?: Record<string, string | number>;
    },
  ): Promise<T> {
    let url = `${this.baseUrl}${path}`;
    if (options?.params) {
      const qs = new URLSearchParams();
      for (const [k, v] of Object.entries(options.params)) {
        qs.set(k, String(v));
      }
      url += `?${qs.toString()}`;
    }

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      let response: Response;
      try {
        response = await fetch(url, {
          method,
          headers: {
            "X-API-Key": this.apiKey,
            "Content-Type": "application/json",
          },
          body: options?.body ? JSON.stringify(options.body) : undefined,
          signal: controller.signal,
        });
      } catch (err) {
        clearTimeout(timer);
        lastError = err instanceof Error ? err : new Error(String(err));
        if (attempt < this.maxRetries) {
          await sleep(Math.min(2 ** attempt * 1000, 8000));
          continue;
        }
        throw new AgentPayError(
          0,
          `Network error after ${this.maxRetries + 1} attempts: ${lastError.message}`,
        );
      } finally {
        clearTimeout(timer);
      }

      if (response.ok) {
        return (await response.json()) as T;
      }

      // Parse error detail
      let detail: string;
      try {
        const body = await response.json();
        detail = (body as Record<string, unknown>).detail as string ?? response.statusText;
      } catch {
        detail = response.statusText;
      }

      // Non-retryable errors
      if (response.status === 401) {
        throw new AuthenticationError(detail);
      }
      if (
        (response.status === 402 || response.status === 400) &&
        detail.toLowerCase().includes("balance")
      ) {
        throw new InsufficientBalanceError(detail);
      }

      // Retryable errors
      if ([429, 500, 502, 503, 504].includes(response.status)) {
        lastError = new AgentPayError(response.status, detail);
        if (attempt < this.maxRetries) {
          let delay: number;
          if (response.status === 429) {
            const retryAfter = response.headers.get("Retry-After");
            delay = retryAfter
              ? parseFloat(retryAfter) * 1000
              : Math.min(2 ** attempt * 1000, 8000);
          } else {
            delay = Math.min(2 ** attempt * 1000, 8000);
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
  async getBalance(): Promise<Balance> {
    return this.request<Balance>("GET", "/v1/balance");
  }

  /** Get the agent's on-chain wallet details. */
  async getWallet(chain: string = "base"): Promise<Wallet> {
    return this.request<Wallet>("GET", "/v1/wallet", {
      params: { chain },
    });
  }

  /** List all supported blockchain networks. */
  async listChains(): Promise<Chain[]> {
    const data = await this.request<{ chains: Chain[] }>("GET", "/v1/chains");
    return data.chains ?? [];
  }

  // -----------------------------------------------------------------
  // Transactions
  // -----------------------------------------------------------------

  /** Spend funds from the agent balance. */
  async spend(
    amount: number,
    description: string,
    idempotencyKey?: string,
  ): Promise<SpendResponse> {
    const body: Record<string, unknown> = { amount, description };
    if (idempotencyKey !== undefined) body.idempotency_key = idempotencyKey;
    return this.request<SpendResponse>("POST", "/v1/spend", { body });
  }

  /** Refund a previous transaction. */
  async refund(transactionId: string): Promise<RefundResponse> {
    return this.request<RefundResponse>("POST", "/v1/refund", {
      body: { transaction_id: transactionId },
    });
  }

  /** Transfer funds to another agent. */
  async transfer(
    toAgentId: string,
    amount: number,
    description?: string,
  ): Promise<TransferResponse> {
    const body: Record<string, unknown> = {
      to_agent_id: toAgentId,
      amount,
    };
    if (description !== undefined) body.description = description;
    return this.request<TransferResponse>("POST", "/v1/transfer", { body });
  }

  /** Retrieve recent transactions. */
  async getTransactions(limit: number = 20): Promise<Transaction[]> {
    return this.request<Transaction[]>("GET", "/v1/transactions", {
      params: { limit },
    });
  }

  // -----------------------------------------------------------------
  // Webhooks
  // -----------------------------------------------------------------

  /** Register a webhook endpoint. */
  async registerWebhook(
    url: string,
    events?: string[],
  ): Promise<WebhookInfo> {
    const body: Record<string, unknown> = { url };
    if (events !== undefined) body.events = events;
    return this.request<WebhookInfo>("POST", "/v1/webhook", { body });
  }

  // -----------------------------------------------------------------
  // x402 Protocol
  // -----------------------------------------------------------------

  /** Pay for an x402-gated resource using the agent's wallet. */
  async x402Pay(url: string, maxAmount?: number): Promise<X402Response> {
    const body: Record<string, unknown> = { url };
    if (maxAmount !== undefined) body.max_price_usd = maxAmount;
    return this.request<X402Response>("POST", "/v1/x402/pay", { body });
  }

  // -----------------------------------------------------------------
  // Identity
  // -----------------------------------------------------------------

  /** Get the agent's identity profile (KYA). */
  async getIdentity(): Promise<AgentIdentity> {
    return this.request<AgentIdentity>("GET", "/v1/agent/identity");
  }

  /** Get the agent's trust score breakdown. */
  async getTrustScore(): Promise<TrustScore> {
    return this.request<TrustScore>("GET", "/v1/agent/identity/score");
  }
}
