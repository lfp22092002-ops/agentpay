import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { AgentPayClient } from "../src/client.js";
import {
  AgentPayError,
  AuthenticationError,
  InsufficientBalanceError,
  RateLimitError,
} from "../src/errors.js";

// ═══════════════════════════════════════
// Mock fetch
// ═══════════════════════════════════════

function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: () => Promise.resolve(body),
  } as unknown as Response);
}

describe("AgentPayClient", () => {
  let client: AgentPayClient;
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    client = new AgentPayClient("ap_test_key_123", {
      baseUrl: "https://test.local",
      timeoutMs: 5000,
    });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  // ─────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────

  it("strips trailing slashes from baseUrl", async () => {
    const c = new AgentPayClient("key", { baseUrl: "https://x.com///" });
    globalThis.fetch = mockFetch(200, { balance_usd: 0 });
    await c.getBalance();
    const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toBe("https://x.com/v1/balance");
  });

  it("uses default baseUrl when none provided", async () => {
    const c = new AgentPayClient("key");
    globalThis.fetch = mockFetch(200, { balance_usd: 0 });
    await c.getBalance();
    const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toContain("leofundmybot.dev");
  });

  // ─────────────────────────────────────
  // getBalance
  // ─────────────────────────────────────

  it("getBalance returns balance object", async () => {
    const body = {
      agent_id: "a1",
      agent_name: "bot",
      balance_usd: 42.5,
      daily_limit_usd: 100,
      daily_spent_usd: 10,
      daily_remaining_usd: 90,
      tx_limit_usd: 50,
      is_active: true,
    };
    globalThis.fetch = mockFetch(200, body);
    const bal = await client.getBalance();
    expect(bal.balance_usd).toBe(42.5);
    expect(bal.is_active).toBe(true);
    const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(call[1].headers["X-API-Key"]).toBe("ap_test_key_123");
  });

  // ─────────────────────────────────────
  // getWallet
  // ─────────────────────────────────────

  it("getWallet passes chain as query param", async () => {
    globalThis.fetch = mockFetch(200, { address: "0xABC", chain: "polygon" });
    const w = await client.getWallet("polygon");
    expect(w.address).toBe("0xABC");
    const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toContain("chain=polygon");
  });

  it("getWallet defaults to base", async () => {
    globalThis.fetch = mockFetch(200, { address: "0xDEF" });
    await client.getWallet();
    const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toContain("chain=base");
  });

  // ─────────────────────────────────────
  // listChains
  // ─────────────────────────────────────

  it("listChains unwraps the chains array", async () => {
    globalThis.fetch = mockFetch(200, {
      chains: [
        { id: "base", name: "Base", type: "evm", native_token: "ETH", usdc_supported: true, explorer: "" },
        { id: "solana", name: "Solana", type: "svm", native_token: "SOL", usdc_supported: true, explorer: "" },
      ],
    });
    const chains = await client.listChains();
    expect(chains).toHaveLength(2);
    expect(chains[0].id).toBe("base");
  });

  // ─────────────────────────────────────
  // spend
  // ─────────────────────────────────────

  it("spend sends correct JSON body", async () => {
    const body = { success: true, transaction_id: "tx1", amount: 1.5, fee: 0.03, remaining_balance: 40, status: "completed" };
    globalThis.fetch = mockFetch(200, body);
    const res = await client.spend(1.5, "GPT-4 call", "idem-1");
    expect(res.success).toBe(true);
    const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const sentBody = JSON.parse(call[1].body);
    expect(sentBody.amount).toBe(1.5);
    expect(sentBody.description).toBe("GPT-4 call");
    expect(sentBody.idempotency_key).toBe("idem-1");
  });

  it("spend omits idempotency_key when not provided", async () => {
    globalThis.fetch = mockFetch(200, { success: true, amount: 1, fee: 0, remaining_balance: 10, status: "completed" });
    await client.spend(1, "test");
    const sentBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(sentBody.idempotency_key).toBeUndefined();
  });

  // ─────────────────────────────────────
  // refund
  // ─────────────────────────────────────

  it("refund sends transaction_id", async () => {
    globalThis.fetch = mockFetch(200, { success: true, refund_transaction_id: "r1", amount_refunded: 5, new_balance: 50 });
    const res = await client.refund("tx_original");
    expect(res.amount_refunded).toBe(5);
    const sentBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(sentBody.transaction_id).toBe("tx_original");
  });

  // ─────────────────────────────────────
  // transfer
  // ─────────────────────────────────────

  it("transfer sends to_agent_id and amount", async () => {
    globalThis.fetch = mockFetch(200, { success: true, transaction_id: "t1", amount: 10, from_balance: 90 });
    const res = await client.transfer("agent_xyz", 10, "Payment for data");
    expect(res.success).toBe(true);
    const sentBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(sentBody.to_agent_id).toBe("agent_xyz");
    expect(sentBody.amount).toBe(10);
    expect(sentBody.description).toBe("Payment for data");
  });

  it("transfer omits description when not provided", async () => {
    globalThis.fetch = mockFetch(200, { success: true, amount: 5, from_balance: 95 });
    await client.transfer("a2", 5);
    const sentBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(sentBody.description).toBeUndefined();
  });

  // ─────────────────────────────────────
  // getTransactions
  // ─────────────────────────────────────

  it("getTransactions passes limit param", async () => {
    globalThis.fetch = mockFetch(200, [{ id: "tx1" }]);
    const txs = await client.getTransactions(5);
    expect(txs).toHaveLength(1);
    const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toContain("limit=5");
  });

  it("getTransactions defaults to limit=20", async () => {
    globalThis.fetch = mockFetch(200, []);
    await client.getTransactions();
    const url = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toContain("limit=20");
  });

  // ─────────────────────────────────────
  // registerWebhook
  // ─────────────────────────────────────

  it("registerWebhook sends url and events", async () => {
    globalThis.fetch = mockFetch(200, { url: "https://x.com/hook", secret: "s1", events: ["spend"] });
    const wh = await client.registerWebhook("https://x.com/hook", ["spend"]);
    expect(wh.secret).toBe("s1");
    const sentBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(sentBody.url).toBe("https://x.com/hook");
    expect(sentBody.events).toEqual(["spend"]);
  });

  it("registerWebhook omits events when not provided", async () => {
    globalThis.fetch = mockFetch(200, { url: "https://x.com/hook" });
    await client.registerWebhook("https://x.com/hook");
    const sentBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(sentBody.events).toBeUndefined();
  });

  // ─────────────────────────────────────
  // x402Pay
  // ─────────────────────────────────────

  it("x402Pay sends url and maxAmount", async () => {
    globalThis.fetch = mockFetch(200, { success: true, paid_usd: 0.5, data: "premium content" });
    const res = await client.x402Pay("https://api.ex.com/data", 2.0);
    expect(res.paid_usd).toBe(0.5);
    const sentBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(sentBody.url).toBe("https://api.ex.com/data");
    expect(sentBody.max_price_usd).toBe(2.0);
  });

  it("x402Pay omits max_price_usd when not provided", async () => {
    globalThis.fetch = mockFetch(200, { success: true, paid_usd: 0.1 });
    await client.x402Pay("https://api.ex.com/free");
    const sentBody = JSON.parse((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body);
    expect(sentBody.max_price_usd).toBeUndefined();
  });

  // ─────────────────────────────────────
  // Identity / KYA
  // ─────────────────────────────────────

  it("getIdentity returns identity object", async () => {
    globalThis.fetch = mockFetch(200, { agent_id: "a1", display_name: "Bot", verified: true, trust_score: 75 });
    const id = await client.getIdentity();
    expect(id.display_name).toBe("Bot");
    expect(id.trust_score).toBe(75);
  });

  it("getTrustScore returns breakdown", async () => {
    globalThis.fetch = mockFetch(200, { total: 60, account_age_pts: 10, account_age_max: 20, transaction_count_pts: 15, transaction_count_max: 25, volume_pts: 10, volume_max: 20, profile_completeness_pts: 15, profile_completeness_max: 20, verified_pts: 10, verified_max: 15, details: {} });
    const score = await client.getTrustScore();
    expect(score.total).toBe(60);
    expect(score.account_age_pts).toBe(10);
  });

  // ─────────────────────────────────────
  // Error handling
  // ─────────────────────────────────────

  it("throws AuthenticationError on 401", async () => {
    globalThis.fetch = mockFetch(401, { detail: "Invalid API key" });
    await expect(client.getBalance()).rejects.toThrow(AuthenticationError);
  });

  it("throws RateLimitError on 429", async () => {
    globalThis.fetch = mockFetch(429, { detail: "Too many requests" });
    await expect(client.spend(1, "test")).rejects.toThrow(RateLimitError);
  });

  it("throws InsufficientBalanceError on 402", async () => {
    globalThis.fetch = mockFetch(402, { detail: "Insufficient balance" });
    await expect(client.spend(999, "big spend")).rejects.toThrow(InsufficientBalanceError);
  });

  it("throws InsufficientBalanceError on 400 with balance message", async () => {
    globalThis.fetch = mockFetch(400, { detail: "Insufficient balance for this transaction" });
    await expect(client.spend(999, "big spend")).rejects.toThrow(InsufficientBalanceError);
  });

  it("throws generic AgentPayError on other errors", async () => {
    globalThis.fetch = mockFetch(500, { detail: "Internal server error" });
    await expect(client.getBalance()).rejects.toThrow(AgentPayError);
    try {
      globalThis.fetch = mockFetch(500, { detail: "Server down" });
      await client.getBalance();
    } catch (e) {
      expect(e).toBeInstanceOf(AgentPayError);
      expect((e as AgentPayError).statusCode).toBe(500);
    }
  });

  it("handles non-JSON error body gracefully", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      json: () => Promise.reject(new Error("not json")),
    } as unknown as Response);
    await expect(client.getBalance()).rejects.toThrow(AgentPayError);
  });
});

// ═══════════════════════════════════════
// Error classes
// ═══════════════════════════════════════

describe("Error classes", () => {
  it("AgentPayError has correct properties", () => {
    const err = new AgentPayError(404, "Not found");
    expect(err.name).toBe("AgentPayError");
    expect(err.statusCode).toBe(404);
    expect(err.detail).toBe("Not found");
    expect(err.message).toContain("404");
    expect(err).toBeInstanceOf(Error);
  });

  it("AuthenticationError extends AgentPayError", () => {
    const err = new AuthenticationError("Bad key");
    expect(err).toBeInstanceOf(AgentPayError);
    expect(err.name).toBe("AuthenticationError");
    expect(err.statusCode).toBe(401);
  });

  it("InsufficientBalanceError extends AgentPayError", () => {
    const err = new InsufficientBalanceError("No funds");
    expect(err).toBeInstanceOf(AgentPayError);
    expect(err.statusCode).toBe(402);
  });

  it("RateLimitError extends AgentPayError", () => {
    const err = new RateLimitError("Slow down");
    expect(err).toBeInstanceOf(AgentPayError);
    expect(err.statusCode).toBe(429);
  });
});
