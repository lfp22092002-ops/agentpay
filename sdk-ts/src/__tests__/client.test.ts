import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { AgentPayClient } from "../client.js";
import { AgentPayError, AuthenticationError, InsufficientBalanceError, RateLimitError } from "../errors.js";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(data: unknown, status = 200, headers?: Record<string, string>) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(data),
    headers: new Headers(headers),
  };
}

describe("AgentPayClient", () => {
  let client: AgentPayClient;

  beforeEach(() => {
    client = new AgentPayClient("ap_test_key_123", {
      baseUrl: "https://test.example.com",
      maxRetries: 0, // no retries in tests for speed
    });
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ---- Construction ----

  it("strips trailing slashes from baseUrl", async () => {
    const c = new AgentPayClient("key", { baseUrl: "https://x.com///", maxRetries: 0 });
    mockFetch.mockResolvedValueOnce(jsonResponse({ balance_usd: "0" }));
    await c.getBalance();
    expect(mockFetch).toHaveBeenCalledWith(
      "https://x.com/v1/balance",
      expect.anything(),
    );
  });

  it("uses default base URL when none provided", async () => {
    const c = new AgentPayClient("key", { maxRetries: 0 });
    mockFetch.mockResolvedValueOnce(jsonResponse({ balance_usd: "10" }));
    await c.getBalance();
    expect(mockFetch.mock.calls[0][0]).toBe("https://leofundmybot.dev/v1/balance");
  });

  // ---- Auth header ----

  it("sends X-API-Key header", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ balance_usd: "5" }));
    await client.getBalance();
    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers["X-API-Key"]).toBe("ap_test_key_123");
  });

  // ---- Balance ----

  it("getBalance returns balance", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ balance_usd: "42.50", currency: "USD" }));
    const result = await client.getBalance();
    expect(result.balance_usd).toBe("42.50");
  });

  // ---- Wallet ----

  it("getWallet passes chain param", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ address: "0xabc", chain: "polygon" }));
    await client.getWallet("polygon");
    expect(mockFetch.mock.calls[0][0]).toContain("chain=polygon");
  });

  it("getWallet defaults to base chain", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ address: "0xabc", chain: "base" }));
    await client.getWallet();
    expect(mockFetch.mock.calls[0][0]).toContain("chain=base");
  });

  // ---- Chains ----

  it("listChains returns array", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ chains: [{ id: "base" }, { id: "polygon" }] }));
    const chains = await client.listChains();
    expect(chains).toHaveLength(2);
  });

  it("listChains handles missing chains key", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({}));
    const chains = await client.listChains();
    expect(chains).toEqual([]);
  });

  // ---- Spend ----

  it("spend sends correct body", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ transaction_id: "tx_1", status: "completed" }));
    await client.spend(5.0, "test payment", "idem_123");
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({ amount: 5.0, description: "test payment", idempotency_key: "idem_123" });
  });

  it("spend omits idempotency_key when not provided", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ transaction_id: "tx_2" }));
    await client.spend(1.0, "no key");
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({ amount: 1.0, description: "no key" });
  });

  // ---- Refund ----

  it("refund sends transaction_id", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ status: "refunded" }));
    await client.refund("tx_99");
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.transaction_id).toBe("tx_99");
  });

  // ---- Transfer ----

  it("transfer sends correct body", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ status: "completed" }));
    await client.transfer("agent_2", 10.0, "payment for service");
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({ to_agent_id: "agent_2", amount: 10.0, description: "payment for service" });
  });

  // ---- Transactions ----

  it("getTransactions passes limit", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse([]));
    await client.getTransactions(5);
    expect(mockFetch.mock.calls[0][0]).toContain("limit=5");
  });

  // ---- Webhook ----

  it("registerWebhook sends url and events", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "wh_1" }));
    await client.registerWebhook("https://hook.example.com", ["spend", "refund"]);
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.url).toBe("https://hook.example.com");
    expect(body.events).toEqual(["spend", "refund"]);
  });

  // ---- x402 ----

  it("x402Pay sends url and max amount", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ paid: true }));
    await client.x402Pay("https://paywall.example.com/article", 0.50);
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.url).toBe("https://paywall.example.com/article");
    expect(body.max_price_usd).toBe(0.50);
  });

  // ---- Identity ----

  it("getIdentity calls correct endpoint", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ agent_id: "a_1", name: "TestBot" }));
    const identity = await client.getIdentity();
    expect(mockFetch.mock.calls[0][0]).toContain("/v1/agent/identity");
    expect(identity.agent_id).toBe("a_1");
  });

  it("getTrustScore calls correct endpoint", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ score: 85 }));
    const score = await client.getTrustScore();
    expect(mockFetch.mock.calls[0][0]).toContain("/v1/agent/identity/score");
    expect(score.score).toBe(85);
  });

  // ---- Error handling ----

  it("throws AuthenticationError on 401", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ detail: "Invalid API key" }, 401));
    await expect(client.getBalance()).rejects.toThrow(AuthenticationError);
  });

  it("throws InsufficientBalanceError on 402 with balance message", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ detail: "Insufficient balance" }, 402));
    await expect(client.spend(999, "too much")).rejects.toThrow(InsufficientBalanceError);
  });

  it("throws RateLimitError on 429 after retries exhausted", async () => {
    const c = new AgentPayClient("key", { baseUrl: "https://test.example.com", maxRetries: 1 });
    mockFetch
      .mockResolvedValueOnce(jsonResponse({ detail: "Too many requests" }, 429, { "Retry-After": "0.01" }))
      .mockResolvedValueOnce(jsonResponse({ detail: "Too many requests" }, 429));
    await expect(c.getBalance()).rejects.toThrow(RateLimitError);
  });

  it("throws AgentPayError on unknown 4xx", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ detail: "Not found" }, 404));
    await expect(client.getBalance()).rejects.toThrow(AgentPayError);
  });

  it("throws on network error", async () => {
    mockFetch.mockRejectedValueOnce(new Error("ECONNREFUSED"));
    await expect(client.getBalance()).rejects.toThrow(AgentPayError);
  });

  // ---- Retries ----

  it("retries on 500 and succeeds", async () => {
    const c = new AgentPayClient("key", { baseUrl: "https://test.example.com", maxRetries: 1 });
    mockFetch
      .mockResolvedValueOnce(jsonResponse({ detail: "Internal error" }, 500))
      .mockResolvedValueOnce(jsonResponse({ balance_usd: "10" }));
    const result = await c.getBalance();
    expect(result.balance_usd).toBe("10");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("retries on network failure and succeeds", async () => {
    const c = new AgentPayClient("key", { baseUrl: "https://test.example.com", maxRetries: 1 });
    mockFetch
      .mockRejectedValueOnce(new Error("timeout"))
      .mockResolvedValueOnce(jsonResponse({ balance_usd: "5" }));
    const result = await c.getBalance();
    expect(result.balance_usd).toBe("5");
  });
});
