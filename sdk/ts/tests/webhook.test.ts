import { describe, it, expect } from "vitest";
import { createHmac } from "node:crypto";
import { verifyWebhook } from "../src/webhook.js";

describe("verifyWebhook", () => {
  const secret = "whsec_test_secret_123";
  const payload = '{"event":"payment.completed","amount":"1.50"}';

  function sign(body: string, key: string): string {
    return createHmac("sha256", key).update(body).digest("hex");
  }

  it("returns true for valid signature", () => {
    const sig = sign(payload, secret);
    expect(verifyWebhook(payload, sig, secret)).toBe(true);
  });

  it("returns true with sha256= prefix", () => {
    const sig = "sha256=" + sign(payload, secret);
    expect(verifyWebhook(payload, sig, secret)).toBe(true);
  });

  it("returns false for wrong signature", () => {
    expect(verifyWebhook(payload, "deadbeef".repeat(8), secret)).toBe(false);
  });

  it("returns false for null/undefined signature", () => {
    expect(verifyWebhook(payload, null, secret)).toBe(false);
    expect(verifyWebhook(payload, undefined, secret)).toBe(false);
  });

  it("returns false for wrong secret", () => {
    const sig = sign(payload, "wrong_secret");
    expect(verifyWebhook(payload, sig, secret)).toBe(false);
  });

  it("returns false for tampered payload", () => {
    const sig = sign(payload, secret);
    expect(verifyWebhook(payload + "tampered", sig, secret)).toBe(false);
  });

  it("works with Buffer payload", () => {
    const buf = Buffer.from(payload);
    const sig = sign(payload, secret);
    expect(verifyWebhook(buf, sig, secret)).toBe(true);
  });
});
