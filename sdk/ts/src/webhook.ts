import { createHmac, timingSafeEqual } from "node:crypto";

/**
 * Verify an AgentPay webhook signature (HMAC-SHA256).
 *
 * @param payload   Raw request body (string or Buffer).
 * @param signature Value of the `X-AgentPay-Signature` header.
 * @param secret    Your webhook signing secret.
 * @returns `true` if the signature is valid.
 *
 * @example
 * ```ts
 * import { verifyWebhook } from "agentpay";
 *
 * app.post("/webhook", (req, res) => {
 *   const valid = verifyWebhook(
 *     req.body,                            // raw body
 *     req.headers["x-agentpay-signature"], // signature header
 *     process.env.AGENTPAY_WEBHOOK_SECRET! // your secret
 *   );
 *   if (!valid) return res.status(401).send("Invalid signature");
 *   // handle event...
 * });
 * ```
 */
export function verifyWebhook(
  payload: string | Buffer,
  signature: string | undefined | null,
  secret: string,
): boolean {
  if (!signature) return false;

  const expected = createHmac("sha256", secret)
    .update(typeof payload === "string" ? payload : payload)
    .digest("hex");

  // Prefix may be "sha256=" — strip it
  const sig = signature.startsWith("sha256=") ? signature.slice(7) : signature;

  if (sig.length !== expected.length) return false;

  return timingSafeEqual(Buffer.from(sig, "hex"), Buffer.from(expected, "hex"));
}
