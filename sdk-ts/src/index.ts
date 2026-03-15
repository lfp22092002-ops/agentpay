/**
 * AgentPay TypeScript SDK
 *
 * Payment infrastructure for AI agents.
 *
 * @example
 * ```ts
 * import { AgentPayClient } from "agentpay";
 *
 * const client = new AgentPayClient("ap_your_api_key");
 *
 * // Check balance
 * const balance = await client.getBalance();
 * console.log(`Balance: $${balance.balance_usd}`);
 *
 * // Spend
 * const tx = await client.spend(0.50, "GPT-4 API call");
 * console.log(`Spent $${tx.amount}, remaining $${tx.remaining_balance}`);
 *
 * // Pay for x402-gated resource
 * const result = await client.x402Pay("https://api.example.com/premium");
 * ```
 *
 * @packageDocumentation
 */

export { AgentPayClient } from "./client.js";
export {
  AgentPayError,
  AuthenticationError,
  InsufficientBalanceError,
  RateLimitError,
} from "./errors.js";
export type {
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
