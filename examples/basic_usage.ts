/**
 * AgentPay — TypeScript Basic Usage
 *
 * Shows balance check, spend, transactions, and error handling with the TS SDK.
 *
 * Requirements:
 *   npm install agentpay
 */

import { AgentPayClient } from "agentpay";

const API_KEY = "ap_your_key_here";
const BASE_URL = "https://leofundmybot.dev";

async function main() {
  const client = new AgentPayClient(API_KEY, { baseUrl: BASE_URL });

  // 1. Check balance
  const balance = await client.getBalance();
  console.log(`Balance: $${balance.balance_usd}`);
  console.log(
    `Daily: $${balance.daily_spent_usd} / $${balance.daily_limit_usd}`
  );

  // 2. Spend
  try {
    const tx = await client.spend(1.5, "Image generation — DALL-E 3", "img-gen-001");
    console.log(`Spent $${tx.amount} — TX: ${tx.transaction_id}`);
    console.log(`Remaining: $${tx.remaining_balance}`);
  } catch (err) {
    console.error(`Spend failed: ${err}`);
  }

  // 3. List transactions
  const txs = await client.getTransactions(5);
  for (const tx of txs) {
    const icon = tx.type === "deposit" ? "↙" : "↗";
    console.log(`  ${icon} $${tx.amount} — ${tx.description}`);
  }

  // 4. Refund
  try {
    const refund = await client.refund("some-tx-id");
    console.log(`Refunded: $${refund.amount_refunded}`);
  } catch (err) {
    console.error(`Refund failed: ${err}`);
  }
}

main().catch(console.error);
