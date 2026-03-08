/**
 * AgentPay — TypeScript/Node.js Basic Usage
 *
 * Shows balance check, spend, and error handling with the TS SDK.
 *
 * Requirements:
 *   npm install agentpay
 *   # or use raw fetch (SDK uses native fetch, Node 18+)
 */

const API_KEY = "ap_your_key_here";
const BASE_URL = "https://leofundmybot.dev";

async function main() {
  // Using raw fetch (no SDK dependency needed)
  const headers = {
    Authorization: `Bearer ${API_KEY}`,
    "Content-Type": "application/json",
  };

  // 1. Check balance
  const balanceRes = await fetch(`${BASE_URL}/v1/balance`, { headers });
  const balance = await balanceRes.json();
  console.log(`Balance: $${balance.balance_usd}`);
  console.log(
    `Daily: $${balance.daily_spent_usd} / $${balance.daily_limit_usd}`
  );

  // 2. Spend
  const spendRes = await fetch(`${BASE_URL}/v1/spend`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      amount: 1.5,
      description: "Image generation — DALL-E 3",
      idempotency_key: "img-gen-001",
    }),
  });

  if (spendRes.ok) {
    const tx = await spendRes.json();
    console.log(`Spent $${tx.amount_usd} — TX: ${tx.id}`);
  } else {
    const err = await spendRes.json();
    console.error(`Spend failed: ${err.detail}`);
  }

  // 3. List transactions
  const txRes = await fetch(`${BASE_URL}/v1/transactions?limit=5`, {
    headers,
  });
  const txs = await txRes.json();
  for (const tx of txs) {
    const icon = tx.tx_type === "deposit" ? "↙" : "↗";
    console.log(`  ${icon} $${tx.amount_usd} — ${tx.description}`);
  }
}

main().catch(console.error);
