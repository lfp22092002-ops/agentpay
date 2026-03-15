/**
 * AgentPay — Vercel AI SDK Agent Integration
 *
 * Give your Vercel AI SDK agent autonomous payment capabilities.
 * Vercel AI SDK is the leading TypeScript agent framework (2M+ npm downloads/month).
 *
 * Requirements:
 *   npm install ai agentpay @ai-sdk/openai
 */

import { generateText, tool } from "ai";
import { openai } from "@ai-sdk/openai";
import { AgentPayClient } from "agentpay";
import { z } from "zod";

const client = new AgentPayClient(
  process.env.AGENTPAY_API_KEY ?? "ap_your_key_here",
);

// ─── AgentPay Tools ─────────────────────────────────

const checkBalance = tool({
  description:
    "Check the agent's current wallet balance, spending limits, and usage.",
  parameters: z.object({}),
  execute: async () => {
    const b = await client.getBalance();
    return {
      balance_usd: b.balance_usd,
      daily_limit_usd: b.daily_limit_usd,
      daily_remaining_usd: b.daily_remaining_usd,
    };
  },
});

const spend = tool({
  description: "Spend money from the agent's wallet for a specific purpose.",
  parameters: z.object({
    amount: z.number().positive().describe("Amount in USD to spend"),
    description: z.string().describe("What the money is being spent on"),
  }),
  execute: async ({ amount, description }) => {
    const tx = await client.spend(amount, description);
    return {
      id: tx.id,
      amount: tx.amount,
      description: tx.description,
      remaining_balance: tx.remaining_balance,
    };
  },
});

const transfer = tool({
  description: "Transfer funds to another agent.",
  parameters: z.object({
    to_agent_id: z.string().describe("Target agent ID"),
    amount: z.number().positive().describe("Amount in USD"),
    description: z.string().optional().describe("Transfer description"),
  }),
  execute: async ({ to_agent_id, amount, description }) => {
    const tx = await client.transfer(to_agent_id, amount, description);
    return {
      id: tx.id,
      amount: tx.amount,
      to_agent_id: tx.to_agent_id,
    };
  },
});

const getTransactions = tool({
  description: "List recent transactions with optional limit.",
  parameters: z.object({
    limit: z
      .number()
      .int()
      .min(1)
      .max(100)
      .default(5)
      .describe("Number of transactions to return"),
  }),
  execute: async ({ limit }) => {
    const txs = await client.getTransactions(limit);
    return txs.map((tx) => ({
      id: tx.id,
      type: tx.type,
      amount: tx.amount,
      description: tx.description,
      created_at: tx.created_at,
    }));
  },
});

const x402Pay = tool({
  description:
    "Pay for an x402-gated API resource using the agent's on-chain wallet.",
  parameters: z.object({
    url: z.string().url().describe("The x402-gated resource URL"),
    max_amount: z
      .number()
      .positive()
      .default(1.0)
      .describe("Maximum price in USD"),
  }),
  execute: async ({ url, max_amount }) => {
    const result = await client.x402Pay(url, max_amount);
    return {
      url,
      amount: result.amount,
      tx_hash: result.tx_hash,
      data: result.data,
    };
  },
});

// ─── Agent ──────────────────────────────────────────

async function main() {
  const result = await generateText({
    model: openai("gpt-4o"),
    system:
      "You are a helpful assistant with a budget. Check your balance before " +
      "spending. Never exceed your daily limit. Report transactions clearly.",
    prompt: "Check my balance, then spend $0.10 on a test transaction.",
    tools: {
      checkBalance,
      spend,
      transfer,
      getTransactions,
      x402Pay,
    },
    maxSteps: 5,
  });

  console.log(result.text);
  console.log(`\nSteps: ${result.steps.length}`);
  console.log(
    `Tokens: ${result.usage.promptTokens} prompt + ${result.usage.completionTokens} completion`,
  );
}

main().catch(console.error);
