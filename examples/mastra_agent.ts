/**
 * AgentPay + Mastra Agent Example
 *
 * Mastra is a TypeScript-first AI agent framework with built-in
 * tool registration, workflows, and observability.
 *
 * Install: npm install mastra @mastra/core agentpay
 */

import { Agent, createTool } from "@mastra/core";
import { z } from "zod";
import { AgentPay } from "agentpay";

const ap = new AgentPay({
  apiKey: process.env.AGENTPAY_API_KEY!,
  baseUrl: process.env.AGENTPAY_BASE_URL || "https://leofundmybot.dev",
});

const AGENT_ID = process.env.AGENTPAY_AGENT_ID || "mastra-agent";

// --- AgentPay tools for Mastra ---

const checkBudget = createTool({
  id: "check-budget",
  description: "Check the agent's current balance and spending limits",
  inputSchema: z.object({}),
  execute: async () => {
    const wallet = await ap.getWallet(AGENT_ID);
    return {
      balance: wallet.balance,
      currency: wallet.currency,
      dailyLimit: wallet.daily_limit,
      dailySpent: wallet.daily_spent,
    };
  },
});

const spendBudget = createTool({
  id: "spend-budget",
  description: "Spend from the agent's budget for a task",
  inputSchema: z.object({
    amount: z.number().positive().describe("Amount to spend in cents"),
    description: z.string().describe("What the spend is for"),
    metadata: z.record(z.string()).optional(),
  }),
  execute: async ({ context }) => {
    const tx = await ap.spend({
      agentId: AGENT_ID,
      amount: context.amount,
      description: context.description,
      metadata: context.metadata,
    });
    return {
      transactionId: tx.id,
      newBalance: tx.balance_after,
      status: tx.status,
    };
  },
});

const getTransactions = createTool({
  id: "get-transactions",
  description: "View recent spending history",
  inputSchema: z.object({
    limit: z.number().min(1).max(50).default(10),
  }),
  execute: async ({ context }) => {
    const txs = await ap.getTransactions(AGENT_ID, { limit: context.limit });
    return txs.map((tx) => ({
      id: tx.id,
      amount: tx.amount,
      description: tx.description,
      createdAt: tx.created_at,
    }));
  },
});

// --- Mastra Agent ---

const agent = new Agent({
  name: "Budget-Aware Assistant",
  instructions: `You are a helpful assistant with a spending budget managed via AgentPay.
Before performing any paid action, check your budget with check-budget.
After spending, confirm the transaction details to the user.
If the budget is too low, inform the user and suggest requesting a top-up.`,
  model: {
    provider: "OPEN_AI",
    name: "gpt-4o-mini",
  },
  tools: { checkBudget, spendBudget, getTransactions },
});

// --- Run ---

async function main() {
  const result = await agent.generate(
    "Check my budget, then spend $0.50 on a research task about AI agent payments."
  );
  console.log("Agent response:", result.text);
}

main().catch(console.error);
