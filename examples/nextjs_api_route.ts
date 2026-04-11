// examples/nextjs_api_route.ts
//
// Next.js App Router API route that gates an AI endpoint behind AgentPay.
// Works with Next.js 13+ (app directory).
//
// Place this file at: app/api/ask/route.ts
//
// Usage:
//   POST /api/ask
//   Headers: { "X-Agent-Key": "<agent_api_key>" }
//   Body:    { "question": "What is the meaning of life?" }

import { NextRequest, NextResponse } from "next/server";
import { AgentPay } from "agentpay";

const ap = new AgentPay({ apiKey: process.env.AGENTPAY_API_KEY! });

export async function POST(req: NextRequest) {
  const agentKey = req.headers.get("x-agent-key");
  if (!agentKey) {
    return NextResponse.json({ error: "Missing X-Agent-Key" }, { status: 401 });
  }

  const { question } = await req.json();
  if (!question) {
    return NextResponse.json({ error: "Missing question" }, { status: 400 });
  }

  // 1. Charge the agent before doing expensive work
  let spend;
  try {
    spend = await ap.spend(0.01, `ask: ${question.slice(0, 80)}`);
  } catch (err: any) {
    if (err.statusCode === 402) {
      return NextResponse.json({ error: "Insufficient balance" }, { status: 402 });
    }
    return NextResponse.json({ error: "Payment failed" }, { status: 500 });
  }

  // 2. Do the expensive work (call your LLM, DB, etc.)
  const answer = `You asked: "${question}" — this is where your LLM response goes.`;

  // 3. Return result with transaction receipt
  return NextResponse.json({
    answer,
    transaction_id: spend.transaction_id,
    charged: spend.amount,
    remaining: spend.remaining_balance,
  });
}
