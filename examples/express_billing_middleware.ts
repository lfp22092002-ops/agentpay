/**
 * Express.js Billing Middleware — AgentPay TypeScript SDK
 *
 * Metered API access: charge per-request or per-token before processing.
 * Drop this middleware into any Express app to monetise your API.
 *
 * Usage:
 *   npm install express agentpay
 *   AGENTPAY_API_KEY=ap_... npx tsx examples/express_billing_middleware.ts
 */

import express, { Request, Response, NextFunction } from "express";
import { AgentPayClient } from "agentpay";

const app = express();
app.use(express.json());

const agentpay = new AgentPayClient(process.env.AGENTPAY_API_KEY!);

// ---------------------------------------------------------------------------
// Pricing tiers — configure per route or globally
// ---------------------------------------------------------------------------

const PRICING: Record<string, number> = {
  "/v1/summarize": 0.02,   // $0.02 per call
  "/v1/translate": 0.05,   // $0.05 per call
  "/v1/generate":  0.10,   // $0.10 per call
};

const DEFAULT_PRICE = 0.01; // fallback for unlisted routes

// ---------------------------------------------------------------------------
// Billing middleware
// ---------------------------------------------------------------------------

function billingMiddleware(price?: number) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const amount = price ?? PRICING[req.path] ?? DEFAULT_PRICE;
    const idempotencyKey = req.headers["x-idempotency-key"] as string | undefined;

    try {
      const tx = await agentpay.spend(amount, `API call: ${req.method} ${req.path}`, idempotencyKey);

      // Attach transaction info to request for downstream handlers
      (req as any).billing = {
        transactionId: tx.transaction_id,
        amount: tx.amount,
        remaining: tx.remaining_balance,
      };

      // Add billing headers to response
      res.setHeader("X-AgentPay-Transaction", tx.transaction_id ?? "");
      res.setHeader("X-AgentPay-Charged", String(tx.amount));
      res.setHeader("X-AgentPay-Remaining", String(tx.remaining_balance));

      next();
    } catch (err: any) {
      if (err.name === "InsufficientBalanceError") {
        res.status(402).json({ error: "Insufficient balance", detail: err.message });
      } else if (err.name === "AuthenticationError") {
        res.status(401).json({ error: "Invalid API key" });
      } else if (err.name === "RateLimitError") {
        res.status(429).json({ error: "Rate limited", retryAfter: 60 });
      } else {
        res.status(500).json({ error: "Billing error", detail: err.message });
      }
    }
  };
}

// ---------------------------------------------------------------------------
// Example routes
// ---------------------------------------------------------------------------

// Health check (free, no billing)
app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

// Paid endpoints — billing middleware charges before handler runs
app.post("/v1/summarize", billingMiddleware(), (req, res) => {
  const text = req.body.text ?? "";
  res.json({
    summary: `Summary of ${text.length} chars (placeholder)`,
    billing: (req as any).billing,
  });
});

app.post("/v1/translate", billingMiddleware(), (req, res) => {
  res.json({
    translated: `[translated] ${req.body.text ?? ""}`,
    billing: (req as any).billing,
  });
});

app.post("/v1/generate", billingMiddleware(0.10), (req, res) => {
  res.json({
    generated: `Generated content for: ${req.body.prompt ?? ""}`,
    billing: (req as any).billing,
  });
});

// Custom priced endpoint
app.post("/v1/premium", billingMiddleware(1.00), (req, res) => {
  res.json({
    result: "Premium analysis complete",
    billing: (req as any).billing,
  });
});

// Balance check (free)
app.get("/v1/balance", async (_req, res) => {
  try {
    const balance = await agentpay.getBalance();
    res.json(balance);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

const PORT = parseInt(process.env.PORT ?? "3000", 10);
app.listen(PORT, () => {
  console.log(`Billing API running on http://localhost:${PORT}`);
  console.log("Pricing:", PRICING);
});
