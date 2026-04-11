/**
 * AgentPay — TypeScript Webhook Receiver
 *
 * Express server that receives and verifies AgentPay webhooks
 * using the SDK's built-in HMAC-SHA256 signature verification.
 *
 * Requirements:
 *   npm install agentpay express
 */

import express from "express";
import { verifyWebhook } from "agentpay";

const WEBHOOK_SECRET = process.env.AGENTPAY_WEBHOOK_SECRET || "whsec_your_secret";
const PORT = 3000;

const app = express();

// IMPORTANT: use raw body for signature verification
app.use("/webhooks/agentpay", express.raw({ type: "application/json" }));

app.post("/webhooks/agentpay", (req, res) => {
  const signature = req.headers["x-agentpay-signature"] as string;
  const timestamp = req.headers["x-agentpay-timestamp"] as string;
  const body = req.body.toString();

  // Verify signature (timing-safe comparison, replay protection)
  if (!verifyWebhook(body, signature, WEBHOOK_SECRET, timestamp)) {
    console.warn("⚠️  Invalid webhook signature — rejecting");
    return res.status(401).json({ error: "Invalid signature" });
  }

  const event = JSON.parse(body);
  console.log(`✅ Verified event: ${event.type} (${event.id})`);

  // Handle event types
  switch (event.type) {
    case "transaction.completed":
      console.log(`💸 Spent $${event.data.amount} — ${event.data.description}`);
      break;
    case "transaction.refunded":
      console.log(`↩️  Refund $${event.data.amount_refunded}`);
      break;
    case "balance.low":
      console.log(`⚠️  Low balance: $${event.data.balance_usd}`);
      break;
    case "approval.requested":
      console.log(`🔐 Approval needed: $${event.data.amount} — ${event.data.description}`);
      break;
    default:
      console.log(`📨 Unhandled event type: ${event.type}`);
  }

  res.status(200).json({ received: true });
});

app.listen(PORT, () => {
  console.log(`🔔 Webhook receiver listening on port ${PORT}`);
});
