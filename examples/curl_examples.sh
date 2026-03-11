#!/usr/bin/env bash
# AgentPay — cURL Examples
#
# Quick-start for any language. Replace ap_your_key_here with your real key.
# Get a key: https://t.me/FundmyAIbot → /newagent my-bot
#
# Base URL: https://leofundmybot.dev (or your self-hosted instance)

API_KEY="ap_your_key_here"
BASE="https://leofundmybot.dev"

# ─── Check balance ──────────────────────────────────────────────
echo "=== Balance ==="
curl -s "$BASE/api/balance" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# ─── Spend money ────────────────────────────────────────────────
echo -e "\n=== Spend ==="
curl -s -X POST "$BASE/api/spend" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 2.50,
    "description": "GPT-4 API call — summarize document",
    "idempotency_key": "doc-summary-001"
  }' | python3 -m json.tool

# ─── List transactions ──────────────────────────────────────────
echo -e "\n=== Transactions ==="
curl -s "$BASE/api/transactions?limit=5" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# ─── Transfer to another agent ──────────────────────────────────
echo -e "\n=== Transfer ==="
curl -s -X POST "$BASE/api/transfer" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "to_agent": "agent_recipient_id",
    "amount": 5.00,
    "description": "Payment for translation service"
  }' | python3 -m json.tool

# ─── Refund a transaction ───────────────────────────────────────
echo -e "\n=== Refund ==="
curl -s -X POST "$BASE/api/refund" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_abc123"
  }' | python3 -m json.tool

# ─── Create a webhook ──────────────────────────────────────────
echo -e "\n=== Create Webhook ==="
curl -s -X POST "$BASE/api/webhooks" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-server.com/hook",
    "events": ["transaction.completed", "balance.low"]
  }' | python3 -m json.tool

# ─── Get wallet addresses ──────────────────────────────────────
echo -e "\n=== Wallets ==="
curl -s "$BASE/api/wallets" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# ─── Agent identity / trust score ──────────────────────────────
echo -e "\n=== Trust Score ==="
curl -s "$BASE/api/identity" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# ─── x402 probe (check price of a gated resource) ──────────────
echo -e "\n=== x402 Probe ==="
curl -s "$BASE/api/v1/x402/probe?url=https://api.example.com/premium-data" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# ─── Export transactions as CSV ─────────────────────────────────
echo -e "\n=== CSV Export ==="
curl -s "$BASE/api/export/csv" \
  -H "Authorization: Bearer $API_KEY" \
  -o transactions.csv
echo "Saved to transactions.csv"
