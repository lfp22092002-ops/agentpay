# Examples

Ready-to-run examples showing how to integrate AgentPay into your AI agents.

## Python

| Example | Description |
|---------|-------------|
| [basic_usage.py](basic_usage.py) | Balance check, spend, transactions, refund (sync SDK) |
| [openai_agent.py](openai_agent.py) | OpenAI GPT agent with budget control |
| [webhook_receiver.py](webhook_receiver.py) | FastAPI webhook receiver with HMAC verification |
| [x402_agent.py](x402_agent.py) | Autonomous x402 micropayments — agent pays for gated APIs |
| [multi_agent_budget.py](multi_agent_budget.py) | Fleet management — multiple agents with shared budgets |

## cURL / Any Language

| Example | Description |
|---------|-------------|
| [curl_examples.sh](curl_examples.sh) | All key endpoints — balance, spend, transfer, webhooks, x402 |

## TypeScript / Node.js

| Example | Description |
|---------|-------------|
| [basic_usage.ts](basic_usage.ts) | Balance, spend, transactions, refund (TS SDK) |

## Setup

1. Create an agent: message [@FundmyAIbot](https://t.me/FundmyAIbot) → `/newagent my-bot`
2. Copy your API key (shown once)
3. Fund with Telegram Stars or crypto
4. Replace `ap_your_key_here` in examples with your real key

## Install the SDK

```bash
# Python
pip install agentpay

# TypeScript / Node.js
npm install agentpay
```

## Self-Hosted

All examples default to `https://leofundmybot.dev`. For self-hosted instances, change `BASE_URL` to your server address.
