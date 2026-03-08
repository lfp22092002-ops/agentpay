# Examples

Ready-to-run examples showing how to integrate AgentPay into your AI agents.

## Python

| Example | Description |
|---------|-------------|
| [basic_usage.py](basic_usage.py) | Balance check, spend, transactions, refund |
| [openai_agent.py](openai_agent.py) | OpenAI GPT agent with budget control |
| [webhook_receiver.py](webhook_receiver.py) | FastAPI webhook receiver with HMAC verification |

## TypeScript / Node.js

| Example | Description |
|---------|-------------|
| [basic_usage.ts](basic_usage.ts) | Balance, spend, transactions using raw `fetch` |

## Setup

1. Create an agent: message [@FundmyAIbot](https://t.me/FundmyAIbot) → `/newagent my-bot`
2. Copy your API key (shown once)
3. Fund with Telegram Stars or crypto
4. Replace `ap_your_key_here` in examples with your real key

## Self-Hosted

All examples default to `https://leofundmybot.dev`. For self-hosted instances, change `BASE_URL` to your server address.
