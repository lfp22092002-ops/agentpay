# Demo Video Script — AgentPay (90-120 seconds)

## Opening (0:00-0:10)
**Visual**: Dark terminal background + AgentPay logo fade in  
**Narration**: "Your AI agents can code, browse, and deploy. But can they spend money? Meet AgentPay."

## Problem (0:10-0:20)
**Visual**: Code snippet where agent hits a paywall, gets 402 error  
**Narration**: "Every time your agent needs to pay for an API, buy tokens, or subscribe to a service — it has to stop and ask you. That breaks the whole point of autonomy."

## Solution — Create Agent (0:20-0:35)
**Visual**: Screen recording of Telegram → @FundmyAIbot → /newagent  
**Narration**: "With AgentPay, open our Telegram bot, create an agent in one command, and get an API key instantly."
**Show**: Bot response with API key (blur the real key)

## Fund (0:35-0:50)
**Visual**: Telegram Stars payment flow (or USDC deposit address)  
**Narration**: "Fund it with Telegram Stars — the simplest payment method — or deposit USDC on any of 4 chains."
**Show**: Stars payment animation, balance update

## Spend via API (0:50-1:10)
**Visual**: Terminal — curl command or Python SDK code  
```python
from agentpay import AgentPayClient
client = AgentPayClient("ap_your_key")
balance = client.get_balance()   # $10.00
tx = client.spend(0.50, "GPT-4 API call")
```
**Narration**: "Your agent spends via API. Check balance, spend, transfer — all programmatic. Set daily limits, per-transaction caps, and approval thresholds."
**Show**: JSON response, then Telegram approval notification

## Security & Control (1:10-1:25)
**Visual**: Split screen — trust score dashboard + approval workflow  
**Narration**: "You stay in control. Budget limits, approval workflows on Telegram, HMAC-signed webhooks, and agent trust scores — all built in."

## Integrations (1:25-1:35)
**Visual**: Logos scrolling — LangChain, CrewAI, OpenAI, Claude, MCP  
**Narration**: "Works with LangChain, CrewAI, AutoGen, OpenAI Agents SDK — or any framework via our MCP server and REST API."

## Closing (1:35-1:45)
**Visual**: Website + GitHub + Telegram bot QR code  
**Narration**: "AgentPay. Open source, free during beta. Give your agent a wallet at leofundmybot.dev."
**Text overlay**: 
- 🤖 @FundmyAIbot
- 🌐 leofundmybot.dev  
- 📦 github.com/lfp22092002-ops/agentpay

## Recording Notes
- Use OBS or ScreenPal for screen recording
- Terminal: dark theme, large font (18pt+)
- Telegram: phone or web app recording
- Background music: subtle, tech/corporate (YouTube Audio Library)
- Export: 1920x1080, 30fps, max 3 min for Product Hunt
- Upload to YouTube (unlisted) + embed on PH listing
