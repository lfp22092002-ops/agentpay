# Installing AgentPay MCP Server

## Option 1: Remote (Recommended)

Add to your MCP config (Claude Desktop, Cursor, Cline, OpenClaw, etc.):

```json
{
  "mcpServers": {
    "agentpay": {
      "url": "https://leofundmybot.dev/mcp",
      "headers": {
        "x-api-key": "YOUR_AGENTPAY_API_KEY"
      }
    }
  }
}
```

## Option 2: Local (stdio)

1. Clone the repo:
```bash
git clone https://github.com/lfp22092002-ops/agentpay.git
cd agentpay
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add to MCP config:
```json
{
  "mcpServers": {
    "agentpay": {
      "command": "python",
      "args": ["mcp/server.py"],
      "env": {
        "AGENTPAY_API_KEY": "YOUR_AGENTPAY_API_KEY"
      }
    }
  }
}
```

## Getting an API Key

1. Open [@FundmyAIbot](https://t.me/FundmyAIbot) in Telegram
2. Send `/newagent` to create an agent and receive your API key
3. Fund via Telegram Stars or deposit USDC to your agent's wallet

## Available Tools

| Tool | Description |
|------|-------------|
| `get_balance` | Check agent wallet balance (USD + per-chain breakdown) |
| `spend` | Deduct from agent wallet with description |
| `transfer` | Agent-to-agent USDC transfers |
| `get_transactions` | Transaction history with filters |
| `create_agent` | Provision new agent with multi-chain wallets |
| `get_identity` | KYA identity profile + trust score |
| `register_webhook` | Real-time event notifications (HMAC-SHA256 signed) |

## Verify Installation

After setup, ask your AI agent:
> "Check my AgentPay balance"

It should call `get_balance` and return your wallet balance.
