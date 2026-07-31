# MCP Directory Auto-Publish Plan

## Completed ✅ (Auto)
| Directory | Status | Notes |
|-----------|--------|-------|
| GitHub topics (`mcp`, `model-context-protocol`) | Done | Repos with these tags auto-discover to PulseMCP, LobeHub, etc. |
| Smithery server card | Done | `.well-known/mcp/server-card.json` served at `/` |
| MCP endpoint live | Done | `https://leofundmybot.dev/mcp` (Streamable HTTP) |
| PulseMCP | Auto | Topics already include `mcp` + `model-context-protocol` — auto-crawled |

## Needs G's Manual Action ⬜
### 1. Smithery (smithery.ai) — HIGHEST PRIORITY
Smithery requires interactive API key login. **One command after setup:**

```bash
# Step 1: Get API key from https://smithery.ai/account/api-keys
export SMITHERY_API_KEY="your_key"

# Step 2: Publish AgentPay
cd projects/agentpay
npx smithery mcp publish "https://leofundmybot.dev/mcp" \
  -n "@lfp22092002/agentpay" \
  --config-schema '{"type":"object","properties":{"apiKey":{"type":"string","title":"AgentPay API Key","x-from":{"header":"x-api-key"}}},"required":["apiKey"]}'
```

### 2. mcp.so — REQUIRED
Go to https://mcp.so/servers/submit and fill:
- **Name:** AgentPay
- **URL:** https://leofundmybot.dev/mcp  
- **GitHub:** lfp22092002-ops/agentpay
- **Description:** Payment layer for autonomous AI agents — balance, spend, transfer, multi-chain wallets

### 3. LobeHub MCP
Fork and submit PR to: https://github.com/lobe-chat/mcp-servers
(Add agentpay config following existing patterns)

## One-Liner Summary
**Just publish to Smithery + mcp.so manually.** Everything else is auto-discovered. Takes <5 minutes total.
