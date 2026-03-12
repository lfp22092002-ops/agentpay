# MCP Directory Submission Plan

AgentPay MCP server is now live at `https://leofundmybot.dev/mcp` (Streamable HTTP) and available as local stdio (`python mcp/server.py`).

## Directories to Submit To

### 1. Smithery (smithery.ai) — HIGHEST PRIORITY
- **Method**: URL publishing
- **URL**: `https://leofundmybot.dev/mcp`
- **Steps**: Go to smithery.ai/new → Enter URL → Complete publishing flow
- **Auth**: Server requires `X-API-Key` header → use config schema with `x-from: { header: "x-api-key" }`
- **Fallback**: Static server card already served at `/.well-known/mcp/server-card.json`
- **Status**: ⬜ Needs G's account (or create one)

### 2. mcp.so
- **Method**: GitHub issue or auto-crawl
- **Steps**: Submit via their GitHub repo (Issues → "New MCP Server" template)
- **URL**: Include GitHub repo link + live endpoint
- **Status**: ⬜ Pending

### 3. PulseMCP (pulsemcp.com)
- **Count**: 8610+ servers indexed
- **Method**: May auto-crawl GitHub repos with MCP tags
- **Steps**: Tag GitHub repo with `mcp`, `model-context-protocol` topics
- **Status**: ⬜ Add GitHub topics

### 4. LobeHub MCP (lobehub.com/mcp)
- **Method**: Submit via LobeHub GitHub marketplace repo
- **Steps**: Fork lobe-chat-plugins, add agentpay config, submit PR
- **Status**: ⬜ Pending

### 5. mcp-marketplace.io
- **Method**: Submit form or GitHub
- **Count**: 2200+ servers
- **Status**: ⬜ Investigate submission process

### 6. mcpserverfinder.com
- **Method**: Unknown — likely submit form
- **Status**: ⬜ Investigate

### 7. AI Agent Store / apitracker.io
- **Method**: Various submission forms
- **Status**: ⬜ Low priority

## Quick Wins (Can Do Now)
1. ✅ Add GitHub topics to repo: `mcp`, `model-context-protocol`, `ai-agent`, `payments`, `crypto`
2. ⬜ Smithery URL publish (needs web UI — ask G or automate)
3. ⬜ mcp.so GitHub issue
4. ⬜ LobeHub PR

## Config Schema for Smithery CLI Publishing
```bash
smithery mcp publish "https://leofundmybot.dev/mcp" \
  -n @agentpay/agentpay \
  --config-schema '{"type":"object","properties":{"apiKey":{"type":"string","title":"AgentPay API Key","x-from":{"header":"x-api-key"}}},"required":["apiKey"]}'
```
