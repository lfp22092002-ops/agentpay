# Launch Checklist

Pre-launch steps before going live. Check off as completed.

## Blocking (needs G)

- [ ] **PyPI publish** — `cd sdk/agentpay && python -m build && twine upload dist/*`
- [ ] **Lithic production key** — swap sandbox → production in `.env`
- [ ] **Cloudflare Tunnel** — verify `agentpay` tunnel routes to API on port 8080
- [ ] **Domain DNS** — confirm `leofundmybot.dev` resolves via Cloudflare

## Ready (already done)

- [x] CI pipeline (test + lint + security + TS SDK build)
- [x] 40+ test files, zero TODOs in codebase
- [x] OpenAPI spec — 39 endpoints documented
- [x] Python SDK (`sdk/agentpay/`) with py.typed, sync + async
- [x] TypeScript SDK (`sdk/ts/`) with full types
- [x] 16 framework examples (OpenAI, Claude, LangChain, CrewAI, etc.)
- [x] MCP server + streamable HTTP + discovery artifacts
- [x] CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CODEOWNERS
- [x] Docker + docker-compose
- [x] Landing page + Mini App
- [x] Telegram bot (@FundmyAIbot) — 19 commands
- [x] Webhook system (HMAC-SHA256 verified)
- [x] Multi-chain wallets (Base + Polygon + BNB + Solana)
- [x] Encryption (Fernet + PBKDF2), rate limiting, CORS

## Post-Launch

- [ ] Product Hunt submission
- [ ] Hacker News Show HN post
- [ ] r/SideProject + r/artificial Reddit posts
- [ ] npm publish TS SDK
- [ ] MCP directory listings go live (mcp.so, Cline)
- [ ] First 10 users → collect feedback
- [ ] v0.2.0 planning based on user feedback
