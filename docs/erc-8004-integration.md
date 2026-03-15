# ERC-8004 Integration Plan — AgentPay

## Why
ERC-8004 is becoming the de facto standard for AI agent identity on Ethereum. 24K+ agents registered on mainnet since Jan 2026. Crypto.com, ConsenSys, Coinbase, and Phala Network all building on it. Our KYA (Know Your Agent) trust scores already align with ERC-8004's three registries. Adding compatibility makes AgentPay agents interoperable with the broader on-chain agent ecosystem.

## Architecture Mapping

| AgentPay Concept | ERC-8004 Equivalent | Status |
|---|---|---|
| Agent ID | Identity Registry (ERC-721 tokenId) | New |
| Trust Score (0-100) | Reputation Registry (feedback signals) | Map existing |
| Verified Badge | Validation Registry (validator checks) | Map existing |
| Agent Name/Description | Registration File (JSON) | Map existing |
| API capabilities | Registration File → capabilities | New |

## Three Registries

### 1. Identity Registry
- Each AgentPay agent gets an ERC-721 NFT on Base (cheapest gas)
- `agentURI` → points to AgentPay's `/api/agents/{id}/registration.json`
- Registration file format follows ERC-8004 spec:
  ```json
  {
    "type": "https://eips.ethereum.org/EIPS/eip-8004#registration-v1",
    "name": "my-agent",
    "description": "AI agent for...",
    "capabilities": ["payment", "balance-check"],
    "agentpay_trust_score": 85
  }
  ```

### 2. Reputation Registry
- Map our trust score (0-100) to on-chain feedback signals
- When an agent completes a transaction → post feedback to Reputation Registry
- Read reputation from chain when evaluating unknown agents

### 3. Validation Registry
- Our approval workflows can act as off-chain validators
- Optionally register as a Validation Oracle for AgentPay-managed agents

## Implementation Phases

### Phase 1: Read-Only Compatibility (1-2 days)
- New endpoint: `GET /api/agents/{id}/registration.json` → ERC-8004 format
- Add `erc8004_agent_id` field to Agent model (nullable)
- Serve `.well-known/erc-8004` discovery endpoint
- Zero smart contract interaction — just format compatibility

### Phase 2: On-Chain Registration (3-5 days)
- Deploy or use existing Identity Registry on Base
- Auto-register new agents on-chain when created (gas from agent's USDC balance)
- Store `agentRegistry` + `agentId` in DB
- Update `agentURI` when agent metadata changes

### Phase 3: Reputation Bridge (3-5 days)
- Post transaction feedback to Reputation Registry after successful payments
- Read external reputation for unknown agents (trust discovery)
- Aggregate on-chain + off-chain trust signals into unified score

## API Changes

### New Endpoints
- `GET /api/agents/{id}/registration.json` — ERC-8004 registration file
- `GET /.well-known/erc-8004` — discovery metadata
- `GET /api/agents/{id}/erc8004` — on-chain registration status

### Modified Endpoints
- `GET /api/agents/{id}` — add `erc8004_agent_id`, `erc8004_registry` fields
- `POST /api/agents` — optionally register on-chain

## Dependencies
- `web3.py` or `eth-brownie` for Base chain interaction
- Base RPC endpoint (Alchemy/Infura free tier)
- Identity Registry contract address on Base

## Security Considerations
- Agent NFTs are owned by AgentPay's deployer wallet, not individual users
- Transfer/burn restricted to admin
- Registration file hosted on our infra (not IPFS yet — simplicity first)
- Gas costs: ~$0.001 per registration on Base (negligible)

## Priority
Start with Phase 1 (read-only compatibility) — zero cost, immediate interoperability signal. Ship in the next code iteration. Phases 2-3 require Base wallet setup and smart contract interaction.
