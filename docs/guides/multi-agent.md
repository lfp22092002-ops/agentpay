# Multi-Agent Payments: Agents Funding Agents

One of AgentPay's most powerful features is agent-to-agent transfers. An orchestrator agent can fund sub-agents, sub-agents can pay each other for services, and you get a full audit trail of every transfer.

## Use Case

Imagine an AI workflow:
- **Orchestrator**: Manages tasks, has a large budget ($10)
- **Researcher**: Gets funded per research task ($0.50 each)
- **Writer**: Gets funded per article ($1.00 each)
- **Publisher**: Gets funded per publish action ($0.25 each)

The orchestrator delegates work and pays sub-agents, who spend only on their assigned tasks.

---

## Setup

### 1. Create All Agents

In [@FundmyAIbot](https://t.me/FundmyAIbot):
```
/newagent Orchestrator
/newagent Researcher
/newagent Writer
/newagent Publisher
```

Save each API key. Fund only the Orchestrator — it will fund the others.

### 2. Fund the Orchestrator

```
/fund  → select Orchestrator → deposit $10 via Stars or USDC
```

---

## Code

```python
from agentpay import AgentPayClient
from decimal import Decimal

# Initialize all agents
orchestrator = AgentPayClient("ap_orchestrator_key")
researcher   = AgentPayClient("ap_researcher_key")
writer       = AgentPayClient("ap_writer_key")
publisher    = AgentPayClient("ap_publisher_key")

# Agent IDs (from /info command in bot, or API)
RESEARCHER_ID = "agent_id_from_bot"
WRITER_ID     = "agent_id_from_bot"
PUBLISHER_ID  = "agent_id_from_bot"


def delegate_research(topic: str) -> str:
    """Fund researcher and run research task."""
    # Orchestrator funds researcher
    transfer = orchestrator.transfer(
        to_agent_id=RESEARCHER_ID,
        amount=0.50,
        description=f"Research task: {topic[:40]}",
    )
    if not transfer.success:
        raise RuntimeError(f"Transfer failed: {transfer.error}")
    
    print(f"  → Funded researcher ${transfer.amount:.2f}")
    
    # Researcher spends on the task
    spend = researcher.spend(
        amount=0.45,
        description=f"Research: {topic[:40]}",
    )
    
    return f"Research results for: {topic}"


def delegate_writing(content: str) -> str:
    """Fund writer and run writing task."""
    transfer = orchestrator.transfer(
        to_agent_id=WRITER_ID,
        amount=1.00,
        description="Article writing",
    )
    if not transfer.success:
        raise RuntimeError(f"Transfer failed: {transfer.error}")
    
    spend = writer.spend(amount=0.90, description="Write article")
    return f"Article written: {content[:50]}..."


def delegate_publish(article: str) -> bool:
    """Fund publisher and publish."""
    transfer = orchestrator.transfer(
        to_agent_id=PUBLISHER_ID,
        amount=0.25,
        description="Publish article",
    )
    if not transfer.success:
        raise RuntimeError(f"Transfer failed: {transfer.error}")
    
    spend = publisher.spend(amount=0.20, description="Publish to platform")
    return spend.success


def run_workflow(topics: list[str]):
    """Run the full multi-agent content pipeline."""
    
    # Check orchestrator budget
    balance = orchestrator.get_balance()
    print(f"Orchestrator budget: ${balance.balance_usd:.2f}")
    
    cost_per_topic = 0.50 + 1.00 + 0.25  # research + write + publish
    if balance.balance_usd < cost_per_topic:
        print("Insufficient budget for even one topic.")
        return
    
    for topic in topics:
        balance = orchestrator.get_balance()
        if balance.balance_usd < cost_per_topic:
            print(f"Budget exhausted after {topics.index(topic)} topics.")
            break
        
        print(f"\n📋 Processing: {topic}")
        
        research = delegate_research(topic)
        print(f"  ✓ Research done")
        
        article = delegate_writing(research)
        print(f"  ✓ Article written")
        
        published = delegate_publish(article)
        print(f"  ✓ Published: {published}")
    
    # Final report
    print("\n📊 Final balances:")
    for name, agent in [("Orchestrator", orchestrator), ("Researcher", researcher),
                         ("Writer", writer), ("Publisher", publisher)]:
        b = agent.get_balance()
        print(f"  {name}: ${b.balance_usd:.4f}")


if __name__ == "__main__":
    topics = [
        "The future of AI payments",
        "How autonomous agents work",
        "Building profitable AI products",
    ]
    run_workflow(topics)
```

---

## Getting Agent IDs

You need the agent ID (not the API key) to transfer funds. Get it via the API:

```python
balance = researcher.get_balance()
print(f"Researcher ID: {balance.agent_id}")
```

Or from the Telegram bot: `/info` → shows your agent ID.

---

## Transaction Audit Trail

Every transfer and spend is logged. View the full history:

```python
# Orchestrator's outgoing transfers
txs = orchestrator.get_transactions(limit=50)
for tx in txs:
    print(f"{tx.created_at[:10]} | {tx.type:10} | ${tx.amount:.4f} | {tx.description}")
```

Output:
```
2026-05-07 | transfer   | $0.5000 | Research task: The future of AI payments
2026-05-07 | transfer   | $1.0000 | Article writing
2026-05-07 | transfer   | $0.2500 | Publish article
...
```

---

## Spending Controls for Sub-Agents

Prevent sub-agents from overspending their allocation:

```python
# In the bot: set researcher's daily limit to $2
# /settings researcher daily_limit 2.00
# /settings researcher tx_limit 0.50
```

Even if the orchestrator over-funds by mistake, the sub-agent is capped.

---

## Advanced: Agent-to-Agent Service Marketplace

Agents can offer services to *other* agents:

```python
# Agent A requests data from Agent B's paid API
# Agent B is running an AgentPay-gated service

import httpx

def call_agent_service(service_url: str, query: str, max_cost: float):
    """Call a paid agent service, paying from our wallet."""
    # x402 probe to check price
    probe = orchestrator._request("GET", f"/v1/x402/probe?url={service_url}")
    if probe.get("price_usd", 0) > max_cost:
        raise ValueError(f"Service costs ${probe['price_usd']}, max is ${max_cost}")
    
    # Pay and get response
    result = orchestrator._request("POST", "/v1/x402/pay", json={
        "url": service_url,
        "method": "POST",
        "body": {"query": query},
        "max_price_usd": max_cost,
    })
    return result
```

See [x402_agent.py](../../examples/x402_agent.py) for the full x402 integration.

---

*Next: [Deploy AgentPay to Railway →](deploy-railway.md)*
