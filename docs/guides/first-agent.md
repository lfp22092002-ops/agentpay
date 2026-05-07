# Build Your First Paid Agent in 10 Minutes

By the end of this guide, you'll have an AI agent that can spend money autonomously — calling paid APIs, tracking its own budget, and stopping when it runs out.

## What You'll Build

A simple Python agent that:
1. Gets its own wallet with a $5 budget
2. Calls a paid API (simulated)
3. Tracks spending
4. Stops when budget is low

**Time required**: 10 minutes  
**Prerequisites**: Python 3.11+, a Telegram account

---

## Step 1 — Create Your Agent

Open [@FundmyAIbot](https://t.me/FundmyAIbot) on Telegram:

```
/start
/newagent MyFirstAgent
```

You'll receive an API key like `ap_xxxx...`. **Copy it — it's shown only once.**

---

## Step 2 — Install the SDK

```bash
pip install agentpay
```

---

## Step 3 — Fund Your Agent

In the Telegram bot:
```
/fund
```

Select "Telegram Stars" and send a small amount (50 Stars ≈ $0.50 USD).

Or deposit USDC directly to your agent's wallet address:
```
/wallet base
```

---

## Step 4 — Write Your Agent

Create `my_agent.py`:

```python
from agentpay import AgentPayClient

# Your API key from Step 1
client = AgentPayClient("ap_your_key_here")

def check_budget():
    """Check if we have enough to continue."""
    balance = client.get_balance()
    print(f"💰 Balance: ${balance.balance_usd:.4f}")
    print(f"📊 Daily remaining: ${balance.daily_remaining_usd:.4f}")
    return balance.balance_usd > 0.01  # Stop if under 1 cent

def call_paid_api(prompt: str) -> str:
    """Simulate calling a paid API (e.g. OpenAI, Anthropic)."""
    # In a real agent, this would be your actual API call
    # Here we simulate it costing $0.002 per call
    
    tx = client.spend(
        amount=0.002,
        description=f"API call: {prompt[:50]}",
        idempotency_key=f"api-{hash(prompt)}",
    )
    
    if not tx.success:
        raise RuntimeError(f"Payment failed: {tx.error}")
    
    print(f"  ✓ Spent ${tx.amount:.4f} (tx: {tx.transaction_id[:8]}...)")
    
    # Your actual API call would go here
    return f"Response to: {prompt}"

def run_agent(tasks: list[str]):
    """Run the agent through a list of tasks."""
    print("🤖 Starting agent...\n")
    
    for i, task in enumerate(tasks, 1):
        print(f"Task {i}/{len(tasks)}: {task}")
        
        # Check budget before each task
        if not check_budget():
            print("⚠️ Budget too low — stopping.")
            break
        
        try:
            result = call_paid_api(task)
            print(f"  → {result}\n")
        except RuntimeError as e:
            print(f"  ✗ Error: {e}\n")
            break
    
    # Final balance
    final = client.get_balance()
    print(f"\n📋 Session complete. Final balance: ${final.balance_usd:.4f}")
    
    # Print transaction history
    print("\n📜 Transactions:")
    txs = client.get_transactions(limit=10)
    for tx in txs:
        print(f"  {tx.created_at[:10]} | -{tx.amount:.4f} | {tx.description}")

if __name__ == "__main__":
    tasks = [
        "Summarize today's news",
        "Write a haiku about AI",
        "Explain quantum computing simply",
        "Generate a business idea",
        "Translate 'hello world' to 10 languages",
    ]
    run_agent(tasks)
```

---

## Step 5 — Run It

```bash
python my_agent.py
```

Output:
```
🤖 Starting agent...

Task 1/5: Summarize today's news
💰 Balance: $0.5000
📊 Daily remaining: $50.0000
  ✓ Spent $0.0020 (tx: a1b2c3d4...)
  → Response to: Summarize today's news

Task 2/5: Write a haiku about AI
💰 Balance: $0.4980
...

📋 Session complete. Final balance: $0.4900

📜 Transactions:
  2026-05-07 | -0.0020 | API call: Summarize today's news
  2026-05-07 | -0.0020 | API call: Write a haiku about AI
  ...
```

---

## Step 6 — Add Spending Controls (Optional)

Set a per-transaction limit so the agent can't accidentally spend too much:

```python
# In Telegram bot
/settings daily_limit 1.00    # Max $1/day
/settings tx_limit 0.10       # Max $0.10/transaction
```

Or via API:
```python
# Coming in v0.2.0: client.update_limits(daily_usd=1.0, tx_usd=0.10)
```

---

## Step 7 — Add Webhooks (Optional)

Get notified whenever your agent spends:

```python
client.register_webhook(
    url="https://your-server.com/hooks/agentpay",
    events=["spend", "low_balance"],
)
```

See [webhook_receiver.py](../../examples/webhook_receiver.py) for a complete example.

---

## What's Next?

- **Real AI integration**: See [examples/openai_agent.py](../../examples/openai_agent.py) for OpenAI, [examples/anthropic_claude_agent.py](../../examples/anthropic_claude_agent.py) for Claude
- **Multi-agent**: [examples/multi_agent_budget.py](../../examples/multi_agent_budget.py) — agent funds sub-agents
- **x402 micropayments**: [examples/x402_agent.py](../../examples/x402_agent.py) — pay per HTTP request
- **MCP integration**: [examples/mcp_client.py](../../examples/mcp_client.py) — use from Claude Code / Cursor

---

*Questions? Open an issue on [GitHub](https://github.com/lfp22092002-ops/agentpay) or message [@FundmyAIbot](https://t.me/FundmyAIbot).*
