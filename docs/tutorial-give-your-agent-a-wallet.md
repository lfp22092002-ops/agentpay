# Give Your AI Agent a Wallet in 5 Minutes

Your agent can write code, browse the web, call APIs — but the moment it needs to *pay* for something, a human has to step in. That's dumb. Let's fix it.

This tutorial walks you through giving your AI agent its own wallet, funding it, and making its first autonomous transaction — all in about 5 minutes.

**What you'll have by the end:**
- An agent with its own balance and on-chain USDC wallets
- Working Python code to spend, transfer, and track funds
- A LangChain tool your agent can call to pay for things

---

## Prerequisites

- Python 3.9+
- A Telegram account (for creating your agent)
- That's it. No KYC, no credit card, no 47-page form.

---

## Step 1: Create Your Agent Account

Open [@FundmyAIbot](https://t.me/FundmyAIbot) in Telegram and send:

```
/newagent
```

The bot walks you through it — pick a name, get your agent created. Takes 30 seconds.

---

## Step 2: Get Your API Key

After creating the agent, the bot gives you an API key that looks like:

```
ap_7f3a1b2c4d5e6f...
```

> ⚠️ **Save this immediately.** API keys are hashed on our end (Stripe-style) — we can't show it again. If you lose it, you'll need to rotate to a new one via `/settings` in the bot.

---

## Step 3: Install the SDK

```bash
pip install agentpay
```

That's one dependency. The SDK is typed, uses Pydantic v2 models, and supports both sync and async clients.

---

## Step 4: Create Your First Client

```python
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_api_key_here")

# Verify it works
balance = client.get_balance()
print(f"Agent: {balance.agent_name}")
print(f"Balance: ${balance.balance_usd}")
print(f"Active: {balance.is_active}")
```

**With curl:**

```bash
curl -s https://leofundmybot.dev/v1/balance \
  -H "X-API-Key: ap_your_api_key_here" | python -m json.tool
```

You should see your agent's name and a $0.00 balance. Time to fix that.

---

## Step 5: Fund Your Agent

You have two options:

### Option A: Telegram Stars (instant)

Go back to [@FundmyAIbot](https://t.me/FundmyAIbot) and use the funding menu. Telegram Stars convert to USD balance instantly. Good for testing and small amounts.

### Option B: USDC Deposit (on-chain)

Your agent has wallets on **four chains** out of the box:

```python
# Get your agent's Base wallet address
wallet = client.get_wallet(chain="base")
print(f"Base address: {wallet.address}")
print(f"USDC balance: {wallet.balance_usdc}")

# Check all supported chains
chains = client.list_chains()
for chain in chains:
    print(f"  {chain.name} ({chain.id}) — {chain.native_token}")
```

Supported chains:
| Chain | Token | Best for |
|-------|-------|----------|
| **Base** | ETH | Low fees, fast |
| **Polygon** | MATIC | Cheap transactions |
| **BNB Chain** | BNB | BSC ecosystem |
| **Solana** | SOL | High throughput |

Send USDC to any of these addresses and your balance updates automatically.

**With curl:**

```bash
# Get Base wallet address
curl -s https://leofundmybot.dev/v1/wallet?chain=base \
  -H "X-API-Key: ap_your_api_key_here"

# List all supported chains
curl -s https://leofundmybot.dev/v1/chains \
  -H "X-API-Key: ap_your_api_key_here"
```

---

## Step 6: Make Your First Transaction

Now the fun part. Let your agent spend money:

```python
# Spend $0.50 on an API call
tx = client.spend(
    amount=0.50,
    description="GPT-4 API call — summarize document"
)

print(f"✅ Transaction: {tx.transaction_id}")
print(f"   Amount: ${tx.amount}")
print(f"   Fee: ${tx.fee}")           # $0.00 during beta!
print(f"   Remaining: ${tx.remaining_balance}")
print(f"   Status: {tx.status}")
```

**With curl:**

```bash
curl -X POST https://leofundmybot.dev/v1/spend \
  -H "X-API-Key: ap_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 0.50,
    "description": "GPT-4 API call — summarize document"
  }'
```

### Preventing Duplicate Charges

Use idempotency keys to avoid charging twice if a request retries:

```python
tx = client.spend(
    amount=0.50,
    description="Embedding generation batch #42",
    idempotency_key="batch-42-embed"
)
```

Same key = same transaction. The API returns the original result instead of charging again.

### Transfer Between Agents

Agents can pay each other directly:

```python
result = client.transfer(
    to_agent_id="agent_xyz",
    amount=5.00,
    description="Payment for data enrichment"
)
print(f"Transferred: ${result.amount}")
print(f"Your new balance: ${result.from_balance}")
```

### Refund a Transaction

Made a mistake? Roll it back:

```python
refund = client.refund("tx_abc123")
print(f"Refunded: ${refund.amount_refunded}")
print(f"New balance: ${refund.new_balance}")
```

---

## Step 7: Check the Dashboard

View your agent's transaction history programmatically:

```python
transactions = client.get_transactions(limit=10)

for tx in transactions:
    print(f"[{tx.created_at}] {tx.type}: ${tx.amount} — {tx.description}")
```

**With curl:**

```bash
curl -s "https://leofundmybot.dev/v1/transactions?limit=10" \
  -H "X-API-Key: ap_your_api_key_here"
```

You can also check the web dashboard at [leofundmybot.dev/app](https://leofundmybot.dev/app/) — log in with your Telegram account to see stats, analytics, and spending breakdowns.

---

## Bonus: LangChain Integration

Here's how to give a LangChain agent the ability to spend money:

```python
from langchain.tools import tool
from agentpay import AgentPayClient, InsufficientBalanceError

pay_client = AgentPayClient("ap_your_api_key_here")

@tool
def check_balance() -> str:
    """Check how much money the agent has available."""
    b = pay_client.get_balance()
    return f"${b.balance_usd:.2f} available | daily remaining: ${b.daily_remaining_usd:.2f}"

@tool
def spend_money(amount: float, description: str) -> str:
    """Spend money from the agent's wallet.
    Args:
        amount: Dollar amount to spend.
        description: What the money is for.
    """
    try:
        tx = pay_client.spend(amount, description)
        return f"✅ Spent ${tx.amount:.2f} — remaining ${tx.remaining_balance:.2f}"
    except InsufficientBalanceError:
        return "❌ Not enough funds. Top up via @FundmyAIbot."

@tool
def get_wallet_address(chain: str = "base") -> str:
    """Get the agent's USDC wallet address for receiving payments."""
    wallet = pay_client.get_wallet(chain=chain)
    return f"Send USDC on {chain} to: {wallet.address}"

# Use with any LangChain agent
tools = [check_balance, spend_money, get_wallet_address]
```

This works with **any agent framework** — CrewAI, AutoGen, OpenClaw, or your own custom loop.

---

## Webhooks: Get Notified in Real-Time

Don't poll for updates. Set up a webhook and get pushed:

```python
webhook = client.register_webhook(
    url="https://your-server.com/agentpay-webhook",
    events=["spend", "refund", "transfer"]
)
print(f"Webhook secret: {webhook.secret}")  # Use this to verify signatures
```

AgentPay signs every webhook with HMAC-SHA256 so you can verify it's legit. Events include: `spend`, `refund`, `transfer`, `deposit`, and more.

---

## Error Handling

The SDK raises specific exceptions so you can handle failures gracefully:

```python
from agentpay import (
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
)

try:
    tx = client.spend(100.00, "Big purchase")
except AuthenticationError:
    print("Bad API key — check your credentials")
except InsufficientBalanceError:
    print("Not enough funds — top up via Telegram")
except RateLimitError:
    print("Slow down — 60 requests/min limit")
except AgentPayError as e:
    print(f"API error {e.status_code}: {e.detail}")
```

---

## x402: HTTP-Native Payments

AgentPay supports the [x402 protocol](https://www.x402.org/) — pay for web resources using standard HTTP:

```python
result = client.x402_pay(
    url="https://api.example.com/premium-dataset",
    max_amount=2.00
)
print(f"Data: {result.data}")
print(f"Paid: ${result.paid_usd}")
```

Your agent hits a 402 paywall, pays automatically, and gets the data. No checkout flows, no redirect chains.

---

## Agent Identity (KYA)

AgentPay includes **Know Your Agent** — a trust and identity layer for the agent economy. Your agent gets a trust score (0-100), verified badges, public directory listing, and category tags. Think of it as a reputation system — services can check trust before accepting payments.

---

## Things Worth Knowing

| | |
|---|---|
| **Fees** | 0% during beta. Yes, really. |
| **Rate limit** | 60 requests/min per API key |
| **Chains** | Base, Polygon, BNB Chain, Solana |
| **Currency** | USD (internal) + USDC (on-chain) |
| **Security** | API keys SHA-256 hashed, wallet keys encrypted (Fernet+PBKDF2) |
| **Self-host** | Fully open source, run your own instance |
| **API docs** | [leofundmybot.dev/docs](https://leofundmybot.dev/docs) (interactive OpenAPI) |

---

## Full Example: Autonomous Agent Loop

Here's a minimal agent that checks its balance, does work, and pays for resources:

```python
from agentpay import AgentPayClient, InsufficientBalanceError

client = AgentPayClient("ap_your_api_key_here")

def autonomous_agent_loop():
    # 1. Check if we can afford to work
    balance = client.get_balance()
    if balance.balance_usd < 1.00:
        print("Low funds — requesting top-up")
        return

    # 2. Do some work that costs money
    try:
        tx = client.spend(0.25, "OpenAI embedding — 1000 tokens")
        print(f"Paid for embeddings: {tx.transaction_id}")
    except InsufficientBalanceError:
        print("Ran out of money mid-task")
        return

    # 3. Check remaining budget
    updated = client.get_balance()
    print(f"Remaining budget: ${updated.balance_usd:.2f}")

autonomous_agent_loop()
```

---

## What's Next

- **TypeScript SDK** — coming soon on npm
- **Virtual Visa cards** — let agents pay anywhere (via Lithic)
- **Approval workflows** — require human sign-off above spending thresholds
- **LangChain/CrewAI native tools** — first-class framework integrations

---

## Links

| | |
|---|---|
| 🤖 **Telegram Bot** | [@FundmyAIbot](https://t.me/FundmyAIbot) |
| 🌐 **Website** | [leofundmybot.dev](https://leofundmybot.dev) |
| 📚 **API Docs** | [leofundmybot.dev/docs](https://leofundmybot.dev/docs) |
| 📖 **Full Docs** | [leofundmybot.dev/docs-site](https://leofundmybot.dev/docs-site/) |
| 💻 **GitHub** | [github.com/lfp22092002-ops/agentpay](https://github.com/lfp22092002-ops/agentpay) |
| 📦 **PyPI** | [pypi.org/project/agentpay](https://pypi.org/project/agentpay/) |
| 🐦 **X/Twitter** | [@autonomousaibot](https://x.com/autonomousaibot) |

---

*Built for the agent economy. Because autonomous agents deserve their own wallets.*
