"""
x402 Agent Example — Autonomous pay-per-request with AgentPay

This example shows an AI agent that:
1. Discovers an x402-gated API endpoint
2. Probes the price
3. Pays automatically if within budget
4. Uses the response

Requirements:
    pip install agentpay httpx
"""

import httpx
from agentpay import AgentPayClient, AgentPayError

# Initialize client
client = AgentPayClient(
    api_key="ap_your_key_here",
    base_url="https://leofundmybot.dev",
)


def autonomous_fetch(url: str, max_price_usd: float = 1.00) -> dict:
    """
    Fetch a resource, paying automatically via x402 if required.

    The agent:
    1. Tries to access the URL normally
    2. If it gets HTTP 402, probes the x402 price
    3. If price <= max_price_usd, pays and retrieves the data
    4. Returns the response data
    """

    # Step 1: Try normal access
    with httpx.Client() as http:
        resp = http.get(url)

        if resp.status_code == 200:
            print(f"✅ Free access: {url}")
            return resp.json()

        if resp.status_code != 402:
            resp.raise_for_status()

    # Step 2: x402 — probe the price
    print(f"💰 x402 payment required for {url}")
    probe = client._get(f"/v1/x402/probe", params={"url": url})
    price = probe.get("price_usd", 0)
    print(f"   Price: ${price:.4f}")

    if price > max_price_usd:
        raise ValueError(f"Price ${price} exceeds budget ${max_price_usd}")

    # Step 3: Pay and access
    result = client._post("/v1/x402/pay", json={
        "url": url,
        "max_price_usd": max_price_usd,
    })

    print(f"✅ Paid ${result.get('amount_paid', price):.4f} — access granted")
    return result.get("data", result)


def main():
    # Example: Agent fetching premium market data
    balance = client.get_balance()
    print(f"Agent balance: ${balance.balance_usd}")
    print()

    # Fetch an x402-gated resource
    try:
        data = autonomous_fetch(
            "https://api.example.com/v1/market-data/btc",
            max_price_usd=0.50,
        )
        print(f"Data received: {data}")
    except AgentPayError as e:
        print(f"Payment failed: {e}")
    except ValueError as e:
        print(f"Budget exceeded: {e}")

    # Check what was spent
    txs = client.get_transactions(limit=3)
    print(f"\nRecent transactions:")
    for tx in txs:
        print(f"  {tx.type}: ${tx.amount} — {tx.description}")


if __name__ == "__main__":
    main()
