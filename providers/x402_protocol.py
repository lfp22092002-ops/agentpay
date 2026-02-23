"""
x402 Protocol Support for AgentPay.

Allows AgentPay agents to:
1. PAY for x402-gated resources (client mode) — agent spends from their balance
2. CHARGE via x402 (server mode) — gate AgentPay API endpoints behind x402 paywall

Uses the official x402 Python SDK from Coinbase.
"""
import json
import logging
import os
from decimal import Decimal

from eth_account import Account
from web3 import Web3

from config.settings import ENVIRONMENT

logger = logging.getLogger("agentpay.x402")

# ═══════════════════════════════════════
# x402 CLIENT — Agent pays for resources
# ═══════════════════════════════════════

FACILITATOR_URL = "https://x402.org/facilitator"


async def pay_x402_resource(
    agent_id: str,
    resource_url: str,
    method: str = "GET",
    body: dict | None = None,
    max_price_usd: float = 1.0,
) -> dict:
    """
    Make an HTTP request to an x402-gated resource using the agent's wallet.

    Flow:
    1. Request the resource → get 402 with payment requirements
    2. Sign payment with agent's private key
    3. Resend request with X-PAYMENT header
    4. Return the resource

    Args:
        agent_id: Agent whose wallet pays
        resource_url: URL of the x402-gated resource
        method: HTTP method (GET, POST, etc.)
        body: Request body for POST
        max_price_usd: Max price willing to pay (safety limit)

    Returns:
        {"success": bool, "status": int, "data": ..., "paid_usd": float}
    """
    import httpx
    from providers.local_wallet import WALLETS_DIR
    from pathlib import Path
    from core.encryption import decrypt

    # Load agent wallet
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return {"success": False, "error": "Agent has no wallet. Create one first."}

    wallet_data = json.loads(wallet_file.read_text())
    private_key = wallet_data["private_key"]
    if wallet_data.get("encrypted", False):
        private_key = decrypt(private_key)

    address = wallet_data["address"]

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: Initial request
        if method.upper() == "GET":
            resp = await client.get(resource_url)
        else:
            resp = await client.request(method.upper(), resource_url, json=body)

        # Not a 402 → return directly
        if resp.status_code != 402:
            return {
                "success": True,
                "status": resp.status_code,
                "data": resp.text,
                "paid_usd": 0,
            }

        # Step 2: Parse payment requirements
        try:
            resp_json = resp.json()
            payment_requirements = resp_json.get("accepts", [])
            if not payment_requirements:
                return {"success": False, "error": "402 response has no payment requirements"}

            # Pick the first requirement
            req = payment_requirements[0]
            max_amount = int(req.get("maxAmountRequired", 0))
            asset = req.get("asset", "")
            pay_to = req.get("payTo", "")
            network = req.get("network", "base-sepolia")
            scheme = req.get("scheme", "exact")

            # Safety check — convert atomic USDC (6 decimals) to USD
            price_usd = max_amount / 1_000_000
            if price_usd > max_price_usd:
                return {
                    "success": False,
                    "error": f"Price ${price_usd:.4f} exceeds max ${max_price_usd:.4f}",
                }

            logger.info(f"x402: paying ${price_usd:.4f} to {pay_to} for {resource_url}")

        except Exception as e:
            return {"success": False, "error": f"Failed to parse 402 response: {e}"}

        # Step 3: Create and sign payment
        try:
            from x402.clients.httpx import x402_httpx_client

            # Use the x402 httpx client which handles the payment flow
            x402_client = x402_httpx_client(private_key)

            if method.upper() == "GET":
                paid_resp = await x402_client.get(resource_url)
            else:
                paid_resp = await x402_client.request(method.upper(), resource_url, json=body)

            return {
                "success": paid_resp.status_code < 400,
                "status": paid_resp.status_code,
                "data": paid_resp.text,
                "paid_usd": price_usd,
            }

        except Exception as e:
            logger.error(f"x402 payment failed: {e}")
            return {"success": False, "error": f"Payment failed: {e}"}


# ═══════════════════════════════════════
# x402 SERVER — Gate endpoints behind x402
# ═══════════════════════════════════════

def get_x402_middleware(
    pay_to_address: str,
    price_usd: float = 0.01,
    path: str | list[str] = "/v1/x402/*",
    description: str = "AgentPay x402 endpoint",
    network: str = "base-sepolia",
):
    """
    Create FastAPI middleware that gates endpoints behind x402 payments.

    Anyone can pay with USDC on Base to access these endpoints — no API key needed.
    Payment goes directly to the specified address.

    Usage in api/main.py:
        from providers.x402_protocol import get_x402_middleware
        app.middleware("http")(get_x402_middleware("0xYourAddress"))
    """
    try:
        from x402.fastapi.middleware import require_payment
        return require_payment(
            price=price_usd,
            pay_to_address=pay_to_address,
            path=path,
            description=description,
            network=network if ENVIRONMENT == "development" else "base",
        )
    except Exception as e:
        logger.error(f"Failed to create x402 middleware: {e}")
        raise


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def estimate_x402_cost(resource_url: str) -> dict:
    """
    Probe an x402 resource to see what it costs without paying.
    Makes a GET request and parses the 402 response.
    """
    import httpx

    try:
        resp = httpx.get(resource_url, timeout=10)
        if resp.status_code != 402:
            return {"gated": False, "status": resp.status_code}

        data = resp.json()
        requirements = data.get("accepts", [])
        costs = []
        for req in requirements:
            max_amount = int(req.get("maxAmountRequired", 0))
            costs.append({
                "network": req.get("network"),
                "asset": req.get("asset"),
                "price_usd": max_amount / 1_000_000,
                "scheme": req.get("scheme"),
            })

        return {"gated": True, "costs": costs}
    except Exception as e:
        return {"error": str(e)}
