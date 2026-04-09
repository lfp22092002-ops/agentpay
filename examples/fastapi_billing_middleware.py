"""
AgentPay + FastAPI Middleware — Metered API billing for AI agents.

This example shows how to add per-request billing to any FastAPI service
using AgentPay as the payment layer. Agents authenticate with their API key,
and each request is automatically charged based on endpoint pricing.

Usage:
    pip install fastapi uvicorn agentpay
    python fastapi_billing_middleware.py

Then agents call your API with their AgentPay key:
    curl -H "X-Agent-Key: ap_xxx" http://localhost:8000/api/summarize -d '{"text": "..."}'
"""

from __future__ import annotations

from typing import Callable

from fastapi import FastAPI, HTTPException, Request, Response

from agentpay import AgentPayClient

app = FastAPI(title="Metered AI API", description="Pay-per-request API powered by AgentPay")

# --- Pricing config (USD per request) ---
ENDPOINT_PRICING: dict[str, float] = {
    "/api/summarize": 0.01,
    "/api/translate": 0.02,
    "/api/generate-image": 0.05,
}
DEFAULT_PRICE = 0.005  # fallback for unlisted endpoints


# --- Middleware ---
@app.middleware("http")
async def billing_middleware(request: Request, call_next: Callable) -> Response:
    # Skip non-API routes
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    # Extract agent's API key
    agent_key = request.headers.get("X-Agent-Key")
    if not agent_key:
        raise HTTPException(401, "Missing X-Agent-Key header")

    # Determine price
    price = ENDPOINT_PRICING.get(request.url.path, DEFAULT_PRICE)

    # Charge the agent via AgentPay
    client = AgentPayClient(agent_key)
    try:
        tx = client.spend(
            amount=price,
            description=f"{request.method} {request.url.path}",
            idempotency_key=f"{agent_key}:{request.headers.get('X-Request-ID', '')}",
        )
    except Exception as e:
        error_msg = str(e)
        if "balance" in error_msg.lower():
            raise HTTPException(402, f"Insufficient balance. Top up your AgentPay wallet. (need ${price})")
        if "401" in error_msg or "auth" in error_msg.lower():
            raise HTTPException(401, "Invalid AgentPay API key")
        raise HTTPException(500, f"Billing error: {error_msg}")

    # Process the actual request
    response = await call_next(request)

    # Add billing headers
    response.headers["X-Charged-USD"] = str(price)
    response.headers["X-Transaction-ID"] = tx.transaction_id or ""
    response.headers["X-Remaining-Balance"] = str(tx.remaining_balance)

    return response


# --- Example endpoints ---

@app.post("/api/summarize")
async def summarize(request: Request):
    body = await request.json()
    text = body.get("text", "")
    # In production, call your LLM here
    return {"summary": f"Summary of {len(text)} chars: {text[:100]}...", "price": "$0.01"}


@app.post("/api/translate")
async def translate(request: Request):
    body = await request.json()
    return {"translated": body.get("text", ""), "target": body.get("lang", "en"), "price": "$0.02"}


@app.post("/api/generate-image")
async def generate_image(request: Request):
    body = await request.json()
    return {"image_url": "https://example.com/generated.png", "prompt": body.get("prompt", ""), "price": "$0.05"}


@app.get("/api/pricing")
async def pricing():
    """Public endpoint (free) showing API pricing."""
    return {"endpoints": ENDPOINT_PRICING, "default": DEFAULT_PRICE, "currency": "USD"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
