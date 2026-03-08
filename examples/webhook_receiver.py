"""
AgentPay — Webhook Receiver Example

Receive real-time notifications when your agent spends, receives funds,
or hits spending limits. Webhooks use HMAC-SHA256 for verification.

Requirements:
    pip install fastapi uvicorn
"""
import hashlib
import hmac
import json
from fastapi import FastAPI, Request, HTTPException

app = FastAPI(title="AgentPay Webhook Receiver")

# Your webhook secret (set via POST /v1/webhook)
WEBHOOK_SECRET = "your-webhook-secret-here"


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 webhook signature."""
    expected = hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@app.post("/webhook/agentpay")
async def receive_webhook(request: Request):
    """Handle AgentPay webhook events."""
    body = await request.body()
    signature = request.headers.get("X-AgentPay-Signature", "")

    if not verify_signature(body, signature, WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event = json.loads(body)
    event_type = event.get("event")
    data = event.get("data", {})

    match event_type:
        case "spend.completed":
            print(f"💸 Agent spent ${data['amount_usd']} — {data.get('description', 'N/A')}")

        case "spend.approval_needed":
            print(f"⚠️ Approval needed: ${data['amount_usd']} — {data.get('description')}")
            # Could forward to Slack, Discord, email, etc.

        case "deposit.completed":
            print(f"💰 Deposit received: ${data['amount_usd']}")

        case "balance.low":
            print(f"🔴 Low balance warning: ${data['balance_usd']} remaining")

        case _:
            print(f"📩 Unknown event: {event_type}")

    return {"received": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
