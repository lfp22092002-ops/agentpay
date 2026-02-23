"""
Lithic Virtual Card Provider for AgentPay.

Creates and manages virtual Visa/Mastercard cards for AI agents.
Sandbox mode for testing, production requires Lithic approval.
PAN and CVV encrypted at rest.
"""
import json
import logging
import os
from pathlib import Path
from lithic import Lithic

from config.settings import ENVIRONMENT
from core.encryption import encrypt, decrypt

logger = logging.getLogger("agentpay.lithic")

CARDS_DIR = Path(__file__).parent.parent / "data" / "cards"
CARDS_DIR.mkdir(parents=True, exist_ok=True)
os.chmod(CARDS_DIR, 0o700)


def _get_client() -> Lithic:
    """Initialize Lithic client."""
    api_key = os.getenv("LITHIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "LITHIC_API_KEY not configured. "
            "Get a free sandbox key at https://app.lithic.com/signup"
        )
    env = "sandbox" if ENVIRONMENT == "development" else "production"
    return Lithic(api_key=api_key, environment=env)


def create_virtual_card(agent_id: str, spend_limit: int = 5000, memo: str = "") -> dict:
    """
    Create a virtual card for an agent.
    
    Args:
        agent_id: The agent's ID
        spend_limit: Monthly spend limit in cents (default $50)
        memo: Optional label for the card
    
    Returns:
        Card details including last4, expiry, and token
    """
    # Check if card already exists
    card_file = CARDS_DIR / f"{agent_id}.json"
    if card_file.exists():
        data = json.loads(card_file.read_text())
        return {
            "already_existed": True,
            "card_token": data["card_token"],
            "last4": data["last4"],
            "exp_month": data["exp_month"],
            "exp_year": data["exp_year"],
            "state": data.get("state", "OPEN"),
        }

    client = _get_client()

    card = client.cards.create(
        type="VIRTUAL",
        spend_limit=spend_limit,
        spend_limit_duration="MONTHLY",
        memo=memo or f"AgentPay - {agent_id[:8]}",
    )

    # Save card data (PAN + CVV encrypted)
    card_data = {
        "agent_id": agent_id,
        "card_token": card.token,
        "last4": card.last_four,
        "exp_month": card.exp_month,
        "exp_year": card.exp_year,
        "pan": encrypt(card.pan) if card.pan else None,
        "cvv": encrypt(card.cvv) if card.cvv else None,
        "encrypted": True,
        "state": card.state,
        "spend_limit": spend_limit,
    }
    card_file.write_text(json.dumps(card_data))
    os.chmod(card_file, 0o600)

    logger.info(f"Created virtual card ****{card.last_four} for agent {agent_id}")

    return {
        "card_token": card.token,
        "last4": card.last_four,
        "exp_month": card.exp_month,
        "exp_year": card.exp_year,
        "state": card.state,
        "spend_limit_cents": spend_limit,
    }


def get_card_details(agent_id: str, show_full: bool = False) -> dict | None:
    """
    Get card details for an agent.
    show_full=True returns PAN + CVV (for the cardholder only).
    """
    card_file = CARDS_DIR / f"{agent_id}.json"
    if not card_file.exists():
        return None

    data = json.loads(card_file.read_text())

    result = {
        "last4": data["last4"],
        "exp_month": data["exp_month"],
        "exp_year": data["exp_year"],
        "state": data.get("state", "OPEN"),
        "spend_limit_cents": data.get("spend_limit", 0),
    }

    if show_full:
        pan_raw = data.get("pan", "N/A")
        cvv_raw = data.get("cvv", "N/A")
        if data.get("encrypted", False):
            pan_raw = decrypt(pan_raw) if pan_raw != "N/A" and pan_raw else "N/A"
            cvv_raw = decrypt(cvv_raw) if cvv_raw != "N/A" and cvv_raw else "N/A"
        result["pan"] = pan_raw
        result["cvv"] = cvv_raw

    return result


def get_card_transactions(agent_id: str, limit: int = 10) -> list[dict]:
    """Get recent transactions for an agent's card."""
    card_file = CARDS_DIR / f"{agent_id}.json"
    if not card_file.exists():
        return []

    data = json.loads(card_file.read_text())
    card_token = data["card_token"]

    try:
        client = _get_client()
        txns = client.transactions.list(card_token=card_token, page_size=limit)

        results = []
        for tx in txns:
            results.append({
                "amount_cents": tx.amount,
                "merchant": tx.merchant.descriptor if tx.merchant else "Unknown",
                "status": tx.status,
                "created": tx.created.isoformat() if tx.created else "",
            })
            if len(results) >= limit:
                break

        return results
    except Exception as e:
        logger.error(f"Failed to get card transactions: {e}")
        return []


def update_card_state(agent_id: str, state: str) -> dict:
    """
    Update card state: OPEN, PAUSED, or CLOSED.
    CLOSED is permanent.
    """
    card_file = CARDS_DIR / f"{agent_id}.json"
    if not card_file.exists():
        return {"success": False, "error": "No card found"}

    data = json.loads(card_file.read_text())
    card_token = data["card_token"]

    if state not in ("OPEN", "PAUSED", "CLOSED"):
        return {"success": False, "error": "State must be OPEN, PAUSED, or CLOSED"}

    try:
        client = _get_client()
        card = client.cards.update(card_token, state=state)

        data["state"] = card.state
        card_file.write_text(json.dumps(data))

        logger.info(f"Card ****{data['last4']} for {agent_id} → {state}")
        return {"success": True, "state": card.state}
    except Exception as e:
        logger.error(f"Failed to update card: {e}")
        return {"success": False, "error": str(e)}


def update_spend_limit(agent_id: str, limit_cents: int) -> dict:
    """Update the monthly spend limit for an agent's card."""
    card_file = CARDS_DIR / f"{agent_id}.json"
    if not card_file.exists():
        return {"success": False, "error": "No card found"}

    data = json.loads(card_file.read_text())
    card_token = data["card_token"]

    try:
        client = _get_client()
        card = client.cards.update(
            card_token,
            spend_limit=limit_cents,
            spend_limit_duration="MONTHLY",
        )

        data["spend_limit"] = limit_cents
        card_file.write_text(json.dumps(data))

        return {"success": True, "spend_limit_cents": limit_cents}
    except Exception as e:
        logger.error(f"Failed to update spend limit: {e}")
        return {"success": False, "error": str(e)}
