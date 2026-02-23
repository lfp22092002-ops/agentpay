"""
Webhook system for AgentPay.

Delivers real-time notifications when events happen (spends, deposits, approvals).
Agents register webhook URLs and get POST requests with signed payloads.
"""
import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import httpx

from config.settings import API_SECRET

logger = logging.getLogger("agentpay.webhooks")

# In-memory registry (loaded from DB on startup, synced on change)
# Format: { agent_id: { "url": "https://...", "secret": "whsec_...", "events": ["spend", "deposit", ...] } }
_webhook_registry: dict[str, dict] = {}


def generate_webhook_secret() -> str:
    import secrets
    return f"whsec_{secrets.token_hex(24)}"


def sign_payload(payload: str, secret: str) -> str:
    """HMAC-SHA256 signature for webhook verification."""
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def build_event(event_type: str, agent_id: str, data: dict) -> dict:
    """Build a standardized webhook event payload."""
    return {
        "id": f"evt_{hashlib.md5(f'{agent_id}{time.time()}'.encode()).hexdigest()[:16]}",
        "type": event_type,
        "agent_id": agent_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }


async def deliver_webhook(agent_id: str, event_type: str, data: dict):
    """
    Fire-and-forget webhook delivery with retry.
    Called after spend/deposit/approval events.
    """
    config = _webhook_registry.get(agent_id)
    if not config:
        return

    # Check if this event type is subscribed
    events = config.get("events", ["all"])
    if "all" not in events and event_type not in events:
        return

    url = config["url"]
    secret = config["secret"]
    event = build_event(event_type, agent_id, data)
    payload = json.dumps(event)
    signature = sign_payload(payload, secret)

    headers = {
        "Content-Type": "application/json",
        "X-AgentPay-Signature": signature,
        "X-AgentPay-Event": event_type,
        "User-Agent": "AgentPay-Webhook/0.1",
    }

    # Retry up to 3 times with exponential backoff
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, content=payload, headers=headers)
                if resp.status_code < 300:
                    logger.info(f"Webhook delivered: {event_type} → {url} ({resp.status_code})")
                    return
                logger.warning(f"Webhook {url} returned {resp.status_code} (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"Webhook delivery failed: {url} attempt {attempt + 1}: {e}")

        if attempt < 2:
            await asyncio.sleep(2 ** attempt)  # 1s, 2s

    logger.error(f"Webhook delivery failed after 3 attempts: {event_type} → {url}")


async def load_webhooks_from_db():
    """Load all webhook registrations from DB into memory on startup."""
    from models.database import async_session
    from models.schema import Agent
    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.webhook_url.isnot(None)))
        for agent in result.scalars().all():
            _webhook_registry[agent.id] = {
                "url": agent.webhook_url,
                "secret": agent.webhook_secret,
                "events": ["all"],
            }
    logger.info(f"Loaded {len(_webhook_registry)} webhooks from DB")


async def register_webhook(agent_id: str, url: str, secret: str, events: list[str] | None = None):
    """Register or update a webhook for an agent (in-memory + DB)."""
    _webhook_registry[agent_id] = {
        "url": url,
        "secret": secret,
        "events": events or ["all"],
    }
    # Persist to DB
    from models.database import async_session
    from models.schema import Agent
    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()
        if agent:
            agent.webhook_url = url
            agent.webhook_secret = secret
            await db.commit()
    logger.info(f"Webhook registered for {agent_id}: {url}")


async def unregister_webhook(agent_id: str):
    """Remove webhook for an agent (in-memory + DB)."""
    _webhook_registry.pop(agent_id, None)
    from models.database import async_session
    from models.schema import Agent
    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()
        if agent:
            agent.webhook_url = None
            agent.webhook_secret = None
            await db.commit()


def get_webhook_config(agent_id: str) -> dict | None:
    """Get webhook config for an agent."""
    return _webhook_registry.get(agent_id)


# ═══════════════════════════════════════
# TELEGRAM NOTIFICATIONS (always-on)
# ═══════════════════════════════════════

_bot_instance = None


def set_bot(bot):
    """Set the bot instance for sending Telegram alerts."""
    global _bot_instance
    _bot_instance = bot


async def notify_telegram(telegram_id: int, text: str):
    """Send a notification to the user via Telegram."""
    if not _bot_instance:
        logger.warning("Bot not set, can't send Telegram notification")
        return
    try:
        await _bot_instance.send_message(telegram_id, text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Telegram notification failed: {e}")


async def notify_spend(agent_name: str, telegram_id: int, amount: float, fee: float, description: str | None, remaining: float):
    """Notify user about a spend event."""
    text = (
        f"💸 <b>{agent_name}</b> spent <b>${amount:.2f}</b>\n"
        f"Fee: ${fee:.4f}\n"
    )
    if description:
        text += f"For: {description}\n"
    text += f"Remaining: ${remaining:.2f}"

    await notify_telegram(telegram_id, text)


async def notify_deposit(agent_name: str, telegram_id: int, amount: float, method: str, new_balance: float):
    """Notify user about a deposit."""
    text = (
        f"💰 <b>{agent_name}</b> received <b>${amount:.2f}</b>\n"
        f"Via: {method}\n"
        f"New balance: ${new_balance:.2f}"
    )
    await notify_telegram(telegram_id, text)


async def notify_approval_request(agent_name: str, telegram_id: int, amount: float, description: str | None, approval_id: str):
    """Send approval request with inline buttons."""
    if not _bot_instance:
        return

    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

    text = (
        f"⚠️ <b>Approval Required</b>\n\n"
        f"<b>{agent_name}</b> wants to spend <b>${amount:.2f}</b>\n"
    )
    if description:
        text += f"For: {description}\n"
    text += f"\nThis exceeds the auto-approve threshold."

    buttons = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Approve", callback_data=f"approve:{approval_id}"),
            InlineKeyboardButton(text="❌ Deny", callback_data=f"deny:{approval_id}"),
        ]
    ])

    try:
        await _bot_instance.send_message(telegram_id, text, parse_mode="HTML", reply_markup=buttons)
    except Exception as e:
        logger.error(f"Approval request notification failed: {e}")
