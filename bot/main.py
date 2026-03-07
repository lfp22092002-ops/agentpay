import asyncio
import logging
from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    Message, CallbackQuery, LabeledPrice, PreCheckoutQuery,
    InlineKeyboardMarkup, InlineKeyboardButton
)
from aiogram.filters import Command
from aiogram.enums import ParseMode
from models.database import async_session, init_db
from models.schema import PaymentMethod, Agent
from core.wallet import (
    get_or_create_user, create_agent, get_user_agents,
    deposit, spend, get_agent_transactions, get_daily_spent,
    execute_approved_spend, refund, transfer_between_agents,
    rotate_api_key
)
from core.webhooks import set_bot as set_webhook_bot
from core.approvals import resolve_approval, get_pending as get_pending_approval
from providers.telegram_stars import stars_to_usd
from providers.local_wallet import create_agent_wallet, get_wallet_address, get_wallet_balance, CHAIN_CONFIGS, SUPPORTED_EVM_CHAINS
from providers.solana_wallet import create_solana_wallet, get_solana_wallet_address, get_solana_balance
from providers.lithic_card import (
    create_virtual_card, get_card_details, update_card_state, update_spend_limit
)
from config.settings import BOT_TOKEN
from decimal import Decimal
from sqlalchemy import select, delete as sql_delete

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentpay")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
router = Router()


# ═══════════════════════════════════════
# AGENT DELETION CLEANUP
# ═══════════════════════════════════════

async def cleanup_agent(agent_id: str):
    """Clean up all resources for an agent before deletion."""
    import os
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Delete wallet files (EVM + Solana)
    wallet_file = os.path.join(_base, "data", "wallets", f"{agent_id}.json")
    if os.path.exists(wallet_file):
        try:
            os.remove(wallet_file)
            logger.info(f"Deleted EVM wallet file for {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to delete wallet file for {agent_id}: {e}")

    solana_wallet_file = os.path.join(_base, "data", "wallets", f"{agent_id}_solana.json")
    if os.path.exists(solana_wallet_file):
        try:
            os.remove(solana_wallet_file)
            logger.info(f"Deleted Solana wallet file for {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to delete Solana wallet file for {agent_id}: {e}")

    # 2. Delete card file
    card_file = os.path.join(_base, "data", "cards", f"{agent_id}.json")
    if os.path.exists(card_file):
        try:
            os.remove(card_file)
            logger.info(f"Deleted card file for {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to delete card file for {agent_id}: {e}")

    # 3. Close Lithic card if exists
    try:
        update_card_state(agent_id, "CLOSED")
        logger.info(f"Closed card for {agent_id}")
    except Exception as e:
        logger.debug(f"Card close for {agent_id} (may not exist): {e}")

    # 4. Unregister webhook
    try:
        from core.webhooks import unregister_webhook
        await unregister_webhook(agent_id)
        logger.info(f"Unregistered webhook for {agent_id}")
    except Exception as e:
        logger.debug(f"Webhook unregister for {agent_id}: {e}")


# ═══════════════════════════════════════
# START & HELP
# ═══════════════════════════════════════

@router.message(Command("start"))
async def cmd_start(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(
            db, message.from_user.id,
            message.from_user.username,
            message.from_user.first_name
        )
        agents = await get_user_agents(db, user)

    has_agents = len(agents) > 0

    if has_agents:
        total = sum(a.balance_usd for a in agents)
        active = sum(1 for a in agents if a.is_active)
        buttons = [
            [InlineKeyboardButton(text="💰 Fund Agent", callback_data="action:fund"),
             InlineKeyboardButton(text="📊 Dashboard", callback_data="action:agents")],
            [InlineKeyboardButton(text="📜 History", callback_data="action:history"),
             InlineKeyboardButton(text="⚙️ Limits", callback_data="action:limits")],
        ]
        await message.answer(
            f"👋 Welcome back!\n\n"
            f"🤖 {active} active agent{'s' if active != 1 else ''} · "
            f"💰 ${total:.2f} total\n\n"
            f"What would you like to do?",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        )
    else:
        await message.answer(
            "⚡ <b>AgentPay</b>\n\n"
            "Give your AI agent a wallet. Fund it, "
            "set spending limits, and let it pay for "
            "APIs, services, and tools autonomously.\n\n"
            "🔹 Create an agent\n"
            "🔹 Fund with Telegram Stars\n"
            "🔹 Agent spends via API\n"
            "🔹 You stay in control\n\n"
            "Ready? 👇",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🤖 Create My First Agent", callback_data="action:newagent_prompt")],
                [InlineKeyboardButton(text="📖 How It Works", callback_data="action:help")],
            ]),
        )


@router.callback_query(F.data == "action:newagent_prompt")
async def callback_newagent_prompt(callback: CallbackQuery):
    await callback.message.answer(
        "Give your agent a name:\n\n"
        "<code>/newagent trading-bot</code>\n\n"
        "Names can be anything — it's just a label for you."
    )
    await callback.answer()


@router.callback_query(F.data == "action:fund")
async def callback_action_fund(callback: CallbackQuery):
    # Delegate to the fund command
    await cmd_fund(callback.message, callback.from_user.id)
    await callback.answer()


@router.callback_query(F.data == "action:agents")
async def callback_action_agents(callback: CallbackQuery):
    await cmd_agents_for_user(callback.message, callback.from_user.id)
    await callback.answer()


@router.callback_query(F.data == "action:history")
async def callback_action_history(callback: CallbackQuery):
    await cmd_history_for_user(callback.message, callback.from_user.id)
    await callback.answer()


@router.callback_query(F.data == "action:limits")
async def callback_action_limits(callback: CallbackQuery):
    await cmd_limits_for_user(callback.message, callback.from_user.id)
    await callback.answer()


@router.callback_query(F.data == "action:help")
async def callback_action_help(callback: CallbackQuery):
    await send_help(callback.message)
    await callback.answer()


@router.message(Command("help"))
async def cmd_help(message: Message):
    await send_help(message)


async def send_help(message: Message):
    await message.answer(
        "📖 <b>AgentPay Commands</b>\n\n"
        "<b>Getting Started</b>\n"
        "/newagent <code>name</code> — Create agent\n"
        "/fund — Add funds (Stars)\n"
        "/apikey — View API key prefixes\n"
        "/rotatekey — New API key\n\n"
        "<b>Money</b>\n"
        "/balance — Check balances\n"
        "/history — Transactions\n"
        "/refund <code>id</code> — Refund a spend\n"
        "/transfer <code>from to amount</code>\n"
        "/export — CSV download\n\n"
        "<b>Integrations</b>\n"
        "/wallet — On-chain wallet\n"
        "/card — Virtual Visa card\n\n"
        "<b>Controls</b>\n"
        "/limits — View limits\n"
        "/setlimit <code>agent daily 100</code>\n"
        "/setapprove <code>agent 25</code>\n"
        "/approvals — Pending approvals\n\n"
        "<b>Account</b>\n"
        "/agents — List agents\n"
        "/stats — Usage stats\n"
        "/delete — Remove agent\n"
        "/demo — See a live demo\n\n"
        "📊 <a href='https://leofundmybot.dev/app'>Dashboard</a> · "
        "📖 <a href='https://leofundmybot.dev/docs-site'>API Docs</a>",
        disable_web_page_preview=True,
    )


# ═══════════════════════════════════════
# AGENT MANAGEMENT
# ═══════════════════════════════════════

@router.message(Command("newagent"))
async def cmd_new_agent(message: Message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "Give your agent a name:\n\n"
            "<code>/newagent trading-bot</code>"
        )
        return

    name = args[1].strip()[:50]

    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

        if len(agents) >= 5 and not user.is_pro:
            await message.answer(
                "⚠️ Free tier: max 5 agents.\n"
                "Delete one or upgrade to Pro."
            )
            return

        agent, full_key = await create_agent(db, user, name)

    buttons = [
        [InlineKeyboardButton(text="💰 Fund This Agent", callback_data=f"pick:{agent.id}")],
        [InlineKeyboardButton(text="📊 View All Agents", callback_data="action:agents")],
    ]

    await message.answer(
        f"✅ <b>{name}</b> created!\n\n"
        f"🔑 API Key:\n<code>{full_key}</code>\n\n"
        f"⚠️ Save this — shown only once.\n\n"
        f"Daily limit: ${agent.daily_limit_usd} · Per-tx: ${agent.tx_limit_usd}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.message(Command("agents"))
async def cmd_agents(message: Message):
    await cmd_agents_for_user(message, message.from_user.id)


async def cmd_agents_for_user(message: Message, user_id: int):
    async with async_session() as db:
        user = await get_or_create_user(db, user_id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer(
            "No agents yet.\n\n"
            "Create one: <code>/newagent MyBot</code>"
        )
        return

    lines = []
    total = Decimal("0")
    for a in agents:
        status = "🟢" if a.is_active else "🔴"
        lines.append(f"{status} <b>{a.name}</b> — ${a.balance_usd:.2f}")
        total += a.balance_usd

    buttons = [
        [InlineKeyboardButton(text="💰 Fund", callback_data="action:fund"),
         InlineKeyboardButton(text="➕ New Agent", callback_data="action:newagent_prompt")],
    ]

    await message.answer(
        "🤖 <b>Your Agents</b>\n\n"
        + "\n".join(lines)
        + f"\n\n💰 Total: <b>${total:.2f}</b>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.message(Command("balance"))
async def cmd_balance(message: Message):
    await cmd_agents(message)


@router.message(Command("apikey"))
async def cmd_apikey(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents. Create one: <code>/newagent MyBot</code>")
        return

    lines = []
    for a in agents:
        lines.append(f"<b>{a.name}</b>: <code>{a.api_key_prefix}</code>")

    await message.answer(
        "🔑 <b>API Key Prefixes</b>\n\n"
        + "\n".join(lines)
        + "\n\nUse /rotatekey to generate a new key.",
    )


@router.message(Command("rotatekey"))
async def cmd_rotatekey(message: Message):
    args = message.text.split(maxsplit=1) if message.text else []
    if len(args) < 2:
        async with async_session() as db:
            user = await get_or_create_user(db, message.from_user.id)
            agents = await get_user_agents(db, user)
        if not agents:
            await message.answer("No agents. Create one: <code>/newagent MyBot</code>")
            return
        buttons = [[InlineKeyboardButton(
            text=f"🔄 {a.name}",
            callback_data=f"rotate:{a.id}"
        )] for a in agents]
        buttons.append([InlineKeyboardButton(text="Cancel", callback_data="rotate:cancel")])
        await message.answer(
            "🔄 <b>Rotate key for which agent?</b>\n\n"
            "⚠️ Old key stops working immediately.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        )
        return

    name = args[1].strip()
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        result = await db.execute(
            select(Agent).where(Agent.user_id == user.id, Agent.name == name)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            await message.answer(f"Agent '{name}' not found.")
            return
        new_key = await rotate_api_key(db, agent)

    await message.answer(
        f"🔄 Key rotated for <b>{name}</b>\n\n"
        f"🔑 New key:\n<code>{new_key}</code>\n\n"
        f"⚠️ Save this — shown only once."
    )


@router.callback_query(F.data.startswith("rotate:"))
async def callback_rotate(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    if agent_id == "cancel":
        await callback.message.edit_text("Cancelled.")
        await callback.answer()
        return

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()
        if not agent:
            await callback.message.edit_text("Agent not found.")
            await callback.answer()
            return
        name = agent.name
        new_key = await rotate_api_key(db, agent)

    await callback.message.edit_text(
        f"🔄 Key rotated for <b>{name}</b>\n\n"
        f"🔑 New key:\n<code>{new_key}</code>\n\n"
        f"⚠️ Save this — shown only once."
    )
    await callback.answer()


@router.message(Command("delete"))
async def cmd_delete(message: Message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        async with async_session() as db:
            user = await get_or_create_user(db, message.from_user.id)
            agents = await get_user_agents(db, user)
        if not agents:
            await message.answer("No agents to delete.")
            return

        buttons = [[InlineKeyboardButton(
            text=f"🗑 {a.name} (${a.balance_usd:.2f})",
            callback_data=f"del:{a.id}"
        )] for a in agents]
        buttons.append([InlineKeyboardButton(text="Cancel", callback_data="del:cancel")])
        await message.answer(
            "⚠️ <b>Delete which agent?</b>\n\n"
            "This is permanent. Remaining balance will be lost.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        )
        return

    name = args[1].strip()
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        result = await db.execute(
            select(Agent).where(Agent.user_id == user.id, Agent.name == name)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            await message.answer(f"Agent '{name}' not found.")
            return

        await cleanup_agent(agent.id)
        await db.delete(agent)
        await db.commit()

    await message.answer(f"🗑 <b>{name}</b> deleted.")


@router.callback_query(F.data.startswith("del:"))
async def callback_delete(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    if agent_id == "cancel":
        await callback.message.edit_text("Cancelled.")
        await callback.answer()
        return

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()
        if agent:
            name = agent.name
            await cleanup_agent(agent.id)
            await db.delete(agent)
            await db.commit()
            await callback.message.edit_text(f"🗑 <b>{name}</b> deleted.")
        else:
            await callback.message.edit_text("Agent not found.")
    await callback.answer()


# ═══════════════════════════════════════
# FUNDING (Telegram Stars)
# ═══════════════════════════════════════

STAR_OPTIONS = [
    (50, "$0.65"),
    (200, "$2.60"),
    (500, "$6.50"),
    (1000, "$13.00"),
    (2500, "$32.50"),
]


@router.message(Command("fund"))
async def cmd_fund(message: Message, user_id: int = None):
    uid = user_id or message.from_user.id
    async with async_session() as db:
        user = await get_or_create_user(db, uid)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer(
            "Create an agent first:\n<code>/newagent MyBot</code>"
        )
        return

    buttons = [[InlineKeyboardButton(
        text=f"🤖 {a.name} · ${a.balance_usd:.2f}",
        callback_data=f"pick:{a.id}"
    )] for a in agents]

    await message.answer(
        "💰 <b>Fund Agent</b>\n\nWhich agent?",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("pick:"))
async def callback_pick_agent(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    buttons = [[InlineKeyboardButton(
        text=f"⭐ {stars}  →  {usd_str}",
        callback_data=f"fund:{agent_id}:{stars}"
    )] for stars, usd_str in STAR_OPTIONS]

    await callback.message.edit_text(
        "💰 <b>Choose amount</b>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("fund:"))
async def callback_fund(callback: CallbackQuery):
    _, agent_id, stars_str = callback.data.split(":")
    stars = int(stars_str)
    usd = stars_to_usd(stars)

    await callback.message.answer_invoice(
        title=f"Fund Agent — ⭐{stars}",
        description=f"Add ${usd:.2f} to your agent's wallet",
        payload=f"fund:{agent_id}:{stars}",
        currency="XTR",
        prices=[LabeledPrice(label=f"{stars} Stars", amount=stars)],
    )
    await callback.answer()


@router.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery):
    await query.answer(ok=True)


@router.message(F.successful_payment)
async def successful_payment(message: Message):
    payload = message.successful_payment.invoice_payload
    _, agent_id, stars_str = payload.split(":")
    stars = int(stars_str)
    usd = stars_to_usd(stars)

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

        if agent:
            tx = await deposit(
                db, agent, usd,
                PaymentMethod.TELEGRAM_STARS,
                external_ref=message.successful_payment.telegram_payment_charge_id,
                description=f"⭐ {stars} Stars deposit",
            )

            buttons = [
                [InlineKeyboardButton(text="📊 View Agents", callback_data="action:agents")],
            ]

            await message.answer(
                f"✅ <b>${usd:.2f}</b> added to <b>{agent.name}</b>\n\n"
                f"New balance: <b>${agent.balance_usd:.2f}</b>",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
            )
        else:
            await message.answer("❌ Agent not found. Contact support.")


# ═══════════════════════════════════════
# HISTORY & STATS
# ═══════════════════════════════════════

@router.message(Command("history"))
async def cmd_history(message: Message):
    await cmd_history_for_user(message, message.from_user.id)


async def cmd_history_for_user(message: Message, user_id: int):
    async with async_session() as db:
        user = await get_or_create_user(db, user_id)
        agents = await get_user_agents(db, user)

        if not agents:
            await message.answer("No agents yet. Create one: <code>/newagent MyBot</code>")
            return

        lines = ["📜 <b>Recent Activity</b>\n"]
        found_any = False
        for a in agents:
            txs = await get_agent_transactions(db, a, limit=5)
            if txs:
                found_any = True
                lines.append(f"<b>{a.name}</b>")
                for tx in txs:
                    icon = "↙" if tx.tx_type.value == "deposit" else "↗"
                    lines.append(
                        f"  {icon} ${tx.amount_usd:.2f} · {tx.description or 'N/A'}"
                    )
                lines.append("")

    if not found_any:
        lines.append("No transactions yet. Fund an agent with /fund")

    await message.answer("\n".join(lines))


@router.message(Command("stats"))
async def cmd_stats(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents yet.")
        return

    total_balance = sum(a.balance_usd for a in agents)
    active = sum(1 for a in agents if a.is_active)
    tier = "⭐ Pro" if user.is_pro else "Free"

    await message.answer(
        "📊 <b>Account</b>\n\n"
        f"Plan: {tier}\n"
        f"Agents: {active} active / {len(agents)} total\n"
        f"Balance: ${total_balance:.2f}\n"
        f"Joined: {user.created_at.strftime('%b %d, %Y')}",
    )


# ═══════════════════════════════════════
# DEMO COMMAND
# ═══════════════════════════════════════

@router.message(Command("demo"))
async def cmd_demo(message: Message):
    """Simulate an agent spending money in real-time."""
    msg = await message.answer(
        "🎬 <b>Live Demo — Agent Spending</b>\n\n"
        "⏳ Initializing agent <b>demo-bot</b>..."
    )
    await asyncio.sleep(1.5)

    await msg.edit_text(
        "🎬 <b>Live Demo — Agent Spending</b>\n\n"
        "✅ Agent <b>demo-bot</b> connected\n"
        "💰 Balance: $50.00\n\n"
        "⏳ Agent requesting API call..."
    )
    await asyncio.sleep(2)

    await msg.edit_text(
        "🎬 <b>Live Demo — Agent Spending</b>\n\n"
        "✅ Agent <b>demo-bot</b> connected\n"
        "💰 Balance: $50.00\n\n"
        "↗ <b>SPEND</b> $2.50 — GPT-4 API call\n"
        "   Status: ✅ completed\n"
        "   Balance: $47.50\n\n"
        "⏳ Agent making another purchase..."
    )
    await asyncio.sleep(2)

    await msg.edit_text(
        "🎬 <b>Live Demo — Agent Spending</b>\n\n"
        "✅ Agent <b>demo-bot</b> connected\n"
        "💰 Balance: $50.00\n\n"
        "↗ <b>SPEND</b> $2.50 — GPT-4 API call ✅\n"
        "↗ <b>SPEND</b> $0.75 — Web scraping (Firecrawl) ✅\n"
        "   Balance: $46.75\n\n"
        "⏳ Large purchase detected..."
    )
    await asyncio.sleep(2)

    await msg.edit_text(
        "🎬 <b>Live Demo — Agent Spending</b>\n\n"
        "✅ Agent <b>demo-bot</b> connected\n"
        "💰 Balance: $50.00\n\n"
        "↗ $2.50 — GPT-4 API call ✅\n"
        "↗ $0.75 — Web scraping ✅\n"
        "⚠️ <b>APPROVAL NEEDED</b> $35.00 — Cloud GPU rental\n"
        "   Exceeds auto-approve limit ($25)\n"
        "   → Sent to owner for approval",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="✅ Approve", callback_data="demo:approve"),
             InlineKeyboardButton(text="🚫 Deny", callback_data="demo:deny")],
        ]),
    )


@router.callback_query(F.data == "demo:approve")
async def callback_demo_approve(callback: CallbackQuery):
    await callback.message.edit_text(
        "🎬 <b>Demo Complete!</b>\n\n"
        "↗ $2.50 — GPT-4 API call ✅\n"
        "↗ $0.75 — Web scraping ✅\n"
        "↗ $35.00 — Cloud GPU rental ✅ <i>(approved)</i>\n\n"
        "💰 Final balance: $11.75\n"
        "📊 3 transactions · $38.25 spent\n\n"
        "<i>That's AgentPay — your agent spends, you stay in control.</i>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🤖 Create My Agent", callback_data="action:newagent_prompt")],
        ]),
    )
    await callback.answer("Approved! ✅")


@router.callback_query(F.data == "demo:deny")
async def callback_demo_deny(callback: CallbackQuery):
    await callback.message.edit_text(
        "🎬 <b>Demo Complete!</b>\n\n"
        "↗ $2.50 — GPT-4 API call ✅\n"
        "↗ $0.75 — Web scraping ✅\n"
        "🚫 $35.00 — Cloud GPU rental ❌ <i>(denied)</i>\n\n"
        "💰 Final balance: $46.75\n"
        "📊 2 transactions · $3.25 spent\n\n"
        "<i>You denied it — no funds were deducted. Full control.</i>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🤖 Create My Agent", callback_data="action:newagent_prompt")],
        ]),
    )
    await callback.answer("Denied! 🚫")


# ═══════════════════════════════════════
# REVENUE (Admin — G only)
# ═══════════════════════════════════════

@router.message(Command("revenue"))
async def cmd_revenue(message: Message):
    if message.from_user.id != 5360481016:
        return

    from models.schema import PlatformRevenue
    from sqlalchemy import select, func
    from decimal import Decimal
    from datetime import datetime, timezone

    async with async_session() as db:
        total = await db.execute(select(func.sum(PlatformRevenue.amount_usd)))
        total_usd = total.scalar() or Decimal("0")

        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_result = await db.execute(
            select(func.sum(PlatformRevenue.amount_usd)).where(PlatformRevenue.created_at >= today)
        )
        today_usd = today_result.scalar() or Decimal("0")

        count_result = await db.execute(select(func.count(PlatformRevenue.id)))
        tx_count = count_result.scalar() or 0

    await message.answer(
        "💰 <b>Revenue</b>\n\n"
        f"Total: <b>${total_usd:.4f}</b>\n"
        f"Today: <b>${today_usd:.4f}</b>\n"
        f"Transactions: {tx_count}\n\n"
        f"Cashout: <code>0xD51B231F317260FB86b47A38F14eA29Cc81E0073</code>\n"
        f"<i>2% per agent spend</i>"
    )


# ═══════════════════════════════════════
# LIMITS
# ═══════════════════════════════════════

@router.message(Command("limits"))
async def cmd_limits(message: Message):
    await cmd_limits_for_user(message, message.from_user.id)


async def cmd_limits_for_user(message: Message, user_id: int):
    async with async_session() as db:
        user = await get_or_create_user(db, user_id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents yet.")
        return

    lines = ["⚙️ <b>Spending Limits</b>\n"]
    for a in agents:
        lines.append(
            f"<b>{a.name}</b>\n"
            f"  Per tx: ${a.tx_limit_usd:.2f} · Daily: ${a.daily_limit_usd:.2f}\n"
        )

    await message.answer(
        "\n".join(lines) + "\nChange: /setlimit <code>Agent daily 100</code>",
    )


@router.message(Command("setlimit"))
async def cmd_setlimit(message: Message):
    args = message.text.split()
    if len(args) < 4:
        await message.answer(
            "Usage:\n"
            "<code>/setlimit AgentName daily 100</code>\n"
            "<code>/setlimit AgentName tx 25</code>"
        )
        return

    name = args[1]
    limit_type = args[2].lower()
    try:
        value = Decimal(args[3])
    except Exception:
        await message.answer("Invalid amount.")
        return

    if limit_type not in ("daily", "tx"):
        await message.answer("Use <code>daily</code> or <code>tx</code>")
        return

    if value <= 0 or value > 10000:
        await message.answer("Must be $0.01–$10,000")
        return

    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        result = await db.execute(
            select(Agent).where(Agent.user_id == user.id, Agent.name == name)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            await message.answer(f"Agent '{name}' not found.")
            return

        if limit_type == "daily":
            agent.daily_limit_usd = value
        else:
            agent.tx_limit_usd = value

        await db.commit()

    label = "Daily limit" if limit_type == "daily" else "Per-tx limit"
    await message.answer(f"✅ <b>{name}</b> {label}: ${value:.2f}")


# ═══════════════════════════════════════
# WALLET (Multi-Chain)
# ═══════════════════════════════════════

CHAIN_LABELS = {
    "base": ("🔵", "Base"),
    "polygon": ("🟣", "Polygon"),
    "bnb": ("🟡", "BNB Chain"),
    "solana": ("🟢", "Solana"),
}


@router.message(Command("wallet"))
async def cmd_wallet(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents. Create one: <code>/newagent MyBot</code>")
        return

    buttons = []
    for a in agents:
        evm_addr = get_wallet_address(a.id)
        sol_addr = get_solana_wallet_address(a.id)
        if evm_addr or sol_addr:
            label = f"📊 {a.name}"
            if evm_addr:
                label += f" · {evm_addr[:6]}…{evm_addr[-4:]}"
            buttons.append([InlineKeyboardButton(
                text=label,
                callback_data=f"wchains:{a.id}"
            )])
        else:
            buttons.append([InlineKeyboardButton(
                text=f"🔗 Create wallet — {a.name}",
                callback_data=f"wcreate:{a.id}"
            )])

    await message.answer(
        "🔗 <b>On-Chain Wallets</b>\n\n"
        "🔵 Base  🟣 Polygon  🟡 BNB  🟢 Solana\n\n"
        "EVM chains share one address. Solana is separate.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("wchains:"))
async def callback_wallet_chains(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    buttons = [[InlineKeyboardButton(
        text=f"{emoji} {name}",
        callback_data=f"winfo:{agent_id}:{chain_key}"
    )] for chain_key, (emoji, name) in CHAIN_LABELS.items()]

    await callback.message.edit_text(
        "🔗 <b>Select chain:</b>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("wcreate:"))
async def callback_create_wallet(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

    if not agent:
        await callback.answer("Agent not found", show_alert=True)
        return

    await callback.message.edit_text("⏳ Creating wallets...")

    try:
        evm_info = create_agent_wallet(agent_id)
        evm_address = evm_info["address"]

        sol_info = create_solana_wallet(agent_id)
        sol_address = sol_info["address"]

        async with async_session() as db:
            from models.schema import Wallet
            result = await db.execute(
                select(Wallet).where(Wallet.agent_id == agent_id)
            )
            wallet = result.scalar_one_or_none()
            if wallet:
                wallet.wallet_type = "usdc"
                wallet.address = evm_address
            else:
                wallet = Wallet(
                    agent_id=agent_id,
                    wallet_type="usdc",
                    address=evm_address,
                )
                db.add(wallet)
            await db.commit()

        await callback.message.edit_text(
            f"✅ <b>Wallets ready — {agent.name}</b>\n\n"
            f"🔵🟣🟡 EVM:\n<code>{evm_address}</code>\n\n"
            f"🟢 Solana:\n<code>{sol_address}</code>\n\n"
            f"Check balances: /wallet"
        )
    except Exception as e:
        logger.error(f"Wallet creation failed: {e}", exc_info=True)
        await callback.message.edit_text(f"❌ Failed: {e}")

    await callback.answer()


@router.callback_query(F.data.startswith("winfo:"))
async def callback_wallet_info(callback: CallbackQuery):
    parts = callback.data.split(":")
    agent_id = parts[1]
    chain = parts[2] if len(parts) > 2 else "base"

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

    if not agent:
        await callback.answer("Agent not found", show_alert=True)
        return

    emoji, chain_name = CHAIN_LABELS.get(chain, ("🔵", chain))

    if chain == "solana":
        address = get_solana_wallet_address(agent_id)
        if not address:
            sol_info = create_solana_wallet(agent_id)
            address = sol_info["address"]
        balance_info = get_solana_balance(agent_id)
        native_token = "SOL"
        native_balance = balance_info.get("balance_sol", "0")
    else:
        address = get_wallet_address(agent_id)
        if not address:
            evm_info = create_agent_wallet(agent_id, chain=chain)
            address = evm_info["address"]
        balance_info = get_wallet_balance(agent_id, chain=chain)
        native_token = balance_info.get("native_token", "ETH")
        native_balance = balance_info.get("balance_native", "0")

    usdc_balance = balance_info.get("balance_usdc", "0")

    buttons = [[InlineKeyboardButton(text="← Back", callback_data=f"wchains:{agent_id}")]]

    await callback.message.edit_text(
        f"{emoji} <b>{agent.name} · {chain_name}</b>\n\n"
        f"<code>{address}</code>\n\n"
        f"{native_token}: {native_balance}\n"
        f"USDC: {usdc_balance}\n\n"
        f"Send tokens to this address on {chain_name}.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )
    await callback.answer()


# ═══════════════════════════════════════
# VIRTUAL CARD (Lithic)
# ═══════════════════════════════════════

@router.message(Command("card"))
async def cmd_card(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents. Create one: <code>/newagent MyBot</code>")
        return

    buttons = []
    for a in agents:
        details = get_card_details(a.id)
        if details:
            state_icon = "🟢" if details["state"] == "OPEN" else "🔴" if details["state"] == "CLOSED" else "⏸"
            buttons.append([InlineKeyboardButton(
                text=f"{state_icon} {a.name} · ****{details['last4']}",
                callback_data=f"cinfo:{a.id}"
            )])
        else:
            buttons.append([InlineKeyboardButton(
                text=f"💳 Create card — {a.name}",
                callback_data=f"ccreate:{a.id}"
            )])

    await message.answer(
        "💳 <b>Virtual Cards</b>\n\n"
        "Issue a Visa card your agent can use for online purchases.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("ccreate:"))
async def callback_create_card(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

    if not agent:
        await callback.answer("Agent not found", show_alert=True)
        return

    await callback.message.edit_text("⏳ Creating card...")

    try:
        card_info = create_virtual_card(
            agent_id,
            spend_limit=5000,
            memo=f"AgentPay - {agent.name}",
        )

        limit_usd = card_info.get("spend_limit_cents", 5000) / 100

        await callback.message.edit_text(
            f"✅ <b>Card created — {agent.name}</b>\n\n"
            f"Card: ****{card_info['last4']}\n"
            f"Expires: {card_info['exp_month']}/{card_info['exp_year']}\n"
            f"Limit: ${limit_usd:.0f}/mo\n\n"
            f"Use /card to manage."
        )
    except ValueError as e:
        await callback.message.edit_text(
            f"⚠️ Lithic API key needed\n\n"
            f"Get one at: https://app.lithic.com/signup"
        )
    except Exception as e:
        logger.error(f"Card creation failed: {e}", exc_info=True)
        await callback.message.edit_text(f"❌ Failed: {e}")

    await callback.answer()


@router.callback_query(F.data.startswith("cinfo:"))
async def callback_card_info(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

    if not agent:
        await callback.answer("Agent not found", show_alert=True)
        return

    details = get_card_details(agent_id)
    if not details:
        await callback.answer("No card found", show_alert=True)
        return

    state_icon = "🟢" if details["state"] == "OPEN" else "🔴" if details["state"] == "CLOSED" else "⏸"
    limit_usd = details.get("spend_limit_cents", 0) / 100

    buttons = []
    if details["state"] == "OPEN":
        buttons.append([InlineKeyboardButton(text="👁 Show Details", callback_data=f"cfull:{agent_id}")])
        buttons.append([
            InlineKeyboardButton(text="⏸ Pause", callback_data=f"cpause:{agent_id}"),
            InlineKeyboardButton(text="🔴 Close", callback_data=f"cclose:{agent_id}"),
        ])
    elif details["state"] == "PAUSED":
        buttons.append([
            InlineKeyboardButton(text="▶️ Resume", callback_data=f"cresume:{agent_id}"),
            InlineKeyboardButton(text="🔴 Close", callback_data=f"cclose:{agent_id}"),
        ])

    await callback.message.edit_text(
        f"💳 <b>{agent.name}</b>\n\n"
        f"****{details['last4']} · {details['exp_month']}/{details['exp_year']}\n"
        f"Status: {state_icon} {details['state']}\n"
        f"Limit: ${limit_usd:.0f}/mo",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons) if buttons else None,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("cfull:"))
async def callback_card_full(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    details = get_card_details(agent_id, show_full=True)
    if not details:
        await callback.answer("No card found", show_alert=True)
        return

    msg = await callback.message.answer(
        f"🔐 <b>Card Details</b>\n\n"
        f"Number: <code>{details.get('pan', 'N/A')}</code>\n"
        f"Expiry: <code>{details['exp_month']}/{details['exp_year']}</code>\n"
        f"CVV: <code>{details.get('cvv', 'N/A')}</code>\n\n"
        f"⚠️ Deletes in 60s"
    )

    asyncio.get_event_loop().call_later(
        60, lambda: asyncio.ensure_future(msg.delete())
    )
    await callback.answer()


@router.callback_query(F.data.startswith("cpause:"))
async def callback_pause_card(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    result = update_card_state(agent_id, "PAUSED")
    if result["success"]:
        await callback.message.edit_text("⏸ Card paused. /card to resume.")
    else:
        await callback.answer(f"Error: {result['error']}", show_alert=True)
    await callback.answer()


@router.callback_query(F.data.startswith("cresume:"))
async def callback_resume_card(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    result = update_card_state(agent_id, "OPEN")
    if result["success"]:
        await callback.message.edit_text("▶️ Card resumed!")
    else:
        await callback.answer(f"Error: {result['error']}", show_alert=True)
    await callback.answer()


@router.callback_query(F.data.startswith("cclose:"))
async def callback_close_card(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    buttons = [[
        InlineKeyboardButton(text="⚠️ Yes, close", callback_data=f"ccloseconfirm:{agent_id}"),
        InlineKeyboardButton(text="Cancel", callback_data=f"cinfo:{agent_id}"),
    ]]
    await callback.message.edit_text(
        "⚠️ <b>Close card permanently?</b>\n\nCan't be undone.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("ccloseconfirm:"))
async def callback_close_card_confirm(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    result = update_card_state(agent_id, "CLOSED")
    if result["success"]:
        await callback.message.edit_text("🔴 Card closed permanently.")
    else:
        await callback.answer(f"Error: {result['error']}", show_alert=True)
    await callback.answer()


# ═══════════════════════════════════════
# APPROVAL WORKFLOW
# ═══════════════════════════════════════

@router.callback_query(F.data.startswith("approve:"))
async def callback_approve(callback: CallbackQuery):
    approval_id = callback.data.split(":")[1]
    approval = get_pending_approval(approval_id)

    if not approval:
        await callback.answer("Expired or resolved", show_alert=True)
        return

    if approval.telegram_id != callback.from_user.id:
        await callback.answer("Not your approval", show_alert=True)
        return

    resolve_approval(approval_id, approved=True)

    async with async_session() as db:
        tx, error = await execute_approved_spend(
            db, approval.agent_id, approval.amount_usd, approval.description
        )

    if error:
        await callback.message.edit_text(f"❌ Approved but failed: {error}")
    else:
        await callback.message.edit_text(
            f"✅ <b>Approved</b>\n\n"
            f"<b>{approval.agent_name}</b> spent ${float(approval.amount_usd):.2f}\n"
            f"{'· ' + approval.description if approval.description else ''}"
        )
    await callback.answer()


@router.callback_query(F.data.startswith("deny:"))
async def callback_deny(callback: CallbackQuery):
    approval_id = callback.data.split(":")[1]
    approval = get_pending_approval(approval_id)

    if not approval:
        await callback.answer("Expired or resolved", show_alert=True)
        return

    if approval.telegram_id != callback.from_user.id:
        await callback.answer("Not your approval", show_alert=True)
        return

    resolve_approval(approval_id, approved=False, reason="Denied by user")

    await callback.message.edit_text(
        f"🚫 <b>Denied</b>\n\n"
        f"<b>{approval.agent_name}</b> — ${float(approval.amount_usd):.2f}\n"
        f"No funds deducted."
    )
    await callback.answer()


@router.message(Command("approvals"))
async def cmd_approvals(message: Message):
    from core.approvals import get_user_pending
    pending = get_user_pending(message.from_user.id)

    if not pending:
        await message.answer("✅ No pending approvals.")
        return

    text = f"⏳ <b>{len(pending)} Pending</b>\n\n"
    for a in pending:
        text += f"• <b>{a.agent_name}</b>: ${float(a.amount_usd):.2f}\n  {a.description or 'No description'}\n\n"
    await message.answer(text)


@router.message(Command("setapprove"))
async def cmd_set_approve(message: Message):
    parts = message.text.split(maxsplit=2) if message.text else []
    if len(parts) < 3:
        await message.answer(
            "Usage: <code>/setapprove AgentName 25</code>\n\n"
            "Spends above this amount need your approval."
        )
        return

    agent_name = parts[1]
    try:
        threshold = Decimal(parts[2])
    except Exception:
        await message.answer("Invalid amount.")
        return

    if threshold < 0 or threshold > 10000:
        await message.answer("Must be $0–$10,000.")
        return

    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    agent = next((a for a in agents if a.name.lower() == agent_name.lower()), None)
    if not agent:
        await message.answer(f"Agent '{agent_name}' not found.")
        return

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent.id))
        a = result.scalar_one()
        a.auto_approve_usd = threshold
        await db.commit()

    if threshold == 0:
        await message.answer(f"✅ <b>{agent.name}</b>: all spends need approval")
    else:
        await message.answer(f"✅ <b>{agent.name}</b>: auto-approve up to ${threshold}")


# ═══════════════════════════════════════
# REFUND & TRANSFER
# ═══════════════════════════════════════

@router.message(Command("refund"))
async def cmd_refund(message: Message):
    parts = message.text.split(maxsplit=1) if message.text else []
    if len(parts) < 2:
        await message.answer(
            "Usage: <code>/refund TransactionID</code>\n\n"
            "Find IDs in /history or /export"
        )
        return

    tx_id = parts[1].strip()

    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents.")
        return

    for agent in agents:
        async with async_session() as db:
            result = await db.execute(select(Agent).where(Agent.id == agent.id))
            a = result.scalar_one()
            tx, error = await refund(db, a, tx_id)
            if tx:
                await message.answer(
                    f"✅ <b>Refunded</b>\n\n"
                    f"{a.name}: +${float(tx.amount_usd):.2f}\n"
                    f"Balance: ${float(a.balance_usd):.2f}"
                )
                return
            if error and "not found" not in error.lower():
                await message.answer(f"❌ {error}")
                return

    await message.answer("❌ Transaction not found.")


@router.message(Command("transfer"))
async def cmd_transfer(message: Message):
    parts = message.text.split() if message.text else []
    if len(parts) < 4:
        await message.answer(
            "Usage: <code>/transfer From To Amount</code>\n\n"
            "Example: <code>/transfer MyBot Helper 5.00</code>"
        )
        return

    from_name = parts[1]
    to_name = parts[2]
    try:
        amount = Decimal(parts[3])
    except Exception:
        await message.answer("Invalid amount.")
        return

    if amount <= 0:
        await message.answer("Amount must be positive.")
        return

    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    from_agent = next((a for a in agents if a.name.lower() == from_name.lower()), None)
    to_agent = next((a for a in agents if a.name.lower() == to_name.lower()), None)

    if not from_agent:
        await message.answer(f"Agent '{from_name}' not found.")
        return
    if not to_agent:
        await message.answer(f"Agent '{to_name}' not found.")
        return
    if from_agent.id == to_agent.id:
        await message.answer("Can't transfer to the same agent.")
        return

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == from_agent.id))
        fa = result.scalar_one()
        tx, error = await transfer_between_agents(db, fa, to_agent.id, amount, f"Transfer to {to_agent.name}")

    if error:
        await message.answer(f"❌ {error}")
    else:
        await message.answer(
            f"✅ ${float(amount):.2f}: <b>{from_agent.name}</b> → <b>{to_agent.name}</b>"
        )


@router.message(Command("export"))
async def cmd_export(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents.")
        return

    import csv, io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "agent", "type", "amount", "fee", "description", "status", "id"])

    for agent in agents:
        async with async_session() as db:
            txs = await get_agent_transactions(db, agent, limit=500)
            for tx in txs:
                writer.writerow([
                    tx.created_at.isoformat(),
                    agent.name,
                    tx.tx_type.value,
                    float(tx.amount_usd),
                    float(tx.fee_usd),
                    tx.description or "",
                    tx.status.value,
                    tx.id,
                ])

    csv_bytes = output.getvalue().encode("utf-8")
    from aiogram.types import BufferedInputFile
    doc = BufferedInputFile(csv_bytes, filename="agentpay-transactions.csv")
    await message.answer_document(doc, caption="📄 Transaction history")


# ═══════════════════════════════════════
# CATCH-ALL
# ═══════════════════════════════════════

@router.message()
async def catch_all(message: Message):
    logger.info(f"Unhandled message from {message.from_user.id}: {message.text!r}")
    if message.text and message.text.startswith("/"):
        await message.answer(
            "Unknown command. Try /help",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📖 Help", callback_data="action:help")],
            ]),
        )


dp.include_router(router)


@dp.errors()
async def error_handler(event, exception):
    logger.error(f"Error handling update: {exception}", exc_info=True)
    return True


async def main():
    logger.info("🚀 AgentPay starting...")
    await init_db()
    logger.info("✅ Database initialized")

    set_webhook_bot(bot)
    logger.info("✅ Webhook notifications enabled")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
