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
    execute_approved_spend, refund, transfer_between_agents
)
from core.webhooks import set_bot as set_webhook_bot
from core.approvals import resolve_approval, get_pending as get_pending_approval
from providers.telegram_stars import stars_to_usd
from providers.local_wallet import create_agent_wallet, get_wallet_address, get_wallet_balance
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

    # 1. Delete wallet file
    wallet_file = os.path.join(_base, "data", "wallets", f"{agent_id}.json")
    if os.path.exists(wallet_file):
        try:
            os.remove(wallet_file)
            logger.info(f"Deleted wallet file for {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to delete wallet file for {agent_id}: {e}")

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

    await message.answer(
        "⚡ <b>AgentPay</b> — Give your AI agent a wallet\n\n"
        "Fund your agents, let them spend autonomously.\n\n"
        "🔹 /newagent — Create an agent\n"
        "🔹 /agents — List your agents\n"
        "🔹 /fund — Add funds (Telegram Stars)\n"
        "🔹 /balance — Check balances\n"
        "🔹 /history — Transaction history\n"
        "🔹 /apikey — Get your agent's API key\n"
        "🔹 /limits — View/set spending limits\n"
        "🔹 /setlimit — Change limits\n"
        "🔹 /delete — Delete an agent\n"
        "🔹 /stats — Account stats\n"
        "🔹 /help — Full command list\n\n"
        "Ready? Create your first agent with /newagent 👇"
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "💳 <b>AgentPay — Fund Your AI Agents</b>\n\n"
        "<b>🚀 Getting Started</b>\n"
        "/newagent — Create an agent\n"
        "/fund — Add funds (Telegram Stars)\n"
        "/apikey — Get your API key\n\n"
        "<b>💰 Money</b>\n"
        "/balance — Check balances\n"
        "/history — Transaction history\n"
        "/refund — Refund a transaction\n"
        "/transfer — Move funds between agents\n"
        "/export — Download transactions as CSV\n\n"
        "<b>🔗 Integrations</b>\n"
        "/wallet — On-chain USDC wallet (Base)\n"
        "/card — Virtual Visa card\n\n"
        "<b>⚙️ Settings</b>\n"
        "/limits — View spending limits\n"
        "/setlimit — Change limits\n"
        "/setapprove — Auto-approve threshold\n"
        "/approvals — Pending approvals\n\n"
        "<b>📊 Account</b>\n"
        "/agents — List your agents\n"
        "/stats — Usage statistics\n"
        "/delete — Remove an agent\n\n"
        "<b>🌐 Links</b>\n"
        "📊 Dashboard: leofundmybot.dev/app\n"
        "📖 API Docs: leofundmybot.dev/docs-site\n"
        "🤖 SDK: pip install agentpay",
        parse_mode="HTML",
    )


# ═══════════════════════════════════════
# AGENT MANAGEMENT
# ═══════════════════════════════════════

@router.message(Command("newagent"))
async def cmd_new_agent(message: Message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "Usage: <code>/newagent MyBot</code>\n"
            "Example: <code>/newagent trading-bot</code>\n\n"
            "💡 Tap the command above to copy it."
        )
        return

    name = args[1].strip()[:50]

    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

        if len(agents) >= 5 and not user.is_pro:
            await message.answer("⚠️ Free tier: max 5 agents. Upgrade to Pro for unlimited.")
            return

        agent = await create_agent(db, user, name)

    await message.answer(
        f"✅ Agent <b>{name}</b> created!\n\n"
        f"🔑 API Key:\n<code>{agent.api_key}</code>\n\n"
        f"💰 Balance: $0.00\n"
        f"📊 Daily limit: ${agent.daily_limit_usd}\n"
        f"📊 Per-tx limit: ${agent.tx_limit_usd}\n\n"
        f"Fund it with /fund to start spending."
    )


@router.message(Command("agents"))
async def cmd_agents(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents yet. Create one with <code>/newagent MyBot</code>")
        return

    lines = ["🤖 <b>Your Agents</b>\n"]
    for a in agents:
        status = "🟢" if a.is_active else "🔴"
        lines.append(f"{status} <b>{a.name}</b> — ${a.balance_usd:.2f}")

    total = sum(a.balance_usd for a in agents)
    lines.append(f"\n💰 Total: ${total:.2f}")
    await message.answer("\n".join(lines))


@router.message(Command("balance"))
async def cmd_balance(message: Message):
    await cmd_agents(message)


@router.message(Command("apikey"))
async def cmd_apikey(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents. Create one with <code>/newagent MyBot</code>")
        return

    lines = ["🔑 <b>API Keys</b> (keep secret!)\n"]
    for a in agents:
        lines.append(f"<b>{a.name}</b>:\n<code>{a.api_key}</code>\n")

    await message.answer("\n".join(lines))


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

        buttons = []
        for a in agents:
            buttons.append([
                InlineKeyboardButton(
                    text=f"🗑 {a.name} (${a.balance_usd:.2f})",
                    callback_data=f"del:{a.id}"
                )
            ])
        buttons.append([InlineKeyboardButton(text="❌ Cancel", callback_data="del:cancel")])
        await message.answer(
            "⚠️ <b>Delete which agent?</b>\n\nThis is permanent. Remaining balance will be lost.",
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
            await message.answer(f"Agent '{name}' not found. Check /agents")
            return

        await cleanup_agent(agent.id)
        await db.delete(agent)
        await db.commit()

    await message.answer(f"🗑 Agent <b>{name}</b> deleted.")


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
            await callback.message.edit_text(f"🗑 Agent <b>{name}</b> deleted.")
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
async def cmd_fund(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("Create an agent first: <code>/newagent MyBot</code>")
        return

    buttons = []
    for a in agents:
        buttons.append([InlineKeyboardButton(
            text=f"🤖 {a.name} (${a.balance_usd:.2f})",
            callback_data=f"pick:{a.id}"
        )])

    await message.answer(
        "💫 <b>Fund an Agent</b>\n\nPick which agent to fund:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("pick:"))
async def callback_pick_agent(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    buttons = []
    for stars, usd_str in STAR_OPTIONS:
        buttons.append([InlineKeyboardButton(
            text=f"⭐ {stars} ({usd_str})",
            callback_data=f"fund:{agent_id}:{stars}"
        )])

    await callback.message.edit_text(
        "💫 <b>Choose amount:</b>",
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
            await message.answer(
                f"✅ <b>Funded!</b>\n\n"
                f"Agent: {agent.name}\n"
                f"Added: ${usd:.2f} ({stars} ⭐)\n"
                f"New balance: ${agent.balance_usd:.2f}"
            )
        else:
            await message.answer("❌ Agent not found. Contact support.")


# ═══════════════════════════════════════
# HISTORY & STATS
# ═══════════════════════════════════════

@router.message(Command("history"))
async def cmd_history(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

        if not agents:
            await message.answer("No agents. Create one with <code>/newagent MyBot</code>")
            return

        lines = ["📜 <b>Recent Transactions</b>\n"]
        found_any = False
        for a in agents:
            txs = await get_agent_transactions(db, a, limit=5)
            if txs:
                found_any = True
                lines.append(f"<b>{a.name}</b>:")
                for tx in txs:
                    icon = "💰" if tx.tx_type.value == "deposit" else "💸"
                    lines.append(
                        f"  {icon} {tx.tx_type.value.upper()} ${tx.amount_usd:.2f}"
                        f" — {tx.description or 'N/A'}"
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

    lines = [
        "📊 <b>Account Stats</b>\n",
        f"👤 Plan: {tier}",
        f"🤖 Agents: {active} active / {len(agents)} total",
        f"💰 Total balance: ${total_balance:.2f}",
        f"\nJoined: {user.created_at.strftime('%Y-%m-%d')}",
    ]
    await message.answer("\n".join(lines))


# ═══════════════════════════════════════
# REVENUE (Admin — G only)
# ═══════════════════════════════════════

@router.message(Command("revenue"))
async def cmd_revenue(message: Message):
    if message.from_user.id != 5360481016:
        return  # silent ignore

    from models.schema import PlatformRevenue
    from sqlalchemy import select, func
    from decimal import Decimal
    from datetime import datetime

    async with async_session() as db:
        # Total
        total = await db.execute(select(func.sum(PlatformRevenue.amount_usd)))
        total_usd = total.scalar() or Decimal("0")

        # Today
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_result = await db.execute(
            select(func.sum(PlatformRevenue.amount_usd)).where(PlatformRevenue.created_at >= today)
        )
        today_usd = today_result.scalar() or Decimal("0")

        # Count
        count_result = await db.execute(select(func.count(PlatformRevenue.id)))
        tx_count = count_result.scalar() or 0

    lines = [
        "💰 <b>Platform Revenue</b>\n",
        f"📈 Total earned: <b>${total_usd:.4f}</b>",
        f"📅 Today: <b>${today_usd:.4f}</b>",
        f"🔢 Fee transactions: {tx_count}",
        f"\n💳 Cashout wallet:",
        f"<code>0xD51B231F317260FB86b47A38F14eA29Cc81E0073</code>",
        f"\n<i>Revenue = 2% of every agent spend. Fees tracked per-transaction.</i>",
    ]
    await message.answer("\n".join(lines))


# ═══════════════════════════════════════
# LIMITS
# ═══════════════════════════════════════

@router.message(Command("limits"))
async def cmd_limits(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents yet.")
        return

    lines = ["⚙️ <b>Spending Limits</b>\n"]
    for a in agents:
        lines.append(
            f"<b>{a.name}</b>\n"
            f"  Per transaction: ${a.tx_limit_usd:.2f}\n"
            f"  Daily max: ${a.daily_limit_usd:.2f}\n"
        )
    lines.append("Change with /setlimit")
    await message.answer("\n".join(lines))


@router.message(Command("setlimit"))
async def cmd_setlimit(message: Message):
    args = message.text.split()
    # /setlimit AgentName daily 100
    if len(args) < 4:
        await message.answer(
            "Usage:\n"
            "<code>/setlimit AgentName daily 100</code>\n"
            "<code>/setlimit AgentName tx 25</code>\n\n"
            "Set daily max or per-transaction limit in USD."
        )
        return

    name = args[1]
    limit_type = args[2].lower()
    try:
        value = Decimal(args[3])
    except Exception:
        await message.answer("Invalid amount. Use a number like 50 or 100.00")
        return

    if limit_type not in ("daily", "tx"):
        await message.answer("Limit type must be <code>daily</code> or <code>tx</code>")
        return

    if value <= 0 or value > 10000:
        await message.answer("Limit must be between $0.01 and $10,000")
        return

    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        result = await db.execute(
            select(Agent).where(Agent.user_id == user.id, Agent.name == name)
        )
        agent = result.scalar_one_or_none()
        if not agent:
            await message.answer(f"Agent '{name}' not found. Check /agents")
            return

        if limit_type == "daily":
            agent.daily_limit_usd = value
        else:
            agent.tx_limit_usd = value

        await db.commit()

    label = "Daily limit" if limit_type == "daily" else "Per-tx limit"
    await message.answer(f"✅ {label} for <b>{name}</b> set to ${value:.2f}")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

# ═══════════════════════════════════════
# WALLET (On-chain USDC)
# ═══════════════════════════════════════

@router.message(Command("wallet"))
async def cmd_wallet(message: Message):
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents. Create one with <code>/newagent MyBot</code>")
        return

    buttons = []
    for a in agents:
        addr = get_wallet_address(a.id)
        if addr:
            buttons.append([InlineKeyboardButton(
                text=f"📊 {a.name} — {addr[:6]}...{addr[-4:]}",
                callback_data=f"winfo:{a.id}"
            )])
        else:
            buttons.append([InlineKeyboardButton(
                text=f"🔗 Create wallet for {a.name}",
                callback_data=f"wcreate:{a.id}"
            )])

    await message.answer(
        "🔗 <b>On-Chain Wallets</b>\n\n"
        "Each agent can have a USDC wallet on Base network.\n"
        "Receive USDC directly, or use the API to send payments.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("wcreate:"))
async def callback_create_wallet(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

    if not agent:
        await callback.answer("Agent not found", show_alert=True)
        return

    await callback.message.edit_text("⏳ Creating wallet on Base network...")

    try:
        wallet_info = create_agent_wallet(agent_id)
        address = wallet_info["address"]

        # Update agent's wallet record in DB
        async with async_session() as db:
            from models.schema import Wallet
            result = await db.execute(
                select(Wallet).where(Wallet.agent_id == agent_id)
            )
            wallet = result.scalar_one_or_none()
            if wallet:
                wallet.wallet_type = "usdc"
                wallet.address = address
            else:
                wallet = Wallet(
                    agent_id=agent_id,
                    wallet_type="usdc",
                    address=address,
                )
                db.add(wallet)
            await db.commit()

        network_label = "Base Sepolia (testnet)" if "sepolia" in wallet_info["network"] else "Base Mainnet"
        await callback.message.edit_text(
            f"✅ <b>Wallet Created!</b>\n\n"
            f"Agent: {agent.name}\n"
            f"Network: {network_label}\n"
            f"Address:\n<code>{address}</code>\n\n"
            f"You can now receive USDC at this address.\n"
            f"Check balance with /wallet"
        )
    except ValueError as e:
        await callback.message.edit_text(
            f"⚠️ <b>CDP API keys needed</b>\n\n"
            f"To create on-chain wallets, you need Coinbase Developer Platform API keys.\n"
            f"Get them free at: https://portal.cdp.coinbase.com\n\n"
            f"Error: {e}"
        )
    except Exception as e:
        logger.error(f"Wallet creation failed: {e}", exc_info=True)
        await callback.message.edit_text(f"❌ Wallet creation failed: {e}")

    await callback.answer()


@router.callback_query(F.data.startswith("winfo:"))
async def callback_wallet_info(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]

    async with async_session() as db:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = result.scalar_one_or_none()

    if not agent:
        await callback.answer("Agent not found", show_alert=True)
        return

    address = get_wallet_address(agent_id)
    if not address:
        await callback.answer("No wallet found", show_alert=True)
        return

    balance_info = get_wallet_balance(agent_id)

    await callback.message.edit_text(
        f"🔗 <b>{agent.name} — Wallet</b>\n\n"
        f"Address:\n<code>{address}</code>\n\n"
        f"💰 ETH: {balance_info.get('balance_eth', '0')}\n"
        f"💵 USDC: {balance_info.get('balance_usdc', '0')}\n"
        f"🌐 Network: {balance_info.get('network', 'N/A')}\n\n"
        f"Send USDC to this address on Base to fund on-chain."
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
        await message.answer("No agents. Create one with <code>/newagent MyBot</code>")
        return

    buttons = []
    for a in agents:
        details = get_card_details(a.id)
        if details:
            state_icon = "🟢" if details["state"] == "OPEN" else "🔴" if details["state"] == "CLOSED" else "⏸"
            buttons.append([InlineKeyboardButton(
                text=f"{state_icon} {a.name} — ****{details['last4']}",
                callback_data=f"cinfo:{a.id}"
            )])
        else:
            buttons.append([InlineKeyboardButton(
                text=f"💳 Create card for {a.name}",
                callback_data=f"ccreate:{a.id}"
            )])

    await message.answer(
        "💳 <b>Virtual Cards</b>\n\n"
        "Each agent can have a virtual Visa card for online purchases.\n"
        "Your agent can use it to pay for APIs, services, subscriptions.",
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

    await callback.message.edit_text("⏳ Creating virtual card...")

    try:
        card_info = create_virtual_card(
            agent_id,
            spend_limit=5000,  # $50 monthly default
            memo=f"AgentPay - {agent.name}",
        )

        limit_usd = card_info.get("spend_limit_cents", 5000) / 100

        await callback.message.edit_text(
            f"✅ <b>Virtual Card Created!</b>\n\n"
            f"Agent: {agent.name}\n"
            f"Card: ****{card_info['last4']}\n"
            f"Expires: {card_info['exp_month']}/{card_info['exp_year']}\n"
            f"Monthly limit: ${limit_usd:.0f}\n"
            f"Status: {card_info['state']}\n\n"
            f"Use /carddetails to see full card number.\n"
            f"Use /cardpause or /cardclose to manage."
        )
    except ValueError as e:
        await callback.message.edit_text(
            f"⚠️ <b>Lithic API key needed</b>\n\n"
            f"Get a free sandbox key at:\nhttps://app.lithic.com/signup\n\n"
            f"Error: {e}"
        )
    except Exception as e:
        logger.error(f"Card creation failed: {e}", exc_info=True)
        await callback.message.edit_text(f"❌ Card creation failed: {e}")

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
        buttons.append([
            InlineKeyboardButton(text="👁 Show Full Details", callback_data=f"cfull:{agent_id}"),
        ])
        buttons.append([
            InlineKeyboardButton(text="⏸ Pause Card", callback_data=f"cpause:{agent_id}"),
            InlineKeyboardButton(text="🔴 Close Card", callback_data=f"cclose:{agent_id}"),
        ])
    elif details["state"] == "PAUSED":
        buttons.append([
            InlineKeyboardButton(text="▶️ Resume Card", callback_data=f"cresume:{agent_id}"),
            InlineKeyboardButton(text="🔴 Close Card", callback_data=f"cclose:{agent_id}"),
        ])

    await callback.message.edit_text(
        f"💳 <b>{agent.name} — Card</b>\n\n"
        f"Card: ****{details['last4']}\n"
        f"Expires: {details['exp_month']}/{details['exp_year']}\n"
        f"Status: {state_icon} {details['state']}\n"
        f"Monthly limit: ${limit_usd:.0f}",
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

    # Send as a separate message that auto-deletes (sensitive info)
    msg = await callback.message.answer(
        f"🔐 <b>Card Details</b> (sensitive)\n\n"
        f"Number: <code>{details.get('pan', 'N/A')}</code>\n"
        f"Expiry: <code>{details['exp_month']}/{details['exp_year']}</code>\n"
        f"CVV: <code>{details.get('cvv', 'N/A')}</code>\n\n"
        f"⚠️ This message will be deleted in 60 seconds."
    )

    # Schedule deletion
    asyncio.get_event_loop().call_later(
        60, lambda: asyncio.ensure_future(msg.delete())
    )
    await callback.answer()


@router.callback_query(F.data.startswith("cpause:"))
async def callback_pause_card(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    result = update_card_state(agent_id, "PAUSED")
    if result["success"]:
        await callback.message.edit_text("⏸ Card paused. Use /card to resume.")
    else:
        await callback.answer(f"Error: {result['error']}", show_alert=True)
    await callback.answer()


@router.callback_query(F.data.startswith("cresume:"))
async def callback_resume_card(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    result = update_card_state(agent_id, "OPEN")
    if result["success"]:
        await callback.message.edit_text("▶️ Card resumed! Use /card to view.")
    else:
        await callback.answer(f"Error: {result['error']}", show_alert=True)
    await callback.answer()


@router.callback_query(F.data.startswith("cclose:"))
async def callback_close_card(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    buttons = [[
        InlineKeyboardButton(text="⚠️ Yes, close permanently", callback_data=f"ccloseconfirm:{agent_id}"),
        InlineKeyboardButton(text="❌ Cancel", callback_data=f"cinfo:{agent_id}"),
    ]]
    await callback.message.edit_text(
        "⚠️ <b>Close card permanently?</b>\n\nThis cannot be undone.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("ccloseconfirm:"))
async def callback_close_card_confirm(callback: CallbackQuery):
    agent_id = callback.data.split(":")[1]
    result = update_card_state(agent_id, "CLOSED")
    if result["success"]:
        await callback.message.edit_text("🔴 Card permanently closed.")
    else:
        await callback.answer(f"Error: {result['error']}", show_alert=True)
    await callback.answer()


# ═══════════════════════════════════════
# APPROVAL WORKFLOW (Telegram inline buttons)
# ═══════════════════════════════════════

@router.callback_query(F.data.startswith("approve:"))
async def callback_approve(callback: CallbackQuery):
    approval_id = callback.data.split(":")[1]
    approval = get_pending_approval(approval_id)

    if not approval:
        await callback.answer("Expired or already resolved", show_alert=True)
        return

    if approval.telegram_id != callback.from_user.id:
        await callback.answer("Not your approval", show_alert=True)
        return

    # Resolve approval
    resolve_approval(approval_id, approved=True)

    # Execute the spend
    async with async_session() as db:
        tx, error = await execute_approved_spend(
            db, approval.agent_id, approval.amount_usd, approval.description
        )

    if error:
        await callback.message.edit_text(
            f"❌ Approved but failed: {error}\n\n"
            f"(Balance may have changed since request)"
        )
    else:
        await callback.message.edit_text(
            f"✅ <b>Approved!</b>\n\n"
            f"<b>{approval.agent_name}</b> spent ${float(approval.amount_usd):.2f}\n"
            f"{'For: ' + approval.description if approval.description else ''}"
        )
    await callback.answer()


@router.callback_query(F.data.startswith("deny:"))
async def callback_deny(callback: CallbackQuery):
    approval_id = callback.data.split(":")[1]
    approval = get_pending_approval(approval_id)

    if not approval:
        await callback.answer("Expired or already resolved", show_alert=True)
        return

    if approval.telegram_id != callback.from_user.id:
        await callback.answer("Not your approval", show_alert=True)
        return

    resolve_approval(approval_id, approved=False, reason="Denied by user")

    await callback.message.edit_text(
        f"🚫 <b>Denied</b>\n\n"
        f"<b>{approval.agent_name}</b> wanted to spend ${float(approval.amount_usd):.2f}\n"
        f"{'For: ' + approval.description if approval.description else ''}\n\n"
        f"No funds were deducted."
    )
    await callback.answer()


@router.message(Command("approvals"))
async def cmd_approvals(message: Message):
    """Show pending approvals for this user."""
    from core.approvals import get_user_pending
    pending = get_user_pending(message.from_user.id)

    if not pending:
        await message.answer("✅ No pending approvals.")
        return

    text = f"⏳ <b>{len(pending)} Pending Approval(s)</b>\n\n"
    for a in pending:
        text += (
            f"• <b>{a.agent_name}</b>: ${float(a.amount_usd):.2f}\n"
            f"  {a.description or 'No description'}\n\n"
        )
    await message.answer(text)


@router.message(Command("setapprove"))
async def cmd_set_approve(message: Message):
    """Set auto-approve threshold: /setapprove MyBot 25"""
    parts = message.text.split(maxsplit=2) if message.text else []
    if len(parts) < 3:
        await message.answer("Usage: <code>/setapprove AgentName Amount</code>\n\nExample: /setapprove MyBot 25")
        return

    agent_name = parts[1]
    try:
        threshold = Decimal(parts[2])
    except Exception:
        await message.answer("Invalid amount.")
        return

    if threshold < 0 or threshold > 10000:
        await message.answer("Threshold must be between $0 and $10,000.")
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
        await message.answer(f"✅ <b>{agent.name}</b>: ALL spends now require approval.")
    else:
        await message.answer(f"✅ <b>{agent.name}</b>: Auto-approve up to ${threshold}. Above → needs your approval.")


# ═══════════════════════════════════════
# REFUND & TRANSFER COMMANDS
# ═══════════════════════════════════════

@router.message(Command("refund"))
async def cmd_refund(message: Message):
    """Refund a spend transaction: /refund TransactionID"""
    parts = message.text.split(maxsplit=1) if message.text else []
    if len(parts) < 2:
        await message.answer(
            "Usage: <code>/refund TransactionID</code>\n\n"
            "Find transaction IDs in /history"
        )
        return

    tx_id = parts[1].strip()

    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("You have no agents.")
        return

    # Try refund against each agent (user may not know which agent the tx belongs to)
    for agent in agents:
        async with async_session() as db:
            result = await db.execute(select(Agent).where(Agent.id == agent.id))
            a = result.scalar_one()
            tx, error = await refund(db, a, tx_id)
            if tx:
                await message.answer(
                    f"✅ <b>Refunded!</b>\n\n"
                    f"Agent: {a.name}\n"
                    f"Amount: ${float(tx.amount_usd):.2f}\n"
                    f"New balance: ${float(a.balance_usd):.2f}"
                )
                return
            if error and "not found" not in error.lower():
                await message.answer(f"❌ {error}")
                return

    await message.answer("❌ Transaction not found. Check the ID with /history")


@router.message(Command("transfer"))
async def cmd_transfer(message: Message):
    """Transfer between agents: /transfer FromAgent ToAgent Amount"""
    parts = message.text.split() if message.text else []
    if len(parts) < 4:
        await message.answer(
            "Usage: <code>/transfer FromAgent ToAgent Amount</code>\n\n"
            "Example: /transfer MyBot Helper 5.00"
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
            f"✅ <b>Transferred!</b>\n\n"
            f"${float(amount):.2f} from <b>{from_agent.name}</b> → <b>{to_agent.name}</b>"
        )


@router.message(Command("export"))
async def cmd_export(message: Message):
    """Export transaction history as CSV."""
    async with async_session() as db:
        user = await get_or_create_user(db, message.from_user.id)
        agents = await get_user_agents(db, user)

    if not agents:
        await message.answer("No agents found.")
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
    await message.answer_document(doc, caption="📄 Your transaction history")


# ═══════════════════════════════════════
# ERROR HANDLER & CATCH-ALL
# ═══════════════════════════════════════

@router.message()
async def catch_all(message: Message):
    """Log unhandled messages for debugging."""
    logger.info(f"Unhandled message from {message.from_user.id}: {message.text!r}")
    if message.text and message.text.startswith("/"):
        await message.answer("Unknown command. Try /help")


dp.include_router(router)


@dp.errors()
async def error_handler(event, exception):
    logger.error(f"Error handling update: {exception}", exc_info=True)
    return True


async def main():
    logger.info("🚀 AgentPay starting...")
    await init_db()
    logger.info("✅ Database initialized")

    # Register bot instance for Telegram notifications
    set_webhook_bot(bot)
    logger.info("✅ Webhook notifications enabled")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
