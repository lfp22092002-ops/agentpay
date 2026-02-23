#!/usr/bin/env python3
"""Check for new AgentPay users and return count."""
import asyncio
import sys
sys.path.insert(0, "/home/leo/.openclaw/workspace/projects/agentpay")

from models.database import async_session, init_db
from models.schema import User, Agent
from sqlalchemy import select, func
from datetime import datetime, timedelta


async def check():
    await init_db()
    async with async_session() as db:
        # Total users (exclude G = telegram_id 5360481016)
        total = await db.execute(
            select(func.count(User.id)).where(User.telegram_id != 5360481016)
        )
        total_users = total.scalar() or 0

        # New users in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        new = await db.execute(
            select(User).where(
                User.telegram_id != 5360481016,
                User.created_at >= one_hour_ago
            )
        )
        new_users = new.scalars().all()

        # Total agents (exclude G's)
        g_user = await db.execute(select(User).where(User.telegram_id == 5360481016))
        g = g_user.scalar_one_or_none()
        if g:
            total_agents = await db.execute(
                select(func.count(Agent.id)).where(Agent.user_id != g.id)
            )
        else:
            total_agents = await db.execute(select(func.count(Agent.id)))
        agent_count = total_agents.scalar() or 0

        if new_users:
            for u in new_users:
                name = u.first_name or u.username or f"ID:{u.telegram_id}"
                print(f"NEW_USER:{name}:{u.telegram_id}:{u.created_at.isoformat()}")

        print(f"TOTAL_USERS:{total_users}")
        print(f"TOTAL_AGENTS:{agent_count}")


asyncio.run(check())
