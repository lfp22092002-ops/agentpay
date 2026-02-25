#!/usr/bin/env python3
"""
Update the bot's menu button with the current Cloudflare tunnel URL.
Run after tunnel restarts to keep the Mini App link working.
"""
import asyncio
import subprocess
import re
import sys

sys.path.insert(0, "/home/leo/.openclaw/workspace/projects/agentpay")

from aiogram import Bot
from aiogram.types import MenuButtonWebApp, WebAppInfo
from config.settings import BOT_TOKEN


def get_tunnel_url():
    """Get current tunnel URL from journalctl."""
    result = subprocess.run(
        ["sudo", "journalctl", "-u", "agentpay-tunnel", "--no-pager", "-n", "50"],
        capture_output=True, text=True
    )
    matches = re.findall(r"https://[\w-]+\.trycloudflare\.com", result.stdout)
    return matches[-1] if matches else None


async def update_menu_button(url):
    bot = Bot(token=BOT_TOKEN)
    await bot.set_chat_menu_button(
        menu_button=MenuButtonWebApp(
            text="📊 Dashboard",
            web_app=WebAppInfo(url=f"{url}/app/")
        )
    )
    print(f"✅ Menu button → {url}/app/")
    await bot.session.close()


if __name__ == "__main__":
    url = get_tunnel_url()
    if not url:
        print("❌ No tunnel URL found")
        sys.exit(1)
    asyncio.run(update_menu_button(url))
