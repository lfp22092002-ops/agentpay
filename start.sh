#!/bin/bash
# AgentPay launcher — runs both bot and API
cd "$(dirname "$0")"
source venv/bin/activate

echo "🚀 Starting AgentPay..."

# Start API server in background
uvicorn api.main:app --host 0.0.0.0 --port 8080 &
API_PID=$!
echo "✅ API server started (PID: $API_PID)"

# Start Telegram bot
python -m bot.main &
BOT_PID=$!
echo "✅ Bot started (PID: $BOT_PID)"

# Trap to kill both on exit
trap "kill $API_PID $BOT_PID 2>/dev/null; exit" SIGINT SIGTERM

echo "🟢 AgentPay running — API :8080 | Bot active"
wait
