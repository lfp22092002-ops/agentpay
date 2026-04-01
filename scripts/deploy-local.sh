#!/usr/bin/env bash
# Deploy AgentPay locally via Docker Compose
# Usage: ./scripts/deploy-local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Check .env exists
if [ ! -f .env ]; then
  echo "❌ .env not found. Copy .env.example → .env and fill in BOT_TOKEN + API_SECRET"
  echo "   cp .env.example .env"
  echo "   openssl rand -hex 32  # for API_SECRET"
  exit 1
fi

# Check required vars
source .env
if [ -z "${BOT_TOKEN:-}" ]; then
  echo "❌ BOT_TOKEN not set in .env"; exit 1
fi
if [ "${API_SECRET:-change-me-to-a-random-string}" = "change-me-to-a-random-string" ]; then
  echo "⚠️  API_SECRET is still default — generating one..."
  NEW_SECRET=$(openssl rand -hex 32)
  sed -i "s|^API_SECRET=.*|API_SECRET=${NEW_SECRET}|" .env
  echo "✅ Generated API_SECRET in .env"
fi

echo "🚀 Starting AgentPay..."
docker compose up -d --build

echo ""
echo "⏳ Waiting for health check..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8080/v1/health > /dev/null 2>&1; then
    echo "✅ AgentPay is live at http://localhost:8080"
    echo "   📖 API docs: http://localhost:8080/docs"
    echo "   🤖 Bot: @FundmyAIbot"
    exit 0
  fi
  sleep 2
done

echo "❌ Health check failed after 60s. Check logs:"
echo "   docker compose logs --tail 50"
exit 1
