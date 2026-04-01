#!/usr/bin/env bash
# AgentPay one-command deploy for Linux PC
# Usage: ./deploy.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "=== AgentPay Deploy ==="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not installed. Run: curl -fsSL https://get.docker.com | sh"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "❌ Docker Compose plugin not found."; exit 1; }

# Check .env
if [ ! -f .env ]; then
  echo "Creating .env from template..."
  cat > .env <<'EOF'
BOT_TOKEN=your-telegram-bot-token-here
API_SECRET=$(openssl rand -hex 32)
EOF
  echo "⚠️  Edit .env with real values, then re-run this script."
  exit 1
fi

# Validate required vars
source .env
[ -z "${BOT_TOKEN:-}" ] && echo "❌ BOT_TOKEN not set in .env" && exit 1
[ "${API_SECRET:-change-me-in-production}" = "change-me-in-production" ] && echo "❌ API_SECRET not set in .env" && exit 1

# Build and start
echo "Building and starting services..."
docker compose build
docker compose up -d

echo ""
echo "✅ AgentPay is running!"
echo "   API:  http://localhost:8080/docs"
echo "   Bot:  Telegram @FundmyAIbot"
echo ""
echo "Logs: docker compose logs -f"
echo "Stop: docker compose down"
