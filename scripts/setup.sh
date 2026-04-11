#!/usr/bin/env bash
# AgentPay — Quick-start setup script
# Creates .env with random secrets and starts services via Docker Compose.
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }

cd "$(dirname "$0")/.."

# ── .env ───────────────────────────────────────────────────────
if [ -f .env ]; then
  warn ".env already exists — skipping creation (delete it to regenerate)"
else
  cp .env.example .env

  # Generate random secrets
  API_SECRET=$(openssl rand -hex 32 2>/dev/null || head -c 64 /dev/urandom | xxd -p | tr -d '\n' | head -c 64)
  ENCRYPTION_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 64 /dev/urandom | xxd -p | tr -d '\n' | head -c 64)

  sed -i "s|^API_SECRET=.*|API_SECRET=${API_SECRET}|" .env
  sed -i "s|^# ENCRYPTION_KEY=.*|ENCRYPTION_KEY=${ENCRYPTION_KEY}|" .env

  info "Created .env with random API_SECRET and ENCRYPTION_KEY"
  warn "Edit .env to add your BOT_TOKEN (from @BotFather)"
fi

# ── Dependencies check ─────────────────────────────────────────
for cmd in docker; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "❌ $cmd is required but not installed."
    exit 1
  fi
done

# ── Docker Compose ─────────────────────────────────────────────
COMPOSE_FILE="docker-compose.yml"
if [ "${1:-}" = "--dev" ]; then
  COMPOSE_FILE="docker-compose.dev.yml"
  info "Using dev compose (hot-reload)"
fi

if docker compose version &>/dev/null; then
  COMPOSE="docker compose"
else
  COMPOSE="docker-compose"
fi

info "Starting services with ${COMPOSE_FILE}..."
$COMPOSE -f "$COMPOSE_FILE" up -d

echo ""
info "AgentPay is running!"
echo "  API:  http://localhost:8080/docs"
echo "  Bot:  @FundmyAIbot on Telegram"
echo ""
warn "Next steps:"
echo "  1. Add BOT_TOKEN to .env"
echo "  2. Restart: $COMPOSE -f $COMPOSE_FILE restart"
