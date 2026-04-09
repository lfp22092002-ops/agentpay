#!/usr/bin/env bash
# AgentPay Quick-Start Setup
# Usage: bash setup.sh
# Creates .env, generates secrets, and starts services.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[✗]${NC} $1"; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AgentPay — Quick Start Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Check prerequisites
for cmd in docker; do
    if ! command -v "$cmd" &>/dev/null; then
        err "$cmd is required but not installed."
        exit 1
    fi
done

if ! docker compose version &>/dev/null && ! docker-compose version &>/dev/null; then
    err "Docker Compose is required (docker compose or docker-compose)."
    exit 1
fi

info "Prerequisites OK"

# Determine compose command
if docker compose version &>/dev/null; then
    COMPOSE="docker compose"
else
    COMPOSE="docker-compose"
fi

# Create .env if missing
if [ ! -f .env ]; then
    echo
    warn "No .env file found — creating one."

    API_SECRET=$(openssl rand -hex 32)
    PG_PASSWORD=$(openssl rand -hex 16)

    read -rp "  Telegram Bot Token (from @BotFather): " BOT_TOKEN
    if [ -z "$BOT_TOKEN" ]; then
        err "Bot token is required. Get one from https://t.me/BotFather"
        exit 1
    fi

    cat > .env <<EOF
# AgentPay Configuration (generated $(date -Iseconds))

# Telegram Bot
BOT_TOKEN=${BOT_TOKEN}

# API secret (used for signing, keep safe)
API_SECRET=${API_SECRET}

# PostgreSQL (auto-generated, change if needed)
POSTGRES_PASSWORD=${PG_PASSWORD}

# Environment
ENVIRONMENT=production
EOF

    # Also update docker-compose password to match
    if grep -q "agentpay2026" docker-compose.yml; then
        sed -i "s/agentpay2026/${PG_PASSWORD}/g" docker-compose.yml
        info "Updated docker-compose.yml with generated DB password"
    fi

    info "Created .env with secure secrets"
else
    info ".env already exists — skipping"
fi

echo

# Build and start
info "Building containers..."
$COMPOSE build --quiet

info "Starting services..."
$COMPOSE up -d

echo
info "Waiting for health checks..."
sleep 10

# Check health
API_HEALTHY=false
for i in $(seq 1 6); do
    if curl -sf http://localhost:8080/v1/health &>/dev/null; then
        API_HEALTHY=true
        break
    fi
    sleep 5
done

if $API_HEALTHY; then
    info "API is healthy!"
else
    warn "API not responding yet — check logs: $COMPOSE logs api"
fi

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AgentPay is running! 🚀"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "  API:     http://localhost:8080"
echo "  Docs:    http://localhost:8080/docs"
echo "  Health:  http://localhost:8080/v1/health"
echo
echo "  Logs:    $COMPOSE logs -f"
echo "  Stop:    $COMPOSE down"
echo
