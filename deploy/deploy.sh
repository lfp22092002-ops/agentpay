#!/bin/bash
# AgentPay Production Deploy Script
set -euo pipefail

DOMAIN="" BOT_TOKEN="" API_SECRET="" ADMIN_ID="5360481016" PORT=8080 HTTPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --domain) DOMAIN="$2"; shift 2;;
        --bot-token) BOT_TOKEN="$2"; shift 2;;
        --api-secret) API_SECRET="$2"; shift 2;;
        --admin-id) ADMIN_ID="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --https) HTTPS=true; shift;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

if [[ -z "$DOMAIN" || -z "$BOT_TOKEN" || -z "$API_SECRET" ]]; then
    echo "Usage: $0 --domain <host> --bot-token <token> --api-secret <secret>"
    exit 1
fi

echo "🚀 Deploying AgentPay to $DOMAIN"

# Install prerequisites
apt-get update -qq && apt-get install -y curl git postgresql redis-server nginx certbot python3-pip -qq > /dev/null

# Clone & setup
mkdir -p /opt/agentpay
git clone --depth 1 https://github.com/lfp22092002-ops/agentpay.git /opt/agentpay
cd /opt/agentpay
pip install -e . > /dev/null 2>&1

# Generate .env
cat > .env <<EOF
BOT_TOKEN=$BOT_TOKEN
API_SECRET=$API_SECRET
ADMIN_TELEGRAM_ID=$ADMIN_ID
DOMAIN=$DOMAIN
PORT=$PORT
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://agentpay:agentpay2026@localhost:5432/agentpay
REDIS_URL=redis://localhost:6379/0
EOF

# Setup database
sudo -u postgres psql -c "CREATE DATABASE agentpay;" 2>/dev/null || true
sudo -u postgres psql -c "CREATE USER agentpay WITH PASSWORD 'agentpay2026';" 2>/dev/null || true
alembic upgrade head > /dev/null 2>&1

# SSL
if [[ "$HTTPS" == "true" ]]; then
    certbot --nginx -d $DOMAIN -n --agree-tos -m admin@$DOMAIN
fi

# Systemd
cp deploy/agentpay.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now agentpay-api > /dev/null 2>&1

echo "✅ Deployed! Visit https://$DOMAIN"
echo "   Bot: @FundmyAIbot"
echo "   Docs: https://$DOMAIN/docs-site"
