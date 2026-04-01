#!/usr/bin/env bash
# AgentPay one-command deploy for Linux (Ubuntu/Mint/Debian)
# Usage: sudo bash scripts/deploy-linux.sh
set -euo pipefail

AGENTPAY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AGENTPAY_DIR"

echo "═══════════════════════════════════════"
echo "  AgentPay Deploy — Linux"
echo "═══════════════════════════════════════"

# 1. Install Docker if missing
if ! command -v docker &>/dev/null; then
    echo "[1/5] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
    # Add current sudo user to docker group
    if [ -n "${SUDO_USER:-}" ]; then
        usermod -aG docker "$SUDO_USER"
        echo "  → Added $SUDO_USER to docker group (re-login to use without sudo)"
    fi
else
    echo "[1/5] Docker already installed ✓"
fi

# 2. Check .env
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        # Generate API_SECRET
        SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        sed -i "s/^API_SECRET=.*/API_SECRET=$SECRET/" .env
        echo "[2/5] Created .env with generated API_SECRET"
        echo "  ⚠️  You MUST set BOT_TOKEN in .env before starting!"
        echo "  Edit: nano $AGENTPAY_DIR/.env"
        exit 1
    else
        echo "[2/5] ERROR: No .env or .env.example found"
        exit 1
    fi
else
    echo "[2/5] .env exists ✓"
    # Validate BOT_TOKEN is set
    if grep -q "^BOT_TOKEN=$" .env; then
        echo "  ⚠️  BOT_TOKEN is empty in .env — set it first!"
        exit 1
    fi
fi

# 3. Start services
echo "[3/5] Starting AgentPay (docker compose)..."
docker compose up -d --build

# 4. Wait for health
echo "[4/5] Waiting for API health..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/v1/health &>/dev/null; then
        echo "  → API healthy ✓"
        break
    fi
    [ "$i" -eq 30 ] && echo "  ⚠️  API not responding after 30s — check logs: docker compose logs api"
    sleep 1
done

# 5. Create systemd service for auto-start
echo "[5/5] Creating systemd service for auto-start..."
cat > /etc/systemd/system/agentpay.service <<EOF
[Unit]
Description=AgentPay - Telegram Payment Layer for AI Agents
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$AGENTPAY_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable agentpay.service
echo "  → agentpay.service enabled (auto-starts on boot) ✓"

echo ""
echo "═══════════════════════════════════════"
echo "  AgentPay is LIVE!"
echo "  API:  http://localhost:8080/v1/health"
echo "  Docs: http://localhost:8080/docs"
echo "  Logs: docker compose logs -f"
echo "═══════════════════════════════════════"
