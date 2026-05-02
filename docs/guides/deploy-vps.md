# Deploy AgentPay to a VPS (DigitalOcean / Hetzner)

Self-hosting gives you full control. This guide covers deploying AgentPay on any Ubuntu 22.04/24.04 VPS.

**Recommended specs**: 2 vCPU, 4GB RAM, 40GB SSD — ~$6-12/mo on Hetzner or DigitalOcean.

## Step 1 — Provision Your Server

### DigitalOcean
1. Create a Droplet → Ubuntu 24.04 → Basic → $6/mo (1GB) or $12/mo (2GB)
2. Add your SSH key
3. Note the server IP

### Hetzner (cheaper, recommended)
1. New Server → Ubuntu 24.04 → CX22 (2 vCPU, 4GB, €4.15/mo)
2. Add SSH key
3. Note the server IP

## Step 2 — Initial Server Setup

```bash
# SSH in
ssh root@YOUR_SERVER_IP

# Create a non-root user
adduser leo
usermod -aG sudo leo

# Switch to new user
su - leo
```

## Step 3 — Install Dependencies

```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install Git
sudo apt install -y git
```

## Step 4 — Clone AgentPay

```bash
git clone https://github.com/lfp22092002-ops/agentpay.git
cd agentpay
```

## Step 5 — Configure Environment

```bash
cp .env.example .env
nano .env
```

Fill in:
```env
BOT_TOKEN=your_telegram_bot_token
API_SECRET=your_secret_min_32_chars
ADMIN_TELEGRAM_ID=your_telegram_id
ENVIRONMENT=production
BASE_URL=https://yourdomain.com
DATABASE_URL=postgresql+asyncpg://agentpay:strongpassword@db:5432/agentpay
POSTGRES_USER=agentpay
POSTGRES_PASSWORD=strongpassword
POSTGRES_DB=agentpay
```

Generate a secure `API_SECRET`:
```bash
openssl rand -hex 32
```

## Step 6 — Start Services

```bash
docker compose up -d

# Run database migrations
docker compose exec api alembic upgrade head

# Check everything is running
docker compose ps
```

## Step 7 — Set Up HTTPS (Cloudflare Tunnel)

The easiest HTTPS option — no nginx needed:

```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/

# Authenticate (opens browser on your local machine)
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create agentpay

# Route traffic
cloudflared tunnel route dns agentpay yourdomain.com

# Run tunnel (add to systemd for persistence)
cloudflared tunnel run agentpay
```

### Alternative: nginx + Let's Encrypt

```bash
sudo apt install -y nginx certbot python3-certbot-nginx

# Create nginx config
sudo nano /etc/nginx/sites-available/agentpay
```

```nginx
server {
    server_name yourdomain.com;
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/agentpay /etc/nginx/sites-enabled/
sudo certbot --nginx -d yourdomain.com
sudo systemctl restart nginx
```

## Step 8 — Auto-Start on Reboot

```bash
# Install the systemd service
sudo cp deploy/agentpay.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agentpay
sudo systemctl start agentpay
```

## Step 9 — Verify

```bash
curl https://yourdomain.com/v1/health
# Expected: {"status": "ok", "version": "0.1.0"}
```

Send `/start` to your Telegram bot — should respond immediately.

## Maintenance

```bash
# View logs
docker compose logs -f api
docker compose logs -f bot

# Update to latest version
cd agentpay
git pull
docker compose build
docker compose up -d
docker compose exec api alembic upgrade head

# Backup database
docker compose exec db pg_dump -U agentpay agentpay > backup-$(date +%Y%m%d).sql
```

## Firewall

```bash
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

> Port 8080 should **not** be public — all traffic routes through Cloudflare Tunnel or nginx.

---

*For a zero-config cloud option, see [deploy-railway.md](./deploy-railway.md)*
