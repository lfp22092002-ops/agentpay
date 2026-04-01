#!/usr/bin/env bash
# AgentPay Pre-flight Check — run before deployment
# Usage: bash scripts/preflight.sh

set -euo pipefail
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0; WARN=0

ok()   { echo -e "  ${GRN}✓${NC} $1"; ((PASS++)); }
fail() { echo -e "  ${RED}✗${NC} $1"; ((FAIL++)); }
warn() { echo -e "  ${YEL}!${NC} $1"; ((WARN++)); }

echo "═══ AgentPay Pre-flight Check ═══"
echo

# 1. Docker
echo "Docker:"
command -v docker &>/dev/null && ok "docker installed" || fail "docker not found"
command -v docker compose &>/dev/null && ok "docker compose available" || fail "docker compose not found"

# 2. .env file
echo "Environment:"
if [ -f .env ]; then
  ok ".env file exists"
  grep -q '^BOT_TOKEN=.\+' .env && ok "BOT_TOKEN set" || fail "BOT_TOKEN empty or missing"
  grep -q '^API_SECRET=change-me' .env && fail "API_SECRET still default" || ok "API_SECRET customized"
else
  fail ".env file missing (copy from .env.example)"
fi

# 3. Dockerfile
echo "Build:"
[ -f Dockerfile ] && ok "Dockerfile present" || fail "Dockerfile missing"
[ -f docker-compose.yml ] && ok "docker-compose.yml present" || fail "docker-compose.yml missing"
[ -f requirements.txt ] && ok "requirements.txt present" || fail "requirements.txt missing"

# 4. Security
echo "Security:"
grep -q 'agentpay2026' docker-compose.yml && warn "Postgres password is default (change before prod)" || ok "Postgres password customized"

# 5. Ports
echo "Ports:"
ss -tlnp 2>/dev/null | grep -q ':8080' && warn "Port 8080 already in use" || ok "Port 8080 available"
ss -tlnp 2>/dev/null | grep -q ':5432' && warn "Port 5432 already in use" || ok "Port 5432 available"
ss -tlnp 2>/dev/null | grep -q ':6379' && warn "Port 6379 already in use" || ok "Port 6379 available"

# 6. Disk
AVAIL=$(df -BG --output=avail / | tail -1 | tr -d ' G')
[ "$AVAIL" -gt 10 ] && ok "Disk space: ${AVAIL}G available" || fail "Low disk: ${AVAIL}G available"

echo
echo "═══ Results: ${GRN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YEL}${WARN} warnings${NC} ═══"
[ "$FAIL" -eq 0 ] && echo -e "${GRN}Ready to deploy!${NC}" || echo -e "${RED}Fix failures before deploying.${NC}"
exit $FAIL
