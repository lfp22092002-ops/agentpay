#!/usr/bin/env bash
# AgentPay Post-deploy Health Check — validate a live instance
# Usage: bash scripts/healthcheck.sh [BASE_URL]
# Example: bash scripts/healthcheck.sh https://leofundmybot.dev

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
BASE_URL="${BASE_URL%/}"

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0; WARN=0

ok()   { echo -e "  ${GRN}✓${NC} $1"; ((PASS++)); }
fail() { echo -e "  ${RED}✗${NC} $1"; ((FAIL++)); }
warn() { echo -e "  ${YEL}!${NC} $1"; ((WARN++)); }

check_endpoint() {
  local name="$1" path="$2" expected_status="${3:-200}"
  local status
  status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "${BASE_URL}${path}" 2>/dev/null || echo "000")
  if [ "$status" = "$expected_status" ]; then
    ok "$name → $status"
  elif [ "$status" = "000" ]; then
    fail "$name → connection failed"
  else
    fail "$name → $status (expected $expected_status)"
  fi
}

check_json_field() {
  local name="$1" path="$2" field="$3"
  local body
  body=$(curl -s --max-time 10 "${BASE_URL}${path}" 2>/dev/null || echo "{}")
  if echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); assert '$field' in d" 2>/dev/null; then
    ok "$name has '$field' field"
  else
    fail "$name missing '$field' field"
  fi
}

echo "═══ AgentPay Health Check ═══"
echo "Target: ${BASE_URL}"
echo

# Core endpoints
echo "Core:"
check_endpoint "Health" "/v1/health"
check_endpoint "OpenAPI docs" "/docs"
check_endpoint "ReDoc" "/redoc"
check_endpoint "OpenAPI JSON" "/openapi.json"

# Landing & SEO
echo "Landing & SEO:"
check_endpoint "Landing page" "/"
check_endpoint "robots.txt" "/robots.txt"
check_endpoint "sitemap.xml" "/sitemap.xml"
check_endpoint "llms.txt" "/llms.txt"
check_endpoint "llms-full.txt" "/llms-full.txt"

# Auth-required (should return 401/403 without key)
echo "Auth gates (expect 401/403 without API key):"
for ep in "/v1/balance" "/v1/wallet" "/v1/transactions" "/v1/chains"; do
  status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "${BASE_URL}${ep}" 2>/dev/null || echo "000")
  if [ "$status" = "401" ] || [ "$status" = "403" ]; then
    ok "$ep → $status (correctly gated)"
  elif [ "$status" = "200" ]; then
    fail "$ep → 200 (NOT gated — security issue!)"
  elif [ "$status" = "000" ]; then
    fail "$ep → connection failed"
  else
    warn "$ep → $status (unexpected)"
  fi
done

# Response time
echo "Performance:"
total_time=$(curl -s -o /dev/null -w '%{time_total}' --max-time 10 "${BASE_URL}/v1/health" 2>/dev/null || echo "99")
ms=$(echo "$total_time" | python3 -c "import sys; print(int(float(sys.stdin.read().strip())*1000))" 2>/dev/null || echo "9999")
if [ "$ms" -lt 500 ]; then
  ok "Health endpoint: ${ms}ms"
elif [ "$ms" -lt 2000 ]; then
  warn "Health endpoint: ${ms}ms (slow)"
else
  fail "Health endpoint: ${ms}ms (very slow or timeout)"
fi

# TLS (if https)
if [[ "$BASE_URL" == https://* ]]; then
  echo "TLS:"
  domain=$(echo "$BASE_URL" | sed 's|https://||' | cut -d/ -f1)
  expiry=$(echo | openssl s_client -servername "$domain" -connect "${domain}:443" 2>/dev/null | openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
  if [ -n "$expiry" ]; then
    days_left=$(( ($(date -d "$expiry" +%s) - $(date +%s)) / 86400 ))
    if [ "$days_left" -gt 30 ]; then
      ok "TLS cert valid for ${days_left} days"
    elif [ "$days_left" -gt 7 ]; then
      warn "TLS cert expires in ${days_left} days"
    else
      fail "TLS cert expires in ${days_left} days!"
    fi
  else
    warn "Could not check TLS cert expiry"
  fi
fi

echo
echo "═══ Results: ${GRN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YEL}${WARN} warnings${NC} ═══"
[ "$FAIL" -eq 0 ] && echo -e "${GRN}All checks passed!${NC}" || echo -e "${RED}${FAIL} check(s) failed.${NC}"
exit "$FAIL"
