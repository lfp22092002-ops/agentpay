#!/usr/bin/env python3
"""AgentPay CLI — quick admin operations from the terminal.

Usage:
    agentpay-cli health              Check API health
    agentpay-cli wallets             List all wallets
    agentpay-cli wallet <id>         Get wallet details
    agentpay-cli balance <id>        Get wallet balance
    agentpay-cli tx <wallet_id>      List transactions
    agentpay-cli stats               System stats (admin)

Requires AGENTPAY_URL and AGENTPAY_API_KEY env vars (or .env file).
"""

import argparse
import json
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

def load_env():
    """Load .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def api(method: str, path: str, data: dict | None = None) -> dict:
    base = os.environ.get("AGENTPAY_URL", "http://localhost:8080")
    key = os.environ.get("AGENTPAY_API_KEY", "")
    url = f"{base.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)

def pp(obj):
    print(json.dumps(obj, indent=2, default=str))

def cmd_health(_args):
    pp(api("GET", "/v1/health"))

def cmd_wallets(_args):
    pp(api("GET", "/v1/wallets"))

def cmd_wallet(args):
    pp(api("GET", f"/v1/wallets/{args.id}"))

def cmd_balance(args):
    pp(api("GET", f"/v1/wallets/{args.id}/balance"))

def cmd_tx(args):
    pp(api("GET", f"/v1/wallets/{args.wallet_id}/transactions"))

def cmd_stats(_args):
    pp(api("GET", "/v1/admin/stats"))

def main():
    load_env()
    parser = argparse.ArgumentParser(description="AgentPay CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Check API health")
    sub.add_parser("wallets", help="List wallets")

    w = sub.add_parser("wallet", help="Get wallet details")
    w.add_argument("id")

    b = sub.add_parser("balance", help="Get wallet balance")
    b.add_argument("id")

    t = sub.add_parser("tx", help="List transactions")
    t.add_argument("wallet_id")

    sub.add_parser("stats", help="System stats")

    args = parser.parse_args()
    cmds = {
        "health": cmd_health, "wallets": cmd_wallets, "wallet": cmd_wallet,
        "balance": cmd_balance, "tx": cmd_tx, "stats": cmd_stats,
    }
    cmds[args.command](args)

if __name__ == "__main__":
    main()
