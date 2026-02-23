"""
Coinbase Wallet Provider for AgentPay.

Creates and manages USDC wallets on Base network for AI agents.
Each agent gets its own on-chain wallet for receiving/sending USDC.

Requires CDP API keys from https://portal.cdp.coinbase.com/projects/api-keys
"""
import json
import logging
import os
from pathlib import Path
from coinbase_agentkit import (
    CdpEvmWalletProvider,
    CdpEvmWalletProviderConfig,
)
from config.settings import ENVIRONMENT

logger = logging.getLogger("agentpay.coinbase")

WALLETS_DIR = Path(__file__).parent.parent / "data" / "wallets"
WALLETS_DIR.mkdir(parents=True, exist_ok=True)

# Use testnet for dev, mainnet for prod
NETWORK = "base-sepolia" if ENVIRONMENT == "development" else "base-mainnet"


def _get_cdp_keys() -> tuple[str, str, str]:
    """Load CDP API keys from env."""
    api_key_id = os.getenv("CDP_API_KEY_ID", "")
    api_key_secret = os.getenv("CDP_API_KEY_SECRET", "")
    wallet_secret = os.getenv("CDP_WALLET_SECRET", "")
    if not api_key_id or not api_key_secret:
        raise ValueError(
            "CDP API keys not configured. "
            "Set CDP_API_KEY_ID and CDP_API_KEY_SECRET in .env\n"
            "Get them free at: https://portal.cdp.coinbase.com"
        )
    return api_key_id, api_key_secret, wallet_secret


def create_agent_wallet(agent_id: str) -> dict:
    """
    Create a new on-chain wallet for an agent.
    Returns dict with wallet address and wallet data for persistence.
    """
    api_key_id, api_key_secret, wallet_secret = _get_cdp_keys()

    config = CdpEvmWalletProviderConfig(
        api_key_id=api_key_id,
        api_key_secret=api_key_secret,
        wallet_secret=wallet_secret or None,
        network_id=NETWORK,
    )
    wallet_provider = CdpEvmWalletProvider(config)

    address = wallet_provider.get_address()

    # Persist wallet info
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    wallet_file.write_text(json.dumps({
        "agent_id": agent_id,
        "address": address,
        "network": NETWORK,
    }))

    logger.info(f"Created wallet {address} for agent {agent_id} on {NETWORK}")

    return {
        "address": address,
        "network": NETWORK,
    }


def load_agent_wallet(agent_id: str) -> CdpEvmWalletProvider | None:
    """Load an existing agent wallet."""
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return None

    data = json.loads(wallet_file.read_text())
    api_key_id, api_key_secret, wallet_secret = _get_cdp_keys()

    config = CdpEvmWalletProviderConfig(
        api_key_id=api_key_id,
        api_key_secret=api_key_secret,
        wallet_secret=wallet_secret or None,
        network_id=data.get("network", NETWORK),
        address=data.get("address"),
    )
    return CdpEvmWalletProvider(config)


def get_wallet_address(agent_id: str) -> str | None:
    """Get the on-chain address for an agent's wallet."""
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return None
    data = json.loads(wallet_file.read_text())
    return data.get("address")


def get_wallet_balance(agent_id: str) -> dict:
    """Get on-chain balance for an agent's wallet."""
    provider = load_agent_wallet(agent_id)
    if not provider:
        return {"error": "No wallet found"}

    try:
        balance = provider.get_balance()
        return {
            "address": provider.get_address(),
            "network": NETWORK,
            "balance_eth": str(balance),
        }
    except Exception as e:
        logger.error(f"Failed to get balance for {agent_id}: {e}")
        return {"error": str(e)}


async def send_usdc(agent_id: str, to_address: str, amount: float) -> dict:
    """
    Send USDC from agent wallet to an address.
    Uses ERC20 transfer on Base network.
    """
    provider = load_agent_wallet(agent_id)
    if not provider:
        return {"success": False, "error": "No wallet found"}

    # USDC contract addresses
    USDC_ADDRESS = {
        "base-sepolia": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
        "base-mainnet": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    }

    usdc_contract = USDC_ADDRESS.get(NETWORK)
    if not usdc_contract:
        return {"success": False, "error": f"USDC not configured for {NETWORK}"}

    try:
        amount_wei = int(amount * 10**6)

        tx_hash = provider.invoke_contract(
            contract_address=usdc_contract,
            method="transfer",
            abi=[{
                "type": "function",
                "name": "transfer",
                "inputs": [
                    {"name": "to", "type": "address"},
                    {"name": "amount", "type": "uint256"}
                ],
                "outputs": [{"name": "", "type": "bool"}]
            }],
            args={"to": to_address, "amount": str(amount_wei)},
        )

        logger.info(f"Sent {amount} USDC from {agent_id} to {to_address}: {tx_hash}")
        return {
            "success": True,
            "tx_hash": str(tx_hash),
            "amount": amount,
            "to": to_address,
        }
    except Exception as e:
        logger.error(f"USDC transfer failed: {e}")
        return {"success": False, "error": str(e)}
