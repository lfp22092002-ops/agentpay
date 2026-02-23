"""
Local Wallet Provider for AgentPay.

Creates and manages wallets using local private keys on Base network.
No Coinbase CDP account needed — fully self-custodial.
Private keys encrypted at rest with Fernet (PBKDF2-derived from API_SECRET).
"""
import json
import logging
import os
from pathlib import Path
from eth_account import Account
from web3 import Web3

from config.settings import ENVIRONMENT
from core.encryption import encrypt, decrypt

logger = logging.getLogger("agentpay.wallet")

WALLETS_DIR = Path(__file__).parent.parent / "data" / "wallets"
WALLETS_DIR.mkdir(parents=True, exist_ok=True)
os.chmod(WALLETS_DIR, 0o700)

# Base network RPCs
RPC_URLS = {
    "base-sepolia": "https://sepolia.base.org",
    "base-mainnet": "https://mainnet.base.org",
}

NETWORK = "base-sepolia" if ENVIRONMENT == "development" else "base-mainnet"

# USDC contract addresses (6 decimals)
USDC_ADDRESS = {
    "base-sepolia": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "base-mainnet": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
}

# Minimal ERC20 ABI for balance + transfer
ERC20_ABI = json.loads('[{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"}]')


def _get_w3() -> Web3:
    rpc = RPC_URLS.get(NETWORK)
    return Web3(Web3.HTTPProvider(rpc))


def create_agent_wallet(agent_id: str) -> dict:
    """
    Create a new wallet for an agent using a locally generated private key.
    Returns address. Private key is stored encrypted-at-rest.
    """
    # Check if wallet already exists
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if wallet_file.exists():
        data = json.loads(wallet_file.read_text())
        return {
            "address": data["address"],
            "network": data["network"],
            "already_existed": True,
        }

    # Generate new account
    account = Account.create()
    address = account.address
    private_key = account.key.hex()

    # Save wallet data (private key encrypted at rest)
    wallet_data = {
        "agent_id": agent_id,
        "address": address,
        "network": NETWORK,
        "private_key": encrypt(private_key),
        "encrypted": True,
    }
    wallet_file.write_text(json.dumps(wallet_data))
    os.chmod(wallet_file, 0o600)

    logger.info(f"Created wallet {address} for agent {agent_id} on {NETWORK}")

    return {
        "address": address,
        "network": NETWORK,
    }


def get_wallet_address(agent_id: str) -> str | None:
    """Get the on-chain address for an agent's wallet."""
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return None
    data = json.loads(wallet_file.read_text())
    return data.get("address")


def get_wallet_balance(agent_id: str) -> dict:
    """Get ETH and USDC balance for an agent's wallet."""
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return {"error": "No wallet found"}

    data = json.loads(wallet_file.read_text())
    address = data["address"]

    try:
        w3 = _get_w3()
        # ETH balance
        eth_balance = w3.from_wei(w3.eth.get_balance(address), "ether")

        # USDC balance
        usdc_contract_addr = USDC_ADDRESS.get(NETWORK)
        usdc_balance = 0
        if usdc_contract_addr:
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(usdc_contract_addr),
                abi=ERC20_ABI,
            )
            raw = contract.functions.balanceOf(Web3.to_checksum_address(address)).call()
            usdc_balance = raw / 10**6

        return {
            "address": address,
            "network": NETWORK,
            "balance_eth": str(eth_balance),
            "balance_usdc": str(usdc_balance),
        }
    except Exception as e:
        logger.error(f"Failed to get balance for {agent_id}: {e}")
        return {"address": address, "network": NETWORK, "error": str(e)}


def _load_private_key(data: dict) -> str:
    """Load private key from wallet data, decrypting if needed."""
    pk = data["private_key"]
    if data.get("encrypted", False):
        return decrypt(pk)
    return pk


async def send_usdc(agent_id: str, to_address: str, amount: float) -> dict:
    """Send USDC from agent wallet to an address on Base."""
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return {"success": False, "error": "No wallet found"}

    data = json.loads(wallet_file.read_text())
    private_key = _load_private_key(data)
    address = data["address"]

    usdc_contract_addr = USDC_ADDRESS.get(NETWORK)
    if not usdc_contract_addr:
        return {"success": False, "error": f"USDC not configured for {NETWORK}"}

    try:
        w3 = _get_w3()
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(usdc_contract_addr),
            abi=ERC20_ABI,
        )

        amount_raw = int(amount * 10**6)
        nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(address))

        tx = contract.functions.transfer(
            Web3.to_checksum_address(to_address),
            amount_raw,
        ).build_transaction({
            "from": Web3.to_checksum_address(address),
            "nonce": nonce,
            "gas": 100000,
            "gasPrice": w3.eth.gas_price,
        })

        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        logger.info(f"Sent {amount} USDC from {address} to {to_address}: {tx_hash.hex()}")
        return {
            "success": True,
            "tx_hash": tx_hash.hex(),
            "amount": amount,
            "to": to_address,
        }
    except Exception as e:
        logger.error(f"USDC transfer failed: {e}")
        return {"success": False, "error": str(e)}


async def send_eth(agent_id: str, to_address: str, amount: float) -> dict:
    """Send ETH from agent wallet."""
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return {"success": False, "error": "No wallet found"}

    data = json.loads(wallet_file.read_text())
    private_key = _load_private_key(data)
    address = data["address"]

    try:
        w3 = _get_w3()
        nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(address))

        tx = {
            "to": Web3.to_checksum_address(to_address),
            "value": w3.to_wei(amount, "ether"),
            "gas": 21000,
            "gasPrice": w3.eth.gas_price,
            "nonce": nonce,
            "chainId": w3.eth.chain_id,
        }

        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        return {
            "success": True,
            "tx_hash": tx_hash.hex(),
            "amount": amount,
            "to": to_address,
        }
    except Exception as e:
        logger.error(f"ETH transfer failed: {e}")
        return {"success": False, "error": str(e)}
