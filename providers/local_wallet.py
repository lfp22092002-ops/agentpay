"""
Local Wallet Provider for AgentPay.

Creates and manages wallets using local private keys on multiple EVM chains.
Supports: Base, Polygon PoS, BNB Chain.
No Coinbase CDP account needed — fully self-custodial.
Private keys encrypted at rest with Fernet (PBKDF2-derived from API_SECRET).

Key insight: one private key works across ALL EVM chains (same address).
We just talk to different RPCs per chain.
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

# ═══════════════════════════════════════
# MULTI-CHAIN CONFIGURATION
# ═══════════════════════════════════════

CHAIN_CONFIGS = {
    "base": {
        "name": "Base",
        "rpc_mainnet": "https://mainnet.base.org",
        "rpc_testnet": "https://sepolia.base.org",
        "chain_id_mainnet": 8453,
        "chain_id_testnet": 84532,
        "usdc_mainnet": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "usdc_testnet": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
        "native_token": "ETH",
        "explorer": "https://basescan.org",
    },
    "polygon": {
        "name": "Polygon PoS",
        "rpc_mainnet": "https://polygon-rpc.com",
        "rpc_testnet": "https://rpc-amoy.polygon.technology",
        "chain_id_mainnet": 137,
        "chain_id_testnet": 80002,
        "usdc_mainnet": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
        "usdc_testnet": "0x41E94Eb019C0762f9Bfcf9Fb1E58725BfB0e7582",
        "native_token": "POL",
        "explorer": "https://polygonscan.com",
    },
    "bnb": {
        "name": "BNB Chain",
        "rpc_mainnet": "https://bsc-dataseed1.binance.org",
        "rpc_testnet": "https://data-seed-prebsc-1-s1.binance.org:8545",
        "chain_id_mainnet": 56,
        "chain_id_testnet": 97,
        "usdc_mainnet": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
        "usdc_testnet": "0x64544969ed7EBf5f083679233325356EbE738930",
        "native_token": "BNB",
        "explorer": "https://bscscan.com",
    },
}

SUPPORTED_EVM_CHAINS = list(CHAIN_CONFIGS.keys())

# Minimal ERC20 ABI for balance + transfer
ERC20_ABI = json.loads('[{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"}]')


def _get_chain_config(chain: str) -> dict:
    """Get chain configuration, raise if invalid."""
    if chain not in CHAIN_CONFIGS:
        raise ValueError(f"Unsupported chain '{chain}'. Supported: {', '.join(SUPPORTED_EVM_CHAINS)}")
    return CHAIN_CONFIGS[chain]


def _is_testnet() -> bool:
    return ENVIRONMENT == "development"


def _get_rpc(chain: str) -> str:
    """Get RPC URL for the given chain based on environment."""
    config = _get_chain_config(chain)
    return config["rpc_testnet"] if _is_testnet() else config["rpc_mainnet"]


def _get_network_name(chain: str) -> str:
    """Get network name string (e.g. 'base-sepolia' or 'base-mainnet')."""
    suffix = "testnet" if _is_testnet() else "mainnet"
    return f"{chain}-{suffix}"


def _get_usdc_address(chain: str) -> str:
    """Get USDC contract address for the given chain."""
    config = _get_chain_config(chain)
    return config["usdc_testnet"] if _is_testnet() else config["usdc_mainnet"]


def _get_w3(chain: str = "base") -> Web3:
    """Get Web3 instance for the specified chain."""
    rpc = _get_rpc(chain)
    return Web3(Web3.HTTPProvider(rpc))


# Legacy compatibility
NETWORK = "base-sepolia" if ENVIRONMENT == "development" else "base-mainnet"


def create_agent_wallet(agent_id: str, chain: str = "base") -> dict:
    """
    Create a new EVM wallet for an agent using a locally generated private key.
    Returns address. Private key is stored encrypted-at-rest.

    Since EVM wallets share the same private key across all chains,
    this only creates one wallet file per agent. The `chain` parameter
    determines which network name is stored, but the address works everywhere.
    """
    _get_chain_config(chain)  # validate chain

    # Check if wallet already exists (EVM wallets are shared across chains)
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if wallet_file.exists():
        data = json.loads(wallet_file.read_text())
        return {
            "address": data["address"],
            "network": _get_network_name(chain),
            "chain": chain,
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
        "network": _get_network_name(chain),
        "private_key": encrypt(private_key),
        "encrypted": True,
    }
    wallet_file.write_text(json.dumps(wallet_data))
    os.chmod(wallet_file, 0o600)

    logger.info(f"Created EVM wallet {address} for agent {agent_id} (chain={chain})")

    return {
        "address": address,
        "network": _get_network_name(chain),
        "chain": chain,
    }


def get_wallet_address(agent_id: str) -> str | None:
    """Get the on-chain address for an agent's EVM wallet (same across all EVM chains)."""
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return None
    data = json.loads(wallet_file.read_text())
    return data.get("address")


def get_wallet_balance(agent_id: str, chain: str = "base") -> dict:
    """Get native token and USDC balance for an agent's wallet on the specified chain."""
    config = _get_chain_config(chain)

    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return {"error": "No wallet found"}

    data = json.loads(wallet_file.read_text())
    address = data["address"]
    network = _get_network_name(chain)

    try:
        w3 = _get_w3(chain)
        # Native token balance
        native_balance = w3.from_wei(w3.eth.get_balance(address), "ether")

        # USDC balance
        usdc_contract_addr = _get_usdc_address(chain)
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
            "network": network,
            "chain": chain,
            "native_token": config["native_token"],
            "balance_native": str(native_balance),
            "balance_eth": str(native_balance),  # backward compat
            "balance_usdc": str(usdc_balance),
            "explorer": config["explorer"],
        }
    except Exception as e:
        logger.error(f"Failed to get balance for {agent_id} on {chain}: {e}")
        return {"address": address, "network": network, "chain": chain, "error": str(e)}


def _load_private_key(data: dict) -> str:
    """Load private key from wallet data, decrypting if needed."""
    pk = data["private_key"]
    if data.get("encrypted", False):
        return decrypt(pk)
    return pk


async def send_usdc(agent_id: str, to_address: str, amount: float, chain: str = "base") -> dict:
    """Send USDC from agent wallet to an address on the specified EVM chain."""
    _get_chain_config(chain)  # validate

    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return {"success": False, "error": "No wallet found"}

    data = json.loads(wallet_file.read_text())
    private_key = _load_private_key(data)
    address = data["address"]

    usdc_contract_addr = _get_usdc_address(chain)
    if not usdc_contract_addr:
        return {"success": False, "error": f"USDC not configured for {chain}"}

    try:
        w3 = _get_w3(chain)
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

        logger.info(f"Sent {amount} USDC from {address} to {to_address} on {chain}: {tx_hash.hex()}")
        return {
            "success": True,
            "tx_hash": tx_hash.hex(),
            "amount": amount,
            "to": to_address,
            "chain": chain,
        }
    except Exception as e:
        logger.error(f"USDC transfer failed on {chain}: {e}")
        return {"success": False, "error": str(e)}


async def send_native(agent_id: str, to_address: str, amount: float, chain: str = "base") -> dict:
    """Send native token (ETH/POL/BNB) from agent wallet on the specified chain."""
    config = _get_chain_config(chain)

    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return {"success": False, "error": "No wallet found"}

    data = json.loads(wallet_file.read_text())
    private_key = _load_private_key(data)
    address = data["address"]

    try:
        w3 = _get_w3(chain)
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

        logger.info(f"Sent {amount} {config['native_token']} from {address} to {to_address} on {chain}: {tx_hash.hex()}")
        return {
            "success": True,
            "tx_hash": tx_hash.hex(),
            "amount": amount,
            "to": to_address,
            "chain": chain,
            "native_token": config["native_token"],
        }
    except Exception as e:
        logger.error(f"{config['native_token']} transfer failed on {chain}: {e}")
        return {"success": False, "error": str(e)}


# Backward compatibility alias
async def send_eth(agent_id: str, to_address: str, amount: float, chain: str = "base") -> dict:
    """Backward-compatible alias for send_native()."""
    return await send_native(agent_id, to_address, amount, chain)


def get_all_chain_balances(agent_id: str) -> list[dict]:
    """Get balances across all supported EVM chains for an agent."""
    wallet_file = WALLETS_DIR / f"{agent_id}.json"
    if not wallet_file.exists():
        return []

    results = []
    for chain in SUPPORTED_EVM_CHAINS:
        balance = get_wallet_balance(agent_id, chain)
        results.append(balance)
    return results
