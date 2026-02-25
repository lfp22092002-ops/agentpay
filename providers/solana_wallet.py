"""
Solana Wallet Provider for AgentPay.

Creates and manages Solana wallets for agents.
Handles SOL transfers and USDC (SPL token) transfers.
Private keys encrypted at rest with Fernet (PBKDF2-derived from API_SECRET).
"""
import json
import logging
import os
import base64
from pathlib import Path

from config.settings import ENVIRONMENT
from core.encryption import encrypt, decrypt

logger = logging.getLogger("agentpay.solana")

WALLETS_DIR = Path(__file__).parent.parent / "data" / "wallets"
WALLETS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════
# SOLANA CONFIGURATION
# ═══════════════════════════════════════

SOLANA_RPC_MAINNET = "https://api.mainnet-beta.solana.com"
SOLANA_RPC_DEVNET = "https://api.devnet.solana.com"

# USDC SPL token mint addresses
USDC_MINT_MAINNET = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDC_MINT_DEVNET = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"

SOLANA_EXPLORER = "https://solscan.io"

SOL_DECIMALS = 9  # 1 SOL = 10^9 lamports
USDC_DECIMALS = 6  # USDC has 6 decimals


def _is_testnet() -> bool:
    return ENVIRONMENT == "development"


def _get_rpc() -> str:
    return SOLANA_RPC_DEVNET if _is_testnet() else SOLANA_RPC_MAINNET


def _get_usdc_mint() -> str:
    return USDC_MINT_DEVNET if _is_testnet() else USDC_MINT_MAINNET


def _get_network_name() -> str:
    return "solana-devnet" if _is_testnet() else "solana-mainnet"


def _wallet_file(agent_id: str) -> Path:
    """Solana wallets stored separately from EVM wallets."""
    return WALLETS_DIR / f"{agent_id}_solana.json"


def create_solana_wallet(agent_id: str) -> dict:
    """
    Create a new Solana wallet for an agent.
    Generates a Keypair, encrypts the private key, saves to disk.
    """
    from solders.keypair import Keypair

    wf = _wallet_file(agent_id)
    if wf.exists():
        data = json.loads(wf.read_text())
        return {
            "address": data["address"],
            "network": _get_network_name(),
            "chain": "solana",
            "already_existed": True,
        }

    # Generate new keypair
    keypair = Keypair()
    public_key = str(keypair.pubkey())
    # Store full 64-byte secret key (first 32 = private, last 32 = public)
    private_key_bytes = bytes(keypair)
    private_key_b64 = base64.b64encode(private_key_bytes).decode()

    wallet_data = {
        "agent_id": agent_id,
        "address": public_key,
        "network": _get_network_name(),
        "chain": "solana",
        "private_key": encrypt(private_key_b64),
        "encrypted": True,
    }
    wf.write_text(json.dumps(wallet_data))
    os.chmod(wf, 0o600)

    logger.info(f"Created Solana wallet {public_key} for agent {agent_id}")

    return {
        "address": public_key,
        "network": _get_network_name(),
        "chain": "solana",
    }


def get_solana_wallet_address(agent_id: str) -> str | None:
    """Get the Solana address for an agent's wallet."""
    wf = _wallet_file(agent_id)
    if not wf.exists():
        return None
    data = json.loads(wf.read_text())
    return data.get("address")


def _load_keypair(agent_id: str):
    """Load and decrypt the Solana keypair for an agent."""
    from solders.keypair import Keypair

    wf = _wallet_file(agent_id)
    if not wf.exists():
        return None

    data = json.loads(wf.read_text())
    pk_b64 = data["private_key"]
    if data.get("encrypted", False):
        pk_b64 = decrypt(pk_b64)

    pk_bytes = base64.b64decode(pk_b64)
    return Keypair.from_bytes(pk_bytes)


def get_solana_balance(agent_id: str) -> dict:
    """Get SOL and USDC balance for an agent's Solana wallet."""
    from solana.rpc.api import Client
    from solders.pubkey import Pubkey

    wf = _wallet_file(agent_id)
    if not wf.exists():
        return {"error": "No Solana wallet found"}

    data = json.loads(wf.read_text())
    address = data["address"]
    network = _get_network_name()

    try:
        client = Client(_get_rpc())
        pubkey = Pubkey.from_string(address)

        # SOL balance
        resp = client.get_balance(pubkey)
        sol_lamports = resp.value
        sol_balance = sol_lamports / (10 ** SOL_DECIMALS)

        # USDC (SPL token) balance
        usdc_balance = 0.0
        try:
            usdc_mint = Pubkey.from_string(_get_usdc_mint())
            from solana.rpc.types import TokenAccountOpts
            token_resp = client.get_token_accounts_by_owner_json_parsed(
                pubkey,
                TokenAccountOpts(mint=usdc_mint),
            )
            if token_resp.value:
                for acct in token_resp.value:
                    parsed = acct.account.data.parsed
                    info = parsed["info"]
                    token_amount = info["tokenAmount"]
                    usdc_balance += float(token_amount["uiAmount"] or 0)
        except Exception as e:
            logger.warning(f"Failed to get USDC balance for {agent_id} on Solana: {e}")

        return {
            "address": address,
            "network": network,
            "chain": "solana",
            "native_token": "SOL",
            "balance_native": str(sol_balance),
            "balance_sol": str(sol_balance),
            "balance_usdc": str(usdc_balance),
            "explorer": SOLANA_EXPLORER,
        }
    except Exception as e:
        logger.error(f"Failed to get Solana balance for {agent_id}: {e}")
        return {"address": address, "network": network, "chain": "solana", "error": str(e)}


async def send_sol(agent_id: str, to_address: str, amount: float) -> dict:
    """Send SOL from agent wallet to an address."""
    from solana.rpc.api import Client
    from solana.transaction import Transaction
    from solders.pubkey import Pubkey
    from solders.system_program import TransferParams, transfer

    keypair = _load_keypair(agent_id)
    if not keypair:
        return {"success": False, "error": "No Solana wallet found"}

    try:
        client = Client(_get_rpc())
        lamports = int(amount * (10 ** SOL_DECIMALS))

        to_pubkey = Pubkey.from_string(to_address)

        # Build transfer instruction
        ix = transfer(TransferParams(
            from_pubkey=keypair.pubkey(),
            to_pubkey=to_pubkey,
            lamports=lamports,
        ))

        # Get recent blockhash
        recent_blockhash = client.get_latest_blockhash().value.blockhash

        # Build and sign transaction
        tx = Transaction()
        tx.recent_blockhash = recent_blockhash
        tx.fee_payer = keypair.pubkey()
        tx.add(ix)
        tx.sign(keypair)

        # Send
        result = client.send_transaction(tx, keypair)
        tx_sig = str(result.value)

        logger.info(f"Sent {amount} SOL from {keypair.pubkey()} to {to_address}: {tx_sig}")
        return {
            "success": True,
            "tx_hash": tx_sig,
            "amount": amount,
            "to": to_address,
            "chain": "solana",
        }
    except Exception as e:
        logger.error(f"SOL transfer failed: {e}")
        return {"success": False, "error": str(e)}


async def send_solana_usdc(agent_id: str, to_address: str, amount: float) -> dict:
    """Send USDC (SPL token) from agent wallet to an address on Solana."""
    from solana.rpc.api import Client
    from solana.transaction import Transaction
    from solders.pubkey import Pubkey
    from spl.token.instructions import transfer_checked, TransferCheckedParams

    keypair = _load_keypair(agent_id)
    if not keypair:
        return {"success": False, "error": "No Solana wallet found"}

    try:
        client = Client(_get_rpc())
        usdc_mint = Pubkey.from_string(_get_usdc_mint())
        to_pubkey = Pubkey.from_string(to_address)
        amount_raw = int(amount * (10 ** USDC_DECIMALS))

        # Find sender's USDC associated token account
        from spl.token.constants import TOKEN_PROGRAM_ID
        from solders.pubkey import Pubkey as SoldersPubkey

        # Get associated token address for sender
        sender_ata = SoldersPubkey.find_program_address(
            [bytes(keypair.pubkey()), bytes(TOKEN_PROGRAM_ID), bytes(usdc_mint)],
            SoldersPubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"),
        )[0]

        # Get associated token address for receiver
        receiver_ata = SoldersPubkey.find_program_address(
            [bytes(to_pubkey), bytes(TOKEN_PROGRAM_ID), bytes(usdc_mint)],
            SoldersPubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"),
        )[0]

        # Build transfer_checked instruction
        ix = transfer_checked(TransferCheckedParams(
            program_id=TOKEN_PROGRAM_ID,
            source=sender_ata,
            mint=usdc_mint,
            dest=receiver_ata,
            owner=keypair.pubkey(),
            amount=amount_raw,
            decimals=USDC_DECIMALS,
        ))

        # Get recent blockhash
        recent_blockhash = client.get_latest_blockhash().value.blockhash

        # Build and sign transaction
        tx = Transaction()
        tx.recent_blockhash = recent_blockhash
        tx.fee_payer = keypair.pubkey()
        tx.add(ix)
        tx.sign(keypair)

        result = client.send_transaction(tx, keypair)
        tx_sig = str(result.value)

        logger.info(f"Sent {amount} USDC from {keypair.pubkey()} to {to_address} on Solana: {tx_sig}")
        return {
            "success": True,
            "tx_hash": tx_sig,
            "amount": amount,
            "to": to_address,
            "chain": "solana",
        }
    except Exception as e:
        logger.error(f"Solana USDC transfer failed: {e}")
        return {"success": False, "error": str(e)}
