"""
Encryption utilities for AgentPay.
Encrypts/decrypts sensitive data (private keys, card details) using Fernet.
Master key derived from API_SECRET via PBKDF2.
"""
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# Salt stored alongside the app — unique per installation
SALT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", ".salt")


def _get_salt() -> bytes:
    os.makedirs(os.path.dirname(SALT_FILE), exist_ok=True)
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, "rb") as f:
            return f.read()
    salt = os.urandom(16)
    with open(SALT_FILE, "wb") as f:
        f.write(salt)
    os.chmod(SALT_FILE, 0o600)
    return salt


def _get_fernet() -> Fernet:
    secret = os.getenv("API_SECRET", "agentpay-secret-change-me")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_get_salt(),
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
    return Fernet(key)


def encrypt(plaintext: str) -> str:
    """Encrypt a string, return base64-encoded ciphertext."""
    f = _get_fernet()
    return f.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a base64-encoded ciphertext back to string."""
    f = _get_fernet()
    return f.decrypt(ciphertext.encode()).decode()
