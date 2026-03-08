"""
Tests for core/encryption.py — Fernet encryption/decryption with PBKDF2 key derivation.
"""
import os
from unittest.mock import patch

import pytest


class TestSaltManagement:
    """Salt file creation, persistence, and permissions."""

    def test_salt_created_if_missing(self, tmp_path):
        """Salt file is auto-created when it doesn't exist."""
        salt_file = str(tmp_path / "data" / ".salt")
        with patch("core.encryption.SALT_FILE", salt_file):
            from core.encryption import _get_salt
            salt = _get_salt()
            assert len(salt) == 16
            assert os.path.exists(salt_file)

    def test_salt_is_persistent(self, tmp_path):
        """Same salt returned on repeated calls."""
        salt_file = str(tmp_path / "data" / ".salt")
        with patch("core.encryption.SALT_FILE", salt_file):
            from core.encryption import _get_salt
            salt1 = _get_salt()
            salt2 = _get_salt()
            assert salt1 == salt2

    def test_salt_file_permissions(self, tmp_path):
        """Salt file is created with 0600 permissions."""
        salt_file = str(tmp_path / "data" / ".salt")
        with patch("core.encryption.SALT_FILE", salt_file):
            from core.encryption import _get_salt
            _get_salt()
            mode = os.stat(salt_file).st_mode & 0o777
            assert mode == 0o600

    def test_salt_is_16_bytes(self, tmp_path):
        """Salt is exactly 16 bytes."""
        salt_file = str(tmp_path / "data" / ".salt")
        with patch("core.encryption.SALT_FILE", salt_file):
            from core.encryption import _get_salt
            salt = _get_salt()
            assert isinstance(salt, bytes)
            assert len(salt) == 16


class TestEncryptDecrypt:
    """Round-trip encryption and decryption."""

    def test_roundtrip(self):
        """Encrypting then decrypting returns original text."""
        from core.encryption import encrypt, decrypt
        plaintext = "super-secret-private-key-0x1234abcd"
        ciphertext = encrypt(plaintext)
        assert decrypt(ciphertext) == plaintext

    def test_ciphertext_differs_from_plaintext(self):
        """Ciphertext is not the same as plaintext."""
        from core.encryption import encrypt
        plaintext = "my-wallet-key"
        ciphertext = encrypt(plaintext)
        assert ciphertext != plaintext

    def test_ciphertext_is_string(self):
        """Ciphertext is a base64-encoded string."""
        from core.encryption import encrypt
        ciphertext = encrypt("test")
        assert isinstance(ciphertext, str)

    def test_different_plaintexts_different_ciphertexts(self):
        """Different inputs produce different ciphertexts."""
        from core.encryption import encrypt
        c1 = encrypt("secret-one")
        c2 = encrypt("secret-two")
        assert c1 != c2

    def test_same_plaintext_different_ciphertexts(self):
        """Same input encrypted twice produces different ciphertexts (Fernet uses random IV)."""
        from core.encryption import encrypt
        c1 = encrypt("same-text")
        c2 = encrypt("same-text")
        assert c1 != c2  # Fernet adds a random IV

    def test_empty_string(self):
        """Empty string can be encrypted and decrypted."""
        from core.encryption import encrypt, decrypt
        ciphertext = encrypt("")
        assert decrypt(ciphertext) == ""

    def test_unicode_roundtrip(self):
        """Unicode text survives encryption roundtrip."""
        from core.encryption import encrypt, decrypt
        plaintext = "café-Ξ-日本語-🔐"
        assert decrypt(encrypt(plaintext)) == plaintext

    def test_long_text_roundtrip(self):
        """Long text (private key + metadata) survives roundtrip."""
        from core.encryption import encrypt, decrypt
        plaintext = "0x" + "a1b2c3d4e5f6" * 100
        assert decrypt(encrypt(plaintext)) == plaintext

    def test_decrypt_wrong_ciphertext_raises(self):
        """Decrypting garbage raises an error."""
        from core.encryption import decrypt
        with pytest.raises(Exception):  # InvalidToken or base64 error
            decrypt("not-valid-ciphertext-at-all")

    def test_different_secret_cannot_decrypt(self, tmp_path):
        """Data encrypted with one secret can't be decrypted with another."""
        salt_file = str(tmp_path / "data" / ".salt")
        with patch("core.encryption.SALT_FILE", salt_file):
            from core.encryption import encrypt, decrypt, _get_salt
            # Force salt creation
            _get_salt()

            with patch.dict(os.environ, {"API_SECRET": "secret-key-one"}):
                ciphertext = encrypt("private-data")

            with patch.dict(os.environ, {"API_SECRET": "different-secret"}):
                with pytest.raises(Exception):
                    decrypt(ciphertext)
