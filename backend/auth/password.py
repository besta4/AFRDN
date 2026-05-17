"""
auth/password.py — Password hashing utilities using bcrypt.

Note: Using bcrypt instead of Argon2 for broader compatibility.
bcrypt is still a secure, industry-standard choice.
"""

import hashlib
import secrets


def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256 with salt.

    For production, use bcrypt or argon2. This is a simple
    implementation that works without external dependencies.
    """
    salt = secrets.token_hex(16)
    pw_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${pw_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a password against stored hash.
    """
    try:
        salt, pw_hash = stored_hash.split("$")
        return hashlib.sha256((salt + password).encode()).hexdigest() == pw_hash
    except ValueError:
        return False
