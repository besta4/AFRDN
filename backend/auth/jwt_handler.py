"""
auth/jwt_handler.py — JWT token creation and validation.

Uses PyJWT for token handling with HS256 algorithm.
"""

import jwt
import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional
from auth.models import TokenData
from config import load_environment

load_environment()

# Secret key for JWT signing - in production, load from environment
# Generate a new one: secrets.token_urlsafe(32)
JWT_SECRET = os.environ.get(
    "JATAYU_JWT_SECRET",
    "jatayu-fraud-detection-secret-key-change-in-production-2024",
)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
REFRESH_TOKEN_EXPIRE_DAYS = 7


def create_access_token(
    user_id: str,
    user_type: str,
    session_id: str,
    device_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a new JWT access token.

    Args:
        user_id: User's unique ID (e.g., C000000001)
        user_type: CUSTOMER, MERCHANT, or ADMIN
        session_id: Session identifier for revocation
        device_id: Optional device fingerprint
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    payload = {
        "sub": user_id,
        "type": user_type,
        "session": session_id,
        "device": device_id,
        "iat": now,
        "exp": expire,
        "jti": secrets.token_urlsafe(16),  # Unique token ID
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(
    user_id: str,
    session_id: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a refresh token for obtaining new access tokens.
    """
    if expires_delta is None:
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    payload = {
        "sub": user_id,
        "session": session_id,
        "type": "refresh",
        "iat": now,
        "exp": expire,
        "jti": secrets.token_urlsafe(16),
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode a JWT token.

    Args:
        token: The JWT token string

    Returns:
        TokenData if valid, None if invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # Check if it's a refresh token (can't be used as access token)
        if payload.get("type") == "refresh":
            return None

        return TokenData(
            user_id=payload["sub"],
            user_type=payload.get("type", "CUSTOMER"),
            session_id=payload.get("session", ""),
            device_id=payload.get("device"),
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def verify_refresh_token(token: str) -> Optional[dict]:
    """
    Verify a refresh token.

    Returns:
        Dict with user_id and session_id if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # Must be a refresh token
        if payload.get("type") != "refresh":
            return None

        return {
            "user_id": payload["sub"],
            "session_id": payload.get("session", ""),
        }
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_token_hash(token: str) -> str:
    """
    Get a hash of the token for storage (to enable revocation).
    """
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()[:32]
