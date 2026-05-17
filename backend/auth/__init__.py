"""
auth/ — Authentication and authorization module for Jatayu.

Provides:
- JWT token creation and validation
- Password hashing with Argon2
- Role-based access control (RBAC)
- FastAPI dependencies for protected routes
"""

from auth.jwt_handler import create_access_token, verify_token, create_refresh_token
from auth.password import hash_password, verify_password
from auth.dependencies import get_current_user, require_roles
from auth.models import TokenData, LoginRequest, RegisterRequest, TokenResponse

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "hash_password",
    "verify_password",
    "get_current_user",
    "require_roles",
    "TokenData",
    "LoginRequest",
    "RegisterRequest",
    "TokenResponse",
]
