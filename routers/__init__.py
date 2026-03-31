"""
routers/__init__.py — API router package.
"""

from routers.auth import router as auth_router
from routers.users import router as users_router
from routers.transactions import router as transactions_router
from routers.admin import router as admin_router
from routers.merchant import router as merchant_router
from routers.support import router as support_router

__all__ = [
    "auth_router",
    "users_router",
    "transactions_router",
    "admin_router",
    "merchant_router",
    "support_router",
]
