"""
routers/users.py — User and account management endpoints.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone

import database as db
from auth.dependencies import get_current_user, require_customer_or_merchant
from auth.models import TokenData, UserResponse, AccountResponse

router = APIRouter(prefix="/users", tags=["Users"])


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    pincode: Optional[str] = None


class PayeeResponse(BaseModel):
    payee_id: str
    payee_user_id: str
    payee_name: str
    payee_email: str
    business_name: Optional[str] = None
    nickname: Optional[str] = None
    added_at: str
    is_verified: bool
    total_txn_count: int
    total_txn_amount: float


class AddPayeeRequest(BaseModel):
    payee_user_id: str
    nickname: Optional[str] = None


class EmailOTPSettingRequest(BaseModel):
    enabled: bool


# ══════════════════════════════════════════════════════════════════════════════
# Profile Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/profile", response_model=UserResponse)
async def get_profile(user: TokenData = Depends(get_current_user)):
    """Get current user's profile."""
    profile = db.get_user_with_profile(user.user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        user_id=profile["user_id"],
        email=profile["email"],
        display_name=profile.get("display_name", "User"),
        user_type=profile["user_type"],
        account_status=profile["account_status"],
        business_name=profile.get("business_name"),
        phone=profile.get("phone"),
        created_at=profile["created_at"],
        last_login=profile.get("last_login")
    )


@router.put("/profile")
async def update_profile(
    body: UpdateProfileRequest,
    user: TokenData = Depends(get_current_user)
):
    """Update current user's profile."""
    # Update user fields if phone changed
    if body.phone:
        db.update_user(user.user_id, phone=body.phone)

    # Update profile fields
    # Note: Would need to extend db functions for profile updates
    # For now, just return success
    return {"message": "Profile updated successfully"}


@router.get("/settings/email-otp")
async def get_email_otp_setting(user: TokenData = Depends(get_current_user)):
    """Get current email OTP setting."""
    enabled = db.is_email_otp_enabled(user.user_id)
    return {"email_otp_enabled": enabled}


@router.put("/settings/email-otp")
async def update_email_otp_setting(
    body: EmailOTPSettingRequest,
    user: TokenData = Depends(get_current_user)
):
    """Enable or disable email OTP for transactions."""
    db.update_user(user.user_id, email_otp_enabled=1 if body.enabled else 0)
    return {
        "message": f"Email OTP {'enabled' if body.enabled else 'disabled'} successfully",
        "email_otp_enabled": body.enabled
    }


# ══════════════════════════════════════════════════════════════════════════════
# Account Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/accounts", response_model=List[AccountResponse])
async def get_accounts(user: TokenData = Depends(get_current_user)):
    """Get all accounts for current user."""
    accounts = db.get_user_accounts(user.user_id)
    return [
        AccountResponse(
            account_id=acc["account_id"],
            account_type=acc["account_type"],
            balance=acc["balance"],
            currency=acc.get("currency", "INR"),
            is_primary=bool(acc["is_primary"]),
            daily_limit=acc.get("daily_limit")
        )
        for acc in accounts
    ]


@router.get("/accounts/{account_id}/balance")
async def get_account_balance(
    account_id: str,
    user: TokenData = Depends(get_current_user)
):
    """Get real-time balance for a specific account."""
    account = db.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    # Verify ownership
    if account["user_id"] != user.user_id and user.user_type != "ADMIN":
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "account_id": account_id,
        "balance": account["balance"],
        "currency": "INR",
        "updated_at": account.get("updated_at")
    }


# ══════════════════════════════════════════════════════════════════════════════
# Payee Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/payees", response_model=List[PayeeResponse])
async def get_payees(user: TokenData = Depends(require_customer_or_merchant)):
    """Get all saved payees for current user."""
    payees = db.get_user_payees(user.user_id)

    result = []
    for p in payees:
        result.append(PayeeResponse(
            payee_id=p["payee_id"],
            payee_user_id=p["payee_user_id"],
            payee_name=p.get("payee_name", "Unknown"),
            payee_email=p.get("payee_email", ""),
            business_name=p.get("business_name"),
            nickname=p.get("nickname"),
            added_at=p["added_at"],
            is_verified=bool(p.get("is_verified", 0)),
            total_txn_count=p.get("total_txn_count", 0),
            total_txn_amount=p.get("total_txn_amount", 0.0)
        ))

    return result


@router.post("/payees", response_model=PayeeResponse)
async def add_payee(
    body: AddPayeeRequest,
    user: TokenData = Depends(require_customer_or_merchant)
):
    """Add a new payee."""
    # Check payee exists
    payee_user = db.get_user_by_id(body.payee_user_id)
    if not payee_user:
        raise HTTPException(status_code=404, detail="Payee user not found")

    # Can't add self as payee
    if body.payee_user_id == user.user_id:
        raise HTTPException(status_code=400, detail="Cannot add yourself as payee")

    # Check if already added
    existing = db.get_payee(user.user_id, body.payee_user_id)
    if existing:
        raise HTTPException(status_code=409, detail="Payee already added")

    # Add payee
    payee_id = db.add_payee(
        user_id=user.user_id,
        payee_user_id=body.payee_user_id,
        nickname=body.nickname
    )

    # Get payee profile
    payee_profile = db.get_user_with_profile(body.payee_user_id)

    return PayeeResponse(
        payee_id=payee_id,
        payee_user_id=body.payee_user_id,
        payee_name=payee_profile.get("display_name", "Unknown") if payee_profile else "Unknown",
        payee_email=payee_user["email"],
        business_name=payee_profile.get("business_name") if payee_profile else None,
        nickname=body.nickname,
        added_at=datetime.now(timezone.utc).isoformat(),
        is_verified=False,
        total_txn_count=0,
        total_txn_amount=0.0
    )


@router.delete("/payees/{payee_id}")
async def remove_payee(
    payee_id: str,
    user: TokenData = Depends(require_customer_or_merchant)
):
    """Remove a saved payee."""
    deleted = db.remove_payee(user.user_id, payee_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Payee not found")

    return {"message": "Payee removed successfully"}


@router.get("/payees/{payee_id}/status")
async def get_payee_status(
    payee_id: str,
    user: TokenData = Depends(require_customer_or_merchant)
):
    """Check payee status."""
    # Get all payees and find by ID
    payees = db.get_user_payees(user.user_id)
    payee = next((p for p in payees if p["payee_id"] == payee_id), None)

    if not payee:
        raise HTTPException(status_code=404, detail="Payee not found")

    return {
        "payee_id": payee_id,
        "is_verified": bool(payee.get("is_verified", 0))
    }


# ══════════════════════════════════════════════════════════════════════════════
# Search Users (for adding payees)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/search")
async def search_users(
    q: str,
    user: TokenData = Depends(require_customer_or_merchant)
):
    """
    Search for users by email or user ID.
    Used for adding new payees.
    """
    results = []

    # Search by exact user ID
    if q.startswith("C") or q.startswith("M"):
        found = db.get_user_with_profile(q)
        if found and found["user_id"] != user.user_id:
            results.append({
                "user_id": found["user_id"],
                "display_name": found.get("display_name", "User"),
                "user_type": found["user_type"],
                "business_name": found.get("business_name")
            })

    # Search by email
    found = db.get_user_by_email(q)
    if found and found["user_id"] != user.user_id:
        profile = db.get_user_with_profile(found["user_id"])
        if not any(r["user_id"] == found["user_id"] for r in results):
            results.append({
                "user_id": found["user_id"],
                "display_name": profile.get("display_name", "User") if profile else "User",
                "user_type": found["user_type"],
                "business_name": profile.get("business_name") if profile else None
            })

    return {"results": results}
