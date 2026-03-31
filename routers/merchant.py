"""
routers/merchant.py — Merchant-specific endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone, timedelta

import database as db
from auth.dependencies import require_merchant
from auth.models import TokenData

router = APIRouter(prefix="/merchant", tags=["Merchant"])


class PaymentAnalytics(BaseModel):
    period: str
    total_received: float
    transaction_count: int
    average_transaction: float
    top_payers: List[dict]


# ══════════════════════════════════════════════════════════════════════════════
# Payment Reception
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/payments")
async def get_received_payments(
    limit: int = 50,
    user: TokenData = Depends(require_merchant)
):
    """Get payments received by this merchant."""
    transactions = db.get_user_transactions(
        user_id=user.user_id,
        as_sender=False,
        as_receiver=True,
        limit=limit
    )

    # Filter only completed payments
    payments = [
        t for t in transactions
        if t["status"] == "COMPLETED"
    ]

    # Calculate totals
    total_received = sum(p["amount"] for p in payments)

    return {
        "payments": payments,
        "count": len(payments),
        "total_received": total_received
    }


# ══════════════════════════════════════════════════════════════════════════════
# Analytics
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/analytics")
async def get_payment_analytics(
    period: str = "daily",
    user: TokenData = Depends(require_merchant)
):
    """
    Get payment analytics for the merchant.

    Periods: daily, weekly, monthly
    """
    transactions = db.get_user_transactions(
        user_id=user.user_id,
        as_sender=False,
        as_receiver=True,
        limit=500
    )

    # Filter completed transactions
    completed = [t for t in transactions if t["status"] == "COMPLETED"]

    # Time filter
    now = datetime.now(timezone.utc)
    if period == "daily":
        cutoff = now - timedelta(days=1)
    elif period == "weekly":
        cutoff = now - timedelta(weeks=1)
    else:
        cutoff = now - timedelta(days=30)

    # Filter by period
    in_period = []
    for t in completed:
        txn_time = datetime.fromisoformat(t["initiated_at"].replace("Z", "+00:00"))
        if txn_time >= cutoff:
            in_period.append(t)

    # Calculate stats
    total_received = sum(t["amount"] for t in in_period)
    count = len(in_period)
    avg = total_received / count if count > 0 else 0

    # Top payers
    payer_totals = {}
    for t in in_period:
        sender = t["sender_id"]
        payer_totals[sender] = payer_totals.get(sender, 0) + t["amount"]

    top_payers = sorted(
        [{"user_id": k, "total": v} for k, v in payer_totals.items()],
        key=lambda x: x["total"],
        reverse=True
    )[:5]

    return PaymentAnalytics(
        period=period,
        total_received=total_received,
        transaction_count=count,
        average_transaction=round(avg, 2),
        top_payers=top_payers
    )


@router.get("/analytics/hourly")
async def get_hourly_analytics(
    user: TokenData = Depends(require_merchant)
):
    """Get hourly transaction distribution for today."""
    transactions = db.get_user_transactions(
        user_id=user.user_id,
        as_sender=False,
        as_receiver=True,
        limit=200
    )

    # Get today's transactions
    today = datetime.now(timezone.utc).date()
    hourly = {i: {"count": 0, "amount": 0} for i in range(24)}

    for t in transactions:
        if t["status"] != "COMPLETED":
            continue
        try:
            txn_time = datetime.fromisoformat(t["initiated_at"].replace("Z", "+00:00"))
            if txn_time.date() == today:
                hour = txn_time.hour
                hourly[hour]["count"] += 1
                hourly[hour]["amount"] += t["amount"]
        except (ValueError, KeyError):
            continue

    return {
        "date": today.isoformat(),
        "hourly_data": hourly
    }


# ══════════════════════════════════════════════════════════════════════════════
# Settlement Reports
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/settlements")
async def get_settlement_reports(
    user: TokenData = Depends(require_merchant)
):
    """Get settlement/payout information."""
    # Get account balance
    account = db.get_primary_account(user.user_id)
    balance = account["balance"] if account else 0

    # Get recent successful transactions
    transactions = db.get_user_transactions(
        user_id=user.user_id,
        as_sender=False,
        as_receiver=True,
        limit=100
    )

    completed = [t for t in transactions if t["status"] == "COMPLETED"]

    # Calculate daily totals for last 7 days
    now = datetime.now(timezone.utc)
    daily_totals = {}
    for i in range(7):
        day = (now - timedelta(days=i)).date()
        daily_totals[day.isoformat()] = {"amount": 0, "count": 0}

    for t in completed:
        try:
            txn_date = datetime.fromisoformat(t["initiated_at"].replace("Z", "+00:00")).date()
            if txn_date.isoformat() in daily_totals:
                daily_totals[txn_date.isoformat()]["amount"] += t["amount"]
                daily_totals[txn_date.isoformat()]["count"] += 1
        except (ValueError, KeyError):
            continue

    return {
        "current_balance": balance,
        "currency": "INR",
        "daily_settlements": daily_totals,
        "total_transactions": len(completed),
        "total_received": sum(t["amount"] for t in completed)
    }
