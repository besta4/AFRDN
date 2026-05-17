"""
routers/support.py — Support Chat endpoints.

Blocked/Suspended users have VERY limited API access — they can ONLY use
these support endpoints to reach an admin for account resolution.

Design:
  - Any authenticated user (including BLOCKED/SUSPENDED) can access /support/*
  - Admin can view all tickets and reply to any
  - Users can only see their own tickets
  - Once a ticket is resolved, admin can optionally reactivate the user account
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

import database as db
from auth.models import TokenData
from auth.jwt_handler import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter(prefix="/support", tags=["Support"])

# ── Custom auth dependency that ALLOWS blocked/suspended users ────────────────
# Standard get_current_user blocks BLOCKED/SUSPENDED users (HTTP 403).
# For support endpoints we must allow them in — it's the ONLY thing they can do.

_security = HTTPBearer(auto_error=False)


async def get_user_including_blocked(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> TokenData:
    """
    Auth dependency that allows BLOCKED and SUSPENDED users.
    Used exclusively for support chat endpoints.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify user still exists (but don't reject on account_status)
    user = db.get_user_by_id(token_data.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return token_data


async def require_admin_for_support(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> TokenData:
    """Admin-only dependency for support management endpoints."""
    token_data = await get_user_including_blocked(credentials)
    if token_data.user_type != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return token_data


# ── Request / Response Models ─────────────────────────────────────────────────

class CreateTicketRequest(BaseModel):
    subject: str
    message: str


class SendMessageRequest(BaseModel):
    message: str


class ResolveTicketRequest(BaseModel):
    resolution_note: Optional[str] = None
    reactivate_user: bool = False  # Admin can optionally reactivate the user


# ══════════════════════════════════════════════════════════════════════════════
# USER ENDPOINTS — accessible even when BLOCKED/SUSPENDED
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/tickets", status_code=status.HTTP_201_CREATED)
async def create_ticket(
    body: CreateTicketRequest,
    user: TokenData = Depends(get_user_including_blocked),
):
    """
    Create a new support ticket.

    Available to ALL users including BLOCKED and SUSPENDED accounts.
    This is the primary escalation path for users whose accounts have been
    frozen by fraud detection.
    """
    if not body.subject.strip():
        raise HTTPException(status_code=400, detail="Subject cannot be empty")
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Check how many open tickets the user already has (prevent spam)
    existing = db.get_user_support_tickets(user.user_id)
    open_count = sum(1 for t in existing if t["status"] in ("OPEN", "IN_PROGRESS"))
    if open_count >= 3:
        raise HTTPException(
            status_code=429,
            detail="You already have 3 open tickets. Please wait for a response before creating more.",
        )

    ticket_id = db.create_support_ticket(
        user_id=user.user_id,
        subject=body.subject.strip(),
        first_message=body.message.strip(),
    )

    # Get the user's account status to show in response
    db_user = db.get_user_by_id(user.user_id)
    account_status = db_user["account_status"] if db_user else "UNKNOWN"

    return {
        "ticket_id": ticket_id,
        "status": "OPEN",
        "message": "Your support ticket has been created. An admin will respond shortly.",
        "account_status": account_status,
    }


@router.get("/tickets")
async def list_my_tickets(
    user: TokenData = Depends(get_user_including_blocked),
):
    """Get all support tickets for the current user."""
    tickets = db.get_user_support_tickets(user.user_id)
    return {"tickets": tickets, "count": len(tickets)}


@router.get("/tickets/{ticket_id}")
async def get_ticket(
    ticket_id: str,
    user: TokenData = Depends(get_user_including_blocked),
):
    """Get a ticket and its full message thread."""
    ticket = db.get_support_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Users can only see their own tickets; admins see all
    if user.user_type != "ADMIN" and ticket["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    messages = db.get_ticket_messages(ticket_id)
    return {"ticket": ticket, "messages": messages}


@router.post("/tickets/{ticket_id}/messages", status_code=status.HTTP_201_CREATED)
async def send_message(
    ticket_id: str,
    body: SendMessageRequest,
    user: TokenData = Depends(get_user_including_blocked),
):
    """Send a message on an existing ticket."""
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    ticket = db.get_support_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Users can only message their own tickets
    if user.user_type != "ADMIN" and ticket["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Cannot message on closed/resolved tickets
    if ticket["status"] in ("RESOLVED", "CLOSED"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot send message on a {ticket['status']} ticket",
        )

    role = "ADMIN" if user.user_type == "ADMIN" else "USER"
    message_id = db.add_ticket_message(
        ticket_id=ticket_id,
        sender_id=user.user_id,
        sender_role=role,
        message=body.message.strip(),
    )

    return {"message_id": message_id, "status": "sent"}


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS — manage all support tickets
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/admin/tickets")
async def admin_list_tickets(
    ticket_status: Optional[str] = None,
    limit: int = 100,
    user: TokenData = Depends(require_admin_for_support),
):
    """Get all support tickets (admin only)."""
    tickets = db.get_all_support_tickets(status=ticket_status, limit=limit)
    return {"tickets": tickets, "count": len(tickets)}


@router.post("/admin/tickets/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: str,
    body: ResolveTicketRequest,
    user: TokenData = Depends(require_admin_for_support),
):
    """
    Resolve a support ticket.

    Optionally reactivate the user's account at the same time.
    This is the primary way admins unblock a user after reviewing their case.
    """
    ticket = db.get_support_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if ticket["status"] in ("RESOLVED", "CLOSED"):
        raise HTTPException(
            status_code=400,
            detail=f"Ticket is already {ticket['status']}",
        )

    # Add resolution note as a final admin message if provided
    if body.resolution_note:
        db.add_ticket_message(
            ticket_id=ticket_id,
            sender_id=user.user_id,
            sender_role="ADMIN",
            message=f"[RESOLVED] {body.resolution_note}",
        )

    db.update_ticket_status(ticket_id, "RESOLVED")

    # Optionally reactivate the user's account
    reactivated = False
    if body.reactivate_user:
        target_user = db.get_user_by_id(ticket["user_id"])
        if target_user and target_user["account_status"] in ("SUSPENDED", "BLOCKED"):
            db.update_user(ticket["user_id"], account_status="ACTIVE")
            reactivated = True

    return {
        "message": "Ticket resolved",
        "ticket_id": ticket_id,
        "user_reactivated": reactivated,
        "user_id": ticket["user_id"],
    }


@router.post("/admin/tickets/{ticket_id}/close")
async def close_ticket(
    ticket_id: str,
    user: TokenData = Depends(require_admin_for_support),
):
    """Close a ticket without reactivating the user."""
    ticket = db.get_support_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    db.update_ticket_status(ticket_id, "CLOSED")
    return {"message": "Ticket closed", "ticket_id": ticket_id}
