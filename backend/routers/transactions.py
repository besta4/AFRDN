"""
routers/transactions.py — Transaction endpoints.

Handles transaction creation with real-time fraud detection pipeline.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone
import time
import asyncio
from threading import Lock
import uuid

import database as db
from database import TransactionType, TransactionStatus
from auth.dependencies import get_current_user, require_customer_or_merchant, get_client_info
from auth.models import TokenData

router = APIRouter(prefix="/transactions", tags=["Transactions"])

# ── Singleton Orchestrator ────────────────────────────────────────────────────
# The Orchestrator MUST persist across requests so that Agent 2's rolling
# buffer, per-user device/IP history, and all agent state accumulates across
# transactions. Creating a new Orchestrator per request (the old bug) meant
# the buffer was always empty, making MULE_NETWORK / VELOCITY_SPIKE detection
# structurally impossible.
_ORCHESTRATOR_LOCK = Lock()
_ORCHESTRATOR_INSTANCE = None


def _get_orchestrator():
    """Return the singleton Orchestrator, creating it on first call."""
    global _ORCHESTRATOR_INSTANCE
    if _ORCHESTRATOR_INSTANCE is None:
        with _ORCHESTRATOR_LOCK:
            if _ORCHESTRATOR_INSTANCE is None:
                from agents.orchestrator import Orchestrator
                _ORCHESTRATOR_INSTANCE = Orchestrator()
    return _ORCHESTRATOR_INSTANCE


_FLOW_STATE_LOCK = Lock()
_FLOW_STATE_TTL_SECONDS = 180
_FLOW_STATES: dict[str, dict] = {}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cleanup_flow_states() -> None:
    now = _utc_now()
    expired: list[str] = []
    with _FLOW_STATE_LOCK:
        for flow_id, payload in _FLOW_STATES.items():
            updated_at_raw = payload.get("updated_at")
            try:
                updated_at = datetime.fromisoformat(str(updated_at_raw))
            except Exception:
                updated_at = now
            if (now - updated_at).total_seconds() > _FLOW_STATE_TTL_SECONDS:
                expired.append(flow_id)
        for flow_id in expired:
            _FLOW_STATES.pop(flow_id, None)


def _init_flow_state(flow_id: str, transaction_id: str, sender_id: str, receiver_id: str, amount: float) -> None:
    now_iso = _utc_now().isoformat()
    steps = [
        {"agent_key": "agent1", "agent_name": "Transaction Monitoring", "status": "pending", "latency_ms": None, "summary": None, "error": None},
        {"agent_key": "agent2", "agent_name": "Pattern Detection", "status": "pending", "latency_ms": None, "summary": None, "error": None},
        {"agent_key": "agent3", "agent_name": "Risk Assessment", "status": "pending", "latency_ms": None, "summary": None, "error": None},
        {"agent_key": "agent4", "agent_name": "Alert & Block", "status": "pending", "latency_ms": None, "summary": None, "error": None},
        {"agent_key": "agent5", "agent_name": "Compliance Logging", "status": "pending", "latency_ms": None, "summary": None, "error": None},
    ]
    with _FLOW_STATE_LOCK:
        _FLOW_STATES[flow_id] = {
            "flow_id": flow_id,
            "transaction_id": transaction_id,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "amount": amount,
            "status": "running",
            "steps": steps,
            "started_at": now_iso,
            "updated_at": now_iso,
            "ended_at": None,
            "final_result": None,
        }


def _update_flow_step(
    flow_id: str,
    agent_key: str,
    *,
    status: Optional[str] = None,
    latency_ms: Optional[float] = None,
    summary: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    now_iso = _utc_now().isoformat()
    with _FLOW_STATE_LOCK:
        flow = _FLOW_STATES.get(flow_id)
        if not flow:
            return
        for step in flow.get("steps", []):
            if step.get("agent_key") == agent_key:
                if status is not None:
                    step["status"] = status
                if latency_ms is not None:
                    step["latency_ms"] = round(float(latency_ms), 2)
                if summary is not None:
                    step["summary"] = summary
                if error is not None:
                    step["error"] = error
                break
        flow["updated_at"] = now_iso


def _finalize_flow(flow_id: str, *, status: str, final_result: Optional[dict] = None) -> None:
    now_iso = _utc_now().isoformat()
    with _FLOW_STATE_LOCK:
        flow = _FLOW_STATES.get(flow_id)
        if not flow:
            return
        flow["status"] = status
        flow["final_result"] = final_result
        flow["ended_at"] = now_iso
        flow["updated_at"] = now_iso


class CreateTransactionRequest(BaseModel):
    receiver_id: str = Field(..., description="Receiver user ID (e.g., M000000001) or email")
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    type: str = Field(default="TRANSFER", description="PAYMENT or TRANSFER")
    description: Optional[str] = Field(None, max_length=200)
    reference_id: Optional[str] = None


class TransactionResponse(BaseModel):
    transaction_id: str
    status: str
    amount: float
    type: str
    sender_id: str
    receiver_id: str
    sender_new_balance: float
    initiated_at: str
    completed_at: Optional[str] = None
    fraud_check: Optional[dict] = None


class TransactionListItem(BaseModel):
    transaction_id: str
    type: str
    amount: float
    sender_id: str
    receiver_id: str
    status: str
    initiated_at: str
    direction: str  # "sent" or "received"
    fraud_score: Optional[float] = None
    risk_level: Optional[str] = None
    explanation: Optional[str] = None


class VerifyOTPRequest(BaseModel):
    otp: str = Field(..., min_length=6, max_length=6, description="6-digit OTP")


# ══════════════════════════════════════════════════════════════════════════════
# Transaction Creation (Main Endpoint)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/", response_model=TransactionResponse)
async def create_transaction(
    request: Request,
    body: CreateTransactionRequest,
    user: TokenData = Depends(require_customer_or_merchant)
):
    """
    Create a new transaction with real-time fraud detection.

    Flow:
    1. Check account status (suspended/blocked users cannot transact)
    2. Validate sender balance
    3. Check if email OTP is required
    4. Pre-authorize (deduct from sender)
    5. Run 5-agent fraud pipeline
    6. Based on action: COMPLETE, HOLD, or BLOCK
    7. If BLOCKED: suspend sender's account pending admin review
    """
    start_time = time.time()

    _cleanup_flow_states()

    # Get sender's primary account
    sender_account = db.get_primary_account(user.user_id)
    if not sender_account:
        raise HTTPException(status_code=400, detail="No account found")

    # ── Check account status from THE USERS TABLE (accounts table has no status field) ──
    # Suspended or blocked users cannot initiate transactions.
    sender_user = db.get_user_by_id(user.user_id)
    if not sender_user:
        raise HTTPException(status_code=400, detail="User not found")
    user_account_status = sender_user.get("account_status", "ACTIVE")
    if user_account_status == "SUSPENDED":
        raise HTTPException(
            status_code=403,
            detail="Account suspended due to suspicious activity. Contact support to resolve."
        )
    if user_account_status == "BLOCKED":
        raise HTTPException(
            status_code=403,
            detail="Account blocked. Contact support to resolve."
        )

    # Resolve receiver by user ID or email
    resolved_receiver_id = (body.receiver_id or "").strip()
    if not resolved_receiver_id:
        raise HTTPException(status_code=400, detail="Receiver is required")

    if "@" in resolved_receiver_id:
        resolved_receiver = db.get_user_by_email(resolved_receiver_id)
        if not resolved_receiver:
            raise HTTPException(status_code=404, detail="Receiver not found")
        resolved_receiver_id = resolved_receiver["user_id"]
    else:
        resolved_receiver_id = resolved_receiver_id.upper()

    # Check receiver exists
    receiver = db.get_user_by_id(resolved_receiver_id)
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")

    # Get receiver's primary account
    receiver_account = db.get_primary_account(resolved_receiver_id)
    if not receiver_account:
        raise HTTPException(status_code=400, detail="Receiver has no account")

    # Validate transaction type
    txn_type = TransactionType.TRANSFER
    if body.type == "PAYMENT":
        if not resolved_receiver_id.startswith("M"):
            raise HTTPException(
                status_code=400,
                detail="PAYMENT type requires merchant receiver"
            )
        txn_type = TransactionType.PAYMENT

    # ── Step 1: Check balance ──
    if sender_account["balance"] < body.amount:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    # ── Step 2: Check if email OTP is enabled ──
    if db.is_email_otp_enabled(user.user_id):
        # Generate OTP and store pending transaction
        otp_code = db.generate_otp()
        transaction_data = {
            "receiver_id": resolved_receiver_id,
            "amount": body.amount,
            "type": body.type,
            "description": body.description,
            "reference_id": body.reference_id
        }
        otp_id = db.create_pending_otp(user.user_id, otp_code, transaction_data)
        
        # Get user email for OTP
        sender = db.get_user_by_id(user.user_id)
        if not sender:
            raise HTTPException(status_code=400, detail="User not found")
        
        # TODO: Send actual email - for now just log
        email = sender['email']
        print(f"[OTP] User {user.user_id} ({email}): OTP is {otp_code}")
        
        raise HTTPException(
            status_code=202,
            detail={
                "message": "OTP sent to your email. Use /transactions/verify-otp to complete.",
                "otp_required": True,
                "email_hint": email[:3] + "***" + email[email.find('@'):]
            }
        )

    # ── Step 3: Auto-add payee if new ──
    if not resolved_receiver_id.startswith("M"):
        payee = db.get_payee(user.user_id, resolved_receiver_id)
        if not payee:
            db.add_payee(user.user_id, resolved_receiver_id)

    # ── Step 4: Pre-auth hold (deduct from sender) ──
    success, old_balance, new_balance = db.deduct_balance(
        sender_account["account_id"],
        body.amount
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to deduct balance")

    # Get client info
    client_info = get_client_info(request)

    # ── Step 5: Create transaction record ──
    transaction_id = db.create_transaction(
        sender_id=user.user_id,
        sender_account=sender_account["account_id"],
        receiver_id=resolved_receiver_id,
        receiver_account=receiver_account["account_id"],
        amount=body.amount,
        txn_type=txn_type,
        ip_address=client_info["ip_address"],
        device_id=client_info["device_id"],
        old_balance_sender=old_balance,
        new_balance_sender=new_balance,
        old_balance_receiver=receiver_account["balance"],
        description=body.description,
        reference_id=body.reference_id
    )

    flow_id = request.headers.get("X-Flow-ID") or uuid.uuid4().hex
    _init_flow_state(
        flow_id=flow_id,
        transaction_id=transaction_id,
        sender_id=user.user_id,
        receiver_id=resolved_receiver_id,
        amount=body.amount,
    )

    # ── Step 6: Update status to PENDING_FRAUD ──
    db.update_transaction_status(transaction_id, TransactionStatus.PENDING_FRAUD)

    # ── Step 7: Run fraud pipeline ──
    fraud_result = await run_fraud_pipeline(
        transaction_id=transaction_id,
        sender_id=user.user_id,
        receiver_id=resolved_receiver_id,
        amount=body.amount,
        txn_type=txn_type.value,
        old_balance=old_balance,
        new_balance=new_balance,
        ip_address=client_info["ip_address"],
        device_id=client_info["device_id"],
        step=datetime.now(timezone.utc).hour + (datetime.now(timezone.utc).day * 24),
        flow_id=flow_id,
    )

    pipeline_latency = (time.time() - start_time) * 1000

    # ── Step 8: Store fraud results ──
    db.update_transaction_fraud_results(
        transaction_id=transaction_id,
        fraud_score=fraud_result["fraud_score"],
        fraud_label=fraud_result["fraud_label"],
        pattern_type=fraud_result.get("pattern_type"),
        pattern_confidence=fraud_result.get("pattern_confidence"),
        risk_level=fraud_result["risk_level"],
        recommended_action=fraud_result["recommended_action"],
        action_taken=fraud_result["action_taken"],
        explanation=fraud_result.get("explanation"),
        pipeline_latency_ms=pipeline_latency
    )

    # ── Step 9: Execute action ──
    final_status = TransactionStatus.COMPLETED
    new_receiver_balance = receiver_account["balance"]

    if fraud_result["action_taken"] == "BLOCK":
        # Reverse pre-auth
        db.credit_balance(sender_account["account_id"], body.amount)
        final_status = TransactionStatus.BLOCKED

        import logging as _log
        _txn_logger = _log.getLogger(__name__)

        # ── Auto-suspend the CURRENT sender ──
        _txn_logger.warning(
            "[AUTO-SUSPEND] Suspending sender %s — transaction %s BLOCKED (score=%.2f, pattern=%s)",
            user.user_id, transaction_id, fraud_result["fraud_score"],
            fraud_result.get("pattern_type", "NONE")
        )
        db.update_user(user.user_id, account_status="SUSPENDED")

        # ── MULE_NETWORK: suspend the collector AND all prior senders to this collector ──
        # This ensures the ENTIRE mule ring is shut down, not just the current transaction.
        if fraud_result.get("pattern_type") == "MULE_NETWORK":
            _txn_logger.warning(
                "[AUTO-SUSPEND] MULE_NETWORK detected — suspending collector %s and all network participants",
                resolved_receiver_id
            )
            # Suspend the mule collector
            db.update_user(resolved_receiver_id, account_status="SUSPENDED")

            # Query DB for ALL senders who previously transferred to this collector
            mule_senders = db.get_mule_network_senders(resolved_receiver_id)
            for mule_sender_id in mule_senders:
                if mule_sender_id != user.user_id:  # already handled above
                    mule_sender = db.get_user_by_id(mule_sender_id)
                    if mule_sender and mule_sender["account_status"] not in ("SUSPENDED", "BLOCKED"):
                        _txn_logger.warning(
                            "[AUTO-SUSPEND] Suspending mule network participant %s — sent to collector %s",
                            mule_sender_id, resolved_receiver_id
                        )
                        db.update_user(mule_sender_id, account_status="SUSPENDED")

        # ── VELOCITY_SPIKE: suspend the sender (rapid fraudulent transactions) ──
        elif fraud_result.get("pattern_type") == "VELOCITY_SPIKE":
            _txn_logger.warning(
                "[AUTO-SUSPEND] VELOCITY_SPIKE confirmed — sender %s already suspended",
                user.user_id
            )
            # Sender already suspended above; no additional action needed

    elif fraud_result["action_taken"] == "HOLD":
        final_status = TransactionStatus.HELD
    else:  # PASS or SILENT_FLAG
        # Credit receiver
        _, new_receiver_balance = db.credit_balance(
            receiver_account["account_id"],
            body.amount
        )
        final_status = TransactionStatus.COMPLETED

        # Update velocity
        db.update_velocity(user.user_id, body.amount)

        # Update payee stats
        if not resolved_receiver_id.startswith("M"):
            db.update_payee_stats(user.user_id, resolved_receiver_id, body.amount)

    # Update final status
    if final_status == TransactionStatus.COMPLETED:
        db.complete_transaction(transaction_id, new_receiver_balance)
    else:
        db.update_transaction_status(transaction_id, final_status)

    # ── Step 10: Check for compliance reports ──
    should_str, str_reason = db.should_generate_str(
        fraud_result["fraud_score"],
        fraud_result["action_taken"],
        user.user_id
    )
    if should_str:
        db.create_compliance_report(
            report_type="STR",
            trigger_reason=str_reason,
            transaction_id=transaction_id,
            user_id=user.user_id,
            amount=body.amount
        )

    if db.should_generate_ctr(txn_type.value, body.amount):
        db.create_compliance_report(
            report_type="CTR",
            trigger_reason="Cash transaction >= 10 lakh",
            transaction_id=transaction_id,
            user_id=user.user_id,
            amount=body.amount
        )

    response_payload = TransactionResponse(
        transaction_id=transaction_id,
        status=final_status.value,
        amount=body.amount,
        type=txn_type.value,
        sender_id=user.user_id,
        receiver_id=resolved_receiver_id,
        sender_new_balance=new_balance if final_status != TransactionStatus.BLOCKED else old_balance,
        initiated_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat() if final_status == TransactionStatus.COMPLETED else None,
        fraud_check={
            "fraud_score": fraud_result["fraud_score"],
            "risk_level": fraud_result["risk_level"],
            "action_taken": fraud_result["action_taken"],
            "pattern_type": fraud_result.get("pattern_type", "NONE"),
            "pattern_confidence": fraud_result.get("pattern_confidence", 0.0),
            "pipeline_latency_ms": round(pipeline_latency, 1),
            "explanation": fraud_result.get("explanation") if final_status != TransactionStatus.COMPLETED else None,
            "pattern_reasoning": fraud_result.get("pattern_reasoning"),
            "flow_id": flow_id,
        }
    )
    _finalize_flow(
        flow_id,
        status="completed",
        final_result={
            "status": final_status.value,
            "action_taken": fraud_result["action_taken"],
            "risk_level": fraud_result["risk_level"],
            "fraud_score": fraud_result["fraud_score"],
        },
    )
    return response_payload


# ══════════════════════════════════════════════════════════════════════════════
# OTP Verification Endpoint
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/verify-otp", response_model=TransactionResponse)
async def verify_otp_and_complete(
    request: Request,
    body: VerifyOTPRequest,
    user: TokenData = Depends(require_customer_or_merchant)
):
    """
    Verify OTP and complete the pending transaction.
    """
    start_time = time.time()
    
    _cleanup_flow_states()

    # Verify OTP and get pending transaction
    pending = db.verify_and_get_pending_otp(user.user_id, body.otp)
    if not pending:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    txn_data = pending["transaction_data"]
    
    # Get sender's primary account
    sender_account = db.get_primary_account(user.user_id)
    if not sender_account:
        raise HTTPException(status_code=400, detail="No account found")
    
    # Get receiver's primary account
    receiver_account = db.get_primary_account(txn_data["receiver_id"])
    if not receiver_account:
        raise HTTPException(status_code=400, detail="Receiver has no account")
    
    # Check balance again
    if sender_account["balance"] < txn_data["amount"]:
        raise HTTPException(status_code=400, detail="Insufficient balance")
    
    txn_type = TransactionType.PAYMENT if txn_data.get("type") == "PAYMENT" else TransactionType.TRANSFER
    
    # Auto-add payee if new
    if not txn_data["receiver_id"].startswith("M"):
        payee = db.get_payee(user.user_id, txn_data["receiver_id"])
        if not payee:
            db.add_payee(user.user_id, txn_data["receiver_id"])
    
    # Deduct balance
    success, old_balance, new_balance = db.deduct_balance(
        sender_account["account_id"],
        txn_data["amount"]
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to deduct balance")
    
    client_info = get_client_info(request)
    
    # Create transaction record
    transaction_id = db.create_transaction(
        sender_id=user.user_id,
        sender_account=sender_account["account_id"],
        receiver_id=txn_data["receiver_id"],
        receiver_account=receiver_account["account_id"],
        amount=txn_data["amount"],
        txn_type=txn_type,
        ip_address=client_info["ip_address"],
        device_id=client_info["device_id"],
        old_balance_sender=old_balance,
        new_balance_sender=new_balance,
        old_balance_receiver=receiver_account["balance"],
        description=txn_data.get("description"),
        reference_id=txn_data.get("reference_id")
    )

    flow_id = request.headers.get("X-Flow-ID") or uuid.uuid4().hex
    _init_flow_state(
        flow_id=flow_id,
        transaction_id=transaction_id,
        sender_id=user.user_id,
        receiver_id=txn_data["receiver_id"],
        amount=txn_data["amount"],
    )
    
    # Update status to PENDING_FRAUD
    db.update_transaction_status(transaction_id, TransactionStatus.PENDING_FRAUD)
    
    # Run fraud pipeline
    step = datetime.now(timezone.utc).hour + (datetime.now(timezone.utc).day * 24)
    fraud_result = await run_fraud_pipeline(
        transaction_id=transaction_id,
        sender_id=user.user_id,
        receiver_id=txn_data["receiver_id"],
        amount=txn_data["amount"],
        txn_type=txn_type.value,
        old_balance=old_balance,
        new_balance=new_balance,
        ip_address=client_info["ip_address"],
        device_id=client_info["device_id"],
        step=step,
        flow_id=flow_id,
    )
    
    pipeline_latency = (time.time() - start_time) * 1000
    
    # Store fraud results
    db.update_transaction_fraud_results(
        transaction_id=transaction_id,
        fraud_score=fraud_result["fraud_score"],
        fraud_label=fraud_result["fraud_label"],
        pattern_type=fraud_result.get("pattern_type"),
        pattern_confidence=fraud_result.get("pattern_confidence"),
        risk_level=fraud_result["risk_level"],
        recommended_action=fraud_result["recommended_action"],
        action_taken=fraud_result["action_taken"],
        explanation=fraud_result.get("explanation"),
        pipeline_latency_ms=pipeline_latency
    )
    
    # Execute action based on fraud result
    final_status = TransactionStatus(fraud_result["action_taken"])
    
    if final_status == TransactionStatus.COMPLETED:
        db.credit_balance(receiver_account["account_id"], txn_data["amount"])
        db.update_transaction_status(
            transaction_id, final_status,
            new_balance_receiver=receiver_account["balance"] + txn_data["amount"]
        )
    elif final_status == TransactionStatus.BLOCKED:
        db.credit_balance(sender_account["account_id"], txn_data["amount"])  # Refund
        db.update_transaction_status(transaction_id, final_status)

        # ── Auto-suspend sender on BLOCK (OTP path) ──
        import logging as _log
        _otp_logger = _log.getLogger(__name__)
        _otp_logger.warning(
            "[AUTO-SUSPEND] Suspending sender %s — transaction %s BLOCKED (OTP path)",
            user.user_id, transaction_id
        )
        db.update_user(user.user_id, account_status="SUSPENDED")

        if fraud_result.get("pattern_type") == "MULE_NETWORK":
            _otp_logger.warning(
                "[AUTO-SUSPEND] MULE_NETWORK (OTP path) — suspending collector %s and all participants",
                txn_data["receiver_id"]
            )
            db.update_user(txn_data["receiver_id"], account_status="SUSPENDED")

            # Suspend ALL prior senders to this collector from DB
            mule_senders = db.get_mule_network_senders(txn_data["receiver_id"])
            for mule_sender_id in mule_senders:
                if mule_sender_id != user.user_id:
                    mule_sender = db.get_user_by_id(mule_sender_id)
                    if mule_sender and mule_sender["account_status"] not in ("SUSPENDED", "BLOCKED"):
                        _otp_logger.warning(
                            "[AUTO-SUSPEND] Suspending mule participant %s (OTP path)",
                            mule_sender_id
                        )
                        db.update_user(mule_sender_id, account_status="SUSPENDED")
    else:  # HELD
        db.update_transaction_status(transaction_id, final_status)
    
    response_payload = TransactionResponse(
        transaction_id=transaction_id,
        status=final_status.value,
        amount=txn_data["amount"],
        type=txn_type.value,
        sender_id=user.user_id,
        receiver_id=txn_data["receiver_id"],
        sender_new_balance=new_balance if final_status != TransactionStatus.BLOCKED else old_balance,
        initiated_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat() if final_status == TransactionStatus.COMPLETED else None,
        fraud_check={
            "fraud_score": fraud_result["fraud_score"],
            "risk_level": fraud_result["risk_level"],
            "action_taken": fraud_result["action_taken"],
            "pattern_type": fraud_result.get("pattern_type", "NONE"),
            "pattern_confidence": fraud_result.get("pattern_confidence", 0.0),
            "pipeline_latency_ms": round(pipeline_latency, 1),
            "explanation": fraud_result.get("explanation") if final_status != TransactionStatus.COMPLETED else None,
            "pattern_reasoning": fraud_result.get("pattern_reasoning"),
            "flow_id": flow_id,
        }
    )
    _finalize_flow(
        flow_id,
        status="completed",
        final_result={
            "status": final_status.value,
            "action_taken": fraud_result["action_taken"],
            "risk_level": fraud_result["risk_level"],
            "fraud_score": fraud_result["fraud_score"],
        },
    )
    return response_payload


def _run_fraud_pipeline_sync(
    transaction_id: str,
    sender_id: str,
    receiver_id: str,
    amount: float,
    txn_type: str,
    old_balance: float,
    new_balance: float,
    ip_address: str,
    device_id: str,
    step: int,
    flow_id: Optional[str] = None,
) -> dict:
    """
    Run the 5-agent fraud detection pipeline.

    This integrates with the existing Orchestrator.
    For new demo users without GNN embeddings, falls back to rule-based.
    """
    try:
        # Import models (Orchestrator is accessed via module-level singleton)
        from agents.models import TransactionMessage, TransactionType as TxnType

        # Create TransactionMessage for pipeline
        msg = TransactionMessage(
            transaction_id=transaction_id,
            step=step,
            type=TxnType(txn_type),
            amount=amount,
            nameOrig=sender_id,
            nameDest=receiver_id,
            oldbalanceOrg=old_balance,
            newbalanceOrig=new_balance,
            oldbalanceDest=0.0,  # Will be filled
            newbalanceDest=amount,
            ip_address=ip_address,
            device_id=device_id,
        )

        # Use singleton orchestrator so rolling buffer persists across requests
        orchestrator = _get_orchestrator()

        def _flow_step_pause() -> None:
            # Small pause so users can perceive each stage in the live tracker.
            if flow_id:
                time.sleep(0.12)

        if flow_id:
            _update_flow_step(flow_id, "agent1", status="running")
        result = orchestrator.agent1.process(msg)
        if flow_id:
            a1 = result.pipeline_metadata[-1] if result.pipeline_metadata else None
            _update_flow_step(
                flow_id,
                "agent1",
                status="completed",
                latency_ms=a1.latency_ms if a1 else None,
                summary=f"Fraud score {((result.fraud_score or 0.0) * 100):.2f}%",
                error=a1.error if a1 else None,
            )
            _flow_step_pause()

        # Record transaction in Redis dynamic graph cache (between Agent 1 & 2)
        graph_cache = getattr(orchestrator, 'graph_cache', None)
        if graph_cache and graph_cache.available:
            try:
                graph_cache.record_transaction(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    amount=amount,
                    transaction_id=transaction_id,
                )
            except Exception:
                pass  # graceful degradation

        # Add ALL transactions to rolling buffer for pattern detection
        orchestrator.rolling_buffer.append(result)

        if flow_id:
            _update_flow_step(flow_id, "agent2", status="running")
        result = orchestrator.agent2.process(result)
        if flow_id:
            a2 = result.pipeline_metadata[-1] if result.pipeline_metadata else None
            pattern = result.pattern_type.value if result.pattern_type else "NONE"
            conf = result.pattern_confidence or 0.0
            _update_flow_step(
                flow_id,
                "agent2",
                status="completed",
                latency_ms=a2.latency_ms if a2 else None,
                summary=f"Pattern {pattern} ({conf:.2f})",
                error=a2.error if a2 else None,
            )
            _flow_step_pause()

        account_hints = {
            "is_new_device": orchestrator.agent2.is_new_device(result.nameOrig, result.device_id),
            "is_new_ip": orchestrator.agent2.is_new_ip(result.nameOrig, result.ip_address),
        }

        if flow_id:
            _update_flow_step(flow_id, "agent3", status="running")
        result = orchestrator.agent3.process(result, account_hints=account_hints)
        if flow_id:
            a3 = result.pipeline_metadata[-1] if result.pipeline_metadata else None
            risk = result.risk_level.value if result.risk_level else "LOW"
            action = result.recommended_action.value if result.recommended_action else "PASS"
            _update_flow_step(
                flow_id,
                "agent3",
                status="completed",
                latency_ms=a3.latency_ms if a3 else None,
                summary=f"{risk} risk → {action}",
                error=a3.error if a3 else None,
            )
            _flow_step_pause()

        if flow_id:
            _update_flow_step(flow_id, "agent4", status="running")
        result = orchestrator.agent4.process(result)
        if flow_id:
            a4 = result.pipeline_metadata[-1] if result.pipeline_metadata else None
            action = result.action_taken.value if result.action_taken else "PASS"
            _update_flow_step(
                flow_id,
                "agent4",
                status="completed",
                latency_ms=a4.latency_ms if a4 else None,
                summary=f"Action executed: {action}",
                error=a4.error if a4 else None,
            )
            _flow_step_pause()

        result.pipeline_end_ms = time.time() * 1000

        if flow_id:
            _update_flow_step(flow_id, "agent5", status="running")
        result = orchestrator.agent5.process(result)
        if flow_id:
            a5 = result.pipeline_metadata[-1] if result.pipeline_metadata else None
            _update_flow_step(
                flow_id,
                "agent5",
                status="completed",
                latency_ms=a5.latency_ms if a5 else None,
                summary="Audit log written",
                error=a5.error if a5 else None,
            )

        output = {
            "fraud_score": result.fraud_score or 0.0,
            "fraud_label": result.fraud_label or False,
            "pattern_type": result.pattern_type.value if result.pattern_type else "NONE",
            # Ensure confidence is always a number (never None) so callers can display it
            "pattern_confidence": result.pattern_confidence if result.pattern_confidence is not None else 0.0,
            "risk_level": result.risk_level.value if result.risk_level else "LOW",
            "recommended_action": result.recommended_action.value if result.recommended_action else "PASS",
            "action_taken": result.action_taken.value if result.action_taken else "PASS",
            "explanation": result.explanation,
            "pattern_reasoning": result.pattern_reasoning,
        }
        if flow_id:
            _finalize_flow(flow_id, status="completed", final_result=output)
        return output

    except Exception as e:
        # Fallback to simple rule-based scoring
        import logging
        logging.error(f"Pipeline error: {e}, using fallback")

        # Simple rule-based fraud detection
        fraud_score = 0.0
        risk_level = "LOW"
        action = "PASS"
        pattern_type = "NONE"

        # High amount check
        if amount > 100000:
            fraud_score += 0.3
        if amount > 500000:
            fraud_score += 0.3

        # New device check
        is_new_device = db.is_device_new(sender_id, device_id)
        if is_new_device and amount > 50000:
            fraud_score += 0.3
            pattern_type = "ACCOUNT_TAKEOVER"

        # Velocity check
        hourly = db.get_velocity(sender_id, "HOURLY")
        if hourly["txn_count"] > 5:
            fraud_score += 0.2
            pattern_type = "VELOCITY_SPIKE"

        # Balance drain check
        if new_balance == 0 and old_balance > 50000:
            fraud_score += 0.2

        # Determine action
        if fraud_score >= 0.7:
            risk_level = "CRITICAL"
            action = "BLOCK"
        elif fraud_score >= 0.5:
            risk_level = "HIGH"
            action = "HOLD"
        elif fraud_score >= 0.2:
            risk_level = "MEDIUM"
            action = "SILENT_FLAG"
        else:
            risk_level = "LOW"
            action = "PASS"

        output = {
            "fraud_score": min(fraud_score, 1.0),
            "fraud_label": fraud_score >= 0.5,
            "pattern_type": pattern_type,
            "pattern_confidence": fraud_score,
            "risk_level": risk_level,
            "recommended_action": action,
            "action_taken": action,
            "explanation": f"Rule-based: score={fraud_score:.2f}",
        }
        if flow_id:
            _update_flow_step(flow_id, "agent1", status="error", error=str(e))
            _finalize_flow(flow_id, status="error", final_result=output)
        return output


async def run_fraud_pipeline(
    transaction_id: str,
    sender_id: str,
    receiver_id: str,
    amount: float,
    txn_type: str,
    old_balance: float,
    new_balance: float,
    ip_address: str,
    device_id: str,
    step: int,
    flow_id: Optional[str] = None,
) -> dict:
    # Offload to worker thread so /transactions/flow polling can update while
    # the transaction request is still processing.
    return await asyncio.to_thread(
        _run_fraud_pipeline_sync,
        transaction_id,
        sender_id,
        receiver_id,
        amount,
        txn_type,
        old_balance,
        new_balance,
        ip_address,
        device_id,
        step,
        flow_id,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Transaction Listing
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/", response_model=List[TransactionListItem])
async def list_transactions(
    limit: int = 50,
    offset: int = 0,
    user: TokenData = Depends(get_current_user)
):
    """Get transaction history for current user."""
    transactions = db.get_user_transactions(
        user_id=user.user_id,
        limit=limit,
        offset=offset
    )

    result = []
    for txn in transactions:
        direction = "sent" if txn["sender_id"] == user.user_id else "received"
        result.append(TransactionListItem(
            transaction_id=txn["transaction_id"],
            type=txn["type"],
            amount=txn["amount"],
            sender_id=txn["sender_id"],
            receiver_id=txn["receiver_id"],
            status=txn["status"],
            initiated_at=txn["initiated_at"],
            direction=direction,
            fraud_score=txn.get("fraud_score"),
            risk_level=txn.get("risk_level"),
            explanation=txn.get("explanation")
        ))

    return result


@router.get("/{transaction_id}")
async def get_transaction(
    transaction_id: str,
    user: TokenData = Depends(get_current_user)
):
    """Get transaction details."""
    txn = db.get_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Check access
    if txn["sender_id"] != user.user_id and txn["receiver_id"] != user.user_id:
        if user.user_type != "ADMIN":
            raise HTTPException(status_code=403, detail="Access denied")

    return txn


@router.get("/{transaction_id}/timeline")
async def get_transaction_timeline(
    transaction_id: str,
    user: TokenData = Depends(get_current_user)
):
    """Get fraud pipeline timeline for a transaction."""
    txn = db.get_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Check access
    if txn["sender_id"] != user.user_id and txn["receiver_id"] != user.user_id:
        if user.user_type != "ADMIN":
            raise HTTPException(status_code=403, detail="Access denied")

    return {
        "transaction_id": transaction_id,
        "initiated_at": txn["initiated_at"],
        "fraud_check_at": txn.get("fraud_check_at"),
        "completed_at": txn.get("completed_at"),
        "status": txn["status"],
        "fraud_pipeline": {
            "fraud_score": txn.get("fraud_score"),
            "fraud_label": bool(txn.get("fraud_label")),
            "pattern_type": txn.get("pattern_type"),
            "pattern_confidence": txn.get("pattern_confidence"),
            "risk_level": txn.get("risk_level"),
            "action_taken": txn.get("action_taken"),
            "explanation": txn.get("explanation"),
            "pipeline_latency_ms": txn.get("pipeline_latency_ms")
        }
    }


@router.get("/flow/{flow_id}")
async def get_transaction_flow(
    flow_id: str,
    user: TokenData = Depends(get_current_user),
):
    """Get live per-agent fraud-flow status for a transaction submission."""
    _cleanup_flow_states()
    with _FLOW_STATE_LOCK:
        flow = _FLOW_STATES.get(flow_id)
        if not flow:
            raise HTTPException(status_code=404, detail="Flow not found or expired")

        is_owner = user.user_id in {flow.get("sender_id"), flow.get("receiver_id")}
        if not is_owner and user.user_type != "ADMIN":
            raise HTTPException(status_code=403, detail="Access denied")

        return {
            "flow_id": flow.get("flow_id"),
            "transaction_id": flow.get("transaction_id"),
            "status": flow.get("status"),
            "steps": flow.get("steps", []),
            "started_at": flow.get("started_at"),
            "updated_at": flow.get("updated_at"),
            "ended_at": flow.get("ended_at"),
            "final_result": flow.get("final_result"),
        }
