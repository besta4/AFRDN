"""
routers/admin.py — Admin endpoints for fraud monitoring and user management.
"""

from fastapi import APIRouter, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone

import database as db
from database import TransactionStatus, AccountStatus
from auth.dependencies import require_admin
from auth.models import TokenData

router = APIRouter(prefix="/admin", tags=["Admin"])


class UserStatusUpdate(BaseModel):
    account_status: str  # ACTIVE, SUSPENDED, BLOCKED


class TransactionActionRequest(BaseModel):
    action: str  # APPROVE or BLOCK
    reason: Optional[str] = None


class DepositFundsRequest(BaseModel):
    user_id: str
    amount: float
    reason: str = "Admin deposit"


# ══════════════════════════════════════════════════════════════════════════════
# User Management
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/users")
async def list_all_users(
    user_type: Optional[str] = None,
    limit: int = 100,
    user: TokenData = Depends(require_admin)
):
    """List all users (admin only)."""
    from database import UserType
    filter_type = UserType(user_type) if user_type else None
    users = db.list_users(user_type=filter_type, limit=limit)
    return {"users": users, "count": len(users)}


@router.get("/users/{user_id}")
async def get_user_details(
    user_id: str,
    user: TokenData = Depends(require_admin)
):
    """Get detailed user information including accounts and transaction history."""
    target_user = db.get_user_with_profile(user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    accounts = db.get_user_accounts(user_id)
    transactions = db.get_user_transactions(user_id, limit=20)
    devices = db.get_user_devices(user_id)

    # Calculate stats
    blocked_txns = sum(1 for t in transactions if t.get("action_taken") == "BLOCK")
    held_txns = sum(1 for t in transactions if t["status"] == "HELD")

    return {
        "user": target_user,
        "accounts": accounts,
        "recent_transactions": transactions,
        "devices": devices,
        "stats": {
            "total_transactions": len(transactions),
            "blocked_transactions": blocked_txns,
            "held_transactions": held_txns,
            "device_count": len(devices)
        }
    }


@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: str,
    body: UserStatusUpdate,
    user: TokenData = Depends(require_admin)
):
    """Update user account status (suspend/block/activate)."""
    target_user = db.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    if body.account_status not in ["ACTIVE", "SUSPENDED", "BLOCKED"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    # Prevent self-modification
    if user_id == user.user_id:
        raise HTTPException(status_code=400, detail="Cannot modify own account")

    db.update_user(user_id, account_status=body.account_status)

    # If blocking, invalidate all sessions
    if body.account_status == "BLOCKED":
        db.invalidate_all_sessions(user_id)

    return {"message": f"User status updated to {body.account_status}"}


@router.post("/users/deposit")
async def deposit_funds(
    body: DepositFundsRequest,
    user: TokenData = Depends(require_admin)
):
    """Admin deposits funds to any user account."""
    target_user = db.get_user_by_id(body.user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if body.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    
    # Get primary account
    account = db.get_primary_account(body.user_id)
    if not account:
        raise HTTPException(status_code=400, detail="User has no primary account")
    
    # Credit the account
    success = db.credit_balance(account["account_id"], body.amount)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to deposit funds")
    
    # Get updated balance
    updated_account = db.get_account(account["account_id"])
    
    # Log audit
    print(f"[ADMIN DEPOSIT] Admin {user.user_id} deposited ₹{body.amount} to {body.user_id}. Reason: {body.reason}")
    
    return {
        "message": f"Successfully deposited ₹{body.amount:,.2f} to {body.user_id}",
        "new_balance": updated_account["balance"],
        "reason": body.reason
    }


# ══════════════════════════════════════════════════════════════════════════════
# Fraud Monitoring
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/fraud/held")
async def get_held_transactions(
    limit: int = 100,
    user: TokenData = Depends(require_admin)
):
    """Get all HELD transactions requiring admin review."""
    transactions = db.get_held_transactions(limit=limit)
    return {"transactions": transactions, "count": len(transactions)}


@router.get("/fraud/recent")
async def get_recent_transactions(
    limit: int = 100,
    user: TokenData = Depends(require_admin)
):
    """Get recent transactions with fraud scores."""
    transactions = db.get_recent_transactions(limit=limit)
    return {"transactions": transactions, "count": len(transactions)}


@router.post("/fraud/{transaction_id}/approve")
async def approve_transaction(
    transaction_id: str,
    body: TransactionActionRequest,
    user: TokenData = Depends(require_admin)
):
    """Approve a held transaction (release funds)."""
    txn = db.get_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    if txn["status"] != "HELD":
        raise HTTPException(
            status_code=400,
            detail=f"Transaction is not held (status: {txn['status']})"
        )

    # Credit receiver
    receiver_account = db.get_primary_account(txn["receiver_id"])
    if receiver_account:
        _, new_balance = db.credit_balance(receiver_account["account_id"], txn["amount"])
        db.complete_transaction(transaction_id, new_balance)

        # Update velocity
        db.update_velocity(txn["sender_id"], txn["amount"])

    return {"message": "Transaction approved", "transaction_id": transaction_id}


@router.post("/fraud/{transaction_id}/block")
async def block_transaction(
    transaction_id: str,
    body: TransactionActionRequest,
    user: TokenData = Depends(require_admin)
):
    """Block and reverse a held transaction."""
    txn = db.get_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    if txn["status"] != "HELD":
        raise HTTPException(
            status_code=400,
            detail=f"Transaction is not held (status: {txn['status']})"
        )

    # Reverse: credit back to sender
    sender_account = db.get_primary_account(txn["sender_id"])
    if sender_account:
        db.credit_balance(sender_account["account_id"], txn["amount"])

    db.update_transaction_status(transaction_id, TransactionStatus.BLOCKED)

    # Generate STR for blocked transaction with admin attribution
    reason = body.reason or "Admin manually blocked transaction"
    db.create_compliance_report(
        report_type="STR",
        trigger_reason=f"{reason} (reviewed by admin: {user.user_id})",
        transaction_id=transaction_id,
        user_id=txn["sender_id"],
        amount=txn["amount"],
        auto_generated=False  # Manual admin action
    )

    return {"message": "Transaction blocked and reversed", "transaction_id": transaction_id}


@router.get("/fraud/patterns")
async def get_fraud_patterns(
    user: TokenData = Depends(require_admin)
):
    """Get fraud pattern analysis summary."""
    transactions = db.get_recent_transactions(limit=500)

    # Analyze patterns
    patterns = {"NONE": 0, "MULE_NETWORK": 0, "ACCOUNT_TAKEOVER": 0, "VELOCITY_SPIKE": 0}
    risk_levels = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    actions = {"PASS": 0, "SILENT_FLAG": 0, "HOLD": 0, "BLOCK": 0}

    for txn in transactions:
        pattern = txn.get("pattern_type", "NONE")
        if pattern in patterns:
            patterns[pattern] += 1

        risk = txn.get("risk_level", "LOW")
        if risk in risk_levels:
            risk_levels[risk] += 1

        action = txn.get("action_taken", "PASS")
        if action in actions:
            actions[action] += 1

    total = len(transactions)

    return {
        "total_transactions": total,
        "pattern_distribution": patterns,
        "risk_distribution": risk_levels,
        "action_distribution": actions,
        "fraud_rate": round(actions.get("BLOCK", 0) / max(total, 1) * 100, 2)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Compliance Reports
# ══════════════════════════════════════════════════════════════════════════════

class ComplianceStatusUpdate(BaseModel):
    status: str  # SUBMITTED or ACKNOWLEDGED


@router.get("/compliance/str")
async def get_str_reports(
    status: Optional[str] = None,
    limit: int = 100,
    user: TokenData = Depends(require_admin)
):
    """Get Suspicious Transaction Reports with optional status filter."""
    reports = db.get_compliance_reports_filtered(report_type="STR", status=status, limit=limit)
    return {"reports": reports, "count": len(reports)}


@router.get("/compliance/ctr")
async def get_ctr_reports(
    status: Optional[str] = None,
    limit: int = 100,
    user: TokenData = Depends(require_admin)
):
    """Get Cash Transaction Reports with optional status filter."""
    reports = db.get_compliance_reports_filtered(report_type="CTR", status=status, limit=limit)
    return {"reports": reports, "count": len(reports)}


@router.get("/compliance/all")
async def get_all_compliance_reports(
    status: Optional[str] = None,
    limit: int = 100,
    user: TokenData = Depends(require_admin)
):
    """Get all compliance reports with optional status filter."""
    reports = db.get_compliance_reports_filtered(status=status, limit=limit)
    return {"reports": reports, "count": len(reports)}


@router.get("/compliance/{report_id}")
async def get_compliance_report(
    report_id: str,
    user: TokenData = Depends(require_admin)
):
    """Get a single compliance report by ID."""
    report = db.get_compliance_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report": report}


@router.put("/compliance/{report_id}/submit")
async def submit_compliance_report(
    report_id: str,
    user: TokenData = Depends(require_admin)
):
    """Mark a compliance report as SUBMITTED (sent to regulatory authority)."""
    report = db.get_compliance_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report["status"] != "PENDING":
        raise HTTPException(
            status_code=400,
            detail=f"Report cannot be submitted (current status: {report['status']})"
        )

    db.update_compliance_report_status(report_id, "SUBMITTED", admin_user_id=user.user_id)
    return {"message": "Report marked as submitted", "report_id": report_id}


@router.put("/compliance/{report_id}/acknowledge")
async def acknowledge_compliance_report(
    report_id: str,
    user: TokenData = Depends(require_admin)
):
    """Mark a compliance report as ACKNOWLEDGED (confirmed by regulatory authority)."""
    report = db.get_compliance_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report["status"] != "SUBMITTED":
        raise HTTPException(
            status_code=400,
            detail=f"Report must be submitted before acknowledgment (current status: {report['status']})"
        )

    db.update_compliance_report_status(report_id, "ACKNOWLEDGED", admin_user_id=user.user_id)
    return {"message": "Report acknowledged", "report_id": report_id}


# ══════════════════════════════════════════════════════════════════════════════
# System Metrics
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/system/metrics")
async def get_system_metrics(
    user: TokenData = Depends(require_admin)
):
    """Get system performance metrics."""
    # Get recent transactions for latency stats
    transactions = db.get_recent_transactions(limit=100)

    latencies = [t.get("pipeline_latency_ms", 0) for t in transactions if t.get("pipeline_latency_ms")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Count users by type
    all_users = db.list_users(limit=10000)
    user_counts = {"CUSTOMER": 0, "MERCHANT": 0, "ADMIN": 0}
    for u in all_users:
        if u["user_type"] in user_counts:
            user_counts[u["user_type"]] += 1

    return {
        "total_users": len(all_users),
        "user_breakdown": user_counts,
        "total_transactions_today": len(transactions),
        "pipeline_metrics": {
            "avg_latency_ms": round(avg_latency, 2),
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/system/kyc-limits")
async def get_kyc_limits(
    user: TokenData = Depends(require_admin)
):
    """Get current KYC limits configuration."""
    limits = {
        "MINIMUM": db.get_kyc_limits("MINIMUM"),
        "FULL": db.get_kyc_limits("FULL"),
        "ENHANCED": db.get_kyc_limits("ENHANCED")
    }
    return {"limits": limits}


# ══════════════════════════════════════════════════════════════════════════════
# Pattern Detection Testing - Simulates fraud patterns for verification
# ══════════════════════════════════════════════════════════════════════════════

import logging
import time
import uuid

logger = logging.getLogger(__name__)


class SimulateFraudRequest(BaseModel):
    pattern_type: str  # MULE_NETWORK, ACCOUNT_TAKEOVER, VELOCITY_SPIKE
    num_transactions: int = 5


@router.post("/test/simulate-fraud")
async def simulate_fraud_pattern(
    body: SimulateFraudRequest,
    user: TokenData = Depends(require_admin)
):
    """
    Simulate fraud patterns to test the 5-agent pipeline detection.
    
    This creates synthetic transactions that should trigger pattern detection:
    - MULE_NETWORK: Multiple senders to same collector
    - ACCOUNT_TAKEOVER: New device/IP with high amount  
    - VELOCITY_SPIKE: Rapid transactions from same sender
    
    Returns detection results for each simulated transaction.
    """
    from agents.orchestrator import Orchestrator
    from agents.models import TransactionMessage, TransactionType, TrafficMode, PatternType
    
    logger.info("=" * 70)
    logger.info(f"[FRAUD SIMULATION] Starting {body.pattern_type} simulation with {body.num_transactions} transactions")
    logger.info("=" * 70)
    
    results = []
    orchestrator = Orchestrator()
    
    # Shared collector for mule network
    mule_collector = f"M{uuid.uuid4().hex[:9].upper()}"
    # Shared sender for velocity spike
    velocity_sender = f"C{uuid.uuid4().hex[:9].upper()}"
    # ATO victim with established history
    ato_victim = f"C{uuid.uuid4().hex[:9].upper()}"
    ato_known_ip = "ip_100"
    ato_known_device = "device_100"
    
    # Pre-populate history for ATO victim (so they have "known" devices/IPs)
    if body.pattern_type == "ACCOUNT_TAKEOVER":
        orchestrator.agent2._user_ip_history[ato_victim].add(ato_known_ip)
        orchestrator.agent2._user_device_history[ato_victim].add(ato_known_device)
        logger.info(f"[FRAUD SIMULATION] Pre-populated ATO victim {ato_victim} with known IP/device")
    
    base_step = 100  # Starting step for timing
    
    for i in range(body.num_transactions):
        txn_id = f"TEST_{uuid.uuid4().hex[:12].upper()}"
        
        if body.pattern_type == "MULE_NETWORK":
            # Different senders, same collector, within 5 steps
            # Realistic amounts: Rs 8,000 - Rs 45,000 (like in real mule networks)
            sender = f"C{uuid.uuid4().hex[:9].upper()}"
            realistic_amounts = [8000, 12000, 25000, 35000, 45000, 18000, 22000, 9000, 15000, 30000]
            amount = realistic_amounts[i % len(realistic_amounts)]
            msg = TransactionMessage(
                transaction_id=txn_id,
                traffic_mode=TrafficMode.MULE_NETWORK,
                step=base_step + (i % 5),  # Within 5-step window
                type=TransactionType.TRANSFER,
                amount=amount,
                nameOrig=sender,
                nameDest=mule_collector,  # SAME collector
                oldbalanceOrg=100000,
                newbalanceOrig=100000 - amount,
                oldbalanceDest=i * 20000,
                newbalanceDest=(i + 1) * 20000 + amount,
                ip_address=f"ip_{500 + i}",
                device_id=f"device_{500 + i}",
            )
            logger.info(f"[MULE_NETWORK] Txn {i+1}: {sender} -> {mule_collector} (Rs {msg.amount:,.0f})")
            
        elif body.pattern_type == "ACCOUNT_TAKEOVER":
            # Same victim, NEW IP/device each time
            # Realistic amounts: Rs 30,000 - Rs 80,000 (account draining attempts)
            ato_amounts = [35000, 45000, 55000, 65000, 75000, 40000, 50000, 60000]
            amount = ato_amounts[i % len(ato_amounts)]
            msg = TransactionMessage(
                transaction_id=txn_id,
                traffic_mode=TrafficMode.ACCOUNT_TAKEOVER,
                step=base_step + i,
                type=TransactionType.CASH_OUT,
                amount=amount,
                nameOrig=ato_victim,  # SAME victim
                nameDest=f"C{uuid.uuid4().hex[:9].upper()}",
                oldbalanceOrg=200000,
                newbalanceOrig=200000 - amount,
                oldbalanceDest=0,
                newbalanceDest=amount,
                ip_address=f"ip_{900 + i}",  # NEW suspicious IP
                device_id=f"device_{900 + i}",  # NEW suspicious device
            )
            logger.info(f"[ACCOUNT_TAKEOVER] Txn {i+1}: {ato_victim} from NEW device_{900 + i}/ip_{900 + i} (Rs {msg.amount:,.0f})")
            
        elif body.pattern_type == "VELOCITY_SPIKE":
            # Same sender, rapid transactions
            velocity_amounts = [5000, 8000, 12000, 15000, 20000, 10000, 7000, 18000, 9000, 25000]
            amount = velocity_amounts[i % len(velocity_amounts)]
            msg = TransactionMessage(
                transaction_id=txn_id,
                traffic_mode=TrafficMode.NORMAL,  # Normal mode - pattern detected by velocity
                step=base_step,  # Same step = rapid
                type=TransactionType.TRANSFER,
                amount=amount,
                nameOrig=velocity_sender,  # SAME sender
                nameDest=f"C{uuid.uuid4().hex[:9].upper()}",
                oldbalanceOrg=500000 - (i * amount),
                newbalanceOrig=500000 - ((i + 1) * amount),
                oldbalanceDest=0,
                newbalanceDest=amount,
                ip_address=f"ip_{velocity_sender}",
                device_id=f"device_{velocity_sender}",
            )
            logger.info(f"[VELOCITY_SPIKE] Txn {i+1}: {velocity_sender} rapid transaction #{i+1} (Rs {msg.amount:,.0f})")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown pattern type: {body.pattern_type}")
        
        # Run through the FULL pipeline
        start_time = time.time()
        
        # Agent 1: Transaction Monitoring - ALWAYS run to get real ML score
        logger.info(f"  [Agent 1] Processing transaction {txn_id}...")
        msg = orchestrator.agent1.process(msg)
        a1_latency = (time.time() - start_time) * 1000
        logger.info(f"  [Agent 1] Fraud Score: {msg.fraud_score:.4f}, Label: {msg.fraud_label} ({a1_latency:.1f}ms)")
        
        # Add to rolling buffer if:
        # 1. fraud_label is True (Agent 1 flagged it), OR
        # 2. fraud_score > 20% (high enough to warrant pattern analysis), OR
        # 3. It's a pattern simulation (MULE_NETWORK, VELOCITY_SPIKE)
        should_add_to_buffer = (
            msg.fraud_label or 
            (msg.fraud_score and msg.fraud_score > 0.20) or
            body.pattern_type in ["MULE_NETWORK", "VELOCITY_SPIKE"]
        )
        if should_add_to_buffer:
            orchestrator.rolling_buffer.append(msg)
            logger.info(f"  [Buffer] Added to rolling buffer (score={msg.fraud_score:.2%}). Buffer size: {len(orchestrator.rolling_buffer)}")
        
        # Agent 2: Pattern Detection
        a2_start = time.time()
        logger.info(f"  [Agent 2] Analyzing patterns (buffer={len(orchestrator.rolling_buffer)})...")
        msg = orchestrator.agent2.process(msg)
        a2_latency = (time.time() - a2_start) * 1000
        pattern_name = msg.pattern_type.value if msg.pattern_type else "NONE"
        logger.info(f"  [Agent 2] Pattern: {pattern_name}, Confidence: {msg.pattern_confidence:.2f} ({a2_latency:.1f}ms)")
        if msg.pattern_reasoning:
            logger.info(f"  [Agent 2] LLM Reasoning: {msg.pattern_reasoning}")
        
        # Agent 3: Risk Assessment
        a3_start = time.time()
        account_hints = {
            "is_new_device": orchestrator.agent2.is_new_device(msg.nameOrig, msg.device_id),
            "is_new_ip": orchestrator.agent2.is_new_ip(msg.nameOrig, msg.ip_address),
        }
        logger.info(f"  [Agent 3] Assessing risk (new_device={account_hints['is_new_device']}, new_ip={account_hints['is_new_ip']})...")
        msg = orchestrator.agent3.process(msg, account_hints=account_hints)
        a3_latency = (time.time() - a3_start) * 1000
        risk_name = msg.risk_level.value if msg.risk_level else "LOW"
        action_name = msg.recommended_action.value if msg.recommended_action else "PASS"
        logger.info(f"  [Agent 3] Risk: {risk_name}, Action: {action_name} ({a3_latency:.1f}ms)")
        
        # Agent 4: Alert & Block
        a4_start = time.time()
        msg = orchestrator.agent4.process(msg)
        a4_latency = (time.time() - a4_start) * 1000
        final_action = msg.action_taken.value if msg.action_taken else "PASS"
        logger.info(f"  [Agent 4] Final Action: {final_action} ({a4_latency:.1f}ms)")
        
        # Agent 5: Compliance Logging
        a5_start = time.time()
        msg = orchestrator.agent5.process(msg)
        a5_latency = (time.time() - a5_start) * 1000
        logger.info(f"  [Agent 5] Audit logged ({a5_latency:.1f}ms)")
        
        total_latency = (time.time() - start_time) * 1000
        
        # Save transaction to database for admin panel visibility
        # Use existing demo users to satisfy foreign key constraints
        try:
            now = datetime.now(timezone.utc)
            step = now.hour + (now.day * 24)
            # Map action to valid status: BLOCKED, HELD, or COMPLETED
            status_val = "BLOCKED" if final_action == "BLOCK" else ("HELD" if final_action == "HOLD" else "COMPLETED")
            
            # Get existing user IDs for sender/receiver to satisfy FK constraints
            with db.get_conn() as conn:
                users = conn.execute("SELECT user_id FROM users WHERE user_type = 'CUSTOMER' LIMIT 2").fetchall()
                if len(users) >= 2:
                    sender_user_id = users[0][0]
                    receiver_user_id = users[1][0]
                    
                    # Get their accounts
                    sender_acct = conn.execute("SELECT account_id FROM accounts WHERE user_id = ?", (sender_user_id,)).fetchone()
                    receiver_acct = conn.execute("SELECT account_id FROM accounts WHERE user_id = ?", (receiver_user_id,)).fetchone()
                    
                    if sender_acct and receiver_acct:
                        conn.execute(
                            """INSERT INTO transactions
                               (transaction_id, step, type, amount, sender_id, sender_account,
                                receiver_id, receiver_account, old_balance_sender, new_balance_sender,
                                old_balance_receiver, new_balance_receiver, ip_address, device_id,
                                description, status, initiated_at, fraud_score, fraud_label,
                                pattern_type, pattern_confidence, risk_level, recommended_action,
                                action_taken, explanation, pipeline_latency_ms)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (txn_id, step, msg.type.value, msg.amount, 
                             sender_user_id, sender_acct[0],
                             receiver_user_id, receiver_acct[0],
                             msg.oldbalanceOrg or 0, msg.newbalanceOrig or 0,
                             msg.oldbalanceDest or 0, msg.newbalanceDest or 0,
                             msg.ip_address or "127.0.0.1", msg.device_id or "test_device",
                             f"Simulated {body.pattern_type}: {msg.nameOrig} → {msg.nameDest}",
                             status_val, now.isoformat(),
                             msg.fraud_score or 0, 1 if msg.fraud_label else 0,
                             pattern_name, msg.pattern_confidence or 0,
                             risk_name, action_name, final_action,
                             msg.explanation, total_latency)
                        )
                        logger.info(f"  [DB] Transaction saved to database")
                    else:
                        logger.warning("  [DB] Could not find accounts for demo users")
                else:
                    logger.warning("  [DB] Not enough demo users to save transaction")
        except Exception as e:
            logger.error(f"  [DB] Failed to save transaction: {e}")
        
        result = {
            "transaction_id": txn_id,
            "index": i + 1,
            "amount": msg.amount,  # Include amount in results
            "sender": msg.nameOrig,
            "receiver": msg.nameDest,
            "fraud_score": round(msg.fraud_score or 0, 4),
            "fraud_label": msg.fraud_label,
            "pattern_type": pattern_name,
            "pattern_confidence": round(msg.pattern_confidence or 0, 4),
            "pattern_reasoning": msg.pattern_reasoning,
            "risk_level": risk_name,
            "recommended_action": action_name,
            "action_taken": final_action,
            "explanation": msg.explanation,
            "pipeline_latency_ms": round(total_latency, 1),
            "agent_latencies": {
                "agent1_ms": round(a1_latency, 1),
                "agent2_ms": round(a2_latency, 1),
                "agent3_ms": round(a3_latency, 1),
                "agent4_ms": round(a4_latency, 1),
                "agent5_ms": round(a5_latency, 1),
            },
            "buffer_size": len(orchestrator.rolling_buffer),
        }
        results.append(result)
        
        logger.info(f"  [COMPLETE] Txn {i+1}/{body.num_transactions}: {pattern_name} detected, {final_action}")
        logger.info("-" * 50)
    
    # Summary
    patterns_detected = sum(1 for r in results if r["pattern_type"] != "NONE")
    blocked = sum(1 for r in results if r["action_taken"] == "BLOCK")
    held = sum(1 for r in results if r["action_taken"] == "HOLD")
    
    logger.info("=" * 70)
    logger.info(f"[FRAUD SIMULATION COMPLETE] {body.pattern_type}")
    logger.info(f"  Total Transactions: {body.num_transactions}")
    logger.info(f"  Patterns Detected: {patterns_detected}")
    logger.info(f"  Blocked: {blocked}, Held: {held}")
    logger.info("=" * 70)
    
    return {
        "simulation_type": body.pattern_type,
        "total_transactions": body.num_transactions,
        "summary": {
            "patterns_detected": patterns_detected,
            "blocked": blocked,
            "held": held,
            "passed": body.num_transactions - blocked - held,
        },
        "results": results,
        "buffer_state": {
            "final_size": len(orchestrator.rolling_buffer),
            "window_ids": [m.transaction_id for m in list(orchestrator.rolling_buffer)[-5:]]
        }
    }
