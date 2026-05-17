"""
500 realtime UPI-like fraud scenarios for the Jatayu agent pipeline.

This is a standalone verification script, not a pytest test. It exercises the
same Orchestrator used by the live app, with LLM reasoning disabled so the run
does not depend on the optional local llama server. SHAP is sampled on a few
processed transactions when the package is installed.

Run from the project root:
    python tests/test_500_realtime_scenarios.py
"""

from __future__ import annotations

import os
import logging
import random
import sys
import time
import uuid
import warnings
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"
os.environ["JATAYU_AUDIT_FILE"] = str(OUTPUT_DIR / "audit_500.jsonl")
os.environ["JATAYU_DB_PATH"] = str(OUTPUT_DIR / "scenario_500.db")
os.environ["JATAYU_ENABLE_PPO"] = "0"
os.environ.setdefault("JATAYU_REDIS_URL", "redis://localhost:6379/15")

warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
logging.getLogger("agents").setLevel(logging.ERROR)

from agents.models import TransactionMessage, TransactionType  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402


RNG = random.Random(20260517)


def reset_runtime_state() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (Path(os.environ["JATAYU_AUDIT_FILE"]), Path(os.environ["JATAYU_DB_PATH"])):
        if path.exists():
            path.unlink()
    try:
        import redis  # type: ignore
        redis.Redis.from_url(os.environ["JATAYU_REDIS_URL"], decode_responses=True).flushdb()
    except Exception:
        pass


def txn(
    category: str,
    step: int,
    sender: str,
    receiver: str,
    amount: float,
    txn_type: TransactionType = TransactionType.TRANSFER,
    sender_balance: float = 100_000.0,
    receiver_balance: float = 10_000.0,
    ip: str | None = None,
    device: str | None = None,
    label: bool | None = None,
) -> tuple[str, TransactionMessage]:
    amount = round(float(amount), 2)
    return category, TransactionMessage(
        transaction_id=f"UPI_{category}_{uuid.uuid4().hex[:10].upper()}",
        step=step,
        type=txn_type,
        amount=amount,
        nameOrig=sender,
        nameDest=receiver,
        oldbalanceOrg=sender_balance,
        newbalanceOrig=max(sender_balance - amount, 0),
        oldbalanceDest=receiver_balance,
        newbalanceDest=receiver_balance + amount,
        ip_address=ip or f"ip_{RNG.randint(1, 120):04d}",
        device_id=device or f"device_{RNG.randint(1, 160):04d}",
        ground_truth_label=label,
    )


def build_cases() -> list[tuple[str, TransactionMessage]]:
    cases: list[tuple[str, TransactionMessage]] = []
    step = 10_000

    # 250 normal UPI payments/transfers: groceries, bills, cab, rent, small P2P.
    merchants = ["M_GROCERY", "M_CAB", "M_FOOD", "M_RECHARGE", "M_MEDICAL", "M_FUEL"]
    for i in range(250):
        step += RNG.randint(1, 3)
        is_merchant_payment = RNG.random() < 0.65
        sender = f"C_NORMAL_{RNG.randint(1, 90):03d}"
        receiver = RNG.choice(merchants) if is_merchant_payment else f"C_FRIEND_{RNG.randint(1, 160):03d}"
        amount = RNG.choice([99, 149, 249, 499, 799, 1250, 2200, 4500, 8500])
        cases.append(txn(
            "normal_upi",
            step,
            sender,
            receiver,
            amount,
            TransactionType.PAYMENT if is_merchant_payment else TransactionType.TRANSFER,
            sender_balance=RNG.randint(20_000, 180_000),
            receiver_balance=RNG.randint(1_000, 90_000),
            ip=f"ip_home_{RNG.randint(1, 20):02d}",
            device=f"device_known_{RNG.randint(1, 35):02d}",
            label=False,
        ))

    # 40 legitimate split-bill transfers into one collector, then merchant spend.
    split_collector = "C_SPLIT_COLLECTOR"
    for i in range(39):
        step += 1
        cases.append(txn(
            "legit_split_bill",
            step,
            f"C_SPLIT_FRIEND_{i:02d}",
            split_collector,
            RNG.randint(80, 450),
            sender_balance=RNG.randint(5_000, 30_000),
            receiver_balance=RNG.randint(500, 5_000),
            label=False,
        ))
    step += 1
    cases.append(txn(
        "legit_split_bill",
        step,
        split_collector,
        "M_RESTAURANT",
        8_750,
        TransactionType.PAYMENT,
        sender_balance=12_500,
        receiver_balance=50_000,
        label=False,
    ))

    # 45 mule-network cases: fan-in to mule accounts and rapid extraction.
    for cluster in range(3):
        mule = f"C_MULE_{cluster}"
        boss = f"C_MULE_BOSS_{cluster}"
        for i in range(10):
            step += 1
            cases.append(txn(
                "mule_fanin",
                step,
                f"C_VICTIM_{cluster}_{i}",
                mule,
                RNG.randint(2_500, 18_000),
                sender_balance=RNG.randint(20_000, 120_000),
                receiver_balance=0 if i == 0 else RNG.randint(1_000, 8_000),
                ip=f"ip_victim_{i:02d}",
                device=f"device_victim_{i:02d}",
                label=True,
            ))
        for i in range(5):
            step += 1
            cases.append(txn(
                "mule_extraction",
                step,
                mule,
                boss,
                RNG.randint(15_000, 45_000),
                sender_balance=65_000,
                receiver_balance=10_000,
                ip="ip_mule_exit",
                device="device_mule_exit",
                label=True,
            ))

    # 45 velocity/bot cases: many low-value probes from a single account.
    for bot in range(3):
        sender = f"C_BOT_{bot}"
        for i in range(15):
            step += 1
            cases.append(txn(
                "velocity_probe",
                step,
                sender,
                f"C_RANDOM_{bot}_{i}",
                RNG.choice([1, 5, 10, 25, 49, 99]),
                sender_balance=5_000,
                receiver_balance=100,
                ip="ip_botnet",
                device=f"device_bot_{bot}",
                label=True,
            ))

    # 40 account takeover cases: known-user warmup followed by new device/IP cashout.
    for user in range(20):
        victim = f"C_ATO_{user:02d}"
        step += 1
        cases.append(txn(
            "ato_warmup",
            step,
            victim,
            "M_RECHARGE",
            199,
            TransactionType.PAYMENT,
            sender_balance=80_000,
            receiver_balance=30_000,
            ip=f"ip_known_{user:02d}",
            device=f"device_known_{user:02d}",
            label=False,
        ))
        step += 1
        cases.append(txn(
            "account_takeover",
            step,
            victim,
            f"C_DROP_{user:02d}",
            RNG.randint(35_000, 75_000),
            sender_balance=80_000,
            receiver_balance=0,
            ip=f"ip_new_geo_{user:02d}",
            device=f"device_new_{user:02d}",
            label=True,
        ))

    # 30 dormant hijack cases: dormant account receives and nearly drains funds.
    for i in range(15):
        dormant = f"C_DORMANT_{i:02d}"
        step += 1
        cases.append(txn(
            "dormant_inbound",
            step,
            f"C_SOURCE_{i:02d}",
            dormant,
            RNG.randint(20_000, 90_000),
            sender_balance=120_000,
            receiver_balance=0,
            label=True,
        ))
        step += 1
        cases.append(txn(
            "dormant_cashout",
            step,
            dormant,
            f"M_CASHOUT_{i:02d}",
            RNG.randint(18_000, 88_000),
            TransactionType.CASH_OUT,
            sender_balance=90_000,
            receiver_balance=0,
            label=True,
        ))

    # 30 circular-flow cases: ping-pong transfers used for layering.
    for pair in range(5):
        a = f"C_RING_A_{pair}"
        b = f"C_RING_B_{pair}"
        for i in range(3):
            step += 1
            cases.append(txn("circular_flow", step, a, b, 5_000, sender_balance=50_000, receiver_balance=20_000, label=True))
            step += 1
            cases.append(txn("circular_flow", step, b, a, 4_950, sender_balance=45_000, receiver_balance=25_000, label=True))

    # 20 legitimate merchant bursts, expected to remain low friction.
    for i in range(20):
        step += 1
        cases.append(txn(
            "merchant_burst_legit",
            step,
            f"C_CUSTOMER_QR_{i:02d}",
            "M_BUSY_CAFE",
            RNG.randint(80, 650),
            TransactionType.PAYMENT,
            sender_balance=RNG.randint(5_000, 25_000),
            receiver_balance=150_000,
            label=False,
        ))

    assert len(cases) == 500, len(cases)
    return cases


def main() -> int:
    reset_runtime_state()
    cases = build_cases()
    orchestrator = Orchestrator()
    redis_available = bool(orchestrator.graph_cache and orchestrator.graph_cache.available)
    orchestrator.agent2._generate_reasoning = False

    # Keep SHAP checks local and fast: generate SHAP values but skip LLM text.
    orchestrator.agent1._generate_rationale = lambda **_: "SHAP attribution sampled during test."

    started = time.time()
    by_category: dict[str, Counter] = defaultdict(Counter)
    actions = Counter()
    patterns = Counter()
    shap_samples = []

    for category, msg in cases:
        result = orchestrator._process_one(msg)
        action = result.action_taken.value if result.action_taken else "PASS"
        pattern = result.pattern_type.value if result.pattern_type else "NONE"
        actions[action] += 1
        patterns[pattern] += 1
        by_category[category][action] += 1

        assert result.fraud_score is not None
        assert result.risk_level is not None
        assert result.pipeline_metadata

        if len(shap_samples) < 5 and category in {"mule_extraction", "account_takeover", "velocity_probe", "dormant_cashout"}:
            exp = orchestrator.agent1.explain(msg)
            shap_samples.append({
                "transaction_id": msg.transaction_id,
                "category": category,
                "method": exp.get("attribution_method"),
                "top_positive": exp.get("top_positive_features", [])[:3],
            })

    risky_actions = actions["HOLD"] + actions["BLOCK"]
    assert risky_actions > 0, "Expected at least one HOLD/BLOCK across fraud scenarios"

    print("\nJATAYU 500 REALTIME UPI SCENARIO REPORT")
    print("=" * 72)
    print(f"Total cases: {len(cases)}")
    print(f"Runtime seconds: {time.time() - started:.1f}")
    print(f"Redis graph cache: {'enabled' if redis_available else 'disabled'}")
    print(f"Actions: {dict(actions)}")
    print(f"Patterns: {dict(patterns)}")
    print("\nCategory action breakdown:")
    for category in sorted(by_category):
        print(f"  {category:<22} {dict(by_category[category])}")
    print("\nSHAP samples:")
    for sample in shap_samples:
        print(f"  {sample}")
    print("=" * 72)

    assert by_category["legit_split_bill"]["BLOCK"] == 0, "Split-bill traffic should not be blocked"
    assert by_category["merchant_burst_legit"] == Counter({"PASS": 20}), "Merchant QR burst should pass"
    assert by_category["normal_upi"]["PASS"] >= 220, "Normal UPI pass rate is too low"
    assert by_category["normal_upi"]["BLOCK"] <= 3, "Normal UPI block false positives are too high"
    assert by_category["account_takeover"]["BLOCK"] >= 18, "ATO detection should block most takeover cases"
    assert by_category["dormant_inbound"]["BLOCK"] >= 12, "Dormant inbound hijack should be blocked"
    assert by_category["mule_extraction"]["BLOCK"] >= 12, "Mule extraction should be blocked"
    assert by_category["velocity_probe"]["BLOCK"] >= 25, "Velocity probes should be blocked"
    if redis_available:
        assert by_category["circular_flow"]["BLOCK"] >= 20, "Circular flow should be blocked with Redis enabled"
    assert shap_samples and all(sample["method"] == "shap" for sample in shap_samples), "SHAP should explain sampled fraud cases"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
