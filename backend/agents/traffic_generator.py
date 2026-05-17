"""
traffic_generator.py — Synthetic financial transaction stream for Jatayu demos.

Produces PaySim-style TransactionMessage objects in three modes:

  NORMAL           ~0.13% organic fraud, realistic distribution of types/amounts.
  MULE_NETWORK     4–6 users in a coordinated ring funneling money to one merchant
                   within a 5-step burst window.
  ACCOUNT_TAKEOVER Single user transacting from a new IP/device with a large amount
                   to a new payee — classic ATO signal.

Usage:
    gen = SyntheticTrafficGenerator(mode=TrafficMode.MULE_NETWORK, delay_seconds=0.3)
    for msg in gen:
        pipeline.run(msg)          # yields one TransactionMessage at a time
"""

from __future__ import annotations

import random
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from agents.models import (
    Action,
    PatternType,
    RiskLevel,
    TrafficMode,
    TransactionMessage,
    TransactionType,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants that mirror the PaySim / Jatayu training data distribution
# ─────────────────────────────────────────────────────────────────────────────

# Approximate type distribution in PaySim (CASH_OUT + PAYMENT dominate)
_TYPE_WEIGHTS = {
    TransactionType.PAYMENT:  0.34,
    TransactionType.CASH_OUT: 0.35,
    TransactionType.CASH_IN:  0.22,
    TransactionType.TRANSFER: 0.08,
    TransactionType.DEBIT:    0.01,
}

# Amount distributions by type (mean, std) in PaySim — log-normal shape
_AMOUNT_PARAMS: dict[TransactionType, tuple[float, float]] = {
    TransactionType.PAYMENT:  (135_000, 150_000),
    TransactionType.CASH_OUT: (168_000, 200_000),
    TransactionType.CASH_IN:  (150_000, 175_000),
    TransactionType.TRANSFER: (180_000, 220_000),
    TransactionType.DEBIT:    (  5_000,   8_000),
}

_N_IPS     = 2_000   # matches config: n_ips = 2000
_N_DEVICES = 1_998   # matches config: n_devices = 1998
_MAX_STEP  = 743     # matches config: global_max_step = 743

# Organic fraud rate in PaySim (~0.13%)
_ORGANIC_FRAUD_RATE = 0.0013


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_user_id() -> str:
    """Generate a PaySim-style customer ID: C followed by 9 digits."""
    return "C" + "".join(random.choices(string.digits, k=9))


def _random_merchant_id() -> str:
    """Generate a PaySim-style merchant ID: M followed by 9 digits."""
    return "M" + "".join(random.choices(string.digits, k=9))


def _random_ip() -> str:
    return f"ip_{random.randint(1, _N_IPS)}"


def _random_device() -> str:
    return f"device_{random.randint(1, _N_DEVICES)}"


def _random_amount(txn_type: TransactionType) -> float:
    mean, std = _AMOUNT_PARAMS[txn_type]
    return max(1.0, round(random.gauss(mean, std), 2))


def _sample_type() -> TransactionType:
    types = list(_TYPE_WEIGHTS.keys())
    weights = list(_TYPE_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


def _make_balances(amount: float) -> tuple[float, float, float, float]:
    """Return (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)."""
    old_orig = round(random.uniform(amount * 0.5, amount * 5), 2)
    new_orig = max(0.0, round(old_orig - amount, 2))
    old_dest = round(random.uniform(0, amount * 2), 2)
    new_dest = round(old_dest + amount, 2)
    return old_orig, new_orig, old_dest, new_dest


# ─────────────────────────────────────────────────────────────────────────────
# SyntheticTrafficGenerator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticTrafficGenerator:
    """
    Iterator that yields TransactionMessage objects in one of three modes.

    Args:
        mode:          TrafficMode enum — NORMAL, MULE_NETWORK, ACCOUNT_TAKEOVER.
        delay_seconds: Seconds to sleep between yields (0 for maximum speed).
        max_count:     Stop after this many transactions (None = infinite).
        seed:          Optional random seed for reproducibility.
    """

    def __init__(
        self,
        mode: TrafficMode = TrafficMode.NORMAL,
        delay_seconds: float = 0.3,
        max_count: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.mode = mode
        self.delay_seconds = delay_seconds
        self.max_count = max_count
        if seed is not None:
            random.seed(seed)

        self._count = 0

        # ── Mule network state (persists across transactions in one burst) ──
        # When mode == MULE_NETWORK, we generate a "ring" lazily each reset.
        self._mule_ring: list[str] = []          # The 4-6 sending user IDs
        self._mule_collector: str = ""           # The single merchant destination
        self._mule_base_step: int = 0            # Step at which burst started
        self._mule_txn_index: int = 0            # How far into the ring we are
        self._reset_mule_ring()

        # ── ATO state — the "compromised" user persists across the scenario ──
        self._ato_user: str = _random_user_id()
        self._ato_known_ip: str = _random_ip()
        self._ato_known_device: str = _random_device()

    # ── Iterator protocol ─────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[TransactionMessage]:
        return self

    def __next__(self) -> TransactionMessage:
        if self.max_count is not None and self._count >= self.max_count:
            raise StopIteration

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        self._count += 1

        if self.mode == TrafficMode.NORMAL:
            return self._generate_normal()
        elif self.mode == TrafficMode.MULE_NETWORK:
            return self._generate_mule_network()
        elif self.mode == TrafficMode.ACCOUNT_TAKEOVER:
            return self._generate_account_takeover()
        else:
            raise ValueError(f"Unknown traffic mode: {self.mode}")

    # ── Mode: NORMAL ──────────────────────────────────────────────────────────

    def _generate_normal(self) -> TransactionMessage:
        """
        Realistic PaySim-style transaction.
        ~0.13% are organically fraudulent (higher amount, zero new balance on orig).
        These are NOT labelled — Agent 1 must detect them.
        """
        txn_type = _sample_type()
        amount   = _random_amount(txn_type)
        old_orig, new_orig, old_dest, new_dest = _make_balances(amount)

        is_organic_fraud = random.random() < _ORGANIC_FRAUD_RATE
        if is_organic_fraud:
            # Fraud signature: balance wiped to 0 after transfer/cash-out
            new_orig = 0.0
            txn_type = random.choice([TransactionType.TRANSFER, TransactionType.CASH_OUT])
            amount   = old_orig  # Drain the full balance

        name_dest = (
            _random_user_id()
            if txn_type in (TransactionType.TRANSFER, TransactionType.CASH_OUT)
            else _random_merchant_id()
        )

        return TransactionMessage(
            transaction_id=str(uuid.uuid4()),
            traffic_mode=TrafficMode.NORMAL,
            step=random.randint(1, _MAX_STEP),
            type=txn_type,
            amount=round(amount, 2),
            nameOrig=_random_user_id(),
            nameDest=name_dest,
            oldbalanceOrg=old_orig,
            newbalanceOrig=new_orig,
            oldbalanceDest=old_dest,
            newbalanceDest=new_dest,
            ip_address=_random_ip(),
            device_id=_random_device(),
        )

    # ── Mode: MULE_NETWORK ────────────────────────────────────────────────────

    def _reset_mule_ring(self) -> None:
        """Create a fresh coordinated mule ring (4-6 senders → 1 collector)."""
        n          = random.randint(4, 6)
        self._mule_ring      = [_random_user_id() for _ in range(n)]
        self._mule_collector = _random_merchant_id()
        self._mule_base_step = random.randint(1, _MAX_STEP - 10)
        self._mule_txn_index = 0

    def _generate_mule_network(self) -> TransactionMessage:
        """
        Coordinated burst: each sender in the ring sends to the same collector
        within a 5-step window. After all ring members transact, reset the ring.

        The pattern is structurally visible:
          - All nameDest values are the same merchant
          - step values cluster within a narrow window
          - amount is large (draining sender balance)
        """
        if self._mule_txn_index >= len(self._mule_ring):
            self._reset_mule_ring()

        sender = self._mule_ring[self._mule_txn_index]
        self._mule_txn_index += 1

        step    = self._mule_base_step + (self._mule_txn_index % 5)
        amount  = round(random.uniform(50_000, 400_000), 2)
        old_orig = round(amount + random.uniform(0, 10_000), 2)   # near-full drain
        new_orig = 0.0

        old_dest = round(random.uniform(0, 100_000), 2)
        new_dest = round(old_dest + amount, 2)

        return TransactionMessage(
            transaction_id=str(uuid.uuid4()),
            traffic_mode=TrafficMode.MULE_NETWORK,
            step=min(step, _MAX_STEP),
            type=TransactionType.TRANSFER,
            amount=amount,
            nameOrig=sender,
            nameDest=self._mule_collector,   # ← same collector every time
            oldbalanceOrg=old_orig,
            newbalanceOrig=new_orig,          # ← drained to 0
            oldbalanceDest=old_dest,
            newbalanceDest=new_dest,
            ip_address=_random_ip(),          # each mule may use different infra
            device_id=_random_device(),
        )

    # ── Mode: ACCOUNT_TAKEOVER ────────────────────────────────────────────────

    def _generate_account_takeover(self) -> TransactionMessage:
        """
        A single "compromised" user suddenly transacts from a NEW ip/device
        with a large amount to a NEW payee they've never used before.

        ATO signals:
          - nameOrig is always the same victim account
          - ip_address and device_id are NOT self._ato_known_ip / device
          - amount is disproportionately large
          - nameDest is a freshly generated account (unknown payee)
        """
        amount   = round(random.uniform(200_000, 800_000), 2)
        old_orig = round(amount + random.uniform(1000, 20_000), 2)
        new_orig = round(old_orig - amount, 2)
        old_dest = 0.0
        new_dest = amount

        # Use a "suspicious" new IP/device — not the known ones
        suspicious_ip     = f"ip_{random.randint(1, _N_IPS)}"
        suspicious_device = f"device_{random.randint(1, _N_DEVICES)}"
        # Ensure they differ from known good ones (negligible collision risk at 2000 nodes)
        while suspicious_ip == self._ato_known_ip:
            suspicious_ip = f"ip_{random.randint(1, _N_IPS)}"
        while suspicious_device == self._ato_known_device:
            suspicious_device = f"device_{random.randint(1, _N_DEVICES)}"

        return TransactionMessage(
            transaction_id=str(uuid.uuid4()),
            traffic_mode=TrafficMode.ACCOUNT_TAKEOVER,
            step=random.randint(1, _MAX_STEP),
            type=random.choice([TransactionType.TRANSFER, TransactionType.CASH_OUT]),
            amount=amount,
            nameOrig=self._ato_user,               # ← always same victim
            nameDest=_random_user_id(),            # ← unknown new payee
            oldbalanceOrg=old_orig,
            newbalanceOrig=new_orig,
            oldbalanceDest=old_dest,
            newbalanceDest=new_dest,
            ip_address=suspicious_ip,              # ← new / suspicious
            device_id=suspicious_device,           # ← new / suspicious
        )
