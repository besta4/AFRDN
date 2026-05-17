"""
orchestrator.py — Jatayu Agent Pipeline Orchestrator

Connects all 5 specialized agents in sequence and manages the shared
rolling window buffer for Agent 2 (PatternDetectionAgent).

Pipeline order:
    Generator → Agent1 → Agent2 → Agent3 → Agent4 → Agent5

Enhanced with:
  - Redis-backed DynamicGraphCache for real-time graph embeddings
  - Graph context recording after Agent 1 (before pattern detection)

Usage:
    generator = SyntheticTrafficGenerator(mode=TrafficMode.MULE_NETWORK)
    orchestrator = Orchestrator()
    for msg in orchestrator.run(generator):
        print(msg.to_dict())
"""

from __future__ import annotations

import logging
import time
from typing import Generator, Iterator

from config import load_environment
from agents.alert_block_agent import AlertBlockAgent
from agents.compliance_logging_agent import ComplianceLoggingAgent
from agents.models import TransactionMessage
from agents.pattern_detection_agent import PatternDetectionAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.transaction_monitoring_agent import TransactionMonitoringAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Runs each transaction through the full 5-agent pipeline sequentially.

    Agent 2's rolling window buffer is owned here and fed before each
    Agent 2 call: flagged messages (fraud_label=True) are appended AFTER
    Agent 1 runs so the buffer always reflects the latest scored state.

    Error handling:
        Per-transaction exceptions are caught and logged. The pipeline
        continues to the next transaction — a single bad message never
        crashes the stream.
    """

    def __init__(self) -> None:
        import os
        load_environment()

        # ── Initialize Redis-backed dynamic graph cache ───────────────────
        # Gracefully degrades to zero-vectors if Redis is unavailable.
        self._graph_cache = None
        try:
            from agents.graph_embeddings_cache import DynamicGraphCache
            self._graph_cache = DynamicGraphCache()
            if self._graph_cache.available:
                logger.info("[Orchestrator] Redis dynamic graph cache: ENABLED")
            else:
                logger.warning("[Orchestrator] Redis dynamic graph cache: DISABLED (Redis unavailable)")
        except Exception as exc:
            logger.warning("[Orchestrator] Failed to initialize graph cache: %s", exc)

        # ── Initialize all 5 agents ───────────────────────────────────────
        self.agent1 = TransactionMonitoringAgent(graph_cache=self._graph_cache)
        self.agent2 = PatternDetectionAgent(graph_cache=self._graph_cache)

        # PPO is available as an opt-in policy. The rule engine is the default
        # production path because stale checkpoints can over-block normal UPI
        # traffic if they are not recalibrated.
        _policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "risk_ppo_2.pt")
        _enable_ppo = os.getenv("JATAYU_ENABLE_PPO", "0").strip().lower() in {"1", "true", "yes", "on"}
        if _enable_ppo and os.path.exists(_policy_path):
            self.agent3 = RiskAssessmentAgent(use_rl=True, policy_path=_policy_path)
            logger.info("[Orchestrator] Agent 3 loaded PPO policy from %s", _policy_path)
        else:
            if os.path.exists(_policy_path):
                logger.info("[Orchestrator] PPO policy available but disabled. Set JATAYU_ENABLE_PPO=1 to opt in.")
            else:
                logger.warning("[Orchestrator] risk_ppo_2.pt not found at %s — using rule-based Agent 3.", _policy_path)
            self.agent3 = RiskAssessmentAgent()

        self.agent4 = AlertBlockAgent()
        self.agent5 = ComplianceLoggingAgent()

        # Consistent reference to Agent 2's buffer (shared object)
        self.rolling_buffer = self.agent2.buffer

        logger.info("[Orchestrator] All 5 agents initialized.")

    @property
    def graph_cache(self):
        """Expose graph cache for external callers (e.g., transaction router)."""
        return self._graph_cache

    def run(
        self, generator: Iterator[TransactionMessage]
    ) -> Generator[TransactionMessage, None, None]:
        """
        Consume transactions from `generator` and yield fully enriched messages.

        Args:
            generator: Any iterator yielding TransactionMessage objects.

        Yields:
            TransactionMessage — fully enriched with all agent outputs.
        """
        for msg in generator:
            try:
                msg = self._process_one(msg)
                yield msg
            except Exception as exc:
                logger.error(
                    "[Orchestrator] Pipeline error on txn=%s: %s",
                    getattr(msg, "transaction_id", "unknown"),
                    exc,
                    exc_info=True,
                )
                # Yield the partially enriched message so the demo can still show it
                yield msg

    def _process_one(self, msg: TransactionMessage) -> TransactionMessage:
        """Run a single message through all 5 agents."""
        msg.pipeline_start_ms = time.time() * 1000

        # ── Agent 1 — Transaction Monitoring ─────────────────────────────────
        msg = self.agent1.process(msg)

        # ── Record transaction in Redis graph cache ──────────────────────────
        # This must happen AFTER Agent 1 scores the transaction but BEFORE
        # Agent 2 runs pattern detection, so that the current transaction's
        # graph context is available for convergence/velocity queries.
        if self._graph_cache and self._graph_cache.available:
            try:
                self._graph_cache.record_transaction(
                    sender_id=msg.nameOrig,
                    receiver_id=msg.nameDest,
                    amount=msg.amount,
                    transaction_id=msg.transaction_id,
                )
            except Exception as exc:
                logger.debug("[Orchestrator] Graph cache record failed: %s", exc)

        # ── Feed rolling buffer with ALL transactions ─────────────────────────
        # All transactions are added to the pattern detection window regardless
        # of fraud_label. This enables pattern detection (mule networks, velocity
        # spikes, etc.) to analyze the full transaction stream.
        self.rolling_buffer.append(msg)

        # ── Agent 2 — Pattern Detection ────────────────────────────────────────
        msg = self.agent2.process(msg)

        # ── Record pattern suspicion into decay engine ────────────────────────
        # After Agent 2 detects a pattern, store the confidence as a decaying
        # risk score. This ensures historical suspicion fades over time via
        # R(t) = R₀ × e^{-λt} instead of persisting indefinitely in Redis.
        if self._graph_cache and self._graph_cache.available:
            try:
                from agents.models import PatternType
                pat = msg.pattern_type
                conf = msg.pattern_confidence or 0.0
                if pat and pat != PatternType.NONE and conf > 0.0:
                    # Map pattern type → decay tier
                    if pat == PatternType.VELOCITY_SPIKE:
                        tier = "velocity"
                    elif pat in (PatternType.MULE_NETWORK, PatternType.ACCOUNT_TAKEOVER,
                                 PatternType.CIRCULAR_FLOW):
                        tier = "pattern"
                    else:
                        tier = "pattern"

                    # Record for BOTH sender and receiver
                    if msg.nameOrig:
                        self._graph_cache.record_risk_score(
                            msg.nameOrig, tier, conf, accumulate=True
                        )
                    if msg.nameDest and not msg.nameDest.startswith("M"):
                        self._graph_cache.record_risk_score(
                            msg.nameDest, tier, conf * 0.5, accumulate=True
                        )

                    # Also record network-tier risk for graph topology suspicion
                    if pat == PatternType.MULE_NETWORK and conf >= 0.70:
                        if msg.nameOrig:
                            self._graph_cache.record_risk_score(
                                msg.nameOrig, "network", conf * 0.6, accumulate=True
                            )
                        if msg.nameDest and not msg.nameDest.startswith("M"):
                            self._graph_cache.record_risk_score(
                                msg.nameDest, "network", conf * 0.4, accumulate=True
                            )
            except Exception as exc:
                logger.debug("[Orchestrator] Risk decay recording failed: %s", exc)

        # ── Gather observable device/IP signals from Agent 2 for Agent 3 ──────
        account_hints = {
            "is_new_device": self.agent2.is_new_device(msg.nameOrig, msg.device_id),
            "is_new_ip":     self.agent2.is_new_ip(msg.nameOrig, msg.ip_address),
        }

        # ── Inject decayed historical risk into account hints ─────────────────
        # Agent 3 uses this to factor in historical suspicion when making
        # enforcement decisions. The decay ensures old suspicion fades.
        if self._graph_cache and self._graph_cache.available and msg.nameOrig:
            try:
                composite_risk = self._graph_cache.get_composite_risk(msg.nameOrig)
                account_hints["historical_risk"] = composite_risk
                account_hints["decayed_risk_tiers"] = self._graph_cache.get_decayed_risk(msg.nameOrig)
            except Exception:
                account_hints["historical_risk"] = 0.0

        # ── Agent 3 — Risk Assessment ─────────────────────────────────────────
        msg = self.agent3.process(msg, account_hints=account_hints)

        # ── Agent 4 — Alert & Block ───────────────────────────────────────────
        msg = self.agent4.process(msg)

        # ── Stamp decision pipeline end time before compliance logging ─────────
        msg.pipeline_end_ms = time.time() * 1000

        # ── Agent 5 — Compliance Logging ──────────────────────────────────────
        msg = self.agent5.process(msg)

        return msg

    @property
    def audit_trail(self):
        """Convenience accessor to Agent 5's in-memory audit trail."""
        return self.agent5.audit_trail

    async def process_single(self, msg: TransactionMessage) -> TransactionMessage:
        """
        Process a single transaction through the pipeline (async wrapper).

        Used by the real-time transaction API for synchronous fraud checking.
        """
        return self._process_one(msg)
