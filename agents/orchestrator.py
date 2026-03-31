"""
orchestrator.py — Jatayu Agent Pipeline Orchestrator

Connects all 5 specialized agents in sequence and manages the shared
rolling window buffer for Agent 2 (PatternDetectionAgent).

Pipeline order:
    Generator → Agent1 → Agent2 → Agent3 → Agent4 → Agent5

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
        self.agent1 = TransactionMonitoringAgent()
        self.agent2 = PatternDetectionAgent()

        # Load the trained PPO policy (risk_ppo_2.pt sits next to main.py).
        # Resolve relative to this file so it works regardless of cwd.
        _policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "risk_ppo_2.pt")
        if os.path.exists(_policy_path):
            self.agent3 = RiskAssessmentAgent(use_rl=True, policy_path=_policy_path)
            logger.info("[Orchestrator] Agent 3 loaded PPO policy from %s", _policy_path)
        else:
            logger.warning("[Orchestrator] risk_ppo_2.pt not found at %s — using rule-based Agent 3.", _policy_path)
            self.agent3 = RiskAssessmentAgent()

        self.agent4 = AlertBlockAgent()
        self.agent5 = ComplianceLoggingAgent()

        # Consistent reference to Agent 2's buffer (shared object)
        self.rolling_buffer = self.agent2.buffer

        logger.info("[Orchestrator] All 5 agents initialized.")

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

        # ── Feed rolling buffer with ALL transactions ─────────────────────────
        # All transactions are added to the pattern detection window regardless
        # of fraud_label. This enables pattern detection (mule networks, velocity
        # spikes, etc.) to analyze the full transaction stream.
        self.rolling_buffer.append(msg)

        # ── Agent 2 — Pattern Detection ────────────────────────────────────────
        msg = self.agent2.process(msg)

        # ── Gather observable device/IP signals from Agent 2 for Agent 3 ──────
        # Agent 2 maintains per-user device and IP history from prior
        # transactions.  Passing these as hints to Agent 3 replaces the
        # hard-coded is_new_device_stub=True / is_new_ip_stub=True stubs that
        # previously biased the risk policy toward ATO patterns universally.
        account_hints = {
            "is_new_device": self.agent2.is_new_device(msg.nameOrig, msg.device_id),
            "is_new_ip":     self.agent2.is_new_ip(msg.nameOrig, msg.ip_address),
        }

        # ── Agent 3 — Risk Assessment ─────────────────────────────────────────
        msg = self.agent3.process(msg, account_hints=account_hints)

        # ── Agent 4 — Alert & Block ───────────────────────────────────────────
        msg = self.agent4.process(msg)

        # ── Stamp decision pipeline end time before compliance logging ─────────
        # pipeline_end_ms marks the boundary between decision-making (Agents 1–4)
        # and audit logging (Agent 5).  ComplianceLoggingAgent reads this value
        # directly so that decision_latency_ms excludes Agent 5's own overhead.
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
