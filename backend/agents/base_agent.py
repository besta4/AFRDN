"""
base_agent.py — Abstract base class for all Jatayu pipeline agents.

Every agent must implement:
    _process(msg: TransactionMessage) -> TransactionMessage

The base class provides:
    - Standard entry/exit logging with agent name + transaction_id
    - Contract enforcement: exceptions in _process() are caught, logged, and
      recorded in msg.pipeline_metadata — they never propagate to the caller
    - Per-agent AgentMeta provenance entry appended to msg.pipeline_metadata
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from agents.models import AgentMeta, TransactionMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all Jatayu pipeline agents.

    Subclasses must implement `_process()`. They should read from `msg`,
    add their own fields, and return the enriched message.
    """

    name: str = "BaseAgent"

    def __call__(self, msg: TransactionMessage) -> TransactionMessage:
        """Allow agents to be called directly as agent(msg)."""
        return self.process(msg)

    def process(self, msg: TransactionMessage) -> TransactionMessage:
        """
        Process the transaction message and return an enriched copy.

        Enforces the BaseAgent contract:
          - Calls _process() and measures wall-clock latency.
          - If _process() raises, logs the error and appends an AgentMeta
            entry with status="error" — the exception never propagates.
          - Always appends an AgentMeta entry to msg.pipeline_metadata.

        Subclasses should override _process(), NOT this method.
        """
        logger.debug("[%s] → txn=%s", self.name, msg.transaction_id)
        t0 = time.monotonic()
        status = "ok"
        error_str = None
        try:
            result = self._process(msg)
        except Exception as exc:
            logger.error(
                "[%s] Unhandled exception in _process() for txn=%s: %s",
                self.name,
                msg.transaction_id,
                exc,
                exc_info=True,
            )
            status = "error"
            error_str = str(exc)
            result = msg  # return unchanged msg; downstream agents see None fields
        finally:
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            msg.pipeline_metadata.append(
                AgentMeta(
                    agent_name=self.name,
                    status=status,
                    latency_ms=latency_ms,
                    error=error_str,
                )
            )
        logger.debug("[%s] ← txn=%s (%.1f ms)", self.name, msg.transaction_id, latency_ms)
        return result

    @abstractmethod
    def _process(self, msg: TransactionMessage) -> TransactionMessage:
        """
        Agent-specific processing logic. Must be implemented by each subclass.

        Contract:
          - Reads any fields it needs from `msg`
          - Populates ONLY the fields it owns (see models.py for ownership)
          - Returns the same `msg` object (mutated in-place is fine)
          - Should not raise — BaseAgent.process() is the safety net, but
            agents are expected to handle their own recoverable errors locally
            and set status="fallback" on the AgentMeta entry via
            self._record_fallback(msg, reason) when switching to a fallback path
        """
        ...

    # ── Provenance helpers for subclass use ───────────────────────────────────

    def _record_fallback(self, msg: TransactionMessage, reason: str) -> None:
        """
        Mark the most recent AgentMeta entry as a fallback with the given reason.

        Call this when an agent degrades to a fallback path (e.g. PPO → rules)
        so that provenance records the degradation without it appearing as an error.
        """
        for meta in reversed(msg.pipeline_metadata):
            if meta.agent_name == self.name:
                meta.status = "fallback"
                meta.error = reason
                return
