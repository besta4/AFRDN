"""
alert_block_agent.py — Agent 4: Alert & Block

Executes the recommended action from Agent 3 and generates a plain-English
explanation of the decision for downstream systems / human review.

Current implementation: stub explanation generator.
TODO: Replace explanation with an LLM-generated narrative.
"""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from agents.models import (
    Action,
    PatternType,
    RiskLevel,
    TransactionMessage,
)

logger = logging.getLogger(__name__)


class AlertBlockAgent(BaseAgent):
    """
    Agent 4 — Alert & Block.

    Reads from TransactionMessage:
        msg.recommended_action  (from Agent 3)
        msg.fraud_score         (from Agent 1)
        msg.top_features        (from Agent 1)
        msg.pattern_type        (from Agent 2)
        msg.risk_level          (from Agent 3)

    Writes to TransactionMessage:
        msg.action_taken   → Action enum (mirrors recommended_action in stub)
        msg.explanation    → plain-English decision rationale
    """

    name = "AlertBlockAgent"

    def __init__(self) -> None:
        # TODO: Initialize LLM client here (e.g. google.generativeai or openai).
        # self.llm_client = genai.GenerativeModel("gemini-1.5-flash")
        # Keep the client alive across transactions to avoid repeated auth overhead.
        logger.info("[%s] Initialized (stub explanation mode)", self.name)

    def _process(self, msg: TransactionMessage) -> TransactionMessage:
        # Execute the recommended action
        # TODO: If action == BLOCK, call payment gateway API to actually block.
        # TODO: If action == HOLD, push to a human review queue.
        # TODO: If action == SILENT_FLAG, write to a fraud ops dashboard topic.
        msg.action_taken = msg.recommended_action or Action.PASS

        # Generate explanation
        msg.explanation = self._explain(msg)
        return msg

    def _explain(self, msg: TransactionMessage) -> str:
        """
        Generate a plain-English explanation of the fraud decision.

        ── CURRENT STATE: STUB (template-based) ─────────────────────────────────
        The stub builds a readable sentence from structured fields.

        ── TODO: Replace with LLM call ──────────────────────────────────────────
        Replace the template string below with a call like:

            prompt = f\"\"\"
            You are a fraud analyst AI. Explain the following decision in 2-3 sentences
            for a compliance officer. Be specific about the evidence.

            Transaction: {msg.amount:.2f} from {msg.nameOrig} to {msg.nameDest}
            Fraud score: {msg.fraud_score:.4f} (threshold: 0.0224)
            Top contributing features: {', '.join(msg.top_features or [])}
            Detected pattern: {msg.pattern_type}
            Risk level: {msg.risk_level}
            Action taken: {msg.action_taken}
            \"\"\"
            response = self.llm_client.generate_content(prompt)
            return response.text.strip()

        Keep LLM latency in mind — for real-time blocking this should run
        asynchronously or use a streaming response.
        """

        # ── STUB explanation template ─────────────────────────────────────────
        score_pct  = f"{(msg.fraud_score or 0) * 100:.1f}%"
        action     = (msg.action_taken or Action.PASS).value
        pattern    = (msg.pattern_type or PatternType.NONE).value
        risk       = (msg.risk_level or RiskLevel.LOW).value
        top        = ", ".join(msg.top_features or []) or "N/A"
        amount     = f"{msg.amount:,.2f}"

        template = (
            f"Transaction of {msg.type.value if hasattr(msg.type, 'value') else msg.type} "
            f"${amount} from {msg.nameOrig} to {msg.nameDest} received a fraud probability "
            f"of {score_pct}. "
            f"Top contributing signals: [{top}]. "
            f"Pattern analysis flagged this as {pattern} with risk level {risk}. "
            f"Action taken: {action}."
        )
        return template
        # ── END STUB ──────────────────────────────────────────────────────────
