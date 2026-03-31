"""
pattern_detection_agent.py — Agent 2: Pattern Detection

Analyzes a rolling window of recently flagged transactions to detect
coordinated attack patterns (mule networks, ATO, velocity spikes).

The ROLLING BUFFER is real (deque of up to 20 flagged TransactionMessages).
Pattern detection is rule-based, with an optional LLM used only to generate
natural-language reasoning for detected patterns.
"""

from __future__ import annotations

import json
import logging
from collections import deque, defaultdict
from typing import Deque, Dict, Set

import requests

from agents.base_agent import BaseAgent
from agents.models import PatternType, TransactionMessage

logger = logging.getLogger(__name__)

WINDOW_SIZE = 20   # Rolling buffer depth (must match orchestrator config)


class PatternDetectionAgent(BaseAgent):
    """
    Agent 2 — Pattern Detection.

    Reads from TransactionMessage:
        msg.fraud_label       → only flagged transactions are analysed
        msg.nameOrig          → sender identity
        msg.nameDest          → receiver / collector identity
        msg.step              → for timing-based pattern matching
        msg.ip_address        → for ATO detection
        msg.device_id         → for ATO detection
        msg.traffic_mode      → (used in stub; real model ignores this)

    Writes to TransactionMessage:
        msg.pattern_type         → PatternType enum
        msg.pattern_confidence   → float ∈ [0, 1]
        msg.window_snapshot      → list of last ≤5 txn_ids from the buffer
    """

    name = "PatternDetectionAgent"

    def __init__(self, generate_reasoning: bool = True) -> None:
        # ── Real rolling window buffer ────────────────────────────────────────
        # Holds up to WINDOW_SIZE flagged TransactionMessage objects.
        # The Orchestrator appends flagged messages BEFORE calling process().
        self.buffer: Deque[TransactionMessage] = deque(maxlen=WINDOW_SIZE)

        # DONE: At init, also build a per-user device/IP history tracker for
        # ATO detection. Something like:
        #   self._user_device_history: dict[str, set[str]] = defaultdict(set)
        #   self._user_ip_history:     dict[str, set[str]] = defaultdict(set)
        # Update these after each processed transaction.
        self._user_device_history: Dict[str, Set[str]] = defaultdict(set)
        self._user_ip_history: Dict[str, Set[str]] = defaultdict(set)

        # Set to False during offline training to skip LLM calls entirely.
        # pattern_reasoning is not used by the PPO state vector or reward fn.
        self._generate_reasoning = generate_reasoning

        logger.info("[%s] Initialized with window_size=%d", self.name, WINDOW_SIZE)

    def _process(self, msg: TransactionMessage) -> TransactionMessage:
        try:
            # Take a snapshot of current buffer state (up to last 5 IDs)
            snapshot = [m.transaction_id for m in list(self.buffer)[-5:]]
            msg.window_snapshot = snapshot
            
            # Log buffer state for observability
            logger.info(
                "[%s] Processing txn=%s | Buffer size=%d | Sender=%s | Receiver=%s | Amount=₹%.0f",
                self.name, msg.transaction_id[:12] if msg.transaction_id else "?",
                len(self.buffer), msg.nameOrig, msg.nameDest, msg.amount or 0
            )

            pattern, confidence = self._detect_pattern(msg)
            msg.pattern_type = pattern
            msg.pattern_confidence = confidence
            
            # Log detection result
            if pattern is not None and pattern != PatternType.NONE:
                logger.warning(
                    "[%s] 🚨 PATTERN DETECTED: %s (confidence=%.2f) | txn=%s",
                    self.name, pattern.value, confidence, msg.transaction_id[:12] if msg.transaction_id else "?"
                )
            else:
                logger.debug(
                    "[%s] No pattern detected for txn=%s",
                    self.name, msg.transaction_id[:12] if msg.transaction_id else "?"
                )

            # Update the most recent AgentMeta entry with detected confidence.
            for meta in reversed(msg.pipeline_metadata):
                if meta.agent_name == self.name:
                    meta.confidence = confidence
                    break

            # Use the LLM only to generate human-readable reasoning for
            # patterns that the rule-based detector has already identified.
            # Skipped when generate_reasoning=False (e.g. offline RL training).
            if self._generate_reasoning and pattern is not None and pattern is not PatternType.NONE:
                logger.info("[%s] Generating LLM reasoning for %s pattern...", self.name, pattern.value)
                msg.pattern_reasoning = self._call_llm(msg, pattern, confidence)
                if msg.pattern_reasoning:
                    logger.info("[%s] LLM reasoning: %s", self.name, msg.pattern_reasoning[:100])
            else:
                msg.pattern_reasoning = None

        except Exception as exc:
            logger.error("[%s] Pattern detection failed: %s", self.name, exc, exc_info=True)
            # Safe defaults — leave downstream agents functional
            msg.pattern_type = PatternType.NONE
            msg.pattern_confidence = 0.0
            msg.pattern_reasoning = None
            self._record_fallback(msg, str(exc))

        finally:
            # History update must run regardless of detection outcome so that
            # device/IP tracking is never skipped when an exception occurs.
            if msg.nameOrig:
                if msg.device_id:
                    self._user_device_history[msg.nameOrig].add(msg.device_id)
                if msg.ip_address:
                    self._user_ip_history[msg.nameOrig].add(msg.ip_address)

        return msg

    def _detect_pattern(
        self, msg: TransactionMessage
    ) -> tuple[PatternType, float]:
        """
        Inspect the rolling buffer and the current message to detect patterns.

        ── CURRENT STATE ───────────────────────────────────────────────────────
        Rule-based pattern analysis using the rolling window and per-user
        device/IP history. The LLM is not used for classification, only for
        generating reasoning after a pattern is detected.

        ── Rules implemented ───────────────────────────────────────────────────
        MULE_NETWORK detection:
          - Count how many unique nameOrig values in the buffer share the same nameDest
            as the current transaction.
          - If ≥ 3 senders → same collector within last 5 steps → MULE_NETWORK.
          - Confidence = n_senders / WINDOW_SIZE capped at 1.0.

        ACCOUNT_TAKEOVER detection:
          - Maintain per-user IP/device history (self._user_ip_history etc.)
          - If current msg.ip_address ∉ history[msg.nameOrig] → suspicious.
          - If current msg.device_id ∉ history[msg.nameOrig] → suspicious.
          - Combine with high fraud_score for ATO label.
          - Update history after detection.

        VELOCITY_SPIKE detection:
          - Count occurrences of msg.nameOrig in the last 10 buffer entries.
          - If count ≥ 3 → VELOCITY_SPIKE.
          - Confidence = count / 10.

        NONE:
          - Default — no coordinated pattern found.
        """

        # Default: no coordinated pattern found.
        pattern: PatternType = PatternType.NONE
        confidence: float = 0.0

        def _is_merchant(dest: str) -> bool:
            # PaySim-style merchant IDs typically start with "M".
            # We treat this as a weak heuristic (no hard dependency on schema).
            return bool(dest) and dest.startswith("M")

        # ── Rule: MULE_NETWORK ────────────────────────────────────────────────
        # MULE_NETWORK: ≥3 unique senders converging on same nameDest within last 5 steps
        if msg.nameDest and msg.step is not None:
            recent_window = [
                m
                for m in self.buffer
                if m.nameDest == msg.nameDest and (msg.step - m.step) <= 5
            ]
            unique_senders = {m.nameOrig for m in recent_window if m.nameOrig}
            n_unique_senders = len(unique_senders)
            
            logger.debug(
                "[%s] MULE_NETWORK check: nameDest=%s, buffer_matches=%d, unique_senders=%d (need ≥3)",
                self.name, msg.nameDest, len(recent_window), n_unique_senders
            )
            
            if n_unique_senders >= 3:
                # Confidence scales from 0.65 (3 senders) to 1.0 (≥5 senders)
                # 3 senders = minimum trigger = 0.65 base confidence
                mule_confidence = min(1.0, 0.65 + 0.10 * (n_unique_senders - 3))
                pattern = PatternType.MULE_NETWORK
                confidence = mule_confidence
                logger.info(
                    "[%s] ✓ MULE_NETWORK triggered: %d unique senders → %s (conf=%.2f)",
                    self.name, n_unique_senders, msg.nameDest, mule_confidence
                )

        # ── Rule: ACCOUNT_TAKEOVER ────────────────────────────────────────────
        # ACCOUNT_TAKEOVER: novel IP/device for a *known* user + high fraud_score
        #
        # Important: We only treat an IP/device as "new" if we have prior history
        # for that user. Otherwise, every first-seen customer would look like ATO,
        # which creates false positives in merchant-heavy traffic.
        fraud_score = msg.fraud_score or 0.0
        if msg.nameOrig:
            known_ips = self._user_ip_history[msg.nameOrig]
            known_devices = self._user_device_history[msg.nameOrig]
            has_history = bool(known_ips) or bool(known_devices)
            new_ip = bool(msg.ip_address and msg.ip_address not in known_ips)
            new_device = bool(msg.device_id and msg.device_id not in known_devices)
            
            logger.debug(
                "[%s] ATO check: user=%s, has_history=%s, new_ip=%s, new_device=%s, fraud_score=%.2f (need ≥0.7)",
                self.name, msg.nameOrig, has_history, new_ip, new_device, fraud_score
            )
            
            if has_history and (new_ip or new_device) and fraud_score >= 0.7:
                # Scale confidence with fraud_score to emphasize highly suspicious cases.
                ato_confidence = min(1.0, 0.5 + 0.5 * fraud_score)
                # Prefer ACCOUNT_TAKEOVER over previous pattern if more confident.
                if ato_confidence > confidence:
                    pattern = PatternType.ACCOUNT_TAKEOVER
                    confidence = ato_confidence
                    logger.info(
                        "[%s] ✓ ACCOUNT_TAKEOVER triggered: user=%s, new_ip=%s, new_device=%s (conf=%.2f)",
                        self.name, msg.nameOrig, new_ip, new_device, ato_confidence
                    )

        # ── Rule: VELOCITY_SPIKE ──────────────────────────────────────────────
        # VELOCITY_SPIKE: same nameOrig appears ≥3 times in last 10 buffer entries
        if msg.nameOrig:
            last_ten = list(self.buffer)[-10:]
            count_same_origin = sum(1 for m in last_ten if m.nameOrig == msg.nameOrig)
            
            logger.debug(
                "[%s] VELOCITY check: user=%s, occurrences_in_last_10=%d (need ≥3)",
                self.name, msg.nameOrig, count_same_origin
            )
            
            if count_same_origin >= 3:
                # Confidence scales from 0.65 (3 txns) to 1.0 (≥7 txns in last 10)
                # 3 transactions = minimum trigger = 0.65 base confidence
                velocity_confidence = min(1.0, 0.65 + 0.10 * (count_same_origin - 3))
                # Only override existing pattern if this rule is more confident.
                if velocity_confidence > confidence:
                    pattern = PatternType.VELOCITY_SPIKE
                    confidence = velocity_confidence
                    logger.info(
                        "[%s] ✓ VELOCITY_SPIKE triggered: user=%s, %d txns in last 10 (conf=%.2f)",
                        self.name, msg.nameOrig, count_same_origin, velocity_confidence
                    )

        return pattern, confidence

    # ── Observable signal helpers (used by Orchestrator → Agent 3) ───────────

    def is_new_device(self, nameOrig: str, device_id: str) -> bool:
        """
        Return True iff device_id has never been seen for this account AND
        the account has prior device history (so we are not flagging first-time
        customers as suspicious simply because we have no record of them).
        """
        if not nameOrig or not device_id:
            return False
        history = self._user_device_history.get(nameOrig)
        if not history:
            return False  # no history → cannot classify as new vs known
        return device_id not in history

    def is_new_ip(self, nameOrig: str, ip_address: str) -> bool:
        """
        Return True iff ip_address has never been seen for this account AND
        the account has prior IP history.
        """
        if not nameOrig or not ip_address:
            return False
        history = self._user_ip_history.get(nameOrig)
        if not history:
            return False
        return ip_address not in history

    def _call_llm(
        self,
        msg: TransactionMessage,
        pattern: PatternType,
        confidence: float,
        timeout: int = 15,  # Reduced timeout - 15 seconds max
    ) -> str:
        """
        Call the local LFM2.5-1.2B-Instruct model via llama.cpp server
        to generate natural-language reasoning for an already-detected pattern.

        The server exposes an OpenAI-compatible chat API at:
            http://localhost:8080/v1/chat/completions

        Start the server with: python run_llama_server.py

        The model must respond ONLY with:
            {"reasoning": "<one sentence>"}

        Robust to timeouts / parse errors — falls back to pattern-specific default.
        """
        try:
            # Build compact JSON summary of last 10 transactions in the buffer.
            window = list(self.buffer)[-10:]

            def _txn_summary(t: TransactionMessage) -> dict:
                return {
                    "transaction_id": t.transaction_id,
                    "nameOrig": t.nameOrig,
                    "nameDest": t.nameDest,
                    "amount": t.amount,
                    "step": t.step,
                    "ip_address": t.ip_address,
                    "device_id": t.device_id,
                    "fraud_score": t.fraud_score,
                }

            window_payload = [_txn_summary(t) for t in window]
            current_payload = _txn_summary(msg)

            user_content = json.dumps(
                {
                    "pattern_type": pattern.value,
                    "pattern_confidence": confidence,
                    "window": window_payload,
                    "current_transaction": current_payload,
                },
                separators=(",", ":"),
            )

            system_message = (
                "You are a fraud pattern explanation model. "
                "Given a detected pattern label and supporting evidence from a "
                "rolling transaction window, explain in one concise sentence "
                "why this pattern label makes sense. Do not change the label."
            )

            user_instruction = (
                "Here is a JSON summary of the detected pattern, the last 10 "
                "transactions in the buffer, and the current transaction. "
                "Respond ONLY with a single JSON object of the form: "
                '{"reasoning": "<one sentence>"}.\n'
                "Do not include any extra text outside the JSON object.\n"
                "Data:\n"
                f"{user_content}"
            )

            payload = {
                "model": "LFM2.5-1.2B-Instruct",  # LiquidAI model via llama.cpp
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_instruction},
                ],
                "temperature": 0.2,
                "max_tokens": 256,
                "stream": False,
            }

            response = requests.post(
                "http://localhost:8080/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Some local LLMs may wrap the JSON in extra text or code fences.
            # Extract the first top-level JSON object from the content string.
            def _extract_json_object(raw: str) -> str:
                start = raw.find("{")
                if start == -1:
                    raise ValueError("No JSON object found in LLM response.")
                depth = 0
                for idx, ch in enumerate(raw[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return raw[start : idx + 1]
                raise ValueError("Unbalanced JSON object in LLM response.")

            content_stripped = content.strip()
            json_str = _extract_json_object(content_stripped)
            parsed = json.loads(json_str)
            reasoning = str(parsed.get("reasoning", "")).strip()

            logger.debug(
                "[%s] LLM reasoning for pattern=%s confidence=%.3f: %s",
                self.name,
                pattern.value,
                confidence,
                reasoning,
            )

            return reasoning

        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "[%s] LLM reasoning generation failed (%s). Using fallback reasoning.",
                self.name,
                exc,
            )
            # Provide pattern-specific fallback reasoning
            fallback_reasons = {
                PatternType.MULE_NETWORK: f"Multiple unique senders converging on same destination detected with {confidence:.0%} confidence.",
                PatternType.VELOCITY_SPIKE: f"Rapid transaction velocity from same sender detected with {confidence:.0%} confidence.",
                PatternType.ACCOUNT_TAKEOVER: f"Suspicious activity from new device/IP for established user detected with {confidence:.0%} confidence.",
            }
            return fallback_reasons.get(pattern, f"Pattern {pattern.value} detected with {confidence:.0%} confidence.")
