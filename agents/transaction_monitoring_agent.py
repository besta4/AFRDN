"""
transaction_monitoring_agent.py — Agent 1: Transaction Monitoring

Wraps the two-stage GNN + XGBoost model to produce a fraud score and list
of top contributing features for each transaction.

PLUG-IN POINT:
    The `score()` method is clearly marked for replacement with real model
    inference. The stub returns a simulated score for demo purposes only.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any

import requests

import numpy as np

from agents.base_agent import BaseAgent
from agents.models import TrafficMode, TransactionMessage, TransactionType

logger = logging.getLogger(__name__)

# ── Model artifact version (update when loading from real files) ──────────────
_MODEL_VERSION_STUB = "stub_v0.0 — plug real model in score()"

# ── Decision threshold from config_*.json ────────────────────────────────────
# DONE: Load dynamically from data/config_*.json at __init__ time when available.
FRAUD_THRESHOLD = 0.0224


class TransactionMonitoringAgent(BaseAgent):
    """
    Agent 1 — Transaction Monitoring.

    Reads raw transaction fields from TransactionMessage and writes:
        msg.fraud_score     → float ∈ [0, 1]
        msg.fraud_label     → bool  (score >= FRAUD_THRESHOLD)
        msg.top_features    → list[str] of top contributing feature names
        msg.model_version   → str identifier of the artifact set used

    To plug in the real model, replace the body of `score()` according to
    the step-by-step TODO comment there.
    """

    name = "TransactionMonitoringAgent"

    def __init__(self) -> None:
        # Try to load real model artifacts at __init__ time. If anything fails,
        # fall back to stub mode so the demo pipeline remains functional.
        self._stub_mode = True
        self.threshold = FRAUD_THRESHOLD
        self.model_version = _MODEL_VERSION_STUB
        self.global_max_step: int | None = None
        self.user_map: dict[str, int] | None = None
        self.embeddings: np.ndarray | None = None
        self.xgb_model: Any | None = None
        self.feature_names: list[str] | None = None
        self.feature_importances: np.ndarray | None = None
        self._xgb_module: Any | None = None

        try:
            try:
                import xgboost as xgb  # type: ignore
                self._xgb_module = xgb
            except Exception:  # noqa: BLE001
                self._xgb_module = None

            data_dir = Path(__file__).parent.parent / "data"
            config_paths = sorted(data_dir.glob("config_*.json"))
            if not config_paths:
                logger.warning(
                    "[%s] No config_*.json found in %s; staying in stub mode.",
                    self.name,
                    data_dir,
                )
            else:
                cfg_path = config_paths[-1]
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)

                files_cfg = cfg.get("files", {})

                # Mappings: user → index, plus other entity maps if needed later.
                mappings_path = data_dir / files_cfg.get("mappings", "")
                with mappings_path.open("rb") as f:
                    maps = pickle.load(f)
                self.user_map = maps.get("user_map")
                # Other maps available for future use:
                # self.merchant_map = maps.get("merchant_map")
                # self.ip_map       = maps.get("ip_map")
                # self.device_map   = maps.get("device_map")

                # Scalers (currently not used at inference since we rely on precomputed embeddings,
                # but loaded for completeness / future online GNN use).
                scalers_path = data_dir / files_cfg.get("scalers", "")
                with scalers_path.open("rb") as f:
                    scalers = pickle.load(f)
                self.user_scaler = scalers.get("user_scaler")

                # Precomputed user embeddings.
                emb_path = data_dir / files_cfg.get("embeddings", "")
                emb = np.load(emb_path)
                # Use test embeddings as the "live" lookup for known users.
                self.embeddings = emb["test"]

                # XGBoost fraud classifier.
                xgb_path = data_dir / files_cfg.get("xgb_model", "")
                with xgb_path.open("rb") as f:
                    self.xgb_model = pickle.load(f)

                # Thresholds and metadata.
                self.threshold = float(cfg.get("best_threshold", FRAUD_THRESHOLD))
                self.model_version = str(cfg.get("timestamp", _MODEL_VERSION_STUB))
                self.global_max_step = int(cfg.get("global_max_step", 743))

                # Feature metadata (if available) for top feature reporting.
                if hasattr(self.xgb_model, "feature_names_in_"):
                    self.feature_names = list(self.xgb_model.feature_names_in_)
                elif hasattr(self.xgb_model, "feature_names"):
                    self.feature_names = list(self.xgb_model.feature_names)

                if hasattr(self.xgb_model, "feature_importances_"):
                    self.feature_importances = np.asarray(
                        self.xgb_model.feature_importances_, dtype=float
                    )

                self._stub_mode = False
                logger.info(
                    "[%s] Initialized with real model artifacts (version=%s)",
                    self.name,
                    self.model_version,
                )
        except Exception as exc:  # noqa: BLE001
            # Any failure keeps us in stub mode; log and continue.
            self._stub_mode = True
            self.threshold = FRAUD_THRESHOLD
            self.model_version = _MODEL_VERSION_STUB
            logger.warning(
                "[%s] Failed to load real model artifacts (%s). "
                "Falling back to stub scoring.",
                self.name,
                exc,
                exc_info=True,
            )

    # ── Core scoring method — REPLACE THIS ───────────────────────────────────

    def score(self, msg: TransactionMessage) -> tuple[float, list[str]]:
        """
        Compute (fraud_probability, top_contributing_features) for a transaction.

        If real model artifacts were successfully loaded at __init__ time,
        this method performs XGBoost inference using precomputed user
        embeddings. Otherwise it falls back to a traffic_mode-based stub
        for demo purposes.
        """

        if getattr(self, "_stub_mode", True) or self.xgb_model is None or self.embeddings is None:
            return self._stub_score(msg)

        # ── Real inference path: build feature vector for XGBoost ─────────────
        # 1. Look up GNN embedding for the originating user (or fall back to zeros).
        emb_dim = int(self.embeddings.shape[1]) if self.embeddings is not None else 64
        if self.user_map is not None and msg.nameOrig in self.user_map:
            idx = self.user_map[msg.nameOrig]
            try:
                emb = np.asarray(self.embeddings[idx], dtype=float)
            except Exception:  # noqa: BLE001
                emb = np.zeros(emb_dim, dtype=float)
        else:
            emb = np.zeros(emb_dim, dtype=float)

        # 2. Handcrafted features based on transaction balances.
        balance_diff_orig = msg.oldbalanceOrg - msg.newbalanceOrig
        balance_diff_dest = msg.newbalanceDest - msg.oldbalanceDest
        amount_to_balance_ratio = (
            msg.amount / (msg.oldbalanceOrg + 1.0) if msg.oldbalanceOrg is not None else 0.0
        )

        # 3. One-hot encode transaction type (PaySim has 5 types).
        type_order = [
            TransactionType.PAYMENT,
            TransactionType.TRANSFER,
            TransactionType.CASH_IN,
            TransactionType.CASH_OUT,
            TransactionType.DEBIT,
        ]
        type_dummies = [1.0 if msg.type == t else 0.0 for t in type_order]

        # 4. Assemble XGBoost feature vector (order must match training pipeline).
        tabular_cols = [
            float(msg.step),
            float(msg.amount),
            float(msg.oldbalanceOrg),
            float(msg.newbalanceOrig),
            float(msg.oldbalanceDest),
            float(msg.newbalanceDest),
        ]
        extra_cols = [
            float(balance_diff_orig),
            float(balance_diff_dest),
            float(amount_to_balance_ratio),
        ]

        x_vec = np.concatenate(
            [
                np.asarray(tabular_cols, dtype=float),
                np.asarray(extra_cols, dtype=float),
                np.asarray(type_dummies, dtype=float),
                emb,
            ],
            axis=0,
        ).reshape(1, -1)

        # 5. Predict fraud probability.
        # Use Booster.predict(DMatrix) to avoid sklearn wrapper inplace_predict
        # device mismatch warnings when model is on CUDA and input is NumPy/CPU.
        fraud_prob = 0.0
        try:
            if self._xgb_module is not None and hasattr(self.xgb_model, "get_booster"):
                booster = self.xgb_model.get_booster()
                dmatrix = self._xgb_module.DMatrix(x_vec)
                pred = booster.predict(dmatrix)
                fraud_prob = float(np.asarray(pred).reshape(-1)[0])
            else:
                fraud_prob = float(self.xgb_model.predict_proba(x_vec)[0][1])
        except Exception:  # noqa: BLE001
            fraud_prob = float(self.xgb_model.predict_proba(x_vec)[0][1])

        # 6. Top features via model feature_importances_ if available.
        top_features: list[str] = []
        if self.feature_names is not None and self.feature_importances is not None:
            try:
                importances = np.asarray(self.feature_importances, dtype=float)
                names = list(self.feature_names)
                if importances.shape[0] == len(names):
                    idxs = np.argsort(np.abs(importances))[::-1][:5]
                    top_features = [str(names[i]) for i in idxs]
            except Exception:  # noqa: BLE001
                top_features = []

        return fraud_prob, top_features

    def _stub_score(self, msg: TransactionMessage) -> tuple[float, list[str]]:
        """
        Original traffic_mode-driven stub scoring for demo purposes.
        Used only when real model artifacts are unavailable.
        """
        mode = msg.traffic_mode

        if mode == TrafficMode.MULE_NETWORK:
            # Mule transactions should score high — coordinated drain pattern
            base_score = random.uniform(0.45, 0.97)
        elif mode == TrafficMode.ACCOUNT_TAKEOVER:
            # ATO transactions should score high — anomalous large transfer
            base_score = random.uniform(0.55, 0.99)
        else:
            # NORMAL — mostly low, occasional false positives (~5% above threshold)
            if random.random() < 0.92:
                base_score = random.uniform(0.001, 0.018)   # clearly below threshold
            else:
                base_score = random.uniform(0.025, 0.15)    # mild false positive range

        _stub_top_features = [
            "balance_diff_orig",
            "amount_to_balance_ratio",
            "oldbalanceOrg",
            "amount",
            "newbalanceOrig",
            "gnn_emb_3",   # GNN embedding dimensions appear in real top-20
            "gnn_emb_17",
        ]
        top_feats = random.sample(_stub_top_features, k=5)
        return round(base_score, 6), top_feats

    # ── Feature builder — reusable by score() and explain() ─────────────────

    def _build_features(self, msg: TransactionMessage) -> dict:
        """
        Extract and return a dict of all named features used by the XGBoost
        model for the given transaction message.

        Returns a dict: {feature_name: float_value}
        The order matches the assembled x_vec in score().
        """
        # GNN embedding lookup
        emb_dim = int(self.embeddings.shape[1]) if self.embeddings is not None else 64
        if (
            self.embeddings is not None
            and self.user_map is not None
            and msg.nameOrig in self.user_map
        ):
            idx = self.user_map[msg.nameOrig]
            try:
                emb = list(map(float, self.embeddings[idx]))
            except Exception:  # noqa: BLE001
                emb = [0.0] * emb_dim
        else:
            emb = [0.0] * emb_dim

        balance_diff_orig = float(msg.oldbalanceOrg) - float(msg.newbalanceOrig)
        balance_diff_dest = float(msg.newbalanceDest) - float(msg.oldbalanceDest)
        amount_to_balance_ratio = (
            float(msg.amount) / (float(msg.oldbalanceOrg) + 1.0)
            if msg.oldbalanceOrg is not None
            else 0.0
        )

        type_order = [
            TransactionType.PAYMENT,
            TransactionType.TRANSFER,
            TransactionType.CASH_IN,
            TransactionType.CASH_OUT,
            TransactionType.DEBIT,
        ]
        type_dummies = {f"type_{t.value}": (1.0 if msg.type == t else 0.0) for t in type_order}

        tabular = {
            "step": float(msg.step),
            "amount": float(msg.amount),
            "oldbalanceOrg": float(msg.oldbalanceOrg),
            "newbalanceOrig": float(msg.newbalanceOrig),
            "oldbalanceDest": float(msg.oldbalanceDest),
            "newbalanceDest": float(msg.newbalanceDest),
            "balance_diff_orig": balance_diff_orig,
            "balance_diff_dest": balance_diff_dest,
            "amount_to_balance_ratio": amount_to_balance_ratio,
        }
        tabular.update(type_dummies)
        for i, val in enumerate(emb):
            tabular[f"gnn_emb_{i}"] = val

        return tabular

    # ── Explainability method ─────────────────────────────────────────────────

    def explain(self, msg: TransactionMessage) -> dict:
        """
        Compute an end-to-end explanation for a single transaction.

        Returns a structured dict:
          {
            "fraud_score": float,
            "threshold": float,
            "model_version": str,
            "threshold_source": str,
            "top_positive_features": [{"feature": str, "contribution": float}],
            "top_negative_features": [{"feature": str, "contribution": float}],
            "rationale": str,
          }

        Attribution strategy:
          - If real XGBoost model with feature_importances_ is loaded,
            compute signed contributions = importance_weight * feature_value
            (normalised so contributions are in a comparable range).
          - In stub mode, return informative placeholder attributions derived
            from hand-crafted feature values.
        """
        # ── 1. Fraud score ────────────────────────────────────────────────────
        fraud_score, _ = self.score(msg)

        # ── 2. Feature attributions ───────────────────────────────────────────
        features = self._build_features(msg)

        attributions: list[tuple[str, float]] = []

        if (
            not self._stub_mode
            and self.xgb_model is not None
            and self.feature_importances is not None
            and self.feature_names is not None
        ):
            # Real model path: signed contribution = importance * feature_value
            importances = list(map(float, self.feature_importances))
            names = list(self.feature_names)
            total_imp = sum(abs(v) for v in importances) or 1.0
            for name, imp in zip(names, importances):
                feat_val = features.get(name, 0.0)
                # Normalise importance then scale by feature value direction
                contrib = (imp / total_imp) * feat_val
                attributions.append((name, round(contrib, 4)))
        else:
            # Stub path: use known important features with heuristic contributions
            key_features = [
                ("amount_to_balance_ratio", features.get("amount_to_balance_ratio", 0.0) * 0.4),
                ("balance_diff_orig",       features.get("balance_diff_orig", 0.0) / (features.get("amount", 1) + 1) * 0.3),
                ("oldbalanceOrg",           -features.get("oldbalanceOrg", 0.0) / 1e6 * 0.15),
                ("amount",                  features.get("amount", 0.0) / 1e5 * 0.1),
                ("newbalanceOrig",          -features.get("newbalanceOrig", 0.0) / 1e6 * 0.05),
            ]
            # Add some GNN dims
            for i in range(min(3, len([k for k in features if k.startswith("gnn_emb")]))):
                gnn_val = features.get(f"gnn_emb_{i}", 0.0)
                key_features.append((f"gnn_emb_{i}", round(gnn_val * 0.02, 4)))
            attributions = [(k, round(float(v), 4)) for k, v in key_features]

        # Sort by absolute contribution magnitude
        attributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Split into positive (risk-increasing) and negative (risk-reducing)
        top_positive = [
            {"feature": f, "contribution": c}
            for f, c in attributions
            if c > 0
        ][:5]
        top_negative = [
            {"feature": f, "contribution": c}
            for f, c in attributions
            if c < 0
        ][:5]

        # ── 3. LLM rationale ─────────────────────────────────────────────────
        rationale = self._generate_rationale(
            fraud_score=fraud_score,
            threshold=self.threshold,
            top_positive=top_positive,
            top_negative=top_negative,
        )

        return {
            "fraud_score": round(fraud_score, 6),
            "threshold": self.threshold,
            "model_version": self.model_version,
            "threshold_source": "config.json" if not self._stub_mode else "built-in default",
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "rationale": rationale,
        }

    def _generate_rationale(
        self,
        fraud_score: float,
        threshold: float,
        top_positive: list[dict],
        top_negative: list[dict],
        timeout: int = 20,
    ) -> str:
        """
        Call the local LFM2.5-1.2B-Instruct model (llama.cpp server) to generate
        a concise human-readable fraud explanation.  Uses streaming internally
        to collect the response, then returns the assembled text.

        Falls back to a template-based rationale on any error.
        """
        try:
            pos_parts = ", ".join(
                f"{f['feature']} (+{f['contribution']:.2f})" for f in top_positive[:3]
            )
            neg_parts = ", ".join(
                f"{f['feature']} ({f['contribution']:.2f})" for f in top_negative[:3]
            )

            verdict = "HIGH RISK" if fraud_score >= threshold else "LOW RISK"
            user_content = (
                f"Fraud score: {fraud_score:.4f} (threshold: {threshold:.4f}, verdict: {verdict}).\n"
                f"Top risk-increasing features: {pos_parts or 'none'}.\n"
                f"Top risk-reducing features: {neg_parts or 'none'}.\n"
                "Write one concise sentence explaining this fraud decision "
                "in plain English. Reference specific feature names and their "
                "contributions."
            )

            payload = {
                "model": "LFM2.5-1.2B-Instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a fraud analyst. Given a fraud score, "
                            "key contributing features, and their signed contributions, "
                            "explain in one concise sentence why the transaction was scored "
                            "as it was. Be specific about features. Do not add extra text."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.3,
                "max_tokens": 120,
                "stream": True,
            }

            # Streaming call — collect chunks
            collected = []
            with requests.post(
                "http://localhost:8080/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    if decoded.startswith("data: "):
                        raw = decoded[6:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                            delta = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                collected.append(delta)
                        except json.JSONDecodeError:
                            pass

            rationale = "".join(collected).strip()
            if not rationale:
                raise ValueError("Empty rationale from LLM")
            return rationale

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] LLM rationale generation failed (%s). Using template fallback.",
                self.name,
                exc,
            )
            # Template-based fallback
            pos_str = ", ".join(
                f"{f['feature']} (+{f['contribution']:.2f})" for f in top_positive[:3]
            )
            neg_str = ", ".join(
                f"{f['feature']} ({f['contribution']:.2f})" for f in top_negative[:2]
            )
            verdict = "High risk" if fraud_score >= threshold else "Low risk"
            if pos_str and neg_str:
                return (
                    f"{verdict} due to {pos_str}, "
                    f"partially offset by {neg_str}."
                )
            elif pos_str:
                return f"{verdict} driven primarily by {pos_str}."
            else:
                return (
                    f"{verdict} transaction — fraud score {fraud_score:.4f} "
                    f"vs threshold {threshold:.4f}."
                )

    def _process(self, msg: TransactionMessage) -> TransactionMessage:
        fraud_score, top_features = self.score(msg)
        msg.fraud_score   = fraud_score
        msg.fraud_label   = fraud_score >= self.threshold
        msg.top_features  = top_features
        msg.model_version = self.model_version
        return msg
