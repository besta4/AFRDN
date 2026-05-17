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

    def __init__(self, graph_cache: Any = None) -> None:
        # Try to load real model artifacts at __init__ time. If anything fails,
        # fall back to stub mode so the demo pipeline remains functional.
        self._stub_mode: bool = True
        self.threshold: float = FRAUD_THRESHOLD
        self.model_version: str = _MODEL_VERSION_STUB
        self.global_max_step: int | None = None
        # Redis-backed dynamic graph cache (optional, graceful degradation)
        self._graph_cache: Any = graph_cache
        self.user_map: dict[str, int] | None = None
        self.embeddings: np.ndarray | None = None
        self.xgb_model: Any = None
        self.feature_names: list[str] | None = None
        self.feature_importances: np.ndarray | None = None
        self._xgb_module: Any = None
        self.user_scaler: Any = None
        self._shap_explainer: Any = None
        self._shap_unavailable_reason: str | None = None

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
                self._force_xgboost_cpu()

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

    def _force_xgboost_cpu(self) -> None:
        """Keep local inference on CPU even if the persisted booster used CUDA."""
        try:
            if self.xgb_model is None:
                return
            if hasattr(self.xgb_model, "set_params"):
                try:
                    self.xgb_model.set_params(device="cpu")
                except Exception:
                    pass
            if hasattr(self.xgb_model, "get_booster"):
                booster = self.xgb_model.get_booster()
                booster.set_param({"device": "cpu"})
            elif hasattr(self.xgb_model, "set_param"):
                self.xgb_model.set_param({"device": "cpu"})
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] Could not force XGBoost CPU mode: %s", self.name, exc)

    # ── Core scoring method — REPLACE THIS ───────────────────────────────────

    def score(self, msg: TransactionMessage) -> tuple[float, list[str]]:
        """
        Compute (fraud_probability, top_contributing_features) for a transaction.

        If real model artifacts were successfully loaded at __init__ time,
        this method performs XGBoost inference using precomputed user
        embeddings + dynamic graph features from Redis.
        Otherwise it falls back to a traffic_mode-based stub for demo purposes.
        """

        if getattr(self, "_stub_mode", True) or self.xgb_model is None or self.embeddings is None:
            return self._stub_score(msg)

        # ── Real inference path: build feature vector for XGBoost ─────────────
        # 1. Look up static GNN embedding for the originating user (or zeros).
        emb_dim = int(self.embeddings.shape[1]) if self.embeddings is not None else 64
        if self.user_map is not None and msg.nameOrig in self.user_map:
            idx = self.user_map[msg.nameOrig]
            try:
                emb = np.asarray(self.embeddings[idx], dtype=float)
            except Exception:  # noqa: BLE001
                emb = np.zeros(emb_dim, dtype=float)
        else:
            emb = np.zeros(emb_dim, dtype=float)

        # 1b. Get dynamic graph features from Redis (real-time behavioral context).
        #     These 16 dims capture live velocity, counterparty fan-in/fan-out,
        #     burst patterns, and account activity age — things the static GNN
        #     embeddings from training time cannot know.
        dynamic_features = np.zeros(16, dtype=float)
        if self._graph_cache and self._graph_cache.available:
            try:
                dynamic_features = self._graph_cache.get_dynamic_features(msg.nameOrig)
            except Exception:  # noqa: BLE001
                pass  # graceful degradation to zeros

        # 2. Assemble XGBoost feature vector.
        #    Order: tabular + extras + type_dummies + static_emb
        #    Dynamic features are used as a score adjustment (see below)
        #    rather than concatenated into the XGBoost vector, because the
        #    XGBoost model was trained without them. Adding dimensions would
        #    cause a feature-count mismatch.
        x_vec, _ = self._build_feature_array(msg)

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

        # 5b. Dynamic graph score adjustment.
        #     Since the XGBoost model was trained without dynamic features,
        #     we use them as a post-hoc risk boost rather than concatenating
        #     into the feature vector (which would cause dimension mismatch).
        #     The boost is capped at +0.15 so it nudges but doesn't dominate.
        if np.any(dynamic_features != 0):
            # Key signals: burst_5min (idx 13), in_burst_5min (idx 14),
            #              out_count_1h (idx 0), in_count_1h (idx 4)
            burst_signal = float(dynamic_features[13]) + float(dynamic_features[14])
            velocity_signal = float(dynamic_features[0])  # out_count_1h
            # Scale: 5+ bursts or 10+ hourly txns → meaningful boost
            dynamic_boost = min(0.15, (burst_signal / 20.0) + (velocity_signal / 50.0))
            if dynamic_boost > 0.01:
                fraud_prob = min(1.0, fraud_prob + dynamic_boost)
                logger.debug(
                    "[%s] Dynamic graph boost: +%.4f → final=%.4f (burst=%d, vel_1h=%d)",
                    self.name, dynamic_boost, fraud_prob,
                    int(burst_signal), int(velocity_signal)
                )

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

    def _build_feature_array(self, msg: TransactionMessage) -> tuple[np.ndarray, list[str]]:
        """Build the model feature array and the matching feature-name order."""
        features = self._build_features(msg)
        base_names = [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "balance_diff_orig",
            "balance_diff_dest",
            "amount_to_balance_ratio",
            "type_PAYMENT",
            "type_TRANSFER",
            "type_CASH_IN",
            "type_CASH_OUT",
            "type_DEBIT",
        ]
        emb_names = sorted(
            [name for name in features if name.startswith("gnn_emb_")],
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        ordered_names = base_names + emb_names
        values = [float(features.get(name, 0.0)) for name in ordered_names]

        if self.feature_names and len(self.feature_names) == len(values):
            display_names = list(self.feature_names)
        else:
            display_names = ordered_names

        return np.asarray(values, dtype=float).reshape(1, -1), display_names

    def _compute_shap_attributions(
        self,
        x_vec: np.ndarray,
        feature_names: list[str],
    ) -> list[tuple[str, float]]:
        """Return SHAP attributions when the optional shap package is available."""
        try:
            import shap  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._shap_unavailable_reason = f"shap package unavailable: {exc}"
            return []

        if self.xgb_model is None:
            return []

        try:
            if self._shap_explainer is None:
                model = (
                    self.xgb_model.get_booster()
                    if hasattr(self.xgb_model, "get_booster")
                    else self.xgb_model
                )
                self._shap_explainer = shap.TreeExplainer(model)

            shap_values = self._shap_explainer.shap_values(x_vec)
            if isinstance(shap_values, list):
                values = np.asarray(shap_values[-1])
            else:
                values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[0, :, -1]
            else:
                values = values.reshape(values.shape[0], -1)[0]

            values = values[: len(feature_names)]
            return [
                (str(feature_names[i]), round(float(values[i]), 4))
                for i in range(len(values))
            ]
        except Exception as exc:  # noqa: BLE001
            self._shap_unavailable_reason = f"shap attribution failed: {exc}"
            logger.warning("[%s] SHAP attribution unavailable (%s)", self.name, exc)
            return []

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
          - If SHAP is installed and the real XGBoost model is loaded, compute
            local TreeSHAP feature attributions for this transaction.
          - Otherwise, if real XGBoost model with feature_importances_ is loaded,
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
        attribution_method = "heuristic"

        if not self._stub_mode and self.xgb_model is not None:
            x_vec, vector_names = self._build_feature_array(msg)
            attributions = self._compute_shap_attributions(x_vec, vector_names)
            if attributions:
                attribution_method = "shap"

        if (
            not attributions
            and not self._stub_mode
            and self.xgb_model is not None
            and self.feature_importances is not None
            and self.feature_names is not None
        ):
            # Real model path: signed contribution = importance * feature_value
            attribution_method = "feature_importance"
            importances = list(map(float, self.feature_importances))
            names = list(self.feature_names)
            total_imp = sum(abs(v) for v in importances) or 1.0
            for name, imp in zip(names, importances):
                feat_val = features.get(name, 0.0)
                # Normalise importance then scale by feature value direction
                contrib = (imp / total_imp) * feat_val
                attributions.append((name, round(contrib, 4)))
        if not attributions:
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
            "attribution_method": attribution_method,
            "shap_status": "available" if attribution_method == "shap" else (self._shap_unavailable_reason or "fallback"),
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
