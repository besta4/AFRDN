# Jatayu — Autonomous Fraud Detection & Response Network (AFDRN)
## Complete Technical Documentation

---

## 1. Project Overview

**Jatayu** (AFDRN — Autonomous Fraud Detection & Response Network) is a full-stack, real-time financial fraud detection platform built in Python. It combines:

- **Multi-agent AI pipeline** (5 sequential agents)
- **Graph Neural Network + XGBoost** for ML-based fraud scoring
- **Proximal Policy Optimisation (PPO)** reinforcement learning for risk assessment
- **Local LLM (LFM2.5-1.2B-Instruct)** via llama.cpp for natural-language reasoning and intelligence reports
- **FastAPI + SQLite** backend with WebSocket real-time streaming
- **Role-based multi-portal frontend** (Customer, Merchant, Admin, Support)

### Key Numbers
| Attribute | Value |
|---|---|
| Main application entry | `main.py` (1,058 lines) |
| Database layer | `database.py` (1,559 lines) |
| ML/AI agents | 5 specialized agents |
| API routers | 6 routers (auth, users, transactions, admin, merchant, support) |
| Frontend portals | 5 HTML pages |
| Database tables | 14 tables |
| Primary language | Python 3.12+ |
| Web framework | FastAPI + Uvicorn |
| Persistence | SQLite (`jatayu.db`) |

---

## 2. Project Directory Structure

```
jatayu/
├── main.py                        # FastAPI app, WebSocket, CSV pipeline routes
├── database.py                    # SQLite ORM layer — all DB helpers (1,559 lines)
├── start.py                       # System verification & dual-server launcher
├── run_llama_server.py            # llama.cpp local LLM server launcher
├── requirements.txt               # Python dependencies (87 packages)
├── jatayu.db                      # SQLite database (1.1 MB)
├── risk_ppo_2.pt                  # Trained PPO policy weights (28 KB)
├── audit.jsonl                    # Append-only compliance audit log (~20 KB)
├── mule_test.csv                  # Sample mule-network CSV for batch testing
├── simulate_mule_test.py          # Mule network simulation script
├── simulate_velocity_spike.py     # Velocity spike simulation script
│
├── agents/                        # 5-Agent AI pipeline
│   ├── __init__.py
│   ├── base_agent.py              # Abstract base class with provenance tracking
│   ├── models.py                  # TransactionMessage dataclass + all enums
│   ├── orchestrator.py            # Pipeline orchestrator (connects all agents)
│   ├── transaction_monitoring_agent.py   # Agent 1: GNN+XGBoost fraud scoring
│   ├── pattern_detection_agent.py        # Agent 2: Rolling-window pattern detection
│   ├── risk_assessment_agent.py          # Agent 3: PPO risk assessment + rule fallback
│   ├── alert_block_agent.py              # Agent 4: Action execution + explanation
│   ├── compliance_logging_agent.py       # Agent 5: Audit trail + JSONL persistence
│   └── traffic_generator.py             # Synthetic traffic generator for demo/training
│
├── auth/                          # Authentication & authorization
│   ├── __init__.py
│   ├── dependencies.py            # FastAPI auth dependencies + RBAC enforcement
│   ├── jwt_handler.py             # JWT creation/verification (HS256)
│   ├── models.py                  # Pydantic request/response models
│   ├── password.py                # bcrypt password hashing
│   └── rbac.py                   # Permission definitions and role mappings
│
├── routers/                       # FastAPI routers
│   ├── __init__.py
│   ├── auth.py                    # /auth — register, login, logout, refresh
│   ├── users.py                   # /users — profile, accounts, payees
│   ├── transactions.py            # /transactions — create, verify OTP, history
│   ├── admin.py                   # /admin — fraud dashboard, user mgmt, compliance
│   ├── merchant.py                # /merchant — payments, analytics, settlements
│   └── support.py                 # /support — tickets for blocked/suspended users
│
├── data/                          # ML model artifacts
│   ├── config_*.json              # Model config (threshold, file paths, timestamp)
│   ├── xgb_model_*.pkl            # Trained XGBoost classifier (~822 KB each)
│   ├── gnn_model_*.pt             # Trained Graph Neural Network (~1.3 MB each)
│   ├── embeddings_*.npz           # Precomputed GNN user embeddings (~600 MB each)
│   ├── mappings_*.pkl             # user/merchant/IP/device — index maps (~16 MB)
│   ├── scalers_*.pkl              # Feature scaler objects (~1 KB)
│   └── gnn_metadata_*.pkl         # GNN graph metadata
│
├── models/                        # Local LLM
│   └── LFM2.5-1.2B-Instruct-Q4_K_M.gguf   # LiquidAI 1.2B LLM (697 MB)
│
└── static/                        # Frontend (served by FastAPI)
    ├── login.html                 # Login/registration portal
    ├── dashboard.html             # Customer dashboard
    ├── merchant.html              # Merchant portal
    ├── admin.html                 # Admin console (fraud + user management)
    ├── batch.html                 # Batch CSV upload & analysis dashboard
    ├── suspended.html             # Account-restricted support page
    ├── css/styles.css             # Global styles
    └── js/
        ├── main.js                # Customer dashboard logic
        ├── api.js                 # API client wrapper
        ├── charts.js              # Chart.js chart builders
        ├── graph.js               # D3.js transaction network graph
        └── nav.js                 # Navigation helper
```

---

## 3. System Architecture

```
          Browser / Client
  ┌────────────────────────────────┐
  │  Real-Time App (auth required) │   Batch Pipeline (no auth)
  │  login | dashboard | merchant  │   /batch
  │  admin | support | suspended   │         |
  └─────────────┬──────────────────┘         |
                │                            |
          HTTP / WebSocket             HTTP / WebSocket
                │                            |
           FastAPI Application (port 8000)
                │
        ┌───────┴────────────────────────────┐
        │ Real-Time API Routers              │ Batch API (/batch-api/*)
        │ /auth | /transactions | /admin     │ upload, ws, results,
        │ /merchant | /users | /support      │ summary, audit, intelligence
        └───────┬────────────────────────────┘
                │
          5-Agent AI Pipeline
          Agent1 -> 2 -> 3 -> 4 -> 5
                |
        SQLite database (jatayu.db)
        ┌──────────────────────────────────┐
        │ Real-Time Tables  │ Batch Tables │
        │ users, accounts,  │ tasks,       │
        │ transactions, etc │ results, etc │
        └──────────────────────────────────┘
                |
         Llama.cpp LLM Server (port 8080)
         LFM2.5-1.2B-Instruct
```

---

## 4. The 5-Agent AI Pipeline

The heart of Jatayu is a **sequential 5-agent pipeline**. Every transaction flows through all agents in order. Each agent reads from a shared `TransactionMessage` dataclass and writes its own fields before passing the message forward.

### Pipeline Flow

```
Transaction Request
      |
      v
[Agent 1] Transaction Monitoring — XGBoost + GNN fraud scoring
      |
      v
[Agent 2] Pattern Detection — Rolling window: Mule / ATO / Velocity
      |
      v
[Agent 3] Risk Assessment — PPO RL policy (or rule-based fallback)
      |
      v
[Agent 4] Alert & Block — Execute action, generate explanation
      |
      v
[Agent 5] Compliance Logging — Audit entry + JSONL persistence
      |
      v
Final Decision + Audit Log
```

---

### Agent 1 — Transaction Monitoring (`transaction_monitoring_agent.py`)

**Purpose:** Score each transaction with a fraud probability using the trained GNN + XGBoost model.

**Inputs:** step, type, amount, nameOrig, nameDest, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, ip_address, device_id

**Outputs written:**
| Field | Type | Description |
|---|---|---|
| `fraud_score` | float [0,1] | Raw XGBoost fraud probability |
| `fraud_label` | bool | True if fraud_score >= 0.0224 |
| `top_features` | list[str] | Top 5 XGBoost feature names by importance |
| `model_version` | str | Artifact timestamp (e.g. 20260304_120501) |

**Real inference path (when model artifacts found):**
1. Look up GNN user embedding for `nameOrig` from precomputed `embeddings_*.npz`
2. Compute 3 handcrafted balance features (balance_diff_orig, balance_diff_dest, amount_to_balance_ratio)
3. One-hot encode transaction type (5 dims)
4. Concatenate: [6 tabular + 3 extra + 5 type-dummies + 64 GNN-dims] = 78-feature vector
5. Run through XGBoost Booster → fraud probability
6. Return top 5 features by `feature_importances_`

**Stub/fallback path (when model not loaded):**
- MULE_NETWORK traffic → score in [0.45, 0.97]
- ACCOUNT_TAKEOVER traffic → score in [0.55, 0.99]
- NORMAL traffic → score in [0.001, 0.15], ~5% false positive rate

**Fraud detection threshold:** `FRAUD_THRESHOLD = 0.0224` (loaded from config_*.json)

---

### Agent 2 — Pattern Detection (`pattern_detection_agent.py`)

**Purpose:** Detect coordinated attack patterns by analysing a rolling window of recent transactions.

**Key mechanism:** Maintains a `deque(maxlen=20)` rolling buffer. Also maintains per-user device and IP history dictionaries.

**Outputs written:**
| Field | Type | Description |
|---|---|---|
| `pattern_type` | PatternType | NONE / MULE_NETWORK / ACCOUNT_TAKEOVER / VELOCITY_SPIKE |
| `pattern_confidence` | float [0,1] | Confidence in the detected pattern |
| `window_snapshot` | list[str] | Last 5 transaction IDs in the buffer |
| `pattern_reasoning` | str | LLM-generated 1-sentence explanation |

**Detection Rules:**
| Pattern | Rule | Confidence Formula |
|---|---|---|
| MULE_NETWORK | >= 3 unique senders to same receiver within last 5 steps | min(1.0, 0.65 + 0.10 x (n_senders - 3)) |
| ACCOUNT_TAKEOVER | User has prior history AND current IP/device is new AND fraud_score >= 0.70 | min(1.0, 0.5 + 0.5 x fraud_score) |
| VELOCITY_SPIKE | Same sender appears >= 3 times in last 10 buffer entries | min(1.0, 0.65 + 0.10 x (count - 3)) |

---

### Agent 3 — Risk Assessment (`risk_assessment_agent.py`)

**Purpose:** Combine fraud score + pattern + account context to assign risk tier and recommended action.

**Outputs written:**
| Field | Type | Description |
|---|---|---|
| `risk_level` | RiskLevel | LOW / MEDIUM / HIGH / CRITICAL |
| `recommended_action` | Action | PASS / SILENT_FLAG / HOLD / BLOCK |
| `account_context` | dict | Supporting context (device history, velocity) |

**PPO Actor-Critic Architecture:**
```
State vector (11 dims):
  [0]  fraud_score
  [1]  pattern_confidence
  [2-5] pattern_type one-hot (NONE, MULE, ATO, VELOCITY)
  [6]  log10(amount+1) / 6
  [7]  is_flagged_by_model
  [8]  account_age_days_normalized
  [9]  is_new_device
  [10] is_new_ip

Actor:  Linear(11->64) -> ReLU -> Linear(64->64) -> ReLU -> Linear(64->4)
Critic: Linear(11->64) -> ReLU -> Linear(64->1)
```

**Reward function:**
| Ground Truth | Action | Reward |
|---|---|---|
| Fraud | BLOCK | +1.0 |
| Fraud | HOLD | +0.3 |
| Fraud | SILENT_FLAG | -0.5 |
| Fraud | PASS | -2.0 |
| Legitimate | PASS | +0.5 |
| Legitimate | SILENT_FLAG | -0.2 |
| Legitimate | HOLD | -1.0 |
| Legitimate | BLOCK | -2.0 |

**Rule-based fallback decision table:**
| Condition | Risk | Action |
|---|---|---|
| fraud_score >= 0.80 | CRITICAL | BLOCK |
| (MULE/ATO/VELOCITY) AND confidence >= 0.60 | HIGH | BLOCK |
| fraud_score >= 0.50 AND MULE/ATO | HIGH | BLOCK |
| fraud_score >= 0.50 | HIGH | HOLD |
| pattern detected (any) | MEDIUM | HOLD |
| fraud_score >= 0.20 | MEDIUM | SILENT_FLAG |
| fraud_score < 0.20 | LOW | PASS |

---

### Agent 4 — Alert & Block (`alert_block_agent.py`)

**Purpose:** Execute the recommended action and generate a plain-English explanation.

**Outputs written:**
| Field | Type | Description |
|---|---|---|
| `action_taken` | Action | Mirrors recommended_action |
| `explanation` | str | Template-based decision rationale |

**Explanation format:**
```
"Transaction of {type} ${amount} from {nameOrig} to {nameDest} received
 a fraud probability of {score_pct}. Top contributing signals: [{features}].
 Pattern analysis flagged this as {pattern} with risk level {risk}.
 Action taken: {action}."
```

---

### Agent 5 — Compliance Logging (`compliance_logging_agent.py`)

**Purpose:** Generate a structured audit log entry persisted to append-only JSONL file.

**Audit entry fields:**
```
Identity:      transaction_id, generated_at, logged_at_ms
Raw txn:       step, type, amount, nameOrig, nameDest, balances, ip, device
Agent 1:       fraud_score, fraud_label, top_features, model_version
Agent 2:       pattern_type, pattern_confidence, window_snapshot
Agent 3:       risk_level, recommended_action, account_context
Agent 4:       action_taken, explanation
Regulatory:    human_review_flag, agent_versions
Telemetry:     decision_latency_ms, audit_overhead_ms, pipeline_start_ms, pipeline_end_ms
Provenance:    pipeline_metadata (per agent: name, status, latency_ms, error, confidence)
```

---

### Base Agent (`base_agent.py`)

Abstract base class all agents inherit from. Provides:
- Standard `process()` wrapper measuring wall-clock latency
- Automatic `AgentMeta` provenance entry appended to `msg.pipeline_metadata`
- Exception isolation — errors in `_process()` are caught and recorded as `status="error"` without crashing the pipeline
- `_record_fallback(msg, reason)` helper to mark degradation events

---

### Orchestrator (`orchestrator.py`)

**Singleton pattern** — created once and shared across all API requests to preserve state across transactions.

**Pipeline execution order (`_process_one`):**
1. Agent 1 runs — fraud score computed
2. ALL transactions appended to rolling buffer (regardless of fraud label)
3. Agent 2 runs — pattern detection on buffer
4. Account hints (new device/IP) gathered from Agent 2's history
5. Agent 3 runs with hints — risk/action decided
6. `pipeline_end_ms` stamped (before Agent 5)
7. Agent 4 runs — action executed, explanation generated
8. Agent 5 runs — audit logged

---

## 5. TransactionMessage — The Pipeline Dataclass

`agents/models.py` defines the central `TransactionMessage` dataclass. It starts empty and is progressively enriched by each agent.

**Sections:**
- `[RAW]` — Identity + PaySim fields set before pipeline entry
- `[AGENT 1]` — fraud_score, fraud_label, top_features, model_version
- `[AGENT 2]` — pattern_type, pattern_confidence, window_snapshot, pattern_reasoning
- `[AGENT 3]` — risk_level, recommended_action, account_context
- `[AGENT 4]` — action_taken, explanation
- `[AGENT 5]` — audit_log
- `[Orchestrator]` — pipeline_start_ms, pipeline_end_ms, pipeline_metadata

**Enumerations:**
| Enum | Values |
|---|---|
| TransactionType | PAYMENT, TRANSFER, CASH_IN, CASH_OUT, DEBIT |
| TrafficMode | NORMAL, MULE_NETWORK, ACCOUNT_TAKEOVER |
| PatternType | NONE, MULE_NETWORK, ACCOUNT_TAKEOVER, VELOCITY_SPIKE |
| RiskLevel | LOW, MEDIUM, HIGH, CRITICAL |
| Action | PASS, SILENT_FLAG, HOLD, BLOCK |

---

## 6. Database Schema (`database.py`)

14 SQLite tables across 2 layers (1,559 lines).

### Legacy Tables (Batch CSV Processing)
| Table | Purpose |
|---|---|
| `tasks` | CSV upload metadata and status |
| `transaction_results` | JSON blob per row (batch pipeline output) |
| `audit_records` | Agent 5 compliance entries (batch mode) |

### Real-Time System Tables
| Table | Purpose | Key Fields |
|---|---|---|
| `users` | User accounts | user_id, email, password_hash, user_type, account_status, failed_attempts, locked_until |
| `user_profiles` | Extended profile | display_name, business_name |
| `accounts` | Financial accounts | account_id, user_id, account_type, balance, daily_limit |
| `sessions` | JWT sessions | session_id, token_hash, device_id, expires_at, is_active |
| `device_registry` | Trusted device tracking | device_id, user_id, is_trusted, first_seen, last_seen |
| `transactions` | Real-time transactions | transaction_id, sender_id, receiver_id, amount, status, fraud_score, risk_level, action_taken, pipeline_latency_ms |
| `payees` | Saved payee relationships | payee_id, user_id, payee_user_id, total_txn_count |
| `velocity_tracking` | Transaction velocity per window | user_id, window_type (HOURLY/DAILY/WEEKLY/MONTHLY), txn_count, txn_amount |
| `compliance_reports` | STR/CTR regulatory reports | report_id, report_type, trigger_reason, status (PENDING/SUBMITTED/ACKNOWLEDGED) |
| `pending_otp` | Email OTP for high-value transactions | otp_hash, transaction_data, expires_at, attempts |
| `support_tickets` | Blocked user support tickets | ticket_id, user_id, subject, status (OPEN/IN_PROGRESS/RESOLVED/CLOSED) |
| `support_messages` | Messages within tickets | message_id, ticket_id, sender_id, sender_role (USER/ADMIN) |

### Enums
| Enum | Values |
|---|---|
| UserType | CUSTOMER, MERCHANT, ADMIN |
| AccountStatus | PENDING, ACTIVE, SUSPENDED, BLOCKED |
| AccountType | SAVINGS, CURRENT, MERCHANT, ESCROW |
| TransactionStatus | INITIATED, PENDING_FRAUD, HELD, APPROVED, COMPLETED, BLOCKED, FAILED, REVERSED |

### User ID Format
```
CUSTOMER → C000000001
MERCHANT → M000000001
ADMIN    → A000000001
```

---

## 7. Authentication & Authorization (`auth/`)

### JWT Handler (`auth/jwt_handler.py`)
- Algorithm: HS256
- Access token expiry: 24 hours
- Refresh token expiry: 7 days

**JWT payload fields:** sub (user_id), type (user_type), session, device, iat, exp, jti

### Password Security (`auth/password.py`)
- bcrypt hashing with salt
- Lockout after 5 failed attempts (30-minute lock)

### RBAC (`auth/rbac.py`)
| Permission | CUSTOMER | MERCHANT | ADMIN |
|---|---|---|---|
| View own profile | Y | Y | Y |
| Create transaction | Y | Y | Y |
| Receive payments | Y | Y | Y |
| Manage payees | Y | N | Y |
| View payment analytics | N | Y | Y |
| View settlement reports | N | Y | Y |
| View all users | N | N | Y |
| Manage user status | N | N | Y |
| Review held transactions | N | N | Y |
| Approve/block transactions | N | N | Y |
| View compliance reports | N | N | Y |
| Generate STR | N | N | Y |

### Auth Dependencies (`auth/dependencies.py`)
```python
get_current_user                  # JWT + session active + account not blocked
require_customer = require_roles(["CUSTOMER", "ADMIN"])
require_merchant = require_roles(["MERCHANT", "ADMIN"])
require_admin    = require_roles(["ADMIN"])
require_customer_or_merchant = require_roles(["CUSTOMER", "MERCHANT", "ADMIN"])
```

**Special case:** BLOCKED/SUSPENDED users are allowed through `get_user_including_blocked` in `support.py` so they can reach the support portal — this is intentional.

---

## 8. API Routes Reference

### Authentication (`/auth`)
| Method | Path | Description |
|---|---|---|
| POST | /auth/register | Register new user (CUSTOMER or MERCHANT) |
| POST | /auth/login | Login with email/password — returns JWT tokens |
| POST | /auth/logout | Invalidate current session |
| POST | /auth/refresh | Exchange refresh token for new access token |
| GET | /auth/me | Get current user profile |
| PUT | /auth/password | Change password (invalidates all sessions) |

### Transactions (`/transactions`)
| Method | Path | Description |
|---|---|---|
| POST | /transactions/ | Create transaction with full fraud pipeline |
| POST | /transactions/verify-otp | Complete OTP-protected transaction |
| GET | /transactions/ | List user's transaction history |
| GET | /transactions/{id} | Get transaction detail + fraud results |
| GET | /transactions/flow/{flow_id} | Real-time pipeline step tracker |

### Admin (`/admin`)
| Method | Path | Description |
|---|---|---|
| GET | /admin/users | List all users (filterable by type) |
| GET | /admin/users/{id} | Get user detail with accounts + transactions |
| PUT | /admin/users/{id}/status | Update user status (ACTIVE/SUSPENDED/BLOCKED) |
| POST | /admin/users/deposit | Admin deposit funds to any account |
| GET | /admin/fraud/held | Get HELD transactions for review |
| GET | /admin/fraud/recent | Get recent transactions with fraud scores |
| POST | /admin/fraud/{id}/approve | Approve held transaction |
| POST | /admin/fraud/{id}/block | Block and reverse held transaction |
| GET | /admin/fraud/patterns | Fraud pattern distribution summary |
| GET | /admin/compliance/str | Get Suspicious Transaction Reports |
| GET | /admin/compliance/ctr | Get Cash Transaction Reports |
| GET | /admin/compliance/all | Get all compliance reports |
| PUT | /admin/compliance/{id}/submit | Mark STR/CTR as submitted |
| PUT | /admin/compliance/{id}/acknowledge | Acknowledge compliance report |
| GET | /admin/system/metrics | System performance metrics |
| POST | /admin/test/simulate-fraud | Run fraud pattern simulation |

### Merchant (`/merchant`)
| Method | Path | Description |
|---|---|---|
| GET | /merchant/payments | Get payments received |
| GET | /merchant/analytics | Payment analytics (daily/weekly/monthly) |
| GET | /merchant/analytics/hourly | Hourly transaction distribution |
| GET | /merchant/settlements | Settlement reports + daily totals |

### Support (`/support`)
| Method | Path | Description |
|---|---|---|
| POST | /support/tickets | Create support ticket (accessible even if BLOCKED) |
| GET | /support/tickets | List user's own tickets |
| GET | /support/tickets/{id} | Get ticket + message thread |
| POST | /support/tickets/{id}/messages | Send message on ticket |
| GET | /support/admin/tickets | Admin view all tickets |
| POST | /support/admin/tickets/{id}/resolve | Resolve ticket + optionally reactivate user |
| POST | /support/admin/tickets/{id}/close | Close ticket without reactivation |

### Batch CSV Pipeline — Independent (`/batch-api/` prefix)

> **Completely independent** from the real-time application. No authentication required — publicly accessible at `/batch`. Uses its own set of SQLite tables (`tasks`, `transaction_results`, `audit_records`) that are **separate** from all real-time system tables.

| Method | Path | Description |
|---|---|---|
| POST | /batch-api/upload | Upload CSV, returns task_id, starts background processing |
| WS | /batch-api/ws/{task_id} | WebSocket real-time progress stream |
| GET | /batch-api/results/{task_id} | Fetch processed transaction results |
| GET | /batch-api/summary/{task_id} | High-level metrics + network graph data |
| GET | /batch-api/audit | Audit trail (all tasks or filtered by task_id) |
| GET | /batch-api/tasks | List all uploaded tasks |
| GET | /batch-api/agent-outputs/{task_id} | Per-agent sliced outputs |
| GET | /batch-api/intelligence/{task_id} | LLM intelligence report (sync) |
| GET | /batch-api/intelligence-stream/{task_id} | LLM intelligence SSE stream |

### Frontend Pages
| Route | Page | Access |
|---|---|---|
| / | Login | Public |
| /login | Login | Public |
| /dashboard | Customer dashboard | CUSTOMER / ADMIN |
| /merchant | Merchant portal | MERCHANT / ADMIN |
| /admin | Admin console | ADMIN only |
| /batch | Batch CSV upload | Public |
| /suspended | Account-restricted | Any user |

---

## 9. Transaction Fraud Pipeline — Detailed Flow

When `POST /transactions/` is called:

```
1.  Check sender account status (SUSPENDED/BLOCKED → reject)
2.  Resolve receiver by user ID or email
3.  Validate transaction type (PAYMENT requires merchant receiver)
4.  Check sender balance
5.  [Optional] Email OTP check — if enabled, return 202 with otp_id
6.  Auto-add payee if new receiver
7.  Pre-auth hold: deduct amount from sender balance
8.  Create transaction record (status=INITIATED)
9.  Update status → PENDING_FRAUD
10. Run 5-agent fraud pipeline (synchronous, in executor thread)
11. Store fraud results in DB
12. Execute final action:
    BLOCK:
      - Reverse pre-auth (credit back to sender)
      - Update status → BLOCKED
      - Suspend sender (account_status=SUSPENDED)
      - If MULE_NETWORK: suspend collector + all prior senders to that collector
    HOLD:
      - Update status → HELD (goes to admin review queue)
    PASS / SILENT_FLAG:
      - Credit receiver
      - Update status → COMPLETED
      - Update velocity counters
      - Update payee stats
13. Check STR threshold → auto-generate compliance report if triggered
14. Check CTR threshold (>=Rs 10 lakh cash) → auto-generate CTR
15. Return TransactionResponse with fraud_check details
```

---

## 10. Auto-Suspension Logic

When a transaction is BLOCKED by Agent 4:

```
Sender  → always suspended

If MULE_NETWORK:
  Collector account → suspended
  All prior senders to that collector (last 72 hours, from DB) → suspended

If VELOCITY_SPIKE:
  Sender already suspended (no additional action needed)
```

**Suspended user flow:**
- Can still log in (blocking at login defeats support UI access)
- Redirected to `/suspended` page
- Can ONLY use `/support/*` endpoints
- Must create a support ticket for admin review
- Admin resolves ticket + optionally reactivates account

---

## 11. Compliance & Regulatory Features

### Suspicious Transaction Reports (STR)
Auto-generated when:
- fraud_score > 0.8 AND action is BLOCK
- Admin manually blocks a transaction

### Cash Transaction Reports (CTR)
Auto-generated when:
- amount >= Rs 10,00,000 for CASH_IN or CASH_OUT transactions

### Compliance Report Lifecycle
```
PENDING → SUBMITTED → ACKNOWLEDGED
```

Admin can submit STRs/CTRs to regulatory authorities via admin panel.

---

## 12. Local LLM Integration

Uses **LiquidAI LFM2.5-1.2B-Instruct** (Q4_K_M quantised GGUF, 697 MB) served via llama.cpp on port 8080.

**Used for:**
1. **Agent 2** — Generate 1-sentence pattern reasoning after rule-based detection
2. **Intelligence reports** — Full fraud analysis report (streaming SSE + sync endpoints)

**Intelligence report JSON schema:**
```json
{
  "headline": "<2-sentence executive summary>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
  "threat_assessment": "<paragraph about detected threats>",
  "recommendations": ["<rec 1>", "<rec 2>", "<rec 3>"],
  "confidence": "HIGH|MEDIUM|LOW"
}
```

**Fallback:** If LLM unavailable, pattern reasoning uses predefined templates; intelligence endpoint returns raw metrics without LLM analysis.

---

## 13. ML Model Artifacts

### Model Stack
```
CSV Transaction
     |
     v
Graph Neural Network (GNN)
  - Trained on PaySim transaction graph
  - Input: user behaviour graph (edges = transactions)
  - Output: 64-dim user embedding per account
     |
     + (concatenated with tabular features)
     |
XGBoost Classifier
  - Input: [6 tabular + 3 engineered + 5 type-dummies + 64 GNN-dims] = 78 features
  - Output: fraud probability in [0, 1]
  - Threshold: 0.0224 (optimised for F1 score)
     |
     v
fraud_score + fraud_label
```

### Artifact Files (in `data/`)
| File | Purpose | Size |
|---|---|---|
| config_*.json | Threshold, file paths, timestamp | ~707 B |
| xgb_model_*.pkl | XGBoost trained classifier | ~822 KB |
| gnn_model_*.pt | Graph Neural Network weights | ~1.3 MB |
| embeddings_*.npz | Precomputed user embeddings | ~600 MB |
| mappings_*.pkl | user/merchant/IP/device to index maps | ~16 MB |
| scalers_*.pkl | StandardScaler for numerical features | ~1 KB |

### PPO Policy (`risk_ppo_2.pt`)
- Trained Agent 3 policy (28 KB)
- Architecture: MLP actor-critic (11 -> 64 -> 64 -> 4 actions)
- Trained with supervised reward signal from synthetic PaySim data
- Used in deterministic inference mode during production

---

## 14. Frontend Pages

### Login Portal (`static/login.html`)
- Email/password login with device fingerprint generation
- Registration form for CUSTOMER and MERCHANT accounts
- Auto-routes based on user_type in JWT:
  - CUSTOMER → /dashboard
  - MERCHANT → /merchant
  - ADMIN → /admin
  - SUSPENDED/BLOCKED → /suspended
- Stores JWT in localStorage and httpOnly cookie

### Customer Dashboard (`static/dashboard.html`)
Sections:
- Account summary (balance, account ID)
- Send money / payment form
- Transaction history with status badges
- Real-time fraud pipeline tracker (shows each agent's progress)
- Payee management

### Merchant Portal (`static/merchant.html`)
Sections:
- Payment summary (received transactions)
- Analytics charts (daily/weekly/monthly volume)
- Hourly transaction distribution chart
- Settlement reports (7-day daily totals)
- Top payers list

### Admin Console (`static/admin.html`)
Sections:
- **Fraud Dashboard:** Recent + held transactions with fraud scores, risk badges
- **User Management:** List all users, filter by type, update account status
- **Transaction Approval Queue:** Review and approve or block HELD transactions
- **Compliance Reports:** STR/CTR management, submit to regulatory authority
- **Pattern Simulation:** Test MULE_NETWORK, ACCOUNT_TAKEOVER, VELOCITY_SPIKE
- **System Metrics:** User counts, pipeline latency stats
- **Support Tickets:** View all tickets, respond, resolve, reactivate users
- **Deposit Funds:** Admin credit to any user account

### Batch Analysis Dashboard (`static/batch.html`)
**Fully independent** — no login required, publicly accessible at `/batch`. Talks exclusively to `/batch-api/*` endpoints. Data stored in batch-only SQLite tables, completely separate from user/account data.
- Drag-and-drop CSV upload (PaySim format)
- Real-time WebSocket progress bar (via `/batch-api/ws/{task_id}`)
- Post-analysis: metrics cards, charts, exception queue, transaction network graph
- Agent accordion view (per-agent breakdown for each row processed)
- LLM intelligence report (streaming SSE via `/batch-api/intelligence-stream/{task_id}`)

### Suspended Account Page (`static/suspended.html`)
- Explains why the account was restricted
- Support ticket creation form
- Lists existing tickets with status
- Instructions for account reactivation process

---

## 15. JavaScript Frontend (`static/js/`)

### Real-Time Application JS (`/static/js/` — used by login/dashboard/admin/merchant)
| File | Purpose |
|---|---|
| `main.js` | Batch pipeline: upload flow, WebSocket progress, dashboard rendering, agent accordion, intelligence streaming |
| `api.js` | **Batch-only** API client — all calls prefixed with `/batch-api/`. Handles CSV upload, WebSocket socket, SSE stream, summary/results/audit/agent-outputs endpoints |
| `charts.js` | ApexCharts visualisations for the Analytics tab (fraud histogram, risk donut, action radial, pattern polar) |
| `graph.js` | force-graph transaction network (accounts, merchants, NPC background nodes, fraud edges in red) |
| `nav.js` | Floating nav bar with animated sliding indicator pill |

---

## 16. Startup & Deployment

### Development Start (`start.py`)
```
Step 1: Check and install missing packages (fastapi, uvicorn, llama_cpp, etc.)
Step 2: Verify ML model at models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
Step 3: Check SQLite DB (creates on first run)
Step 4: Start Llama server: python run_llama_server.py (port 8080)
        Wait 15 seconds for model load
Step 5: Start main app: uvicorn main:app --reload --port 8000
        Wait 5 seconds for startup
Step 6: Open browser to http://localhost:8000
```

### Manual Start
```bash
# Terminal 1 — LLM Server
python run_llama_server.py

# Terminal 2 — Main Application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access URLs
| Service | URL |
|---|---|
| Dashboard | http://localhost:8000 |
| API Documentation (Swagger) | http://localhost:8000/docs |
| LLM Server | http://localhost:8080 |

---

## 17. Key Technology Stack

| Category | Technology |
|---|---|
| Web Framework | FastAPI + Uvicorn (ASGI) |
| Database | Raw SQLite via sqlite3 contextmanager |
| ML — Graph | PyTorch Geometric (torch_geometric) |
| ML — Classification | XGBoost 3.2.0 |
| ML — Reinforcement | PyTorch 2.10.0 (PPO) |
| Data Processing | Pandas 3.0, NumPy 2.4 |
| LLM | LFM2.5-1.2B-Instruct via llama.cpp |
| Auth | PyJWT (HS256), bcrypt |
| Validation | Pydantic v2 |
| HTTP Client | httpx (async), requests (sync) |
| Real-time | WebSocket (FastAPI native), SSE (StreamingResponse) |
| GPU Support | CUDA 12.8 (nvidia-* packages in requirements.txt) |
| Visualisation | Chart.js, D3.js (frontend) |

---

## 18. Fraud Detection Test Scenarios

### Admin API Simulation (`POST /admin/test/simulate-fraud`)
```json
{
  "pattern_type": "MULE_NETWORK",
  "num_transactions": 5
}
```
Supported patterns: MULE_NETWORK, ACCOUNT_TAKEOVER, VELOCITY_SPIKE

Returns per-transaction detection results including:
- fraud_score, fraud_label
- pattern_type, pattern_confidence, pattern_reasoning
- risk_level, action_taken
- agent latencies (agent1_ms through agent5_ms)
- buffer state (size, last 5 transaction IDs)

### Simulation Scripts
- `simulate_mule_test.py` — Generates MULE_NETWORK pattern transactions directly against the API
- `simulate_velocity_spike.py` — Generates VELOCITY_SPIKE pattern transactions

### Batch CSV Test File
- `mule_test.csv` — Sample mule-network structured CSV for batch pipeline testing

---

## 19. Data Flow — End to End

```
User Initiates Transaction
        |
        v
POST /transactions/
        |
        +-- Balance check
        |
        +-- OTP gate (if enabled)
        |
        +-- Pre-auth (deduct sender balance)
        |
        +-- DB: INSERT transaction (status=PENDING_FRAUD)
        |
        +-- 5-Agent Pipeline
        |     Agent 1: XGBoost fraud score (78-dim GNN+tabular features)
        |     Agent 2: Pattern detection (rolling buffer, 3 rule-based checks)
        |     Agent 3: PPO risk + action (11-dim state vector)
        |     Agent 4: Execute + template explanation
        |     Agent 5: Audit log -> audit.jsonl
        |
        +-- DB: UPDATE transaction with fraud results
        |
        +-- Action execution
        |     BLOCK  -> refund + suspend sender + STR generation
        |     HOLD   -> admin review queue
        |     PASS   -> credit receiver + velocity update
        |
        +-- Return TransactionResponse
```

---

## 20. Security Features Summary

| Feature | Implementation |
|---|---|
| Password storage | bcrypt (8+ rounds) |
| Session tokens | JWT HS256 with per-session JTI |
| Account lockout | 5 failed attempts → 30-min lock |
| Session revocation | DB-tracked is_active flag per session |
| Device tracking | Per-user device registry |
| Suspended user isolation | Can only reach /support/* endpoints |
| Admin self-modification | Prevented (cannot modify own account) |
| Compliance | STR/CTR auto-generation + managed lifecycle |
| Audit trail | Immutable append-only JSONL per transaction |

---

## 21. Known Production TODOs

| Component | TODO |
|---|---|
| Agent 4 | Call real payment gateway for BLOCK action |
| Agent 4 | LLM-generated explanation instead of template |
| Agent 3 | Load real account age from DB |
| OTP | Send actual email (currently only logs OTP code to console) |
| Auth | Move JWT secret to environment variable |
| Velocity | DB-backed velocity limits per account tier |
| CORS | Restrict to known origins in production |
| GNN online | Currently uses precomputed embeddings; online inference not wired |

---

## 22. AgentMeta Provenance

Every agent appends an `AgentMeta` entry to `msg.pipeline_metadata`:

```python
@dataclass
class AgentMeta:
    agent_name: str     # e.g. "PatternDetectionAgent"
    status: str         # "ok" | "fallback" | "error"
    latency_ms: float   # Wall-clock time in _process()
    error: Optional[str]       # Exception message if status=="error"
    confidence: Optional[float]  # Agent-reported confidence (pattern_confidence for Agent 2)
```

This enables:
- Per-agent latency profiling without a separate side-channel
- Fault attribution — which agent degraded
- Full provenance in audit log (pipeline_metadata field in Agent 5 entry)

---

*This document was auto-generated by reading every source file in the Jatayu codebase as of 2026-03-31.*
