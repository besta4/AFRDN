# Jatayu AFDRN - Complete Technical Documentation

**Autonomous Fraud Detection & Response Network**  
**Last Updated:** May 2026

## 1. Overview

Jatayu is a FastAPI application for UPI-like realtime payment fraud detection.
It combines model scoring, graph-aware pattern detection, risk assessment,
automated actioning, compliance logging, admin analytics, support workflows,
and explainability.

Core goals:

- Detect account takeover, mule networks, velocity probes, dormant account
  hijack, circular flows, and suspicious cashout behavior.
- Keep legitimate UPI usage such as split bills and merchant QR bursts low
  friction.
- Give admins historical trends, STR/CTR reporting, review tools, and SHAP
  explanations.

## 2. Project Layout

```text
jatayu/
  backend/              FastAPI app, routers, agents, auth, SQLite, models
  frontend/             Static HTML/CSS/JS served by FastAPI
  docs/                 Documentation and research notes
  sample_data/          CSV samples for batch testing
  tests/                Realtime verification scripts
  deploy/               PM2 and Nginx templates
  start.py              Root local launcher
  .env.example          Environment template
  README.md             Operator guide
```

The frontend folder is flat. FastAPI mounts `frontend/` at `/static`, so URLs
like `/static/js/main.js` and `/static/assets/jatayu-logo.png` still work
without a nested `frontend/static/` folder.

## 3. Runtime Configuration

Configuration loads from process environment first, then root `.env`, then
`backend/.env` if present.

| Variable | Default | Purpose |
| --- | --- | --- |
| `JATAYU_JWT_SECRET` | development fallback | JWT signing secret |
| `JATAYU_DB_PATH` | `data/jatayu.db` | SQLite DB path, relative paths resolve under `backend/` |
| `JATAYU_AUDIT_FILE` | `data/audit.jsonl` | Append-only compliance audit file |
| `JATAYU_REDIS_URL` | `redis://localhost:6379/0` | Redis dynamic graph context |
| `JATAYU_ENABLE_PPO` | `0` | Set to `1` only after validating the bundled PPO checkpoint on current traffic |

`.env` is gitignored. `.env.example` is safe to commit.

## 4. Backend Architecture

FastAPI entrypoint: `backend/main.py`

Routers:

| Router | Purpose |
| --- | --- |
| `auth.py` | Register, login, refresh, logout, password change |
| `users.py` | Customer account/profile operations |
| `transactions.py` | Realtime transaction creation, fraud flow, history |
| `admin.py` | Users, fraud review, analytics, compliance, simulation, risk decay |
| `merchant.py` | Merchant dashboard and transaction views |
| `support.py` | Support tickets for restricted users |

Persistence: `backend/database.py` uses SQLite. The same DB stores users,
accounts, sessions, devices, transactions, velocity windows, compliance reports,
OTP state, support tickets, and legacy batch tables.

## 5. Five-Agent Fraud Pipeline

1. **Transaction Monitoring:** XGBoost + static GNN embeddings produce
   `fraud_score`, `fraud_label`, top features, and model version.
2. **Pattern Detection:** Detects mule networks, account takeover, velocity
   spikes, dormant hijack, and circular flows using a 75-transaction rolling
   window, SQLite time-window history, and Redis graph context.
3. **Risk Assessment:** Uses calibrated rules by default. PPO from
   `backend/risk_ppo_2.pt` is available with `JATAYU_ENABLE_PPO=1` after
   validation. The PPO state remains the original 11-dimensional vector, with
   circular-flow patterns mapped into the mule-network policy bucket while rule
   logic still treats them distinctly. Outputs risk tier and recommended action.
4. **Alert & Block:** Executes PASS, SILENT_FLAG, HOLD, or BLOCK.
5. **Compliance Logging:** Writes a structured audit log and supports STR/CTR
   reporting.

## 6. Redis Dynamic Graph Context

Redis is optional but recommended. If unavailable, the app keeps running with
zero-vector graph context and SQLite-backed checks.

The Redis feature vector is intentionally **16-dimensional**. It is not
concatenated into the XGBoost vector because the model was trained without those
extra dimensions. Instead, Agent 1 uses it as a bounded post-model risk
adjustment. This keeps the current model valid while adding realtime graph
awareness.

Tracked Redis signals include:

- outbound/inbound counts and amounts for 1h and 24h
- unique senders/receivers
- fan-in and fan-out ratios
- account activity age
- 5-minute inbound/outbound bursts
- decaying risk scores for velocity, burst, pattern, and network suspicion

## 7. SHAP Explainability

SHAP is implemented at the right layer: Agent 1, where model scoring and the
XGBoost feature vector are built. The admin explanation endpoint reconstructs a
transaction, calls `TransactionMonitoringAgent.explain()`, and streams:

- fraud score
- threshold
- top risk-increasing SHAP features
- top risk-reducing SHAP features
- rationale
- attribution method metadata

If `shap` is installed, the method is `shap`. If not, the app falls back to
feature-importance or heuristic attribution and the UI still works.

Endpoint:

```text
GET /admin/fraud/{transaction_id}/explain
```

## 8. Admin Analytics

The analytics endpoint aggregates historical transaction data into daily/hourly
buckets:

```text
GET /admin/analytics/trends?days=7|30|90
```

Returned data:

- daily totals, held, approved, blocked, fraud-labeled counts
- daily block rate, fraud rate, average score, average latency
- hourly suspicious and blocked counts
- score distribution buckets
- held review outcomes and false-positive rate

The Admin Analytics tab renders these with Chart.js.

## 9. STR/CTR, Admin, User, Merchant, Graph, And Ticket Connectivity

- STR/CTR reports are created from compliance logic and manual admin block
  actions, then managed through `/admin/compliance/*`.
- Admin review uses `/admin/fraud/held`, `/admin/fraud/recent`,
  `/admin/fraud/{id}/approve`, and `/admin/fraud/{id}/block`.
- Users create and view transactions through `/transactions/*`.
- Merchants use `/merchant/*` for merchant-specific dashboards and histories.
- Graph detection uses Redis when available and SQLite fallbacks otherwise.
- Suspended/blocked users can still log in to the restricted support UI and use
  `/support/*` routes.

## 10. Test And Validation

Primary scenario script:

```powershell
python tests\test_500_realtime_scenarios.py
```

It creates 500 UPI-like realtime scenarios:

| Category | Count | Intent |
| --- | ---: | --- |
| Normal UPI | 250 | Everyday P2P/merchant transfers |
| Legit split bill | 40 | Many-to-one benign collection |
| Mule fan-in | 30 | Many senders to mule |
| Mule extraction | 15 | Mule forwarding funds |
| Velocity probe | 45 | Rapid low-value probes |
| Account takeover | 40 | Known user then new device/IP transfer |
| Dormant hijack | 30 | Dormant inbound and cashout |
| Circular flow + merchant burst | 50 | Loop detection and merchant-safe bursts |

The script exercises the real Orchestrator with LLM reasoning disabled for
speed and samples SHAP attributions for suspicious cases.

## 11. Local Run

```powershell
cd C:\Users\besta\Downloads\jatayu
copy .env.example .env
python -m pip install -r backend\requirements.txt
python start.py
```

`start.py` is enough for local development. It checks dependencies, model files,
SQLite, Redis, optionally starts the local LLM server, and starts the FastAPI
app on port `8000`.

Backend-only:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## 12. EC2 Deployment

Recommended single-instance layout:

```text
/var/www/jatayu/
  backend/
  frontend/
  deploy/
  .env
```

Install packages:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip nginx redis-server nodejs npm
sudo npm install -g pm2
```

Copy application:

```bash
sudo mkdir -p /var/www/jatayu
sudo rsync -a backend/ /var/www/jatayu/backend/
sudo rsync -a frontend/ /var/www/jatayu/frontend/
sudo rsync -a deploy/ /var/www/jatayu/deploy/
sudo cp .env.example /var/www/jatayu/.env
sudo chown -R $USER:www-data /var/www/jatayu
```

Install Python dependencies:

```bash
cd /var/www/jatayu/backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Configure `/var/www/jatayu/.env`:

```env
JATAYU_JWT_SECRET=replace-with-a-long-random-secret
JATAYU_DB_PATH=data/jatayu.db
JATAYU_AUDIT_FILE=data/audit.jsonl
JATAYU_REDIS_URL=redis://localhost:6379/0
JATAYU_ENABLE_PPO=0
```

Start Redis:

```bash
sudo systemctl enable --now redis-server
redis-cli ping
```

Start backend with PM2:

```bash
cd /var/www/jatayu
pm2 start deploy/ecosystem.config.cjs
pm2 save
pm2 startup
```

Nginx and SSL:

```bash
sudo cp deploy/nginx-jatayu.conf /etc/nginx/sites-available/jatayu
sudo nano /etc/nginx/sites-available/jatayu  # replace server_name domains
sudo ln -s /etc/nginx/sites-available/jatayu /etc/nginx/sites-enabled/jatayu
sudo nginx -t
sudo systemctl reload nginx
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d example.com -d www.example.com
```

SQLite can live on the same EC2 instance. Back up `backend/data/jatayu.db` and
`backend/data/audit.jsonl` regularly.
