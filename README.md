# Jatayu

Jatayu is a FastAPI-based fraud detection and response system for UPI-like
payments. It combines transaction scoring, pattern detection, risk decisions,
admin review, STR/CTR compliance reporting, support tickets, graph context, and
SHAP explanations.

## Project Layout

```text
jatayu/
  backend/              FastAPI app, routers, agents, auth, SQLite, models
  frontend/             Static HTML/CSS/JS served by FastAPI
  docs/                 Full project documentation and research notes
  sample_data/          Small CSV samples for batch testing
  tests/                Scenario verification scripts
  start.py              Root launcher for local development
  .env.example          Copy to .env and set local secrets
```

The frontend folder is flat on purpose. FastAPI mounts it at `/static`, so
browser URLs like `/static/js/main.js` still work without needing a nested
`frontend/static` directory.

## Local Setup

```powershell
cd C:\Users\besta\Downloads\jatayu
copy .env.example .env
python -m pip install -r backend\requirements.txt
python start.py
```

`start.py` is enough for the full local launcher. It checks dependencies,
checks Redis/model availability, optionally starts the local LLM server, and
starts the main app on `http://localhost:8000`.

For a simpler backend-only run:

```powershell
cd C:\Users\besta\Downloads\jatayu\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Main URLs

- Login: `http://localhost:8000/login`
- Customer dashboard: `http://localhost:8000/dashboard`
- Merchant dashboard: `http://localhost:8000/merchant`
- Admin console: `http://localhost:8000/admin`
- Batch upload: `http://localhost:8000/batch`
- API docs: `http://localhost:8000/docs`

Demo admin login:

```text
admin@jatayu.com / admin123
```

## Fraud Pipeline

1. Transaction Monitoring: XGBoost + static GNN embeddings, with Redis graph
   features used as a bounded live risk adjustment.
2. Pattern Detection: 75-transaction rolling window plus SQLite and Redis
   checks for mule networks, account takeover, velocity spikes, and graph loops.
3. Risk Assessment: calibrated rules by default, with the bundled PPO policy
   available through `JATAYU_ENABLE_PPO=1` after validation. The PPO checkpoint
   keeps the original 11-value state vector; circular-flow signals map to the
   mule-network policy bucket while rules still handle them as a distinct
   pattern.
4. Alert & Block: applies PASS, SILENT_FLAG, HOLD, or BLOCK.
5. Compliance Logging: writes audit trail and supports STR/CTR reporting.

SHAP is used in the admin transaction explanation flow when installed. If SHAP
is unavailable, the same endpoint falls back to feature-importance attribution.

## Verification

```powershell
python -m compileall backend start.py tests
python tests\test_500_realtime_scenarios.py
```

The 500-scenario script covers normal UPI payments, merchant QR bursts, split
bills, mule fan-in/fan-out, velocity probes, dormant account hijack, circular
flows, and account takeover. It samples SHAP attributions for suspicious cases.

## EC2 Deployment Summary

Recommended single-instance layout:

```text
/var/www/jatayu/
  backend/
  frontend/
  .env
```

Install system packages:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip nginx redis-server nodejs npm
sudo npm install -g pm2
```

Copy the project:

```bash
sudo mkdir -p /var/www/jatayu
sudo rsync -a backend/ /var/www/jatayu/backend/
sudo rsync -a frontend/ /var/www/jatayu/frontend/
sudo rsync -a deploy/ /var/www/jatayu/deploy/
sudo cp .env.example /var/www/jatayu/.env
sudo chown -R $USER:www-data /var/www/jatayu
```

Create the Python environment:

```bash
cd /var/www/jatayu/backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Edit `/var/www/jatayu/.env`:

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
```

Start backend with PM2:

```bash
cd /var/www/jatayu
pm2 start deploy/ecosystem.config.cjs
pm2 save
pm2 startup
```

Install Nginx config and SSL:

```bash
sudo cp deploy/nginx-jatayu.conf /etc/nginx/sites-available/jatayu
sudo ln -s /etc/nginx/sites-available/jatayu /etc/nginx/sites-enabled/jatayu
sudo nginx -t
sudo systemctl reload nginx
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Replace `your-domain.com` in the Nginx config before enabling it.
