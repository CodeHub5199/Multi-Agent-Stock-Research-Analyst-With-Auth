# 🧠 Multi-Agent Stock Research Analyst

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-FF6B6B?style=for-the-badge)
![Supabase](https://img.shields.io/badge/Supabase-Auth%20%2B%20DB-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Institutional-grade, parallel multi-agent AI system for Indian equity research.**

[Features](#-features) • [Architecture](#-architecture) • [Authentication](#-authentication) • [Installation](#-installation) • [API Reference](#-api-reference) • [Contributing](#-contributing)

🌐 **Live Preview:** https://codehub5199.github.io/Multi-Agent-Stock-Research-Analyst/
🐙 **GitHub:** https://github.com/CodeHub5199/Multi-Agent-Stock-Research-Analyst

</div>

---

## 📖 Overview

StockMind is a production-ready AI research platform that mimics how an institutional equity desk works — running **specialized agents in parallel**, synthesizing their signals, and stress-testing the thesis through a critic layer — all delivered in seconds.

Built on **LangGraph** for orchestration, **Groq** for ultra-fast LLM inference, **Tavily** for real-time news, and **yfinance** for market data. Optimized for **NSE/BSE-listed equities** with INR formatting and RBI/SEBI regulatory context.

Users sign in via **Supabase Auth** (email/password or Google OAuth), and every analysis they run is **automatically saved** to their private account — accessible anytime from the collapsible report history sidebar.

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **Parallel Agent Execution** | Fundamentals, Technical, and News agents run concurrently via LangGraph |
| 🔀 **Synthesis Agent** | Triangulates multi-agent signals with weighted consensus scoring |
| 🎯 **Critic Agent** | Stress-tests the bull thesis, surfaces bear cases, data gaps, and risks |
| 📊 **Institutional Outputs** | Signal scores, anomaly detection, S/R levels, high-impact events |
| 🖥️ **Live Dashboard** | Responsive single-page UI with animated charts and verdict banners |
| 🚀 **FastAPI Backend** | Async-first, production-ready REST API with OpenAPI docs |
| 🇮🇳 **India Market Optimized** | NSE/BSE focus, INR formatting, RBI/SEBI regulatory awareness |
| 🔍 **Smart Autocomplete** | Search from the full NSE stock universe as you type |
| 🔐 **User Authentication** | Secure sign-in via email/password or Google OAuth powered by Supabase |
| 📂 **Report History** | Every analysis auto-saved per user — revisit past reports from the sidebar |
| 🏠 **Hero Landing Page** | Engaging public-facing landing page with feature highlights and auth modal |

---

## 🏗 Architecture

```
                        ┌──────────────────────────────────┐
                        │         FastAPI Backend          │
                        │  GET  /           → index.html   │
                        │  GET  /dashboard  → dashboard    │
                        │  POST /analyze    → LangGraph    │
                        │  GET  /reports    → Supabase DB  │
                        └────────────────┬─────────────────┘
                                         │
                    ┌────────────────────▼──────────────────┐
                    │          LangGraph State Machine      │
                    │                    │                  │
          ┌─────────▼──────┐  ┌──────────▼───────┐  ┌───────▼─────────┐
          │  Fundamentals  │  │    Technical     │  │      News       │
          │     Agent      │  │     Agent        │  │     Agent       │
          │  (yfinance +   │  │  (yfinance +     │  │  (Tavily +      │
          │     LLM)       │  │     LLM)         │  │     LLM)        │
          └─────────┬──────┘  └──────────┬───────┘  └────────┬────────┘
                    └────────────────────▼───────────────────┘
                                         │
                              ┌──────────▼───────────┐
                              │   Synthesis Agent    │
                              │  Bull/Bear Signals   │
                              │  Consensus Scoring   │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │    Critic Agent      │
                              │  Risk Assessment     │
                              │  Confidence Audit    │
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────▼─────────────────────┐
                    │              Supabase                    │
                    │  Auto-save report payload per user       │
                    │  Row Level Security — private per user   │
                    └──────────────────────────────────────────┘
```

### Agent Responsibilities

- **Fundamentals Agent** — P/E, P/B, revenue growth, margins, debt ratios, promoter holding via yfinance
- **Technical Agent** — RSI, MACD, moving averages, support/resistance, volume analysis
- **News Agent** — Real-time news via Tavily, sentiment scoring, event impact classification
- **Synthesis Agent** — Weighted signal aggregation, bull/bear case extraction, overall verdict
- **Critic Agent** — Assumption stress-testing, bear case deepening, gap identification, confidence scoring

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph (stateful multi-agent DAG) |
| **Backend** | FastAPI + Uvicorn |
| **LLM Inference** | Groq (Llama-3 / Mixtral) |
| **Market Data** | yfinance |
| **News & Search** | Tavily API |
| **Auth & Database** | Supabase (Auth + PostgreSQL + Row Level Security) |
| **Validation** | Pydantic v2 |
| **Frontend** | Vanilla HTML/CSS/JS (zero dependencies) |

---

## 🔐 Authentication

StockMind uses **Supabase Auth** for secure, production-grade authentication. Two sign-in methods are supported out of the box.

### Sign-in Flow

```
User visits /           →  Hero landing page (public)
Clicks "Get started"    →  Auth modal opens (Login / Sign up)
Signs in                →  Supabase issues JWT access token
Token sent with /analyze →  FastAPI verifies token server-side
Report auto-saved       →  Stored in Supabase, tied to user ID
User revisits /dashboard →  Past reports load in the left sidebar
```

### Sign-in Methods

| Method | Notes |
|---|---|
| **Email / Password** | Standard signup with email confirmation |
| **Google OAuth** | One-click sign-in via Google account |

### Security Design

- JWT tokens verified **server-side** on every `/analyze` call via `api/auth.py`
- Reports table protected by **Row Level Security (RLS)** — users can only access their own rows
- Supabase `service_role` key stays strictly in `.env` — never sent to the browser
- Supabase `anon` key used in HTML — safe for frontend, RLS enforces data isolation

### New Files Added for Auth

| File | Purpose |
|---|---|
| `api/auth.py` | FastAPI `Depends(get_current_user)` — verifies Supabase JWT |
| `api/supabase_client.py` | Cached Supabase admin client (server-side only) |
| `api/models_auth.py` | Pydantic models for report list/detail endpoints |
| `templates/index.html` | Public hero/landing page with auth modal |

---

## 📂 Report History

Every analysis run by an authenticated user is **automatically persisted** to Supabase after the pipeline completes. No manual saving required.

### What Gets Saved Per Report

| Field | Description |
|---|---|
| `ticker` | Stock symbol (e.g. `SBIN`, `ITC`) |
| `company_name` | Resolved company name from fundamentals agent |
| `verdict` | Final critic-adjusted verdict (BUY / HOLD / SELL) |
| `elapsed_seconds` | Total pipeline runtime in seconds |
| `payload` | Complete JSON output of all 5 agents |
| `created_at` | UTC timestamp of the analysis |

### Accessing Past Reports

- A **collapsible left sidebar** lists the 20 most recent reports for the logged-in user
- Click any report row to instantly reload the full analysis — no re-running the pipeline
- Reports are sorted newest-first, showing ticker, company name, verdict chip, and date
- The sidebar collapses via a toggle button (`◀ / ▶`) to maximise dashboard workspace

---

## 📋 Prerequisites

- Python **3.10+**
- [Groq API key](https://console.groq.com) — free tier available
- [Tavily API key](https://tavily.com) — for the news agent
- [Supabase project](https://supabase.com) — free tier, Auth + PostgreSQL
- Google Cloud OAuth credentials — for Google sign-in (optional)

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/CodeHub5199/Multi-Agent-Stock-Research-Analyst.git
cd Multi-Agent-Stock-Research-Analyst
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# LLM & Search
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
GROQ_MODEL=openai/gpt-oss-120b        # or llama3-70b-8192, mixtral-8x7b-32768

# Supabase — server-side only, never expose to frontend
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

> **Model tip:** `llama3-70b-8192` works well on the free tier. `openai/gpt-oss-120b` gives the best output quality.

### 5. Set Supabase frontend keys

In both `templates/index.html` and `templates/dashboard.html`, replace the two placeholder values near the bottom of the `<script>` block:

```javascript
const SUPABASE_URL  = 'https://xxxxxxxxxxxx.supabase.co';  // Project URL
const SUPABASE_ANON = 'eyJ...';                             // anon / public key
```

### 6. Create the reports table

Run this SQL once in your **Supabase → SQL Editor**:

```sql
CREATE TABLE reports (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  ticker          TEXT NOT NULL,
  company_name    TEXT,
  verdict         TEXT,
  elapsed_seconds FLOAT,
  payload         JSONB NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

-- Indexes for fast lookups
CREATE INDEX reports_user_id_idx    ON reports(user_id);
CREATE INDEX reports_created_at_idx ON reports(created_at DESC);

-- Row Level Security
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see own reports"
  ON reports FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users insert own reports"
  ON reports FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users delete own reports"
  ON reports FOR DELETE USING (auth.uid() = user_id);
```

### 7. (Optional) Enable Google OAuth

1. Create OAuth credentials in [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Credentials → OAuth 2.0 Client ID
2. Add the Supabase callback URL as an authorised redirect URI:
   ```
   https://xxxxxxxxxxxx.supabase.co/auth/v1/callback
   ```
3. In Supabase → Authentication → Providers → Google, paste your Client ID and Client Secret

---

## ▶️ Running the Application

```bash
uvicorn main:app --reload --port 8000
```

| URL | Description |
|---|---|
| `http://localhost:8000/` | Hero landing page (public) |
| `http://localhost:8000/dashboard` | Research dashboard (requires login) |
| `http://localhost:8000/docs` | Swagger UI — interactive API docs |
| `http://localhost:8000/redoc` | ReDoc API reference |

---

## 📡 API Reference

### `POST /analyze`
Run the full multi-agent pipeline. If a valid Bearer token is present the report is auto-saved to the database.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <supabase_access_token>" \
  -d '{"ticker": "SBIN"}'
```

**Response:**
```json
{
  "ticker": "SBIN",
  "elapsed_seconds": 34.2,
  "fundamentals_output": { ... },
  "technical_output": { ... },
  "news_output": { ... },
  "synthesis_output": { ... },
  "critic_output": { ... }
}
```

### `GET /reports`
Returns the authenticated user's saved reports, most recent first.

```bash
curl http://localhost:8000/reports \
  -H "Authorization: Bearer <supabase_access_token>"
```

### `GET /reports/{report_id}`
Returns the full payload of a specific saved report.

```bash
curl http://localhost:8000/reports/abc-123 \
  -H "Authorization: Bearer <supabase_access_token>"
```

### `DELETE /reports/{report_id}`
Deletes a report. Ownership enforced — users can only delete their own reports.

### `GET /stocks`
Returns all NSE-listed tickers for autocomplete. No authentication required.

```bash
curl http://localhost:8000/stocks
# ["SBIN", "HDFCBANK", "RELIANCE", ...]
```

### `GET /health`
Liveness probe for uptime monitoring.

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0"}
```

---

## 📊 Sample Tickers

| Sector | Tickers |
|---|---|
| **Banking** | `SBIN`, `HDFCBANK`, `ICICIBANK`, `KOTAKBANK` |
| **IT** | `INFY`, `TCS`, `WIPRO`, `HCLTECH` |
| **Auto** | `TATAMOTORS`, `MARUTI`, `BAJAJ-AUTO` |
| **Energy** | `RELIANCE`, `ONGC`, `POWERGRID` |
| **FMCG** | `HINDUNILVR`, `ITC`, `NESTLEIND` |
| **US Stocks** | `AAPL`, `MSFT`, `GOOGL` |

---

## 📁 Project Structure

```
multi-agent-stock-research/
├── agents/                      # Core research agents (Pydantic outputs)
│   ├── fundamentals_research.py
│   ├── technical_research.py
│   ├── news_research.py
│   ├── synthesis_research.py
│   └── critic_research.py
├── api/                         # FastAPI layer
│   ├── auth.py                  # JWT verification — Depends(get_current_user)
│   ├── supabase_client.py       # Supabase admin client (server-side only)
│   ├── models.py                # Core request/response Pydantic models
│   ├── config.py                # Settings loaded from .env
│   └── pipeline.py              # LangGraph pipeline runner
├── graph/
│   └── research_graph.py        # LangGraph state machine definition
├── templates/
│   ├── index.html               # Hero landing page + auth modal (public)
│   └── dashboard.html           # Research dashboard (auth-gated)
├── static/                      # Static assets
├── main.py                      # FastAPI entrypoint + all route definitions
├── requirements.txt
├── .env                         # Secret keys — never commit to git
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m 'Add some feature'`
4. Push to the branch — `git push origin feature/your-feature`
5. Open a Pull Request

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is **not financial advice**. Always conduct your own due diligence and consult a SEBI-registered investment advisor before making any investment decisions. Past analysis does not guarantee future performance.

---

<div align="center">

Built with ❤️ for the Indian retail investor

⭐ **Star this repo** if you find it useful!

</div>