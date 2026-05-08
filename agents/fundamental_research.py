# ============================================================
# fundamental_research_agent.py
# Fundamental Research Agent — Indian Stock Market
# Multi-Agent Stock Analysis System
# ============================================================

import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq



# ============================================================
# PYDANTIC MODELS
# ============================================================

class AnomalyFlag(BaseModel):
    flag_id:  str
    severity: str   # "critical" | "medium" | "low" | "positive"
    metric:   str
    message:  str
    action:   str   # "monitor" | "note" | "investigate"


class SignalScores(BaseModel):
    revenue_growth:   float
    profitability:    float
    valuation:        float
    financial_health: float
    overall:          float


class FundamentalsOutput(BaseModel):
    metadata:            dict
    current_snapshot:    Optional[dict]
    quarterly_trend:     Optional[dict]
    industry_comparison: Optional[dict]
    anomaly_flags:       list[AnomalyFlag]
    signal_scores:       Optional[SignalScores]
    llm_summary:         str


# ============================================================
# HELPERS
# ============================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_inr(value: Optional[float]) -> str:
    """Format raw rupee value → ₹X,XX,XXX Cr  (Indian numbering system)."""
    if value is None:
        return "N/A"
    cr = value / 1e7                        # convert to Crores
    if cr >= 1_00_000:                      # ≥ 1 lakh crore
        return f"₹{cr/1_00_000:,.2f} L Cr"
    return f"₹{cr:,.0f} Cr"


def _safe(info: dict, *keys, default=None):
    """Try multiple key names and return first match."""
    for k in keys:
        v = info.get(k)
        if v is not None and v != "":
            return v
    return default


def _quarter_label(date: pd.Timestamp) -> str:
    """Convert a pandas Timestamp to Indian FY quarter label e.g. Q1 FY25."""
    month, year = date.month, date.year
    if month in (4, 5, 6):
        return f"Q1 FY{str(year + 1)[-2:]}"
    elif month in (7, 8, 9):
        return f"Q2 FY{str(year + 1)[-2:]}"
    elif month in (10, 11, 12):
        return f"Q3 FY{str(year + 1)[-2:]}"
    else:
        return f"Q4 FY{str(year)[-2:]}"


def _qoq_growth(values: list) -> list:
    """Return QoQ growth % list; first element is always None."""
    result = [None]
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev and prev != 0:
            result.append(round((values[i] - prev) / abs(prev) * 100, 2))
        else:
            result.append(None)
    return result


def _detect_trend(values: list) -> str:
    """Classify a list of 4 numbers as increasing / decreasing / stable / volatile."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return "insufficient_data"
    increases = sum(clean[i] > clean[i - 1] for i in range(1, len(clean)))
    decreases = sum(clean[i] < clean[i - 1] for i in range(1, len(clean)))
    n = len(clean) - 1
    if increases == n:
        return "increasing"
    if decreases == n:
        return "decreasing"
    if increases >= n * 0.67:
        return "mostly_increasing"
    if decreases >= n * 0.67:
        return "mostly_decreasing"
    return "stable"


def _compare_to_industry(stock_val, industry_val, metric: str) -> str:
    """Return a verdict string comparing stock metric vs industry average."""
    if stock_val is None or industry_val is None or industry_val == 0:
        return "unavailable"
    diff_pct = (stock_val - industry_val) / abs(industry_val) * 100
    # For D/E and P/E — lower is better
    lower_is_better = metric in ("debt_to_equity", "pe_ratio")
    if abs(diff_pct) <= 5:
        return "inline"
    if lower_is_better:
        return "outperforming" if diff_pct < -5 else ("slightly_elevated" if diff_pct < 20 else "elevated")
    else:
        return "outperforming" if diff_pct > 5 else "underperforming"


# ── Industry benchmark averages for major NSE sectors ────────────
INDUSTRY_BENCHMARKS = {
    "Banks":               {"pe": 12.0, "pb": 1.6, "roe": 14.0, "de": 10.5},
    "Financial Services":  {"pe": 18.0, "pb": 2.5, "roe": 16.0, "de": 3.0},
    "IT":                  {"pe": 28.0, "pb": 6.0, "roe": 22.0, "de": 0.1},
    "Pharma":              {"pe": 30.0, "pb": 4.0, "roe": 14.0, "de": 0.5},
    "FMCG":               {"pe": 45.0, "pb": 10.0,"roe": 30.0, "de": 0.2},
    "Auto":                {"pe": 20.0, "pb": 3.5, "roe": 16.0, "de": 1.0},
    "Energy":              {"pe": 10.0, "pb": 1.8, "roe": 12.0, "de": 0.8},
    "Metals":              {"pe": 10.0, "pb": 1.5, "roe": 14.0, "de": 0.7},
    "Realty":              {"pe": 35.0, "pb": 3.0, "roe": 10.0, "de": 1.5},
    "Telecom":             {"pe": 40.0, "pb": 3.5, "roe": 8.0,  "de": 2.5},
    "Default":             {"pe": 22.0, "pb": 3.0, "roe": 15.0, "de": 1.5},
}

def _get_benchmarks(sector: str) -> dict:
    for key in INDUSTRY_BENCHMARKS:
        if key.lower() in (sector or "").lower():
            return INDUSTRY_BENCHMARKS[key]
    return INDUSTRY_BENCHMARKS["Default"]


# ============================================================
# RULE-BASED ANOMALY DETECTOR
# ============================================================

def _detect_anomalies(
    current:     dict,
    q_trend:     dict,
    ind_compare: dict,
) -> list[AnomalyFlag]:
    flags: list[AnomalyFlag] = []

    # ── Revenue trend ─────────────────────────────────────────
    rev_trend = q_trend.get("revenue", {}).get("trend", "")
    rev_vals  = q_trend.get("revenue", {}).get("values", [])
    if rev_trend == "decreasing":
        flags.append(AnomalyFlag(
            flag_id="REVENUE_DECLINING_CRITICAL",
            severity="critical",
            metric="revenue",
            message=f"Revenue has declined for {len(rev_vals)} consecutive quarters ({rev_vals[0]:,.0f} → {rev_vals[-1]:,.0f} Cr)",
            action="investigate",
        ))
    elif rev_trend == "mostly_decreasing":
        flags.append(AnomalyFlag(
            flag_id="REVENUE_DECLINING_MEDIUM",
            severity="medium",
            metric="revenue",
            message="Revenue shows a mostly declining trend over the last 4 quarters.",
            action="monitor",
        ))
    elif rev_trend in ("increasing", "mostly_increasing"):
        flags.append(AnomalyFlag(
            flag_id="REVENUE_GROWING",
            severity="positive",
            metric="revenue",
            message=f"Revenue has grown consistently over the last 4 quarters ({rev_vals[0]:,.0f} → {rev_vals[-1]:,.0f} Cr).",
            action="note",
        ))

    # ── Debt-to-Equity trend ──────────────────────────────────
    de_trend = q_trend.get("debt_to_equity", {}).get("trend", "")
    de_vals  = q_trend.get("debt_to_equity", {}).get("values", [])
    de_curr  = current.get("debt_to_equity")
    if de_trend == "increasing" and de_curr is not None:
        severity = "critical" if de_curr > 3.0 else "medium"
        flags.append(AnomalyFlag(
            flag_id="DEBT_RISING",
            severity=severity,
            metric="debt_to_equity",
            message=f"Debt-to-Equity has risen for 4 consecutive quarters ({de_vals[0]} → {de_vals[-1]})",
            action="investigate" if severity == "critical" else "monitor",
        ))

    # ── EPS trend ─────────────────────────────────────────────
    eps_trend = q_trend.get("eps", {}).get("trend", "")
    eps_vals  = q_trend.get("eps", {}).get("values", [])
    if eps_trend == "decreasing":
        flags.append(AnomalyFlag(
            flag_id="EPS_DECLINING",
            severity="critical",
            metric="eps",
            message=f"EPS has declined for 4 consecutive quarters ({eps_vals[0]} → {eps_vals[-1]})",
            action="investigate",
        ))
    elif eps_trend in ("increasing", "mostly_increasing"):
        flags.append(AnomalyFlag(
            flag_id="EPS_GROWING",
            severity="positive",
            metric="eps",
            message=f"EPS growing consistently over last 4 quarters ({eps_vals[0]} → {eps_vals[-1]})",
            action="note",
        ))

    # ── Margin trend ──────────────────────────────────────────
    margin_trend = q_trend.get("net_profit_margin_percent", {}).get("trend", "")
    margin_vals  = q_trend.get("net_profit_margin_percent", {}).get("values", [])
    if margin_trend == "decreasing" and len(margin_vals) >= 2:
        flags.append(AnomalyFlag(
            flag_id="MARGIN_DECLINING",
            severity="medium",
            metric="net_profit_margin_percent",
            message=f"Net profit margin declining ({margin_vals[0]}% → {margin_vals[-1]}%)",
            action="monitor",
        ))
    elif margin_trend == "stable" and len(margin_vals) >= 2:
        flags.append(AnomalyFlag(
            flag_id="MARGIN_FLAT",
            severity="low",
            metric="net_profit_margin_percent",
            message=f"Net profit margin plateaued ({margin_vals[-2]}% → {margin_vals[-1]}%) after earlier growth.",
            action="monitor",
        ))

    # ── ROE vs industry ───────────────────────────────────────
    roe_verdict = ind_compare.get("roe_percent", {}).get("verdict", "")
    roe_stock   = ind_compare.get("roe_percent", {}).get("stock")
    roe_ind     = ind_compare.get("roe_percent", {}).get("industry")
    if roe_verdict == "outperforming" and roe_stock and roe_ind:
        flags.append(AnomalyFlag(
            flag_id="ROE_STRONG",
            severity="positive",
            metric="roe_percent",
            message=f"ROE of {roe_stock}% is {round(roe_stock - roe_ind, 1)} points above industry average — strong capital efficiency.",
            action="note",
        ))
    elif roe_verdict == "underperforming" and roe_stock and roe_ind:
        flags.append(AnomalyFlag(
            flag_id="ROE_WEAK",
            severity="medium",
            metric="roe_percent",
            message=f"ROE of {roe_stock}% is below industry average of {roe_ind}% — weak capital efficiency.",
            action="monitor",
        ))

    # ── Free Cash Flow ────────────────────────────────────────
    fcf = current.get("free_cash_flow")
    if fcf is not None and fcf < 0:
        flags.append(AnomalyFlag(
            flag_id="NEGATIVE_FCF",
            severity="medium",
            metric="free_cash_flow",
            message=f"Negative Free Cash Flow ({_format_inr(fcf)}) — company spending more than it earns.",
            action="investigate",
        ))

    return flags


# ============================================================
# SIGNAL SCORER  (rule-based, feeds Synthesis Agent)
# ============================================================

def _compute_signal_scores(
    current:     dict,
    q_trend:     dict,
    ind_compare: dict,
    anomaly_flags: list[AnomalyFlag],
) -> SignalScores:

    # ── Revenue Growth (0–10) ─────────────────────────────────
    rev_trend = q_trend.get("revenue", {}).get("trend", "")
    rev_score = {"increasing": 9, "mostly_increasing": 7,
                 "stable": 5, "mostly_decreasing": 3, "decreasing": 1}.get(rev_trend, 5)

    # ── Profitability (0–10): EPS trend + margin ──────────────
    eps_trend    = q_trend.get("eps", {}).get("trend", "")
    margin_trend = q_trend.get("net_profit_margin_percent", {}).get("trend", "")
    eps_score    = {"increasing": 9, "mostly_increasing": 7,
                    "stable": 5, "mostly_decreasing": 3, "decreasing": 1}.get(eps_trend, 5)
    margin_score = {"increasing": 9, "stable": 6,
                    "mostly_decreasing": 4, "decreasing": 2}.get(margin_trend, 5)
    profitability = round((eps_score + margin_score) / 2, 1)

    # ── Valuation (0–10): PE + PB vs industry ─────────────────
    pe_verdict  = ind_compare.get("pe_ratio", {}).get("verdict", "")
    pb_verdict  = ind_compare.get("pb_ratio", {}).get("verdict", "")
    val_map     = {"outperforming": 9, "inline": 6,
                   "slightly_elevated": 4, "elevated": 2, "unavailable": 5}
    valuation   = round((val_map.get(pe_verdict, 5) + val_map.get(pb_verdict, 5)) / 2, 1)

    # ── Financial Health (0–10): D/E trend + FCF ──────────────
    de_trend   = q_trend.get("debt_to_equity", {}).get("trend", "")
    de_score   = {"decreasing": 9, "mostly_decreasing": 7,
                  "stable": 6, "mostly_increasing": 4, "increasing": 2}.get(de_trend, 5)
    fcf        = current.get("free_cash_flow")
    fcf_score  = 8 if (fcf and fcf > 0) else 3 if fcf is not None else 5
    fin_health = round((de_score + fcf_score) / 2, 1)

    overall = round((rev_score + profitability + valuation + fin_health) / 4, 1)

    return SignalScores(
        revenue_growth=rev_score,
        profitability=profitability,
        valuation=valuation,
        financial_health=fin_health,
        overall=overall,
    )


# ============================================================
# LLM SUMMARY GENERATOR
# ============================================================

def _generate_llm_summary(
    llm:           ChatGroq,
    ticker:        str,
    company_name:  str,
    current:       dict,
    q_trend:       dict,
    ind_compare:   dict,
    anomaly_flags: list[AnomalyFlag],
    signal_scores: SignalScores,
) -> str:
    """Ask the LLM to write a 3-4 sentence analyst-grade summary of fundamentals."""

    flags_text = "\n".join(
        f"  - [{f.severity.upper()}] {f.metric}: {f.message}"
        for f in anomaly_flags
    )

    prompt = f"""
You are a senior equity research analyst specializing in Indian stock markets.
Write a concise 3–4 sentence fundamental analysis summary for inclusion in an
institutional research report. Be specific, data-driven, and balanced.
Do NOT use bullet points. Output ONLY the summary paragraph — no preamble,
no headers, no markdown.

COMPANY: {company_name} ({ticker})

CURRENT SNAPSHOT:
- Revenue TTM: {current.get("revenue_ttm_formatted")}
- EPS TTM: ₹{current.get("eps_ttm")}
- P/E: {current.get("pe_ratio")}x   |  P/B: {current.get("pb_ratio")}x
- ROE: {current.get("roe_percent")}%  |  D/E: {current.get("debt_to_equity")}
- Free Cash Flow: {current.get("free_cash_flow_formatted")}

QUARTERLY TRENDS (last 4 quarters):
- Revenue trend   : {q_trend.get("revenue", {}).get("trend")}
- EPS trend       : {q_trend.get("eps", {}).get("trend")}
- Margin trend    : {q_trend.get("net_profit_margin_percent", {}).get("trend")}
- D/E trend       : {q_trend.get("debt_to_equity", {}).get("trend")}

INDUSTRY COMPARISON:
- P/E vs peers  : {ind_compare.get("pe_ratio", {}).get("verdict")}
- P/B vs peers  : {ind_compare.get("pb_ratio", {}).get("verdict")}
- ROE vs peers  : {ind_compare.get("roe_percent", {}).get("verdict")}
- D/E vs peers  : {ind_compare.get("debt_to_equity", {}).get("verdict")}

KEY FLAGS:
{flags_text if flags_text else "  No significant anomalies detected."}

SIGNAL SCORES (out of 10):
- Revenue Growth   : {signal_scores.revenue_growth}
- Profitability    : {signal_scores.profitability}
- Valuation        : {signal_scores.valuation}
- Financial Health : {signal_scores.financial_health}
- Overall          : {signal_scores.overall}
"""

    messages = [
        SystemMessage(content=(
            "You are a senior equity research analyst for Indian markets. "
            "Write crisp, factual, institutionally-toned summaries. "
            "Always mention both strengths and risks. No fluff."
        )),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def fundamental_research_agent(ticker, llm: Optional[ChatGroq] = None):
    """
    LangGraph-compatible Fundamental Research Agent.

    Reads  : state["ticker"]
    Writes : state["fundamentals_output"]

    Args:
        state : LangGraph state dict — must contain key "ticker"
        llm   : ChatAnthropic (or any LangChain LLM) instance.
                If None, LLM summary is skipped gracefully.

    Returns:
        Updated state dict with "fundamentals_output" populated.
    """

    # ticker: str = state.get("ticker", "").strip().upper()

    # ── Ensure NSE suffix for Indian tickers ─────────────────────
    if ticker and not ticker.endswith((".NS", ".BO")):
        ticker = ticker + ".NS"

    timestamp = _now_iso()

    # ── Error builder (DRY helper) ────────────────────────────────
    def _error_state(code: str, message: str) -> dict:
        output = FundamentalsOutput(
            metadata={
                "ticker":            ticker,
                "status":            "error",
                "error_code":        code,
                "error_message":     message,
                "analysis_timestamp": timestamp,
            },
            current_snapshot=None,
            quarterly_trend=None,
            industry_comparison=None,
            anomaly_flags=[],
            signal_scores=None,
            llm_summary="Fundamentals data unavailable — analysis based on remaining signals only.",
        )

        return output
        # return {**state, "fundamentals_output": output.model_dump()}

    # ── 1. FETCH DATA FROM YFINANCE ───────────────────────────────
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        # Validate — yfinance returns an empty / stub dict for bad tickers
        company_name = _safe(info, "longName", "shortName", "displayName")
        if not company_name or _safe(info, "regularMarketPrice", "currentPrice") is None:
            return _error_state(
                "TICKER_NOT_FOUND",
                f"No data found for ticker {ticker} on yfinance. Verify the symbol and suffix (.NS / .BO).",
            )

    except Exception as e:
        return _error_state("FETCH_ERROR", f"yfinance fetch failed: {str(e)}")

    # ── 2. CURRENT SNAPSHOT ───────────────────────────────────────
    try:
        revenue_ttm = _safe(info, "totalRevenue")
        eps_ttm     = _safe(info, "trailingEps")
        pe          = _safe(info, "trailingPE", "forwardPE")
        pb          = _safe(info, "priceToBook")
        roe         = _safe(info, "returnOnEquity")
        de          = _safe(info, "debtToEquity")
        fcf         = _safe(info, "freeCashflow")
        mkt_cap     = _safe(info, "marketCap")
        div_yield   = _safe(info, "dividendYield")
        price       = _safe(info, "currentPrice", "regularMarketPrice")
        exchange    = _safe(info, "exchange", default="NSE")
        currency    = _safe(info, "currency", default="INR")
        sector      = _safe(info, "sector", default="")
        industry    = _safe(info, "industry", default="")

        # yfinance returns ROE as decimal (0.18 = 18%) — normalise
        roe_pct = round(roe * 100, 2) if roe is not None else None
        # D/E from yfinance is already a ratio
        de_val  = round(de, 2) if de is not None else None

        current_snapshot = {
            "revenue_ttm":              revenue_ttm,
            "revenue_ttm_formatted":    _format_inr(revenue_ttm),
            "eps_ttm":                  round(eps_ttm, 2) if eps_ttm else None,
            "pe_ratio":                 round(pe, 2) if pe else None,
            "pb_ratio":                 round(pb, 2) if pb else None,
            "roe_percent":              roe_pct,
            "debt_to_equity":           de_val,
            "free_cash_flow":           fcf,
            "free_cash_flow_formatted": _format_inr(fcf),
            "market_cap":               mkt_cap,
            "market_cap_formatted":     _format_inr(mkt_cap),
            "dividend_yield_percent":   round(div_yield * 100, 2) if div_yield else None,
            "current_price":            price,
        }

    except Exception as e:
        return _error_state("SNAPSHOT_ERROR", f"Failed to build current snapshot: {str(e)}")

    # ── 3. QUARTERLY TREND (last 4 quarters) ─────────────────────
    try:
        financials  = stock.quarterly_financials   # income statement
        balance     = stock.quarterly_balance_sheet

        def _extract_row(df: pd.DataFrame, *row_names) -> Optional[pd.Series]:
            for name in row_names:
                for idx in df.index:
                    if name.lower() in str(idx).lower():
                        return df.loc[idx]
            return None

        def _build_trend_series(series: Optional[pd.Series], unit_divisor=1e7, decimals=0) -> dict:
            """
            Extract last 4 quarters from a pandas Series (columns = dates).
            unit_divisor=1e7 converts raw INR → Crores.
            """
            if series is None or series.empty:
                return {"unit": "INR Crores", "quarters": [], "values": [],
                        "qoq_growth_percent": [], "trend": "insufficient_data"}
            # Most recent 4 columns (yfinance orders newest first)
            cols   = series.index[:4][::-1]          # reverse → oldest first
            labels = [_quarter_label(pd.Timestamp(c)) for c in cols]
            vals   = [
                round(float(series[c]) / unit_divisor, decimals)
                if pd.notna(series[c]) else None
                for c in cols
            ]
            return {
                "unit":               "INR Crores",
                "quarters":           labels,
                "values":             vals,
                "qoq_growth_percent": _qoq_growth(vals),
                "trend":              _detect_trend(vals),
            }

        # Revenue
        rev_series = _extract_row(financials, "Total Revenue", "Revenue")
        revenue_trend = _build_trend_series(rev_series)

        # Net Income (for EPS proxy per quarter)
        ni_series  = _extract_row(financials, "Net Income")
        shares     = _safe(info, "sharesOutstanding", default=1)
        if ni_series is not None and shares:
            cols   = ni_series.index[:4][::-1]
            labels = [_quarter_label(pd.Timestamp(c)) for c in cols]
            eps_vals = [
                round(float(ni_series[c]) / shares, 2) if pd.notna(ni_series[c]) else None
                for c in cols
            ]
            eps_trend = {
                "unit":               "INR per share",
                "quarters":           labels,
                "values":             eps_vals,
                "qoq_growth_percent": _qoq_growth(eps_vals),
                "trend":              _detect_trend(eps_vals),
            }
        else:
            eps_trend = {"unit": "INR per share", "quarters": [], "values": [],
                         "qoq_growth_percent": [], "trend": "insufficient_data"}

        # Net Profit Margin %
        if rev_series is not None and ni_series is not None:
            cols   = rev_series.index[:4][::-1]
            labels = [_quarter_label(pd.Timestamp(c)) for c in cols]
            margin_vals = []
            for c in cols:
                rev_v = float(rev_series[c]) if pd.notna(rev_series.get(c, float("nan"))) else None
                ni_v  = float(ni_series[c])  if pd.notna(ni_series.get(c, float("nan")))  else None
                if rev_v and rev_v != 0 and ni_v is not None:
                    margin_vals.append(round(ni_v / rev_v * 100, 2))
                else:
                    margin_vals.append(None)
            margin_trend = {
                "quarters": labels,
                "values":   margin_vals,
                "trend":    _detect_trend(margin_vals),
            }
        else:
            margin_trend = {"quarters": [], "values": [], "trend": "insufficient_data"}

        # Debt-to-Equity (quarterly balance sheet)
        debt_series   = _extract_row(balance, "Total Debt", "Long Term Debt")
        equity_series = _extract_row(balance, "Stockholders Equity", "Total Equity")
        if debt_series is not None and equity_series is not None:
            cols   = debt_series.index[:4][::-1]
            labels = [_quarter_label(pd.Timestamp(c)) for c in cols]
            de_vals = []
            for c in cols:
                d = float(debt_series[c])   if pd.notna(debt_series.get(c, float("nan")))   else None
                e = float(equity_series[c]) if pd.notna(equity_series.get(c, float("nan"))) else None
                de_vals.append(round(d / e, 2) if d is not None and e and e != 0 else None)
            de_trend = {
                "quarters": labels,
                "values":   de_vals,
                "trend":    _detect_trend(de_vals),
            }
        else:
            de_trend = {"quarters": [], "values": [], "trend": "insufficient_data"}

        quarterly_trend = {
            "revenue":                   revenue_trend,
            "eps":                       eps_trend,
            "net_profit_margin_percent": margin_trend,
            "debt_to_equity":            de_trend,
        }

    except Exception as e:
        quarterly_trend = {
            "revenue": {}, "eps": {}, "net_profit_margin_percent": {}, "debt_to_equity": {}
        }

    # ── 4. INDUSTRY COMPARISON ────────────────────────────────────
    try:
        benchmarks   = _get_benchmarks(sector)
        ind_pe       = benchmarks["pe"]
        ind_pb       = benchmarks["pb"]
        ind_roe      = benchmarks["roe"]
        ind_de       = benchmarks["de"]

        industry_comparison = {
            "industry":     industry or sector or "Unknown",
            "pe_ratio": {
                "stock":    current_snapshot["pe_ratio"],
                "industry": ind_pe,
                "verdict":  _compare_to_industry(current_snapshot["pe_ratio"], ind_pe, "pe_ratio"),
            },
            "pb_ratio": {
                "stock":    current_snapshot["pb_ratio"],
                "industry": ind_pb,
                "verdict":  _compare_to_industry(current_snapshot["pb_ratio"], ind_pb, "pb_ratio"),
            },
            "roe_percent": {
                "stock":    current_snapshot["roe_percent"],
                "industry": ind_roe,
                "verdict":  _compare_to_industry(current_snapshot["roe_percent"], ind_roe, "roe_percent"),
            },
            "debt_to_equity": {
                "stock":    current_snapshot["debt_to_equity"],
                "industry": ind_de,
                "verdict":  _compare_to_industry(current_snapshot["debt_to_equity"], ind_de, "debt_to_equity"),
            },
        }

    except Exception as e:
        industry_comparison = {"industry": "Unknown", "error": str(e)}

    # ── 5. ANOMALY FLAGS ──────────────────────────────────────────
    anomaly_flags = _detect_anomalies(
        current_snapshot, quarterly_trend, industry_comparison
    )

    # ── 6. SIGNAL SCORES ─────────────────────────────────────────
    signal_scores = _compute_signal_scores(
        current_snapshot, quarterly_trend, industry_comparison, anomaly_flags
    )

    # ── 7. LLM SUMMARY ───────────────────────────────────────────
    if llm is not None:
        try:
            llm_summary = _generate_llm_summary(
                llm          = llm,
                ticker       = ticker,
                company_name = company_name,
                current      = current_snapshot,
                q_trend      = quarterly_trend,
                ind_compare  = industry_comparison,
                anomaly_flags= anomaly_flags,
                signal_scores= signal_scores,
            )
        except Exception as e:
            llm_summary = (
                f"LLM summary generation failed ({str(e)}). "
                "Rule-based signal scores are still available for downstream agents."
            )
    else:
        llm_summary = (
            f"{company_name} ({ticker}) — fundamentals extracted successfully. "
            "Pass an LLM instance to generate a narrative summary."
        )

    # ── 8. ASSEMBLE FINAL OUTPUT ──────────────────────────────────
    output = FundamentalsOutput(
        metadata={
            "ticker":             ticker,
            "company_name":       company_name,
            "exchange":           exchange,
            "currency":           currency,
            "sector":             sector,
            "industry":           industry,
            "analysis_timestamp": timestamp,
            "data_source":        "yfinance",
            "status":             "success",
        },
        current_snapshot    = current_snapshot,
        quarterly_trend     = quarterly_trend,
        industry_comparison = industry_comparison,
        anomaly_flags       = anomaly_flags,
        signal_scores       = signal_scores,
        llm_summary         = llm_summary,
    )

    return output
    # return {**state, "fundamentals_output": output.model_dump()}



# ============================================================
# QUICK TEST  — run directly: python fundamental_research_agent.py
# ============================================================
# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     load_dotenv()

#     llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

#     test_state = {"ticker": "SBIN"}
#     result = fundamental_research_agent(test_state, llm=llm)

#     print(json.dumps(result["fundamentals_output"], indent=2, default=str))