# ============================================================
# critic_research_agent.py
# Critic Research Agent — Indian Stock Market
# Multi-Agent Stock Analysis System
# ============================================================

import json
import re
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# PYDANTIC MODELS
# ============================================================

class BearCasePoint(BaseModel):
    point_id:       str             # "BEAR_1" | "BEAR_2" | "BEAR_3"
    argument:       str             # the invalidation argument
    source:         str             # which agent's data supports this bear point
    invalidates:    str             # which bull point this directly counters
    severity:       str             # "high" | "medium" | "low"
    probability:    str             # "likely" | "possible" | "unlikely"


class DataGap(BaseModel):
    gap_id:         str
    gap_type:       str             # "missing_agent" | "stale_data" | "thin_coverage"
                                    # | "missing_metric" | "low_confidence"
    description:    str
    affected_agent: str             # which agent is affected
    impact:         str             # "high" | "medium" | "low" — impact on verdict reliability
    recommendation: str             # what should be done to fill the gap


class StaleDataFlag(BaseModel):
    agent:          str
    metric:         str
    last_updated:   Optional[str]   # ISO date or None
    days_old:       Optional[int]
    threshold_days: int             # acceptable freshness threshold
    is_stale:       bool


class RiskToMonitor(BaseModel):
    risk_id:        str
    category:       str             # "macro" | "regulatory" | "competitive"
                                    # | "financial" | "technical" | "event_driven"
    description:    str
    trigger:        str             # what specific event would activate this risk
    time_horizon:   str             # "immediate" | "near_term" | "medium_term"
    severity:       str             # "critical" | "high" | "medium" | "low"
    source:         str             # which agent flagged this


class ConfidenceChallenge(BaseModel):
    challenge:      str             # specific challenge to the synthesis confidence
    reason:         str             # why the confidence may be inflated/deflated
    suggested_adjustment: str       # "reduce" | "maintain" | "increase"
    adjustment_pct: float           # suggested confidence adjustment (-30 to +10)


class CriticVerdict(BaseModel):
    overall_stance:         str     # "agree" | "cautious" | "disagree"
    synthesis_verdict:      str     # original verdict being reviewed
    critic_adjusted_verdict:str     # critic's adjusted verdict (may be same)
    confidence_delta:       float   # positive = more confident, negative = less
    critical_miss:          Optional[str]   # most important thing synthesis missed


class CriticOutput(BaseModel):
    metadata:               dict
    bear_case:              list[BearCasePoint]
    data_gaps:              list[DataGap]
    stale_data_flags:       list[StaleDataFlag]
    risks_to_monitor:       list[RiskToMonitor]
    confidence_challenges:  list[ConfidenceChallenge]
    critic_verdict:         CriticVerdict
    llm_summary:            str


# ============================================================
# HELPERS
# ============================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```$",           "", raw, flags=re.MULTILINE)
    return raw.strip()


def _safe_get(obj, *keys, default=None):
    for key in keys:
        if obj is None:
            return default
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj if obj is not None else default


def _days_since(iso_timestamp: Optional[str]) -> Optional[int]:
    if not iso_timestamp:
        return None
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return None


# ============================================================
# STEP 1 — DATA GAP DETECTOR
# ============================================================

def _detect_data_gaps(state: dict) -> list[DataGap]:
    """
    Systematically check every agent output for missing,
    incomplete, or low-confidence data.
    """
    gaps: list[DataGap] = []

    # ── Missing agent outputs ─────────────────────────────────
    agent_map = {
        "fundamentals_output": "fundamentals",
        "news_output":         "news",
        "technical_output":    "technical",
    }
    for key, label in agent_map.items():
        if state.get(key) is None:
            gaps.append(DataGap(
                gap_id        = f"MISSING_{label.upper()}_AGENT",
                gap_type      = "missing_agent",
                description   = f"Complete {label} agent output is missing from state.",
                affected_agent= label,
                impact        = "high",
                recommendation= f"Re-run {label}_research_agent before synthesizing.",
            ))

    # ── Fundamentals gaps ─────────────────────────────────────
    fund = state.get("fundamentals_output")
    if fund:
        status = _safe_get(fund, "metadata", "status", default="")
        if status == "error":
            gaps.append(DataGap(
                gap_id        = "FUNDAMENTALS_ERROR_STATUS",
                gap_type      = "missing_agent",
                description   = f"Fundamentals agent returned error: {_safe_get(fund, 'metadata', 'error_message', default='Unknown')}",
                affected_agent= "fundamentals",
                impact        = "high",
                recommendation= "Verify ticker symbol and retry fundamentals agent.",
            ))

        snapshot = _safe_get(fund, "current_snapshot") or {}
        missing_metrics = [
            m for m in ["pe_ratio", "pb_ratio", "roe_percent", "debt_to_equity", "free_cash_flow"]
            if snapshot.get(m) is None
        ]
        if missing_metrics:
            gaps.append(DataGap(
                gap_id        = "FUNDAMENTALS_MISSING_METRICS",
                gap_type      = "missing_metric",
                description   = f"Key metrics unavailable: {', '.join(missing_metrics)}.",
                affected_agent= "fundamentals",
                impact        = "medium",
                recommendation= "Try alternate data source (Financial Modeling Prep / Screener.in) for missing metrics.",
            ))

        q_trend = _safe_get(fund, "quarterly_trend") or {}
        thin_quarters = [
            metric for metric, data in q_trend.items()
            if isinstance(data, dict) and len(data.get("values", [])) < 4
        ]
        if thin_quarters:
            gaps.append(DataGap(
                gap_id        = "QUARTERLY_TREND_THIN",
                gap_type      = "thin_coverage",
                description   = f"Less than 4 quarters of data for: {', '.join(thin_quarters)}.",
                affected_agent= "fundamentals",
                impact        = "medium",
                recommendation= "Trend analysis unreliable — consider extending lookback or using annual data.",
            ))

    # ── News gaps ─────────────────────────────────────────────
    news = state.get("news_output")
    if news:
        articles_analyzed = _safe_get(news, "metadata", "articles_analyzed", default=0)
        status            = _safe_get(news, "metadata", "status", default="")

        if status in ("error", "insufficient_data"):
            gaps.append(DataGap(
                gap_id        = "NEWS_INSUFFICIENT_DATA",
                gap_type      = "thin_coverage",
                description   = f"News agent returned status '{status}' — sentiment signal unreliable.",
                affected_agent= "news",
                impact        = "high",
                recommendation= "Retry with broader query or alternate domains for Indian financial news.",
            ))
        elif articles_analyzed < 5:
            gaps.append(DataGap(
                gap_id        = "NEWS_THIN_COVERAGE",
                gap_type      = "thin_coverage",
                description   = f"Only {articles_analyzed} articles analyzed — sentiment may not be representative.",
                affected_agent= "news",
                impact        = "medium",
                recommendation= "Increase max_results in TavilySearch or add more news domains.",
            ))

        rejected = _safe_get(news, "metadata", "articles_rejected", default=0)
        fetched  = _safe_get(news, "metadata", "articles_fetched", default=1)
        if fetched > 0 and (rejected / max(fetched, 1)) > 0.6:
            gaps.append(DataGap(
                gap_id        = "NEWS_HIGH_REJECTION_RATE",
                gap_type      = "thin_coverage",
                description   = f"{rejected}/{fetched} articles rejected — Tavily query may be returning irrelevant results.",
                affected_agent= "news",
                impact        = "medium",
                recommendation= "Refine Tavily query with company full name + 'NSE' + 'India'.",
            ))

    # ── Technical gaps ────────────────────────────────────────
    tech = state.get("technical_output")
    if tech:
        trading_days = _safe_get(tech, "metadata", "total_trading_days", default=0)
        if trading_days < 200:
            gaps.append(DataGap(
                gap_id        = "TECHNICAL_INSUFFICIENT_HISTORY",
                gap_type      = "thin_coverage",
                description   = f"Only {trading_days} trading days available — MA200 and long-term signals unreliable.",
                affected_agent= "technical",
                impact        = "medium",
                recommendation= "Stock may be recently listed. Use shorter MAs (50/100) for trend analysis.",
            ))

        ma200 = _safe_get(tech, "moving_averages", "ma_200")
        if ma200 is None:
            gaps.append(DataGap(
                gap_id        = "TECHNICAL_MA200_UNAVAILABLE",
                gap_type      = "missing_metric",
                description   = "200-day MA unavailable — Golden/Death Cross signals cannot be calculated.",
                affected_agent= "technical",
                impact        = "medium",
                recommendation= "Extend data fetch period or use 100-day MA as proxy.",
            ))

    # ── Synthesis gaps ────────────────────────────────────────
    synth = state.get("synthesis_research")
    if synth:
        agents_missing = _safe_get(synth, "metadata", "agents_missing") or []
        if agents_missing:
            gaps.append(DataGap(
                gap_id        = "SYNTHESIS_INCOMPLETE_INPUTS",
                gap_type      = "missing_agent",
                description   = f"Synthesis ran with missing agents: {', '.join(agents_missing)}. Verdict based on partial data.",
                affected_agent= "synthesis",
                impact        = "high",
                recommendation= "Complete all agent runs before synthesis for reliable recommendation.",
            ))

        confidence = _safe_get(synth, "recommendation", "confidence_pct", default=0)
        if confidence < 40:
            gaps.append(DataGap(
                gap_id        = "LOW_SYNTHESIS_CONFIDENCE",
                gap_type      = "low_confidence",
                description   = f"Synthesis confidence is only {confidence}% — verdict should not be acted upon.",
                affected_agent= "synthesis",
                impact        = "high",
                recommendation= "Gather more data, resolve agent divergences, then re-run synthesis.",
            ))

    return gaps


# ============================================================
# STEP 2 — STALE DATA DETECTOR
# ============================================================

def _detect_stale_data(state: dict) -> list[StaleDataFlag]:
    """
    Check each agent's analysis timestamp against acceptable thresholds.
    Fundamental data: 7 days | News: 1 day | Technical: 1 day
    """
    flags: list[StaleDataFlag] = []

    thresholds = {
        "fundamentals_output": ("fundamentals", 7),    # financials change slowly
        "news_output":         ("news",          1),    # news is time-sensitive
        "technical_output":    ("technical",     1),    # price data must be fresh
    }

    for key, (label, threshold_days) in thresholds.items():
        output = state.get(key)
        if output is None:
            continue

        timestamp = _safe_get(output, "metadata", "analysis_timestamp")
        days_old  = _days_since(timestamp)
        is_stale  = days_old is not None and days_old > threshold_days

        flags.append(StaleDataFlag(
            agent          = label,
            metric         = "analysis_timestamp",
            last_updated   = timestamp,
            days_old       = days_old,
            threshold_days = threshold_days,
            is_stale       = is_stale,
        ))

    return flags


# ============================================================
# STEP 3 — BEAR CASE GENERATOR (Rule-based)
# ============================================================

def _generate_rule_based_bear_case(
    state:       dict,
    synth_bull:  list[dict],
) -> list[BearCasePoint]:
    """
    Generate bear points that directly counter the synthesis bull case.
    Rule-based first — LLM will enhance these later.
    """
    bear_points: list[BearCasePoint] = []

    fund  = state.get("fundamentals_output") or {}
    news  = state.get("news_output")         or {}
    tech  = state.get("technical_output")    or {}
    synth = state.get("synthesis_research")  or {}

    # ── Fundamental counter-signals ───────────────────────────
    de = _safe_get(fund, "current_snapshot", "debt_to_equity")
    if de and de > 2.0:
        bear_points.append(BearCasePoint(
            point_id    = "BEAR_1",
            argument    = f"Debt-to-Equity of {de} creates vulnerability in a rising interest rate environment. RBI rate hikes could significantly increase debt servicing costs and compress margins.",
            source      = "fundamentals",
            invalidates = "Revenue growth bull case — growth financed by debt is fragile.",
            severity    = "high" if de > 4.0 else "medium",
            probability = "possible",
        ))

    rev_trend = _safe_get(fund, "quarterly_trend", "revenue", "trend", default="")
    margin_trend = _safe_get(fund, "quarterly_trend", "net_profit_margin_percent", "trend", default="")
    if rev_trend in ("increasing", "mostly_increasing") and margin_trend in ("decreasing", "stable"):
        bear_points.append(BearCasePoint(
            point_id    = "BEAR_2",
            argument    = "Revenue growth is not translating to margin expansion — costs are rising faster than revenue. This signals deteriorating operational efficiency and earnings quality.",
            source      = "fundamentals",
            invalidates = "EPS growth bull case — revenue growth without margin expansion is unsustainable.",
            severity    = "medium",
            probability = "likely",
        ))

    pe = _safe_get(fund, "current_snapshot", "pe_ratio")
    ind_pe_verdict = _safe_get(fund, "industry_comparison", "pe_ratio", "verdict", default="")
    if pe and pe > 25 and ind_pe_verdict in ("elevated", "slightly_elevated"):
        bear_points.append(BearCasePoint(
            point_id    = "BEAR_3",
            argument    = f"P/E of {pe}x is elevated vs industry. Any earnings miss or guidance cut will trigger a sharp de-rating — downside risk is asymmetric at current valuations.",
            source      = "fundamentals",
            invalidates = "Valuation bull case — premium valuations leave no margin of safety.",
            severity    = "medium",
            probability = "possible",
        ))

    # ── News counter-signals ──────────────────────────────────
    for event in (_safe_get(news, "all_high_impact_events") or []):
        impact   = event.get("impact")   if isinstance(event, dict) else getattr(event, "impact", "")
        severity = event.get("severity") if isinstance(event, dict) else getattr(event, "severity", "")
        desc     = event.get("description") if isinstance(event, dict) else getattr(event, "description", "")
        etype    = event.get("event_type") if isinstance(event, dict) else getattr(event, "event_type", "")

        if impact == "bearish" and severity == "high":
            bear_points.append(BearCasePoint(
                point_id    = f"BEAR_NEWS_{etype.upper()}",
                argument    = f"High-severity bearish event detected: {desc}. This could materially impact near-term price performance and investor sentiment.",
                source      = "news",
                invalidates = "Positive news sentiment bull case.",
                severity    = "high",
                probability = "likely",
            ))

    # ── Technical counter-signals ─────────────────────────────
    rsi_val = _safe_get(tech, "rsi", "current")
    if rsi_val and rsi_val > 68:
        bear_points.append(BearCasePoint(
            point_id    = "BEAR_TECH_OVERBOUGHT",
            argument    = f"RSI at {rsi_val} is in overbought territory. Historically, RSI above 70 precedes mean reversion — short-term pullback of 5-8% is likely before next leg up.",
            source      = "technical",
            invalidates = "Uptrend continuation bull case — momentum indicators signal near-term exhaustion.",
            severity    = "medium",
            probability = "likely",
        ))

    for cross in (_safe_get(tech, "cross_signals") or []):
        detected    = cross.get("detected")    if isinstance(cross, dict) else getattr(cross, "detected", False)
        signal_type = cross.get("signal_type") if isinstance(cross, dict) else getattr(cross, "signal_type", "")
        days_ago    = cross.get("days_ago")    if isinstance(cross, dict) else getattr(cross, "days_ago", None)

        if detected and signal_type == "death_cross":
            bear_points.append(BearCasePoint(
                point_id    = "BEAR_DEATH_CROSS",
                argument    = f"Death Cross confirmed {days_ago} days ago — MA50 below MA200. Institutional algorithms systematically sell on this signal, creating persistent selling pressure.",
                source      = "technical",
                invalidates = "Technical trend bull case — long-term structure is bearish.",
                severity    = "high",
                probability = "likely",
            ))

    # ── Synthesis divergence as bear point ────────────────────
    divergence_flag   = _safe_get(synth, "signal_agreement", "divergence_flag", default=False)
    divergence_detail = _safe_get(synth, "signal_agreement", "divergence_detail")
    if divergence_flag and divergence_detail:
        bear_points.append(BearCasePoint(
            point_id    = "BEAR_SIGNAL_DIVERGENCE",
            argument    = f"Agent signals are significantly divergent: {divergence_detail} Divergence between fundamental value and market technicals often resolves against the prevailing trend.",
            source      = "synthesis",
            invalidates = "Consensus bull case — divergent signals undermine conviction.",
            severity    = "medium",
            probability = "possible",
        ))

    # Ensure exactly 3 — pad with macro if needed
    if len(bear_points) < 3:
        bear_points.append(BearCasePoint(
            point_id    = "BEAR_MACRO_INDIA",
            argument    = "Indian equities face macro headwinds: RBI monetary tightening, FII outflows, INR depreciation pressure, and global risk-off sentiment could compress multiples regardless of company-specific fundamentals.",
            source      = "macro",
            invalidates = "Overall bull case — macro environment can override company fundamentals.",
            severity    = "medium",
            probability = "possible",
        ))

    # Sort by severity and return top 3
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    bear_points.sort(key=lambda x: severity_rank.get(x.severity, 2))
    return bear_points[:3]


# ============================================================
# STEP 4 — RISKS TO MONITOR (Rule-based)
# ============================================================

def _generate_risks_to_monitor(state: dict) -> list[RiskToMonitor]:
    """Extract and categorize all risks across all agent outputs."""
    risks: list[RiskToMonitor] = []

    fund  = state.get("fundamentals_output") or {}
    news  = state.get("news_output")         or {}
    tech  = state.get("technical_output")    or {}

    # ── From fundamentals anomaly flags ───────────────────────
    for flag in (_safe_get(fund, "anomaly_flags") or []):
        flag_id  = flag.get("flag_id")  if isinstance(flag, dict) else getattr(flag, "flag_id", "")
        severity = flag.get("severity") if isinstance(flag, dict) else getattr(flag, "severity", "")
        message  = flag.get("message")  if isinstance(flag, dict) else getattr(flag, "message", "")
        action   = flag.get("action")   if isinstance(flag, dict) else getattr(flag, "action", "")

        if severity in ("critical", "medium") and action in ("investigate", "monitor"):
            risks.append(RiskToMonitor(
                risk_id      = f"RISK_FUND_{flag_id}",
                category     = "financial",
                description  = message,
                trigger      = "Next quarterly earnings release or balance sheet update.",
                time_horizon = "near_term" if severity == "critical" else "medium_term",
                severity     = "critical" if severity == "critical" else "medium",
                source       = "fundamentals",
            ))

    # ── From news high-impact events ──────────────────────────
    for event in (_safe_get(news, "all_high_impact_events") or []):
        impact   = event.get("impact")      if isinstance(event, dict) else getattr(event, "impact", "")
        severity = event.get("severity")    if isinstance(event, dict) else getattr(event, "severity", "")
        etype    = event.get("event_type")  if isinstance(event, dict) else getattr(event, "event_type", "")
        desc     = event.get("description") if isinstance(event, dict) else getattr(event, "description", "")

        if impact == "bearish" or severity in ("high", "medium"):
            category_map = {
                "regulatory_action": "regulatory",
                "lawsuit":           "regulatory",
                "leadership_change": "event_driven",
                "merger_acquisition":"event_driven",
                "macro_event":       "macro",
                "divestment":        "event_driven",
                "earnings":          "financial",
                "financial_result":  "financial",
            }
            risks.append(RiskToMonitor(
                risk_id      = f"RISK_NEWS_{etype.upper()}",
                category     = category_map.get(etype, "event_driven"),
                description  = desc,
                trigger      = f"Follow-up announcement or regulatory response related to {etype.replace('_', ' ')}.",
                time_horizon = "immediate" if severity == "high" else "near_term",
                severity     = severity if severity in ("critical", "high", "medium", "low") else "medium",
                source       = "news",
            ))

    # ── From technical flags ──────────────────────────────────
    for flag in (_safe_get(tech, "technical_flags") or []):
        flag_id  = flag.get("flag_id")  if isinstance(flag, dict) else getattr(flag, "flag_id", "")
        severity = flag.get("severity") if isinstance(flag, dict) else getattr(flag, "severity", "")
        message  = flag.get("message")  if isinstance(flag, dict) else getattr(flag, "message", "")
        action   = flag.get("action")   if isinstance(flag, dict) else getattr(flag, "action", "")

        if action in ("sell_signal", "monitor") and severity in ("critical", "medium"):
            risks.append(RiskToMonitor(
                risk_id      = f"RISK_TECH_{flag_id}",
                category     = "technical",
                description  = message,
                trigger      = "Price breaking key support level or volume confirmation of breakdown.",
                time_horizon = "immediate" if severity == "critical" else "near_term",
                severity     = severity,
                source       = "technical",
            ))

    # ── Macro risks always included for Indian market ─────────
    macro_risks = [
        RiskToMonitor(
            risk_id      = "RISK_MACRO_RBI",
            category     = "macro",
            description  = "RBI monetary policy change — unexpected rate hike or hawkish guidance could compress equity valuations, especially for leveraged companies.",
            trigger      = "RBI MPC meeting decision or inflation data surprise.",
            time_horizon = "near_term",
            severity     = "medium",
            source       = "macro",
        ),
        RiskToMonitor(
            risk_id      = "RISK_MACRO_FII",
            category     = "macro",
            description  = "FII outflow risk — global risk-off events (Fed policy, geopolitical tensions) trigger FII selling in Indian markets, causing broad-based corrections.",
            trigger      = "US Fed hawkish surprise or emerging market risk-off episode.",
            time_horizon = "near_term",
            severity     = "medium",
            source       = "macro",
        ),
    ]
    risks.extend(macro_risks)

    # Deduplicate and sort
    seen  = set()
    dedup = []
    for r in risks:
        if r.risk_id not in seen:
            seen.add(r.risk_id)
            dedup.append(r)

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    dedup.sort(key=lambda x: severity_rank.get(x.severity, 3))
    return dedup[:6]


# ============================================================
# STEP 5 — CONFIDENCE CHALLENGE ANALYZER
# ============================================================

def _analyze_confidence_challenges(
    state:      dict,
    data_gaps:  list[DataGap],
    stale_flags:list[StaleDataFlag],
) -> list[ConfidenceChallenge]:
    """
    Identify specific reasons why synthesis confidence
    may be inflated or unreliable.
    """
    challenges: list[ConfidenceChallenge] = []
    synth = state.get("synthesis_research") or {}

    stated_confidence = _safe_get(synth, "recommendation", "confidence_pct", default=50)
    agents_missing    = _safe_get(synth, "metadata", "agents_missing") or []
    divergence_flag   = _safe_get(synth, "signal_agreement", "divergence_flag", default=False)

    # Missing agents challenge
    if agents_missing:
        challenges.append(ConfidenceChallenge(
            challenge  = f"Confidence of {stated_confidence}% was calculated with missing agents: {', '.join(agents_missing)}.",
            reason     = "Synthesis without all agents produces incomplete signal triangulation.",
            suggested_adjustment = "reduce",
            adjustment_pct = -20.0 * len(agents_missing),
        ))

    # Divergence challenge
    if divergence_flag:
        challenges.append(ConfidenceChallenge(
            challenge  = "Significant signal divergence detected between agents.",
            reason     = "When agents disagree by >3 points, the verdict is inherently uncertain — one agent's signal will be wrong.",
            suggested_adjustment = "reduce",
            adjustment_pct = -15.0,
        ))

    # Stale data challenges
    stale_agents = [f.agent for f in stale_flags if f.is_stale]
    if stale_agents:
        challenges.append(ConfidenceChallenge(
            challenge  = f"Stale data detected for: {', '.join(stale_agents)}.",
            reason     = "Analysis based on outdated data may not reflect current market conditions.",
            suggested_adjustment = "reduce",
            adjustment_pct = -10.0 * len(stale_agents),
        ))

    # High-impact data gap challenge
    high_impact_gaps = [g for g in data_gaps if g.impact == "high"]
    if high_impact_gaps:
        challenges.append(ConfidenceChallenge(
            challenge  = f"{len(high_impact_gaps)} high-impact data gap(s) identified.",
            reason     = "Critical data gaps mean the synthesis is built on an incomplete picture.",
            suggested_adjustment = "reduce",
            adjustment_pct = -10.0,
        ))

    # Overconfidence check
    if stated_confidence >= 80 and divergence_flag:
        challenges.append(ConfidenceChallenge(
            challenge  = f"Confidence of {stated_confidence}% appears inflated given agent divergence.",
            reason     = "High confidence requires agent consensus — divergence contradicts this level of certainty.",
            suggested_adjustment = "reduce",
            adjustment_pct = -25.0,
        ))

    return challenges


# ============================================================
# STEP 6 — CRITIC VERDICT
# ============================================================

def _generate_critic_verdict(
    synth:               dict,
    bear_points:         list[BearCasePoint],
    data_gaps:           list[DataGap],
    confidence_challenges: list[ConfidenceChallenge],
) -> CriticVerdict:

    synthesis_verdict    = _safe_get(synth, "recommendation", "verdict", default="Hold")

    # Calculate confidence delta from all challenges
    total_delta = sum(c.adjustment_pct for c in confidence_challenges)
    total_delta = max(-60.0, min(10.0, total_delta))   # cap between -60 and +10

    # High-severity bear points reduce confidence further
    high_sev_bears  = sum(1 for b in bear_points if b.severity == "high")
    total_delta    -= high_sev_bears * 5.0

    # High-impact gaps reduce confidence
    high_gaps    = sum(1 for g in data_gaps if g.impact == "high")
    total_delta -= high_gaps * 3.0

    total_delta = max(-60.0, min(10.0, total_delta))

    # Determine overall stance
    if total_delta <= -20:
        stance = "disagree"
    elif total_delta <= -8:
        stance = "cautious"
    else:
        stance = "agree"

    # Critic may adjust verdict based on stance
    verdict_map = {
        "Strong Buy": ["Strong Buy", "Buy",      "Hold"],
        "Buy":        ["Buy",        "Hold",      "Hold"],
        "Hold":       ["Hold",       "Hold",      "Sell"],
        "Sell":       ["Sell",       "Sell",      "Strong Sell"],
        "Strong Sell":["Strong Sell","Strong Sell","Strong Sell"],
    }
    stance_idx = {"agree": 0, "cautious": 1, "disagree": 2}
    adjusted_verdict = verdict_map.get(
        synthesis_verdict, ["Hold", "Hold", "Hold"]
    )[stance_idx.get(stance, 1)]

    # Most critical miss
    critical_miss = None
    if high_gaps > 0:
        critical_miss = f"High-impact data gap: {data_gaps[0].description}"
    elif high_sev_bears > 0:
        critical_miss = bear_points[0].argument[:100] + "..."

    return CriticVerdict(
        overall_stance          = stance,
        synthesis_verdict       = synthesis_verdict,
        critic_adjusted_verdict = adjusted_verdict,
        confidence_delta        = round(total_delta, 1),
        critical_miss           = critical_miss,
    )


# ============================================================
# STEP 7 — LLM: ENHANCED BEAR CASE + SUMMARY
# ============================================================

def _llm_enhance_and_summarize(
    llm:           ChatGroq,
    ticker:        str,
    company:       str,
    synth:         dict,
    bear_points:   list[BearCasePoint],
    risks:         list[RiskToMonitor],
    data_gaps:     list[DataGap],
    critic_verdict:CriticVerdict,
) -> tuple[list[BearCasePoint], str]:
    """
    LLM does two things in one call:
    1. Enhances/validates the rule-based bear case arguments
    2. Generates the critic narrative summary
    """

    verdict         = _safe_get(synth, "recommendation", "verdict",       default="Hold")
    confidence      = _safe_get(synth, "recommendation", "confidence_pct",default=50)
    reasoning       = _safe_get(synth, "recommendation", "reasoning",     default="")
    bull_case       = _safe_get(synth, "recommendation", "bull_case")     or []
    consensus_score = _safe_get(synth, "signal_agreement", "consensus_score", default=5.0)

    bull_text  = "\n".join(
        f"  {i+1}. [{b.get('source') if isinstance(b,dict) else b.source}] "
        f"{b.get('point') if isinstance(b,dict) else b.point}"
        for i, b in enumerate(bull_case[:3])
    ) or "  None provided."

    bear_text  = "\n".join(
        f"  {i+1}. [{b.source}|{b.severity}] {b.argument}"
        for i, b in enumerate(bear_points)
    )

    risk_text  = "\n".join(
        f"  - [{r.severity}|{r.time_horizon}] {r.description}"
        for r in risks[:4]
    )

    gap_text   = "\n".join(
        f"  - [{g.impact}] {g.description}"
        for g in data_gaps[:4]
    ) or "  No significant gaps."

    prompt = f"""
You are a contrarian equity research analyst — your job is to stress-test investment theses
for Indian stocks and find what the bull case is missing.

STOCK: {company} ({ticker})
SYNTHESIS VERDICT: {verdict} ({confidence}% confidence | consensus score: {consensus_score}/10)

SYNTHESIS REASONING:
{reasoning}

SYNTHESIS BULL CASE:
{bull_text}

RULE-BASED BEAR CASE (validate and enhance these):
{bear_text}

RISKS TO MONITOR:
{risk_text}

DATA GAPS:
{gap_text}

CRITIC VERDICT: {critic_verdict.overall_stance} | Adjusted to: {critic_verdict.critic_adjusted_verdict}

Generate a JSON response with EXACTLY this structure:
{{
  "enhanced_bear_case": [
    {{
      "point_id": "BEAR_1",
      "argument": "<enhanced, specific, data-driven bear argument. Reference Indian market context.>",
      "source": "<fundamentals|news|technical|macro>",
      "invalidates": "<which specific bull point this counters>",
      "severity": "<high|medium|low>",
      "probability": "<likely|possible|unlikely>"
    }},
    {{
      "point_id": "BEAR_2",
      "argument": "<enhanced bear argument>",
      "source": "<source>",
      "invalidates": "<bull point>",
      "severity": "<severity>",
      "probability": "<probability>"
    }},
    {{
      "point_id": "BEAR_3",
      "argument": "<enhanced bear argument>",
      "source": "<source>",
      "invalidates": "<bull point>",
      "severity": "<severity>",
      "probability": "<probability>"
    }}
  ],
  "llm_summary": "<3-4 sentence critic summary. Start with overall stance. Identify the single most important risk the synthesis may have underweighted. State what would need to be true for the bear case to play out. End with what investors should watch.>"
}}

Rules:
- enhanced_bear_case: improve the rule-based arguments with specific data and Indian market context
- Each bear point must directly counter a specific bull point
- Be specific — mention RBI, SEBI, FII, Budget, sector-specific risks where relevant
- llm_summary: balanced — acknowledge the bull case before presenting the critic view
- Return pure JSON only, no markdown
"""

    messages = [
        SystemMessage(content=(
            "You are a senior contrarian analyst. Your job is to find risks, not confirm biases. "
            "Be rigorous, specific, and grounded in Indian market realities. Return only valid JSON."
        )),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    parsed   = json.loads(_clean_json(response.content))

    # Rebuild enhanced bear case as BearCasePoint objects
    enhanced_bears: list[BearCasePoint] = []
    for b in parsed.get("enhanced_bear_case", [])[:3]:
        enhanced_bears.append(BearCasePoint(
            point_id    = b.get("point_id", "BEAR_X"),
            argument    = b.get("argument", ""),
            source      = b.get("source", "macro"),
            invalidates = b.get("invalidates", ""),
            severity    = b.get("severity", "medium"),
            probability = b.get("probability", "possible"),
        ))

    llm_summary = parsed.get("llm_summary", "")
    return enhanced_bears, llm_summary


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def critic_research_agent(
    state: dict,
    llm:   Optional[ChatGroq],
) -> dict:
    """
    LangGraph-compatible Critic Research Agent.

    Reads  : state["synthesis_research"]  (primary)
              state["fundamentals_output"]
              state["news_output"]
              state["technical_output"]

    Writes : state["critic_output"]

    Args:
        state : LangGraph ResearchState dict
        llm   : ChatGroq instance

    Returns:
        Updated state dict with "critic_output" populated.
    """

    ticker    = state.get("ticker", "UNKNOWN").upper()
    timestamp = _now_iso()

    if not ticker.endswith((".NS", ".BO")):
        ticker = ticker + ".NS"

    # Company name
    fund = state.get("fundamentals_output")
    company_name = ticker.replace(".NS", "")
    if fund:
        meta = (fund.get("metadata") or {}) if isinstance(fund, dict) \
               else (fund.metadata or {})
        company_name = meta.get("company_name", company_name)

    # ── Error builder ─────────────────────────────────────────────
    def _error_output(code: str, message: str) -> dict:
        out = CriticOutput(
            metadata={
                "ticker":             ticker,
                "status":             "error",
                "error_code":         code,
                "error_message":      message,
                "analysis_timestamp": timestamp,
            },
            bear_case=[], data_gaps=[], stale_data_flags=[],
            risks_to_monitor=[], confidence_challenges=[],
            critic_verdict=CriticVerdict(
                overall_stance="cautious",
                synthesis_verdict="Hold",
                critic_adjusted_verdict="Hold",
                confidence_delta=-10.0,
                critical_miss=message,
            ),
            llm_summary=f"Critic analysis unavailable for {ticker}. Reason: {message}",
        )

        return out
        # return {**state, "critic_output": out.model_dump()}

    synth = state.get("synthesis_output")
    if synth is None:
        return _error_output(
            "MISSING_SYNTHESIS_OUTPUT",
            "synthesis_research is None — run synthesis_research_agent first.",
        )

    # ============================================================
    # STEP 1 — DATA GAP DETECTION
    # ============================================================
    data_gaps = _detect_data_gaps(state)

    # ============================================================
    # STEP 2 — STALE DATA DETECTION
    # ============================================================
    stale_flags = _detect_stale_data(state)

    # ============================================================
    # STEP 3 — RULE-BASED BEAR CASE
    # ============================================================
    synth_bull = _safe_get(synth, "recommendation", "bull_case") or []
    bear_case  = _generate_rule_based_bear_case(state, synth_bull)

    # ============================================================
    # STEP 4 — RISKS TO MONITOR
    # ============================================================
    risks = _generate_risks_to_monitor(state)

    # ============================================================
    # STEP 5 — CONFIDENCE CHALLENGES
    # ============================================================
    confidence_challenges = _analyze_confidence_challenges(
        state, data_gaps, stale_flags
    )

    # ============================================================
    # STEP 6 — CRITIC VERDICT (rule-based)
    # ============================================================
    critic_verdict = _generate_critic_verdict(
        synth, bear_case, data_gaps, confidence_challenges
    )

    # ============================================================
    # STEP 7 — LLM: ENHANCE BEAR CASE + GENERATE SUMMARY
    # ============================================================
    llm_summary = ""

    if llm is not None:
        try:
            bear_case, llm_summary = _llm_enhance_and_summarize(
                llm            = llm,
                ticker         = ticker,
                company        = company_name,
                synth          = synth,
                bear_points    = bear_case,
                risks          = risks,
                data_gaps      = data_gaps,
                critic_verdict = critic_verdict,
            )
        except Exception as e:
            llm_summary = (
                f"LLM enhancement failed ({e}). "
                f"Critic stance: {critic_verdict.overall_stance}. "
                f"Adjusted verdict: {critic_verdict.critic_adjusted_verdict}. "
                f"Confidence delta: {critic_verdict.confidence_delta}%."
            )
    else:
        llm_summary = (
            f"Critic analysis for {company_name} ({ticker}): "
            f"Stance is {critic_verdict.overall_stance}. "
            f"Synthesis verdict {critic_verdict.synthesis_verdict} adjusted to "
            f"{critic_verdict.critic_adjusted_verdict} "
            f"(confidence delta: {critic_verdict.confidence_delta}%). "
            f"{critic_verdict.critical_miss or ''}"
        )

    # ============================================================
    # STEP 8 — ASSEMBLE FINAL OUTPUT
    # ============================================================
    output = CriticOutput(
        metadata={
            "ticker":                   ticker,
            "company_name":             company_name,
            "analysis_timestamp":       timestamp,
            "synthesis_verdict_reviewed":_safe_get(synth, "recommendation", "verdict"),
            "data_gaps_found":          len(data_gaps),
            "stale_agents":             [f.agent for f in stale_flags if f.is_stale],
            "status":                   "success",
        },
        bear_case             = bear_case,
        data_gaps             = data_gaps,
        stale_data_flags      = stale_flags,
        risks_to_monitor      = risks,
        confidence_challenges = confidence_challenges,
        critic_verdict        = critic_verdict,
        llm_summary           = llm_summary,
    )

    return output
    # return {**state, "critic_output": output.model_dump()}


# ============================================================
# QUICK TEST — python critic_research_agent.py
# ============================================================
# if __name__ == "__main__":

#     llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

#     mock_state = {
#         "ticker":     "SBIN",
#         "depth":      "standard",
#         "focus_area": "fundamentals,news,technical",

#         "fundamentals_output": {
#             "metadata":         {"company_name": "State Bank of India", "analysis_timestamp": _now_iso()},
#             "current_snapshot": {"pe_ratio": 10.2, "roe_percent": 18.6, "debt_to_equity": 12.4},
#             "quarterly_trend":  {
#                 "revenue":                   {"trend": "increasing",  "values": [108000, 112000, 118000, 122000]},
#                 "net_profit_margin_percent": {"trend": "stable",      "values": [14.2, 15.1, 16.0, 15.8]},
#             },
#             "industry_comparison": {"pe_ratio": {"verdict": "inline"}},
#             "signal_scores":    {"overall": 7.2},
#             "anomaly_flags":    [
#                 {"flag_id": "DEBT_RISING", "severity": "medium",
#                  "message": "D/E rising for 4 quarters", "action": "monitor", "metric": "debt_to_equity"},
#             ],
#             "llm_summary": "SBI fundamentals are strong with rising revenue and ROE above industry.",
#         },

#         "news_output": {
#             "metadata":           {"analysis_timestamp": _now_iso(), "articles_analyzed": 8,
#                                    "articles_fetched": 10, "articles_rejected": 2, "status": "success"},
#             "signal_score":       {"overall": 7.8},
#             "sentiment_breakdown":{"overall_label": "Predominantly Positive", "positive_percent": 60.0},
#             "all_high_impact_events": [
#                 {"event_type": "financial_result", "description": "Record Q4 profit",
#                  "impact": "bullish", "severity": "high"},
#                 {"event_type": "regulatory_action", "description": "RBI fine ₹2 Cr KYC",
#                  "impact": "bearish", "severity": "low"},
#             ],
#             "llm_summary": "News flow positive, anchored by record Q4 earnings beat.",
#         },

#         "technical_output": {
#             "metadata":       {"analysis_timestamp": _now_iso(), "total_trading_days": 252},
#             "signal_score":   {"overall": 6.5},
#             "rsi":            {"current": 58.2, "zone": "neutral"},
#             "macd":           {"trend": "bullish_momentum"},
#             "moving_averages":{"ma_200": 780.0},
#             "trend":          {"primary_trend": "uptrend", "trend_strength": "moderate", "adx_value": 28.4},
#             "cross_signals":  [{"signal_type": "golden_cross", "detected": True, "days_ago": 12,
#                                 "description": "Golden Cross 12 days ago."}],
#             "technical_flags":[],
#             "llm_summary":    "Moderate uptrend with bullish MACD and Golden Cross confirmed.",
#         },

#         "synthesis_research": {
#             "metadata":        {"company_name": "State Bank of India",
#                                 "agents_missing": [], "analysis_timestamp": _now_iso()},
#             "signal_agreement":{"consensus": "bull", "consensus_score": 7.2,
#                                 "divergence_flag": False, "divergence_detail": None,
#                                 "agreeing_agents": ["fundamentals", "news", "technical"],
#                                 "disagreeing_agents": []},
#             "recommendation":  {
#                 "verdict":           "Buy",
#                 "confidence_pct":    72.0,
#                 "investment_horizon":"medium_term",
#                 "reasoning":         "SBI presents a compelling buy with strong fundamentals, positive news flow, and a confirmed technical uptrend.",
#                 "bull_case":         [
#                     {"point": "Revenue growing consistently 4 quarters", "source": "fundamentals", "strength": "strong"},
#                     {"point": "News sentiment 60% positive with earnings beat", "source": "news",         "strength": "strong"},
#                     {"point": "Golden Cross confirmed — bullish long-term signal", "source": "technical", "strength": "strong"},
#                 ],
#                 "bear_case":         [],
#                 "risk_factors":      [],
#                 "key_catalysts":     ["Next RBI policy meeting", "Q1 FY27 results"],
#             },
#             "llm_summary": "SBI is a Buy with 72% confidence based on strong tri-agent consensus.",
#         },
#     }

#     result = critic_research_agent(mock_state, llm=llm)
#     print(json.dumps(result["critic_output"], indent=2, default=str))