# ============================================================
# synthesis_research_agent.py
# Synthesis Research Agent — Indian Stock Market
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

class SignalAlignment(BaseModel):
    agent:          str           # "fundamentals" | "news" | "technical"
    score:          Optional[float]
    direction:      str           # "bullish" | "bearish" | "neutral" | "unavailable"
    key_point:      str           # one-line summary of what this agent says


class SignalAgreement(BaseModel):
    agreeing_agents:    list[str]
    disagreeing_agents: list[str]
    consensus:          str        # "strong_bull" | "bull" | "neutral"
                                   # | "bear" | "strong_bear" | "mixed"
    consensus_score:    float      # weighted composite 0–10
    divergence_flag:    bool       # True if gap between any two scores > 3.0
    divergence_detail:  Optional[str]


class BullPoint(BaseModel):
    point:      str                # the bull case argument
    source:     str                # which agent provided this signal
    strength:   str                # "strong" | "moderate" | "weak"


class BearPoint(BaseModel):
    point:      str
    source:     str
    strength:   str


class RiskFactor(BaseModel):
    risk:       str
    source:     str                # "fundamentals" | "news" | "technical" | "macro"
    severity:   str                # "high" | "medium" | "low"


class Recommendation(BaseModel):
    verdict:            str        # "Strong Buy"|"Buy"|"Hold"|"Sell"|"Strong Sell"
    confidence_pct:     float      # 0–100
    investment_horizon: str        # "short_term"|"medium_term"|"long_term"
    reasoning:          str        # 3–5 sentence narrative
    bull_case:          list[BullPoint]
    bear_case:          list[BearPoint]
    risk_factors:       list[RiskFactor]
    key_catalysts:      list[str]  # upcoming events that could change verdict


class SynthesisOutput(BaseModel):
    metadata:           dict
    signal_alignments:  list[SignalAlignment]
    signal_agreement:   SignalAgreement
    recommendation:     Recommendation
    llm_summary:        str


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
    """Safely navigate nested dict or Pydantic object."""
    for key in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj if obj is not None else default


def _score_to_direction(score: Optional[float]) -> str:
    if score is None:               
        return "unavailable"
    if score >= 7.0:                
        return "bullish"
    if score >= 5.5:                
        return "slightly_bullish"
    if score >= 4.5:                
        return "neutral"
    if score >= 3.0:                
        return "slightly_bearish"
    return "bearish"


def _verdict_from_score(score: float, confidence: float) -> str:
    if score >= 7.5 and confidence >= 65:   
        return "Strong Buy"
    if score >= 6.5:                        
        return "Buy"
    if score >= 4.5:                        
        return "Hold"
    if score >= 3.5:                        
        return "Sell"
    return "Strong Sell"


def _consensus_label(score: float) -> str:
    if score >= 7.5:   
        return "strong_bull"
    if score >= 6.0:   
        return "bull"
    if score >= 4.5:   
        return "neutral"
    if score >= 3.0:   
        return "bear"
    return "strong_bear"


# ============================================================
# STEP 1 — EXTRACT SCORES FROM ALL AGENTS
# ============================================================

def _extract_agent_scores(state: dict) -> dict:
    """
    Pull the signal score, key point, and direction from each agent.
    Returns a clean dict even if any agent output is missing.
    """
    scores = {}

    # ── Fundamentals ─────────────────────────────────────────────
    fund = state.get("fundamentals_output")
    fund_score = _safe_get(fund, "signal_scores", "overall")
    fund_summary = _safe_get(fund, "llm_summary", default="")
    scores["fundamentals"] = {
        "score":        fund_score,
        "direction":    _score_to_direction(fund_score),
        "key_point":    (fund_summary[:120] + "...") if fund_summary else "Fundamentals data unavailable.",
        "pe":           _safe_get(fund, "current_snapshot", "pe_ratio"),
        "roe":          _safe_get(fund, "current_snapshot", "roe_percent"),
        "de":           _safe_get(fund, "current_snapshot", "debt_to_equity"),
        "revenue_trend":_safe_get(fund, "quarterly_trend", "revenue", "trend"),
        "eps_trend":    _safe_get(fund, "quarterly_trend", "eps", "trend"),
        "anomaly_flags":_safe_get(fund, "anomaly_flags") or [],
    }

    # ── News ──────────────────────────────────────────────────────
    news = state.get("news_output")
    news_score   = _safe_get(news, "signal_score", "overall")
    news_summary = _safe_get(news, "llm_summary", default="")
    news_sentiment_label = _safe_get(news, "sentiment_breakdown", "overall_label", default="")
    scores["news"] = {
        "score":            news_score,
        "direction":        _score_to_direction(news_score),
        "key_point":        (news_summary[:120] + "...") if news_summary else "News data unavailable.",
        "sentiment_label":  news_sentiment_label,
        "positive_pct":     _safe_get(news, "sentiment_breakdown", "positive_percent"),
        "high_impact_events":_safe_get(news, "all_high_impact_events") or [],
        "top_stories":      _safe_get(news, "top_stories") or [],
    }

    # ── Technical ─────────────────────────────────────────────────
    tech = state.get("technical_output")
    tech_score   = _safe_get(tech, "signal_score", "overall")
    tech_summary = _safe_get(tech, "llm_summary", default="")
    scores["technical"] = {
        "score":        tech_score,
        "direction":    _score_to_direction(tech_score),
        "key_point":    (tech_summary[:120] + "...") if tech_summary else "Technical data unavailable.",
        "trend":        _safe_get(tech, "trend", "primary_trend"),
        "trend_strength":_safe_get(tech, "trend", "trend_strength"),
        "rsi":          _safe_get(tech, "rsi", "current"),
        "macd_trend":   _safe_get(tech, "macd", "trend"),
        "cross_signals":_safe_get(tech, "cross_signals") or [],
        "tech_flags":   _safe_get(tech, "technical_flags") or [],
    }

    return scores


# ============================================================
# STEP 2 — SIGNAL ALIGNMENT BUILDER
# ============================================================

def _build_signal_alignments(scores: dict) -> list[SignalAlignment]:
    alignments = []
    for agent, data in scores.items():
        alignments.append(SignalAlignment(
            agent      = agent,
            score      = data["score"],
            direction  = data["direction"],
            key_point  = data["key_point"],
        ))
    return alignments


# ============================================================
# STEP 3 — SIGNAL AGREEMENT ANALYSIS
# ============================================================

def _analyze_signal_agreement(
    scores:     dict,
    alignments: list[SignalAlignment],
) -> SignalAgreement:
    """
    Compute weighted consensus score and detect divergences.

    Weights (must sum to 1.0):
      fundamentals → 0.40  (what is the business actually worth?)
      news         → 0.30  (what just happened?)
      technical    → 0.30  (what is the market doing right now?)
    """
    weights = {"fundamentals": 0.40, "news": 0.30, "technical": 0.30}

    available_scores   = {k: v["score"] for k, v in scores.items() if v["score"] is not None}
    unavailable_agents = [k for k, v in scores.items() if v["score"] is None]

    # Re-normalise weights if any agent is missing
    if unavailable_agents:
        total_weight = sum(weights[k] for k in available_scores)
        weights = {k: weights[k] / total_weight for k in available_scores}

    # Weighted composite
    consensus_score = sum(
        available_scores[k] * weights[k]
        for k in available_scores
    ) if available_scores else 5.0

    # Agreement / disagreement
    bullish_agents  = [k for k, v in scores.items() if v["direction"] in ("bullish", "slightly_bullish")]
    bearish_agents  = [k for k, v in scores.items() if v["direction"] in ("bearish", "slightly_bearish")]
    neutral_agents  = [k for k, v in scores.items() if v["direction"] == "neutral"]

    # Determine agreeing vs disagreeing
    if len(bullish_agents) >= 2:
        agreeing    = bullish_agents
        disagreeing = bearish_agents + neutral_agents
    elif len(bearish_agents) >= 2:
        agreeing    = bearish_agents
        disagreeing = bullish_agents + neutral_agents
    else:
        agreeing    = neutral_agents
        disagreeing = bullish_agents + bearish_agents

    # Divergence detection: any two scores differ by > 3 points
    score_values = list(available_scores.values())
    divergence_flag   = False
    divergence_detail = None
    if len(score_values) >= 2:
        max_gap = max(
            abs(score_values[i] - score_values[j])
            for i in range(len(score_values))
            for j in range(i + 1, len(score_values))
        )
        if max_gap > 3.0:
            divergence_flag = True
            # Identify which pair has the biggest gap
            for i, (k1, v1) in enumerate(available_scores.items()):
                for k2, v2 in list(available_scores.items())[i+1:]:
                    if abs(v1 - v2) == max_gap:
                        high_agent = k1 if v1 > v2 else k2
                        low_agent  = k2 if v1 > v2 else k1
                        divergence_detail = (
                            f"{high_agent.title()} is {_score_to_direction(available_scores[high_agent])} "
                            f"(score: {available_scores[high_agent]}) while "
                            f"{low_agent.title()} is {_score_to_direction(available_scores[low_agent])} "
                            f"(score: {available_scores[low_agent]}) — gap of {round(max_gap, 1)} points. "
                            f"This divergence warrants caution."
                        )

    return SignalAgreement(
        agreeing_agents    = agreeing,
        disagreeing_agents = [a for a in disagreeing if a not in unavailable_agents],
        consensus          = _consensus_label(consensus_score),
        consensus_score    = round(consensus_score, 2),
        divergence_flag    = divergence_flag,
        divergence_detail  = divergence_detail,
    )


# ============================================================
# STEP 4 — EXTRACT BULL / BEAR / RISK FROM RAW AGENT DATA
# ============================================================

def _extract_bull_bear_risks(scores: dict) -> tuple[list, list, list]:
    """
    Mine raw agent outputs for bull points, bear points, and risk factors.
    Rule-based extraction — no LLM needed here.
    """
    bull_points:  list[BullPoint]  = []
    bear_points:  list[BearPoint]  = []
    risk_factors: list[RiskFactor] = []

    # ── FUNDAMENTALS bull/bear ────────────────────────────────
    rev_trend = scores["fundamentals"].get("revenue_trend", "")
    eps_trend = scores["fundamentals"].get("eps_trend", "")
    roe       = scores["fundamentals"].get("roe")
    de        = scores["fundamentals"].get("de")

    if rev_trend in ("increasing", "mostly_increasing"):
        bull_points.append(BullPoint(
            point    = f"Revenue on consistent uptrend ({rev_trend}) — top-line growth intact.",
            source   = "fundamentals",
            strength = "strong" if rev_trend == "increasing" else "moderate",
        ))
    elif rev_trend in ("decreasing", "mostly_decreasing"):
        bear_points.append(BearPoint(
            point    = f"Revenue showing {rev_trend} trend — top-line growth under pressure.",
            source   = "fundamentals",
            strength = "strong" if rev_trend == "decreasing" else "moderate",
        ))

    if eps_trend in ("increasing", "mostly_increasing"):
        bull_points.append(BullPoint(
            point    = f"EPS growing {eps_trend} — earnings power strengthening.",
            source   = "fundamentals",
            strength = "strong" if eps_trend == "increasing" else "moderate",
        ))

    if roe and roe > 18:
        bull_points.append(BullPoint(
            point    = f"Strong ROE of {roe}% — exceptional capital efficiency.",
            source   = "fundamentals",
            strength = "strong",
        ))
    elif roe and roe < 10:
        bear_points.append(BearPoint(
            point    = f"Weak ROE of {roe}% — poor capital efficiency vs peers.",
            source   = "fundamentals",
            strength = "moderate",
        ))

    if de and de > 3.0:
        risk_factors.append(RiskFactor(
            risk     = f"High Debt-to-Equity of {de} — leverage risk in rising rate environment.",
            source   = "fundamentals",
            severity = "high" if de > 5.0 else "medium",
        ))

    # Anomaly flags from fundamentals
    for flag in scores["fundamentals"].get("anomaly_flags", []):
        severity = flag.get("severity", "") if isinstance(flag, dict) else getattr(flag, "severity", "")
        message  = flag.get("message", "") if isinstance(flag, dict) else getattr(flag, "message", "")

        if severity == "critical":
            bear_points.append(BearPoint(
                point    = message,
                source   = "fundamentals",
                strength = "strong",
            ))
            risk_factors.append(RiskFactor(
                risk=message, source="fundamentals", severity="high"
            ))
        elif severity == "positive":
            bull_points.append(BullPoint(
                point=message, source="fundamentals", strength="moderate"
            ))

    # ── NEWS bull/bear ────────────────────────────────────────
    positive_pct = scores["news"].get("positive_pct") or 0
    if positive_pct >= 60:
        bull_points.append(BullPoint(
            point    = f"News sentiment predominantly positive ({positive_pct}% positive articles).",
            source   = "news",
            strength = "strong" if positive_pct >= 75 else "moderate",
        ))
    elif positive_pct <= 30:
        bear_points.append(BearPoint(
            point    = f"News sentiment predominantly negative (only {positive_pct}% positive articles).",
            source   = "news",
            strength = "strong" if positive_pct <= 15 else "moderate",
        ))

    for event in scores["news"].get("high_impact_events", []):
        event_type  = event.get("event_type", "") if isinstance(event, dict) else getattr(event, "event_type", "")
        impact      = event.get("impact", "")     if isinstance(event, dict) else getattr(event, "impact", "")
        description = event.get("description", "")if isinstance(event, dict) else getattr(event, "description", "")
        severity    = event.get("severity", "")   if isinstance(event, dict) else getattr(event, "severity", "")

        if impact == "bullish" and severity in ("high", "medium"):
            bull_points.append(BullPoint(
                point    = f"[{event_type.replace('_', ' ').title()}] {description}",
                source   = "news",
                strength = "strong" if severity == "high" else "moderate",
            ))
        elif impact == "bearish" and severity in ("high", "medium"):
            bear_points.append(BearPoint(
                point    = f"[{event_type.replace('_', ' ').title()}] {description}",
                source   = "news",
                strength = "strong" if severity == "high" else "moderate",
            ))
            if severity == "high":
                risk_factors.append(RiskFactor(
                    risk=description, source="news", severity="high"
                ))

    # ── TECHNICAL bull/bear ───────────────────────────────────
    trend        = scores["technical"].get("trend", "")
    trend_str    = scores["technical"].get("trend_strength", "")
    rsi          = scores["technical"].get("rsi")
    macd_trend   = scores["technical"].get("macd_trend", "")

    if trend == "uptrend":
        bull_points.append(BullPoint(
            point    = f"{trend_str.title()} uptrend confirmed — price structure intact.",
            source   = "technical",
            strength = "strong" if trend_str == "strong" else "moderate",
        ))
    elif trend == "downtrend":
        bear_points.append(BearPoint(
            point    = f"{trend_str.title()} downtrend in place — avoid long positions.",
            source   = "technical",
            strength = "strong" if trend_str == "strong" else "moderate",
        ))

    if rsi and rsi <= 35:
        bull_points.append(BullPoint(
            point    = f"RSI at {rsi} — oversold zone, potential mean-reversion bounce.",
            source   = "technical",
            strength = "moderate",
        ))
    elif rsi and rsi >= 70:
        risk_factors.append(RiskFactor(
            risk     = f"RSI at {rsi} — overbought, short-term pullback risk.",
            source   = "technical",
            severity = "medium",
        ))

    if macd_trend == "bullish_momentum":
        bull_points.append(BullPoint(
            point    = "MACD showing bullish momentum — positive price acceleration.",
            source   = "technical",
            strength = "moderate",
        ))
    elif macd_trend == "bearish_momentum":
        bear_points.append(BearPoint(
            point    = "MACD in bearish momentum — negative price acceleration.",
            source   = "technical",
            strength = "moderate",
        ))

    for cross in scores["technical"].get("cross_signals", []):
        detected    = cross.get("detected") if isinstance(cross, dict) else getattr(cross, "detected", False)
        signal_type = cross.get("signal_type", "") if isinstance(cross, dict) else getattr(cross, "signal_type", "")
        description = cross.get("description", "") if isinstance(cross, dict) else getattr(cross, "description", "")
        if detected:
            if signal_type == "golden_cross":
                bull_points.append(BullPoint(
                    point=description, source="technical", strength="strong"
                ))
            elif signal_type == "death_cross":
                bear_points.append(BearPoint(
                    point=description, source="technical", strength="strong"
                ))
                risk_factors.append(RiskFactor(
                    risk=description, source="technical", severity="high"
                ))

    # ── Deduplicate (by first 60 chars) ──────────────────────
    def _dedup(items):
        seen, result = set(), []
        for item in items:
            # BullPoint/BearPoint use .point, RiskFactor uses .risk
            key_text = getattr(item, "point", None) or getattr(item, "risk", "")
            key = key_text[:60]
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    # Sort by strength and cap
    strength_rank = {"strong": 0, "moderate": 1, "weak": 2}
    bull_points  = sorted(_dedup(bull_points),  key=lambda x: strength_rank.get(x.strength, 2))[:5]
    bear_points  = sorted(_dedup(bear_points),  key=lambda x: strength_rank.get(x.strength, 2))[:5]
    risk_factors = sorted(_dedup(risk_factors), key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.severity, 2))[:5]

    return bull_points, bear_points, risk_factors


# ============================================================
# STEP 5 — CONFIDENCE CALCULATOR
# ============================================================

def _calculate_confidence(
    agreement:   SignalAgreement,
    bull_points: list,
    bear_points: list,
    scores:      dict,
) -> float:
    """
    Confidence = how sure we are about the verdict, not how bullish.
    High confidence when agents agree and signals are strong.
    Low confidence when agents diverge or data is thin.
    """
    base = 50.0

    # Agreement bonus
    available = [k for k, v in scores.items() if v["score"] is not None]
    if len(available) == 3:
        base += 15   # all 3 agents available
    elif len(available) == 2:
        base += 5
    else:
        base -= 15   # only 1 agent — low confidence

    # Divergence penalty
    if agreement.divergence_flag:
        base -= 20

    # Strong consensus bonus
    if agreement.consensus in ("strong_bull", "strong_bear"):
        base += 15
    elif agreement.consensus in ("bull", "bear"):
        base += 8

    # Signal count bonus
    if len(bull_points) >= 4 or len(bear_points) >= 4:
        base += 8

    # Score proximity bonus (agents close in score = more confident)
    avail_scores = [v["score"] for v in scores.values() if v["score"] is not None]
    if len(avail_scores) >= 2:
        spread = max(avail_scores) - min(avail_scores)
        if spread < 1.5:  
            base += 10
        elif spread < 3:  
            base += 3

    return round(min(95.0, max(20.0, base)), 1)


# ============================================================
# STEP 6 — LLM: REASONING + CATALYSTS
# ============================================================

def _llm_generate_reasoning(
    llm:          ChatGroq,
    ticker:       str,
    company:      str,
    scores:       dict,
    agreement:    SignalAgreement,
    recommendation: str,
    confidence:   float,
    bull_points:  list[BullPoint],
    bear_points:  list[BearPoint],
    risk_factors: list[RiskFactor],
) -> tuple[str, str, list[str]]:
    """
    Returns (reasoning_paragraph, llm_summary, key_catalysts_list).
    Single LLM call for all three outputs.
    """
    bull_text = "\n".join(f"  {i+1}. [{b.source}|{b.strength}] {b.point}"
                          for i, b in enumerate(bull_points[:3]))
    bear_text = "\n".join(f"  {i+1}. [{b.source}|{b.strength}] {b.point}"
                          for i, b in enumerate(bear_points[:3]))
    risk_text = "\n".join(f"  {i+1}. [{r.source}|{r.severity}] {r.risk}"
                          for i, r in enumerate(risk_factors[:3]))

    fund_summary = scores["fundamentals"].get("key_point", "N/A")
    news_summary = scores["news"].get("key_point", "N/A")
    tech_summary = scores["technical"].get("key_point", "N/A")

    prompt = f"""
You are a senior equity research analyst at an institutional fund specializing in Indian markets.

STOCK: {company} ({ticker})
RECOMMENDATION: {recommendation}
CONFIDENCE: {confidence}%
CONSENSUS: {agreement.consensus} (score: {agreement.consensus_score}/10)

AGENT SIGNAL SCORES:
  Fundamentals : {scores["fundamentals"].get("score")} — {scores["fundamentals"].get("direction")}
  News         : {scores["news"].get("score")} — {scores["news"].get("direction")}
  Technical    : {scores["technical"].get("score")} — {scores["technical"].get("direction")}

DIVERGENCE: {agreement.divergence_detail or "No significant divergence."}

AGENT SUMMARIES:
  Fundamentals: {fund_summary}
  News:         {news_summary}
  Technical:    {tech_summary}

BULL CASE:
{bull_text or "  No strong bull signals."}

BEAR CASE:
{bear_text or "  No strong bear signals."}

KEY RISKS:
{risk_text or "  No major risks identified."}

Generate a JSON response with EXACTLY this structure:
{{
  "reasoning": "<3-5 sentence paragraph — institutional grade reasoning for the {recommendation} verdict. Reference specific signals from all 3 agents. No bullet points.>",
  "llm_summary": "<2-3 sentence executive summary — what an analyst would say in a morning briefing. Start with the ticker and verdict.>",
  "key_catalysts": [
    "<upcoming event or trigger that could change this verdict>",
    "<upcoming event or trigger>",
    "<upcoming event or trigger>"
  ]
}}

Rules:
- reasoning: must cite specific numbers (e.g. RSI, ROE, sentiment %)
- reasoning: must acknowledge the strongest opposing signal
- key_catalysts: be specific to Indian markets (RBI policy, earnings date, SEBI, Budget, FII flows)
- No markdown, return pure JSON only
"""

    messages = [
        SystemMessage(content=(
            "You are a CFA-qualified senior equity analyst. "
            "Write institutional-quality research. Be specific and data-driven. "
            "Return only valid JSON."
        )),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    parsed   = json.loads(_clean_json(response.content))

    reasoning    = parsed.get("reasoning", "")
    llm_summary  = parsed.get("llm_summary", "")
    catalysts    = parsed.get("key_catalysts", [])

    return reasoning, llm_summary, catalysts


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def synthesis_research_agent(
    state: dict,
    llm:   Optional[ChatGroq] = None,
) -> dict:
    """
    LangGraph-compatible Synthesis Research Agent.

    Reads  : state["fundamentals_output"]
              state["news_output"]
              state["technical_output"]

    Writes : state["synthesis_output"]

    Args:
        state : LangGraph ResearchState dict
        llm   : ChatGroq instance

    Returns:
        Updated state dict with "synthesis_output" populated.
    """

    ticker    = state.get("ticker", "UNKNOWN").upper()
    timestamp = _now_iso()

    if not ticker.endswith((".NS", ".BO")):
        ticker = ticker + ".NS"

    # Company name from fundamentals
    fund  = state.get("fundamentals_output")
    company_name = ticker.replace(".NS", "")
    if fund:
        meta = (fund.get("metadata") or {}) if isinstance(fund, dict) \
               else (fund.metadata or {})
        company_name = meta.get("company_name", company_name)

    # ── Error builder ─────────────────────────────────────────────
    def _error_output(code: str, message: str) -> dict:
        out = SynthesisOutput(
            metadata={
                "ticker":             ticker,
                "status":             "error",
                "error_code":         code,
                "error_message":      message,
                "analysis_timestamp": timestamp,
            },
            signal_alignments=[],
            signal_agreement=SignalAgreement(
                agreeing_agents=[], disagreeing_agents=[],
                consensus="neutral", consensus_score=5.0,
                divergence_flag=False, divergence_detail=None,
            ),
            recommendation=Recommendation(
                verdict="Hold", confidence_pct=20.0,
                investment_horizon="medium_term",
                reasoning=f"Synthesis failed: {message}",
                bull_case=[], bear_case=[],
                risk_factors=[], key_catalysts=[],
            ),
            llm_summary=f"Synthesis unavailable for {ticker}. Reason: {message}",
        )
        return out
        # return {**state, "synthesis_output": out.model_dump()}

    # Ensure at least one agent output exists
    available_outputs = [
        k for k in ["fundamentals_output", "news_output", "technical_output"]
        if state.get(k) is not None
    ]
    if not available_outputs:
        return _error_output("NO_AGENT_DATA", "All agent outputs are None — cannot synthesize.")

    # ============================================================
    # STEP 1 — EXTRACT ALL AGENT SCORES
    # ============================================================
    scores = _extract_agent_scores(state)

    # ============================================================
    # STEP 2 — BUILD SIGNAL ALIGNMENTS
    # ============================================================
    alignments = _build_signal_alignments(scores)

    # ============================================================
    # STEP 3 — ANALYZE AGREEMENT / DIVERGENCE
    # ============================================================
    agreement = _analyze_signal_agreement(scores, alignments)

    # ============================================================
    # STEP 4 — EXTRACT BULL / BEAR / RISKS
    # ============================================================
    bull_points, bear_points, risk_factors = _extract_bull_bear_risks(scores)

    # ============================================================
    # STEP 5 — VERDICT + CONFIDENCE
    # ============================================================
    confidence = _calculate_confidence(agreement, bull_points, bear_points, scores)
    verdict    = _verdict_from_score(agreement.consensus_score, confidence)

    # Investment horizon based on signal mix
    tech_trend = scores["technical"].get("trend", "")
    fund_score = scores["fundamentals"].get("score") or 5.0
    if fund_score >= 7 and tech_trend == "uptrend":
        horizon = "medium_term"
    elif fund_score >= 7:
        horizon = "long_term"
    elif tech_trend == "uptrend":
        horizon = "short_term"
    else:
        horizon = "medium_term"

    # ============================================================
    # STEP 6 — LLM REASONING + SUMMARY + CATALYSTS
    # ============================================================
    reasoning    = ""
    llm_summary  = ""
    key_catalysts = []

    if llm is not None:
        try:
            reasoning, llm_summary, key_catalysts = _llm_generate_reasoning(
                llm           = llm,
                ticker        = ticker,
                company       = company_name,
                scores        = scores,
                agreement     = agreement,
                recommendation= verdict,
                confidence    = confidence,
                bull_points   = bull_points,
                bear_points   = bear_points,
                risk_factors  = risk_factors,
            )
        except Exception as e:
            reasoning   = (
                f"LLM reasoning generation failed ({e}). "
                f"Rule-based verdict: {verdict} with {confidence}% confidence. "
                f"Consensus score: {agreement.consensus_score}/10 ({agreement.consensus})."
            )
            llm_summary = reasoning
    else:
        reasoning = (
            f"{company_name} ({ticker}) receives a {verdict} recommendation with "
            f"{confidence}% confidence based on a consensus score of "
            f"{agreement.consensus_score}/10. "
            f"Agents in agreement: {', '.join(agreement.agreeing_agents) or 'None'}. "
            f"{agreement.divergence_detail or ''}"
        )
        llm_summary = reasoning

    # ============================================================
    # STEP 7 — ASSEMBLE FINAL OUTPUT
    # ============================================================
    output = SynthesisOutput(
        metadata={
            "ticker":              ticker,
            "company_name":        company_name,
            "analysis_timestamp":  timestamp,
            "agents_available":    available_outputs,
            "agents_missing":      [
                a for a in ["fundamentals_output", "news_output", "technical_output"]
                if a not in available_outputs
            ],
            "status": "success",
        },
        signal_alignments = alignments,
        signal_agreement  = agreement,
        recommendation    = Recommendation(
            verdict             = verdict,
            confidence_pct      = confidence,
            investment_horizon  = horizon,
            reasoning           = reasoning,
            bull_case           = bull_points[:3],
            bear_case           = bear_points[:3],
            risk_factors        = risk_factors[:3],
            key_catalysts       = key_catalysts[:3],
        ),
        llm_summary = llm_summary,
    )
    return output
    # return {**state, "synthesis_output": output.model_dump()}


# ============================================================
# QUICK TEST — python synthesis_research_agent.py
# ============================================================
# if __name__ == "__main__":

#     llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

#     # Minimal mock state — replace with real agent outputs in production
#     mock_state = {
#         "ticker": "SBIN",
#         "depth":  "standard",
#         "focus_area": "fundamentals,news,technical",
#         "fundamentals_output": {
#             "metadata":       {"company_name": "State Bank of India"},
#             "current_snapshot": {"pe_ratio": 10.2, "roe_percent": 18.6, "debt_to_equity": 12.4},
#             "quarterly_trend": {
#                 "revenue": {"trend": "increasing"},
#                 "eps":     {"trend": "increasing"},
#             },
#             "signal_scores":  {"overall": 7.2},
#             "anomaly_flags":  [{"flag_id": "ROE_STRONG", "severity": "positive",
#                                 "message": "ROE 18.6% — above industry average", "metric": "roe"}],
#             "llm_summary": "SBI shows strong revenue growth and ROE well above peers.",
#         },
#         "news_output": {
#             "signal_score":       {"overall": 7.8},
#             "sentiment_breakdown": {"overall_label": "Predominantly Positive", "positive_percent": 60.0},
#             "all_high_impact_events": [
#                 {"event_type": "financial_result", "description": "Record Q4 profit",
#                  "impact": "bullish", "severity": "high"},
#             ],
#             "top_stories":  [],
#             "llm_summary": "News is predominantly positive anchored by record Q4 earnings.",
#         },
#         "technical_output": {
#             "signal_score": {"overall": 6.5},
#             "trend":        {"primary_trend": "uptrend", "trend_strength": "moderate", "adx_value": 28.4},
#             "rsi":          {"current": 58.2, "zone": "neutral"},
#             "macd":         {"trend": "bullish_momentum", "crossover": None},
#             "cross_signals": [{"signal_type": "golden_cross", "detected": True,
#                                "days_ago": 12, "description": "Golden Cross 12 days ago."}],
#             "technical_flags": [],
#             "llm_summary": "Moderate uptrend with bullish MACD and recent Golden Cross.",
#         },
#     }

#     result = synthesis_research_agent(mock_state, llm=llm)
#     print(json.dumps(result["synthesis_output"], indent=2, default=str))