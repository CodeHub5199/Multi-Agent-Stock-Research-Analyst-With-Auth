# ============================================================
# technical_research_agent.py
# Technical Research Agent — Indian Stock Market
# Multi-Agent Stock Analysis System
# ============================================================

import json
import re
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Optional
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# PYDANTIC MODELS
# ============================================================

class MovingAverages(BaseModel):
    ma_50:              Optional[float]   # 50-day SMA
    ma_200:             Optional[float]   # 200-day SMA
    ma_20:              Optional[float]   # 20-day SMA (for Bollinger)
    price_vs_ma50:      Optional[str]     # "above" | "below"
    price_vs_ma200:     Optional[str]     # "above" | "below"
    ma50_vs_ma200:      Optional[str]     # "above" | "below"


class RSIData(BaseModel):
    current:            Optional[float]   # latest RSI value
    prev_week:          Optional[float]   # RSI 5 days ago
    zone:               str               # "overbought" | "oversold" | "neutral"
    divergence:         Optional[str]     # "bullish" | "bearish" | None


class MACDData(BaseModel):
    macd_line:          Optional[float]
    signal_line:        Optional[float]
    histogram:          Optional[float]
    crossover:          Optional[str]     # "bullish" | "bearish" | None
    trend:              Optional[str]     # "bullish_momentum" | "bearish_momentum" | "weakening"


class BollingerBands(BaseModel):
    upper:              Optional[float]
    middle:             Optional[float]   # 20-day SMA
    lower:              Optional[float]
    bandwidth:          Optional[float]   # (upper - lower) / middle * 100
    price_position:     Optional[str]     # "above_upper" | "near_upper" | "middle"
                                          # | "near_lower" | "below_lower"
    squeeze:            bool              # bandwidth < 10% → low volatility


class SupportResistanceLevel(BaseModel):
    price:              float
    level_type:         str               # "support" | "resistance"
    strength:           str               # "strong" | "moderate" | "weak"
    touches:            int               # how many times price bounced
    distance_pct:       float             # % away from current price


class CrossSignal(BaseModel):
    signal_type:        str               # "golden_cross" | "death_cross"
    detected:           bool
    date:               Optional[str]     # ISO date when cross occurred
    days_ago:           Optional[int]
    description:        str


class TechnicalTrend(BaseModel):
    primary_trend:      str               # "uptrend" | "downtrend" | "consolidation"
    trend_strength:     str               # "strong" | "moderate" | "weak"
    trend_duration_days:Optional[int]
    higher_highs:       Optional[bool]
    higher_lows:        Optional[bool]
    adx_value:          Optional[float]   # trend strength indicator


class PriceAction(BaseModel):
    current_price:      float
    week_52_high:       Optional[float]
    week_52_low:        Optional[float]
    pct_from_52w_high:  Optional[float]
    pct_from_52w_low:   Optional[float]
    avg_volume_30d:     Optional[float]
    current_volume:     Optional[float]
    volume_spike:       bool              # current > 2x average


class TechnicalSignalScore(BaseModel):
    trend_score:        float             # 0-10
    momentum_score:     float             # 0-10  (RSI + MACD)
    volatility_score:   float             # 0-10  (Bollinger)
    volume_score:       float             # 0-10
    overall:            float             # weighted composite


class TechnicalFlag(BaseModel):
    flag_id:            str
    severity:           str               # "critical"|"medium"|"low"|"positive"
    metric:             str
    message:            str
    action:             str               # "buy_signal"|"sell_signal"|"monitor"|"note"


class TechnicalOutput(BaseModel):
    metadata:           dict
    price_action:       Optional[PriceAction]
    moving_averages:    Optional[MovingAverages]
    rsi:                Optional[RSIData]
    macd:               Optional[MACDData]
    bollinger_bands:    Optional[BollingerBands]
    trend:              Optional[TechnicalTrend]
    support_resistance: list[SupportResistanceLevel]
    cross_signals:      list[CrossSignal]
    technical_flags:    list[TechnicalFlag]
    signal_score:       Optional[TechnicalSignalScore]
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


def _round(val, n=2) -> Optional[float]:
    try:
        return round(float(val), n) if val is not None and not np.isnan(val) else None
    except Exception:
        return None


# ============================================================
# INDICATOR CALCULATORS
# ============================================================

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l
    return 100 - (100 / (1 + rs))


def _calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window=period).mean()
    std    = close.rolling(window=period).std()
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    return upper, middle, lower


def _calc_adx(
    high: pd.Series,
    low:  pd.Series,
    close:pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index — measures trend strength (not direction)."""
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm  < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr       = tr.ewm(span=period, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(span=period,  adjust=False).mean() / atr
    minus_di  = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di))
    adx       = dx.ewm(span=period, adjust=False).mean()
    return adx


# ============================================================
# SUPPORT / RESISTANCE DETECTOR
# ============================================================

def _find_support_resistance(
    df:            pd.DataFrame,
    current_price: float,
    window:        int = 10,
    max_levels:    int = 6,
) -> list[SupportResistanceLevel]:
    """
    Identify S/R levels using local minima/maxima (pivot points).
    Groups nearby levels within 1.5% of each other.
    """
    highs = df["High"].values
    lows  = df["Low"].values
    levels: list[tuple[float, str]] = []

    for i in range(window, len(df) - window):
        # Resistance: local high
        if highs[i] == max(highs[i - window: i + window + 1]):
            levels.append((highs[i], "resistance"))
        # Support: local low
        if lows[i] == min(lows[i - window: i + window + 1]):
            levels.append((lows[i], "support"))

    # Cluster nearby levels (within 1.5%)
    clustered: list[tuple[float, str, int]] = []
    used = set()
    for idx, (price, lvl_type) in enumerate(levels):
        if idx in used:
            continue
        group       = [price]
        touch_count = 1
        for jdx, (p2, t2) in enumerate(levels):
            if jdx != idx and jdx not in used:
                if abs(p2 - price) / price < 0.015:  # within 1.5%
                    group.append(p2)
                    touch_count += 1
                    used.add(jdx)
        used.add(idx)
        avg_price = float(np.mean(group))
        clustered.append((avg_price, lvl_type, touch_count))

    # Score strength by touch count
    def strength(touches: int) -> str:
        if touches >= 4: return "strong"
        if touches >= 2: return "moderate"
        return "weak"

    # Sort by distance from current price, take closest max_levels
    result: list[SupportResistanceLevel] = []
    for price, lvl_type, touches in sorted(
        clustered, key=lambda x: abs(x[0] - current_price)
    )[:max_levels * 2]:
        dist_pct = round((price - current_price) / current_price * 100, 2)
        # Only keep relevant levels (within 15% of current price)
        if abs(dist_pct) <= 15:
            result.append(SupportResistanceLevel(
                price         = _round(price),
                level_type    = lvl_type,
                strength      = strength(touches),
                touches       = touches,
                distance_pct  = dist_pct,
            ))

    # Final: closest max_levels, sorted by distance
    result.sort(key=lambda x: abs(x.distance_pct))
    return result[:max_levels]


# ============================================================
# CROSS SIGNAL DETECTOR
# ============================================================

def _detect_cross_signals(df: pd.DataFrame) -> list[CrossSignal]:
    """Detect Golden Cross (MA50 > MA200) and Death Cross (MA50 < MA200)."""
    signals: list[CrossSignal] = []

    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()

    # Need at least 200 days
    if ma200.dropna().empty or ma50.dropna().empty:
        return signals

    # Find crossover points in last 90 days
    recent   = df.loc[df.index >= df.index[-1] - pd.Timedelta(days=90)]
    ma50_r   = ma50.reindex(recent.index)
    ma200_r  = ma200.reindex(recent.index)
    diff     = ma50_r - ma200_r
    sign_chg = diff.diff().apply(np.sign)

    golden_dates = diff.index[sign_chg > 0]
    death_dates  = diff.index[sign_chg < 0]

    today = pd.Timestamp.now(tz=None)

    for d in golden_dates[-1:]:   # most recent only
        days_ago = (today - d.replace(tzinfo=None)).days
        signals.append(CrossSignal(
            signal_type = "golden_cross",
            detected    = True,
            date        = d.strftime("%Y-%m-%d"),
            days_ago    = days_ago,
            description = f"Golden Cross detected {days_ago} days ago — MA50 crossed above MA200 (bullish long-term signal).",
        ))

    for d in death_dates[-1:]:
        days_ago = (today - d.replace(tzinfo=None)).days
        signals.append(CrossSignal(
            signal_type = "death_cross",
            detected    = True,
            date        = d.strftime("%Y-%m-%d"),
            days_ago    = days_ago,
            description = f"Death Cross detected {days_ago} days ago — MA50 crossed below MA200 (bearish long-term signal).",
        ))

    # If no cross in last 90 days, report current state
    if not signals:
        current_diff = (ma50.iloc[-1] - ma200.iloc[-1])
        if current_diff > 0:
            signals.append(CrossSignal(
                signal_type = "golden_cross",
                detected    = False,
                date        = None,
                days_ago    = None,
                description = "No recent Golden Cross — MA50 is currently above MA200 (bullish posture, cross > 90 days ago).",
            ))
        else:
            signals.append(CrossSignal(
                signal_type = "death_cross",
                detected    = False,
                date        = None,
                days_ago    = None,
                description = "No recent Death Cross — MA50 is currently below MA200 (bearish posture, cross > 90 days ago).",
            ))

    return signals


# ============================================================
# TREND IDENTIFIER
# ============================================================

def _identify_trend(
    df:    pd.DataFrame,
    ma50:  float,
    ma200: float,
    adx:   Optional[float],
    price: float,
) -> TechnicalTrend:
    """
    Classify trend using MA alignment, higher highs/lows, and ADX.
    """

    # Use last 60 days for swing analysis
    recent = df["Close"].loc[df.index >= df.index[-1] - pd.Timedelta(days=60)]
    if len(recent) < 20:
        recent = df["Close"].tail(20)

    # Higher Highs / Higher Lows on weekly pivots
    weekly = df["Close"].resample("W").last().dropna()
    hh = hl = None
    if len(weekly) >= 4:
        last4 = weekly.tail(4).values
        hh    = bool(last4[-1] > last4[-2] > last4[-3])
        hl    = bool(df["Low"].resample("W").min().tail(4).values[-1] >
                     df["Low"].resample("W").min().tail(4).values[-2])

    # MA alignment scoring
    score = 0
    if price > ma50:  score += 1
    if price > ma200: score += 1
    if ma50  > ma200: score += 1
    if hh:            score += 1
    if hl:            score += 1

    if score >= 4:
        primary = "uptrend"
    elif score <= 1:
        primary = "downtrend"
    else:
        primary = "consolidation"

    # Trend strength via ADX
    if adx is not None:
        if adx >= 35:   strength = "strong"
        elif adx >= 20: strength = "moderate"
        else:           strength = "weak"
    else:
        strength = "moderate"

    # Approximate trend duration (days since MA50 > MA200 or vice versa)
    ma50_series  = df["Close"].rolling(50).mean()
    ma200_series = df["Close"].rolling(200).mean()
    diff_sign    = np.sign(ma50_series - ma200_series)
    changes      = diff_sign != diff_sign.shift()
    last_change  = changes[changes].index
    duration     = None
    if len(last_change) > 0:
        duration = (df.index[-1] - last_change[-1]).days

    return TechnicalTrend(
        primary_trend       = primary,
        trend_strength      = strength,
        trend_duration_days = duration,
        higher_highs        = hh,
        higher_lows         = hl,
        adx_value           = _round(adx),
    )


# ============================================================
# TECHNICAL FLAG ENGINE
# ============================================================

def _detect_technical_flags(
    price:    float,
    rsi:      RSIData,
    macd:     MACDData,
    bb:       BollingerBands,
    ma:       MovingAverages,
    trend:    TechnicalTrend,
    crosses:  list[CrossSignal],
    vol:      PriceAction,
) -> list[TechnicalFlag]:

    flags: list[TechnicalFlag] = []

    # ── RSI flags ─────────────────────────────────────────────
    if rsi.current is not None:
        if rsi.current >= 75:
            flags.append(TechnicalFlag(
                flag_id  = "RSI_OVERBOUGHT",
                severity = "medium",
                metric   = "rsi",
                message  = f"RSI at {rsi.current} — strongly overbought. Reversal or consolidation likely.",
                action   = "sell_signal",
            ))
        elif rsi.current >= 65:
            flags.append(TechnicalFlag(
                flag_id  = "RSI_ELEVATED",
                severity = "low",
                metric   = "rsi",
                message  = f"RSI at {rsi.current} — approaching overbought zone. Monitor closely.",
                action   = "monitor",
            ))
        elif rsi.current <= 25:
            flags.append(TechnicalFlag(
                flag_id  = "RSI_OVERSOLD",
                severity = "positive",
                metric   = "rsi",
                message  = f"RSI at {rsi.current} — deeply oversold. Potential reversal / bounce opportunity.",
                action   = "buy_signal",
            ))
        elif rsi.current <= 35:
            flags.append(TechnicalFlag(
                flag_id  = "RSI_NEAR_OVERSOLD",
                severity = "low",
                metric   = "rsi",
                message  = f"RSI at {rsi.current} — approaching oversold zone.",
                action   = "monitor",
            ))

    # RSI divergence
    if rsi.divergence == "bullish":
        flags.append(TechnicalFlag(
            flag_id  = "RSI_BULLISH_DIVERGENCE",
            severity = "positive",
            metric   = "rsi",
            message  = "Bullish RSI divergence — price making lower lows but RSI making higher lows. Reversal signal.",
            action   = "buy_signal",
        ))
    elif rsi.divergence == "bearish":
        flags.append(TechnicalFlag(
            flag_id  = "RSI_BEARISH_DIVERGENCE",
            severity = "medium",
            metric   = "rsi",
            message  = "Bearish RSI divergence — price making higher highs but RSI making lower highs. Weakness signal.",
            action   = "sell_signal",
        ))

    # ── MACD flags ────────────────────────────────────────────
    if macd.crossover == "bullish":
        flags.append(TechnicalFlag(
            flag_id  = "MACD_BULLISH_CROSSOVER",
            severity = "positive",
            metric   = "macd",
            message  = "MACD line crossed above signal line — bullish momentum building.",
            action   = "buy_signal",
        ))
    elif macd.crossover == "bearish":
        flags.append(TechnicalFlag(
            flag_id  = "MACD_BEARISH_CROSSOVER",
            severity = "medium",
            metric   = "macd",
            message  = "MACD line crossed below signal line — bearish momentum building.",
            action   = "sell_signal",
        ))

    # ── Bollinger Band flags ───────────────────────────────────
    if bb.price_position == "above_upper":
        flags.append(TechnicalFlag(
            flag_id  = "BB_BREAKOUT_UP",
            severity = "positive",
            metric   = "bollinger_bands",
            message  = "Price above upper Bollinger Band — strong upward breakout. Can signal continuation or reversal.",
            action   = "monitor",
        ))
    elif bb.price_position == "below_lower":
        flags.append(TechnicalFlag(
            flag_id  = "BB_BREAKOUT_DOWN",
            severity = "medium",
            metric   = "bollinger_bands",
            message  = "Price below lower Bollinger Band — strong downward pressure. Potential oversold bounce.",
            action   = "monitor",
        ))
    if bb.squeeze:
        flags.append(TechnicalFlag(
            flag_id  = "BB_SQUEEZE",
            severity = "low",
            metric   = "bollinger_bands",
            message  = f"Bollinger Band squeeze detected (bandwidth {bb.bandwidth}%) — low volatility. Big move imminent.",
            action   = "monitor",
        ))

    # ── MA flags ──────────────────────────────────────────────
    if ma.price_vs_ma200 == "below":
        flags.append(TechnicalFlag(
            flag_id  = "BELOW_200MA",
            severity = "medium",
            metric   = "moving_averages",
            message  = "Price trading below 200-day MA — long-term bearish structure.",
            action   = "sell_signal",
        ))
    elif ma.price_vs_ma50 == "above" and ma.price_vs_ma200 == "above":
        flags.append(TechnicalFlag(
            flag_id  = "ABOVE_BOTH_MAS",
            severity = "positive",
            metric   = "moving_averages",
            message  = "Price above both 50-day and 200-day MA — bullish structure intact.",
            action   = "note",
        ))

    # ── Cross signal flags ────────────────────────────────────
    for cross in crosses:
        if cross.detected:
            if cross.signal_type == "golden_cross":
                flags.append(TechnicalFlag(
                    flag_id  = f"GOLDEN_CROSS_{cross.days_ago}D_AGO",
                    severity = "positive",
                    metric   = "moving_averages",
                    message  = cross.description,
                    action   = "buy_signal",
                ))
            elif cross.signal_type == "death_cross":
                flags.append(TechnicalFlag(
                    flag_id  = f"DEATH_CROSS_{cross.days_ago}D_AGO",
                    severity = "critical",
                    metric   = "moving_averages",
                    message  = cross.description,
                    action   = "sell_signal",
                ))

    # ── Volume spike flag ──────────────────────────────────────
    if vol.volume_spike:
        flags.append(TechnicalFlag(
            flag_id  = "VOLUME_SPIKE",
            severity = "medium",
            metric   = "volume",
            message  = f"Volume spike detected — current volume significantly above 30-day average. Confirms price move.",
            action   = "monitor",
        ))

    # ── Trend flags ────────────────────────────────────────────
    if trend.primary_trend == "uptrend" and trend.trend_strength == "strong":
        flags.append(TechnicalFlag(
            flag_id  = "STRONG_UPTREND",
            severity = "positive",
            metric   = "trend",
            message  = f"Strong uptrend confirmed (ADX: {trend.adx_value}) with higher highs and higher lows.",
            action   = "note",
        ))
    elif trend.primary_trend == "downtrend" and trend.trend_strength == "strong":
        flags.append(TechnicalFlag(
            flag_id  = "STRONG_DOWNTREND",
            severity = "critical",
            metric   = "trend",
            message  = f"Strong downtrend confirmed (ADX: {trend.adx_value}). Avoid long positions.",
            action   = "sell_signal",
        ))

    return flags


# ============================================================
# SIGNAL SCORER
# ============================================================

def _compute_signal_score(
    trend: TechnicalTrend,
    rsi:   RSIData,
    macd:  MACDData,
    bb:    BollingerBands,
    vol:   PriceAction,
    flags: list[TechnicalFlag],
) -> TechnicalSignalScore:

    # ── Trend score (0–10) ────────────────────────────────────
    trend_base = {"uptrend": 8, "consolidation": 5, "downtrend": 2}
    strength_adj = {"strong": 1.5, "moderate": 0, "weak": -1}
    trend_score = min(10, max(0,
        trend_base.get(trend.primary_trend, 5) +
        strength_adj.get(trend.trend_strength, 0)
    ))

    # ── Momentum score (0–10): RSI + MACD ────────────────────
    rsi_val = rsi.current or 50
    if 40 <= rsi_val <= 60:   rsi_score = 6.0
    elif 60 < rsi_val <= 70:  rsi_score = 8.0
    elif rsi_val > 70:        rsi_score = 4.0   # overbought — risky
    elif 30 <= rsi_val < 40:  rsi_score = 4.0
    else:                     rsi_score = 7.0   # oversold bounce potential

    macd_scores = {
        "bullish_momentum": 8.0,
        "bearish_momentum": 3.0,
        "weakening":        5.0,
    }
    macd_score = macd_scores.get(macd.trend or "", 5.0)
    if macd.crossover == "bullish":  macd_score = min(10, macd_score + 1.5)
    if macd.crossover == "bearish":  macd_score = max(0,  macd_score - 1.5)

    momentum_score = round((rsi_score + macd_score) / 2, 1)

    # ── Volatility score (0–10): Bollinger ───────────────────
    vol_map = {
        "above_upper":  6.0,   # breakout — risky
        "near_upper":   8.0,   # strong momentum
        "middle":       6.0,   # neutral
        "near_lower":   4.0,   # weak
        "below_lower":  3.0,   # breakdown
    }
    volatility_score = vol_map.get(bb.price_position or "middle", 5.0)
    if bb.squeeze: volatility_score = max(0, volatility_score - 1)  # uncertainty

    # ── Volume score (0–10) ───────────────────────────────────
    volume_score = 7.0 if vol.volume_spike else 5.0

    overall = round(
        trend_score      * 0.40 +
        momentum_score   * 0.30 +
        volatility_score * 0.20 +
        volume_score     * 0.10,
        2,
    )

    return TechnicalSignalScore(
        trend_score      = round(trend_score, 1),
        momentum_score   = momentum_score,
        volatility_score = round(volatility_score, 1),
        volume_score     = volume_score,
        overall          = overall,
    )


# ============================================================
# LLM SUMMARY
# ============================================================

def _llm_generate_summary(
    llm:     ChatGroq,
    ticker:  str,
    company: str,
    price:   PriceAction,
    ma:      MovingAverages,
    rsi:     RSIData,
    macd:    MACDData,
    bb:      BollingerBands,
    trend:   TechnicalTrend,
    crosses: list[CrossSignal],
    sr:      list[SupportResistanceLevel],
    flags:   list[TechnicalFlag],
    score:   TechnicalSignalScore,
) -> str:

    cross_text = "\n".join(f"  - {c.description}" for c in crosses) or "  None in last 90 days"
    sr_text    = "\n".join(
        f"  - {s.level_type.title()} at ₹{s.price} ({s.strength}, {s.distance_pct:+.1f}% from price)"
        for s in sr[:4]
    ) or "  Not detected"
    flag_text  = "\n".join(
        f"  [{f.severity.upper()}] {f.flag_id}: {f.message}"
        for f in flags[:6]
    ) or "  None"

    prompt = f"""
You are a technical analyst specializing in Indian equity markets (NSE/BSE).
Write a concise 4-sentence technical analysis summary for {company} ({ticker}).

PRICE ACTION:
  Current Price  : ₹{price.current_price}
  52W High       : ₹{price.week_52_high}  ({price.pct_from_52w_high:+.1f}% from high)
  52W Low        : ₹{price.week_52_low}   ({price.pct_from_52w_low:+.1f}% from low)
  Volume Spike   : {price.volume_spike}

MOVING AVERAGES:
  50-day MA      : ₹{ma.ma_50}   | Price is {ma.price_vs_ma50} MA50
  200-day MA     : ₹{ma.ma_200}  | Price is {ma.price_vs_ma200} MA200
  MA50 vs MA200  : {ma.ma50_vs_ma200}

RSI (14):        {rsi.current} — {rsi.zone}
MACD:            {macd.trend} | Crossover: {macd.crossover}
Bollinger:       {bb.price_position} | Bandwidth: {bb.bandwidth}% | Squeeze: {bb.squeeze}
Trend:           {trend.primary_trend} ({trend.trend_strength}) | ADX: {trend.adx_value} | Duration: {trend.trend_duration_days} days

CROSS SIGNALS:
{cross_text}

KEY SUPPORT / RESISTANCE:
{sr_text}

KEY FLAGS:
{flag_text}

SIGNAL SCORE: {score.overall}/10

Rules:
- Sentence 1: State the dominant trend and its strength using MA + ADX evidence
- Sentence 2: Describe momentum (RSI + MACD) and what it implies for near-term
- Sentence 3: Key support/resistance levels traders should watch
- Sentence 4: Overall technical posture — bullish / bearish / wait-and-watch
- Use ₹ for prices. Be specific with numbers. No markdown, no bullets.
Output ONLY the 4-sentence paragraph.
"""

    messages = [
        SystemMessage(content=(
            "You are a senior technical analyst at a leading Indian brokerage. "
            "Write crisp, chart-based, institutional-quality technical summaries."
        )),
        HumanMessage(content=prompt),
    ]
    return llm.invoke(messages).content.strip()


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def technical_research_agent(
    state: dict,
    llm:   Optional[ChatGroq],
) -> dict:
    """
    LangGraph-compatible Technical Research Agent.

    Reads  : state["ticker"]
    Writes : state["technical_output"]

    Args:
        state : LangGraph ResearchState dict
        llm   : ChatGroq instance (optional — skips LLM summary if None)

    Returns:
        Updated state dict with "technical_output" populated.
    """

    ticker:    str = state.get("ticker", "").strip().upper()
    timestamp: str = _now_iso()

    if ticker and not ticker.endswith((".NS", ".BO")):
        ticker = ticker + ".NS"

    company_name = ticker.replace(".NS", "").replace(".BO", "")
    try:
        import yfinance as yf
        company_name = yf.Ticker(ticker).info['longName']
    except:
        company_name = company_name  # fallback

    # ── Error builder ─────────────────────────────────────────────
    def _error_output(code: str, message: str) -> dict:
        out = TechnicalOutput(
            metadata={
                "ticker":             ticker,
                "status":             "error",
                "error_code":         code,
                "error_message":      message,
                "analysis_timestamp": timestamp,
            },
            price_action=None, moving_averages=None,
            rsi=None, macd=None, bollinger_bands=None,
            trend=None, support_resistance=[],
            cross_signals=[], technical_flags=[],
            signal_score=None,
            llm_summary=f"Technical analysis unavailable for {ticker}. Reason: {message}",
        )
        return out
        # return {**state, "technical_output": out.model_dump()}

    # ============================================================
    # STEP 1 — FETCH 1 YEAR OHLCV
    # ============================================================
    try:
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=365 + 60)  # +60 for MA200 warmup

        df = yf.download(
            ticker,
            start    = start_date.strftime("%Y-%m-%d"),
            end      = end_date.strftime("%Y-%m-%d"),
            interval = "1d",
            progress = False,
            auto_adjust= True,
        )

        if df.empty or len(df) < 50:
            return _error_output("INSUFFICIENT_DATA",
                f"Only {len(df)} days of data returned — need at least 50.")

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Close", "High", "Low", "Open", "Volume"])
        df.index = pd.to_datetime(df.index).tz_localize(None)

    except Exception as e:
        return _error_output("FETCH_ERROR", f"yfinance download failed: {str(e)}")

    current_price = float(df["Close"].iloc[-1])

    # ============================================================
    # STEP 2 — PRICE ACTION
    # ============================================================
    df_1y          = df.loc[df.index >= df.index[-1] - pd.Timedelta(days=365)]
    week_52_high   = _round(df_1y["High"].max())
    week_52_low    = _round(df_1y["Low"].min())
    avg_vol_30     = _round(df["Volume"].loc[df.index >= df.index[-1] - pd.Timedelta(days=30)].mean())
    current_vol    = _round(df["Volume"].iloc[-1])
    vol_spike      = bool(current_vol and avg_vol_30 and current_vol > avg_vol_30 * 2)

    price_action = PriceAction(
        current_price      = _round(current_price),
        week_52_high       = week_52_high,
        week_52_low        = week_52_low,
        pct_from_52w_high  = _round((current_price - week_52_high) / week_52_high * 100) if week_52_high else None,
        pct_from_52w_low   = _round((current_price - week_52_low) / week_52_low * 100) if week_52_low else None,
        avg_volume_30d     = avg_vol_30,
        current_volume     = current_vol,
        volume_spike       = vol_spike,
    )

    # ============================================================
    # STEP 3 — MOVING AVERAGES
    # ============================================================
    close  = df["Close"]
    ma50   = close.rolling(50).mean().iloc[-1]
    ma200  = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else None
    ma20   = close.rolling(20).mean().iloc[-1]

    moving_averages = MovingAverages(
        ma_50         = _round(ma50),
        ma_200        = _round(ma200),
        ma_20         = _round(ma20),
        price_vs_ma50 = "above" if current_price > ma50  else "below",
        price_vs_ma200= "above" if (ma200 and current_price > ma200) else "below",
        ma50_vs_ma200 = "above" if (ma200 and ma50 > ma200)          else "below",
    )

    # ============================================================
    # STEP 4 — RSI
    # ============================================================
    rsi_series  = _calc_rsi(close)
    rsi_current = _round(rsi_series.iloc[-1])
    rsi_prev5   = _round(rsi_series.iloc[-6]) if len(rsi_series) >= 6 else None

    if rsi_current >= 70:   rsi_zone = "overbought"
    elif rsi_current <= 30: rsi_zone = "oversold"
    else:                   rsi_zone = "neutral"

    # Simple divergence: price up but RSI down (bearish) or vice versa
    price_5d_chg = (close.iloc[-1] - close.iloc[-6]) if len(close) >= 6 else 0
    rsi_5d_chg   = (rsi_series.iloc[-1] - rsi_series.iloc[-6]) if len(rsi_series) >= 6 else 0
    divergence   = None
    if price_5d_chg > 0 and rsi_5d_chg < -3:  divergence = "bearish"
    if price_5d_chg < 0 and rsi_5d_chg > 3:   divergence = "bullish"

    rsi_data = RSIData(
        current    = rsi_current,
        prev_week  = rsi_prev5,
        zone       = rsi_zone,
        divergence = divergence,
    )

    # ============================================================
    # STEP 5 — MACD
    # ============================================================
    macd_line, signal_line, histogram = _calc_macd(close)

    macd_curr = macd_line.iloc[-1]
    sig_curr  = signal_line.iloc[-1]
    hist_curr = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2] if len(histogram) >= 2 else 0

    # Crossover in last 3 days
    macd_cross = None
    if len(macd_line) >= 4:
        for i in range(-3, 0):
            prev_diff = macd_line.iloc[i-1] - signal_line.iloc[i-1]
            curr_diff = macd_line.iloc[i]   - signal_line.iloc[i]
            if prev_diff < 0 and curr_diff >= 0: macd_cross = "bullish"
            if prev_diff > 0 and curr_diff <= 0: macd_cross = "bearish"

    # Momentum direction
    if macd_curr > 0 and hist_curr > hist_prev:   macd_trend = "bullish_momentum"
    elif macd_curr < 0 and hist_curr < hist_prev: macd_trend = "bearish_momentum"
    else:                                          macd_trend = "weakening"

    macd_data = MACDData(
        macd_line   = _round(macd_curr),
        signal_line = _round(sig_curr),
        histogram   = _round(hist_curr),
        crossover   = macd_cross,
        trend       = macd_trend,
    )

    # ============================================================
    # STEP 6 — BOLLINGER BANDS
    # ============================================================
    bb_upper, bb_middle, bb_lower = _calc_bollinger(close)

    bbu = bb_upper.iloc[-1]
    bbm = bb_middle.iloc[-1]
    bbl = bb_lower.iloc[-1]
    bandwidth = _round((bbu - bbl) / bbm * 100) if bbm else None

    if   current_price > bbu:                                  bb_position = "above_upper"
    elif current_price > bbu - (bbu - bbm) * 0.2:             bb_position = "near_upper"
    elif current_price < bbl:                                  bb_position = "below_lower"
    elif current_price < bbl + (bbm - bbl) * 0.2:             bb_position = "near_lower"
    else:                                                      bb_position = "middle"

    bollinger = BollingerBands(
        upper          = _round(bbu),
        middle         = _round(bbm),
        lower          = _round(bbl),
        bandwidth      = bandwidth,
        price_position = bb_position,
        squeeze        = bool(bandwidth and bandwidth < 10),
    )

    # ============================================================
    # STEP 7 — ADX + TREND
    # ============================================================
    adx_series = _calc_adx(df["High"], df["Low"], df["Close"])
    adx_val    = _round(adx_series.iloc[-1]) if not adx_series.empty else None

    trend = _identify_trend(df, float(ma50), float(ma200 or ma50), adx_val, current_price)

    # ============================================================
    # STEP 8 — SUPPORT / RESISTANCE
    # ============================================================
    sr_levels = _find_support_resistance(df_1y, current_price)

    # ============================================================
    # STEP 9 — CROSS SIGNALS
    # ============================================================
    cross_signals = _detect_cross_signals(df)

    # ============================================================
    # STEP 10 — TECHNICAL FLAGS
    # ============================================================
    tech_flags = _detect_technical_flags(
        price_action, rsi_data, macd_data,
        bollinger, moving_averages, trend,
        cross_signals, price_action,
    )

    # ============================================================
    # STEP 11 — SIGNAL SCORE
    # ============================================================
    signal_score = _compute_signal_score(
        trend, rsi_data, macd_data, bollinger, price_action, tech_flags
    )

    # ============================================================
    # STEP 12 — LLM SUMMARY
    # ============================================================
    if llm is not None:
        try:
            llm_summary = _llm_generate_summary(
                llm=llm, ticker=ticker, company=company_name,
                price=price_action, ma=moving_averages,
                rsi=rsi_data, macd=macd_data, bb=bollinger,
                trend=trend, crosses=cross_signals,
                sr=sr_levels, flags=tech_flags, score=signal_score,
            )
        except Exception as e:
            llm_summary = (
                f"LLM summary failed ({e}). "
                f"Technical posture: {trend.primary_trend} ({trend.trend_strength}). "
                f"Signal score: {signal_score.overall}/10."
            )
    else:
        llm_summary = (
            f"{company_name} — {trend.primary_trend} ({trend.trend_strength}). "
            f"RSI: {rsi_data.current}. MACD: {macd_data.trend}. "
            f"Signal score: {signal_score.overall}/10."
        )

    # ============================================================
    # STEP 13 — ASSEMBLE OUTPUT
    # ============================================================
    output = TechnicalOutput(
        metadata={
            "ticker":             ticker,
            "company_name":       company_name,
            "data_start":         df.index[0].strftime("%Y-%m-%d"),
            "data_end":           df.index[-1].strftime("%Y-%m-%d"),
            "total_trading_days": len(df),
            "analysis_timestamp": timestamp,
            "data_source":        "yfinance",
            "status":             "success",
        },
        price_action       = price_action,
        moving_averages    = moving_averages,
        rsi                = rsi_data,
        macd               = macd_data,
        bollinger_bands    = bollinger,
        trend              = trend,
        support_resistance = sr_levels,
        cross_signals      = cross_signals,
        technical_flags    = tech_flags,
        signal_score       = signal_score,
        llm_summary        = llm_summary,
    )

    return output
    # return {**state, "technical_output": output.model_dump()}


# ============================================================
# QUICK TEST — python technical_research_agent.py
# ============================================================
# if __name__ == "__main__":

#     llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

#     test_state = {
#         "ticker":              "SBIN",
#         "depth":               "standard",
#         "focus_area":          "fundamentals,news,technical",
#         "fundamentals_output": None,
#         "news_output":         None,
#     }

#     result = technical_research_agent(test_state, llm=llm)
#     print(json.dumps(result["technical_output"], indent=2, default=str))