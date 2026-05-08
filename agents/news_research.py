# ============================================================
# news_research_agent.py
# News Research Agent — Indian Stock Market
# Multi-Agent Stock Analysis System
# ============================================================

import json
import re
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ArticleSentiment(BaseModel):
    label:     str      # "Positive" | "Negative" | "Neutral"
    score:     float    # 0.0 – 1.0 confidence
    reasoning: str      # one sentence


class HighImpactEvent(BaseModel):
    event_type:  str    # see EVENT_TYPES below
    description: str
    impact:      str    # "bullish" | "bearish" | "neutral"
    severity:    str    # "high" | "medium" | "low"


class AnalyzedArticle(BaseModel):
    article_id:           int
    title:                str
    url:                  str
    source:               str
    published_date:       Optional[str]
    relevance_score:      float
    sentiment:            ArticleSentiment
    high_impact_events:   list[HighImpactEvent]
    one_line_summary:     str


class RejectedArticle(BaseModel):
    url:              str
    title:            str
    rejection_reason: str
    raw_score:        float


class TopStory(BaseModel):
    rank:             int
    article_id:       int
    title:            str
    url:              str
    one_line_summary: str
    sentiment_label:  str
    impact_reason:    str


class NewsSentimentBreakdown(BaseModel):
    positive_count:   int
    negative_count:   int
    neutral_count:    int
    total_articles:   int
    positive_percent: Optional[float]
    negative_percent: Optional[float]
    neutral_percent:  Optional[float]
    overall_label:    str
    overall_score:    Optional[float]


class NewsSignalScore(BaseModel):
    news_sentiment: Optional[float]
    event_severity: Optional[float]
    source_quality: Optional[float]
    recency:        Optional[float]
    overall:        Optional[float]


class NewsOutput(BaseModel):
    metadata:               dict
    analyzed_articles:      list[AnalyzedArticle]
    rejected_articles:      list[RejectedArticle]
    top_stories:            list[TopStory]
    sentiment_breakdown:    NewsSentimentBreakdown
    all_high_impact_events: list[HighImpactEvent]
    signal_score:           NewsSignalScore
    llm_summary:            str


# ============================================================
# CONSTANTS
# ============================================================

INDIAN_FINANCE_DOMAINS = [
    "economictimes.indiatimes.com",
    "moneycontrol.com",
    "livemint.com",
    "business-standard.com",
    "financialexpress.com",
    "nseindia.com",
    "bseindia.com",
    "ndtvprofit.com",
    "thehinduBusinessline.com",
    "reuters.com",
    "bloombergquint.com",
]

# All supported high-impact event types
EVENT_TYPES = [
    "earnings",             # quarterly / annual results
    "financial_result",     # revenue, profit announcements
    "dividend",             # dividend declaration / cut
    "merger_acquisition",   # M&A activity
    "product_launch",       # new product / service
    "leadership_change",    # CEO / Chairman / board changes
    "lawsuit",              # legal proceedings
    "regulatory_action",    # RBI / SEBI / govt penalties or approvals
    "fundraising",          # QIP, FPO, rights issue, NCD
    "macro_event",          # RBI policy, budget, interest rate impact
    "insider_trading",      # bulk/block deals, promoter buying/selling
    "credit_rating",        # rating upgrade / downgrade
    "divestment",           # stake sale, govt dilution
    "expansion",            # new branches, geographies, partnerships
    "none",                 # no high-impact event detected
]

# Trusted source quality scores (0–10)
SOURCE_QUALITY_MAP = {
    "economictimes.indiatimes.com": 8.5,
    "moneycontrol.com":             8.0,
    "livemint.com":                 8.5,
    "business-standard.com":        8.5,
    "financialexpress.com":         7.5,
    "reuters.com":                  9.0,
    "bloombergquint.com":           9.0,
    "ndtvprofit.com":               7.0,
    "thehinduBusinessline.com":     8.0,
    "bseindia.com":                 9.5,
    "nseindia.com":                 9.5,
}

# ============================================================
# HELPERS
# ============================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_domain(url: str) -> str:
    """https://moneycontrol.com/... → moneycontrol.com"""
    match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    return match.group(1).lower() if match else "unknown"


def _clean_json_response(raw: str) -> str:
    """Strip markdown fences LLMs sometimes wrap JSON in."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```$",           "", raw, flags=re.MULTILINE)
    return raw.strip()


def _is_valid_article(article: dict, ticker_hint: str) -> tuple[bool, str]:
    """
    Content quality gate — runs BEFORE any LLM call.
    Returns (is_valid, rejection_reason).
    """
    content = (article.get("content") or "").strip()
    title   = (article.get("title")   or "").strip()
    score   = article.get("score", 0)

    if score < 0.5:
        return False, "LOW_RELEVANCE_SCORE"

    if len(content) < 100:
        return False, "CONTENT_TOO_SHORT"

    junk_phrases = [
        "best gifts", "latest headlines", "stock news & headlines",
        "sign in to", "subscribe to", "cookie policy",
        "javascript is required", "enable javascript",
        "page not found", "404",
    ]
    combined = (title + " " + content).lower()
    for phrase in junk_phrases:
        if phrase in combined:
            return False, f"JUNK_CONTENT: '{phrase}'"

    # Ticker / company mismatch guard
    # e.g. querying SBIN should not return Japanese SBI Holdings
    ticker_root = ticker_hint.replace(".NS", "").replace(".BO", "").lower()
    if ticker_root not in combined and "state bank" not in combined:
        return False, "COMPANY_MISMATCH"

    return True, ""


def _build_queries(ticker: str, company_name: str) -> list[str]:
    """Ordered list of queries — most specific first."""
    base = company_name or ticker.replace(".NS", "").replace(".BO", "")
    return [
        f"{base} NSE stock news 2026",
        f"{base} share price BSE NSE latest news",
        f"{ticker} quarterly results earnings India",
        f"{base} RBI SEBI announcement India",
    ]


def _source_quality_score(articles: list[AnalyzedArticle]) -> float:
    """Average quality score across all analyzed article sources."""
    if not articles:
        return 0.0
    scores = [SOURCE_QUALITY_MAP.get(a.source, 5.0) for a in articles]
    return round(sum(scores) / len(scores), 2)


def _recency_score(articles: list[AnalyzedArticle]) -> float:
    """
    Score based on how recent the articles are.
    Published today = 10, 1 week ago = 7, 1 month ago = 4, older = 2.
    """
    if not articles:
        return 0.0
    now    = datetime.now(timezone.utc)
    scores = []
    for a in articles:
        if not a.published_date:
            scores.append(5.0)
            continue
        try:
            pub = datetime.fromisoformat(a.published_date).replace(tzinfo=timezone.utc)
            days_old = (now - pub).days
            if days_old <= 1:   scores.append(10.0)
            elif days_old <= 7: scores.append(7.0)
            elif days_old <= 30:scores.append(4.0)
            else:               scores.append(2.0)
        except Exception:
            scores.append(5.0)
    return round(sum(scores) / len(scores), 2)


def _event_severity_score(events: list[HighImpactEvent]) -> float:
    """Score based on severity and count of high-impact events."""
    if not events:
        return 3.0
    severity_weights = {"high": 10.0, "medium": 6.0, "low": 3.0}
    scores = [severity_weights.get(e.severity, 5.0) for e in events]
    return round(min(10.0, sum(scores) / len(scores) + len(events) * 0.3), 2)


def _sentiment_to_score(breakdown: NewsSentimentBreakdown) -> float:
    """Convert sentiment breakdown to 0–10 score."""
    if breakdown.total_articles == 0:
        return 5.0
    pos = breakdown.positive_percent or 0
    neg = breakdown.negative_percent or 0
    # 100% positive = 10,  50/50 = 5,  100% negative = 0
    return round((pos - neg + 100) / 20, 2)


# ============================================================
# LLM: ANALYZE SINGLE ARTICLE BATCH
# ============================================================

def _llm_analyze_articles(
    llm:        ChatGroq,
    articles:   list[dict],
    ticker:     str,
    company:    str,
) -> list[dict]:
    """
    Send all valid articles to LLM in ONE batched call.
    Returns list of per-article analysis dicts.
    """

    articles_text = ""
    for i, a in enumerate(articles, 1):
        articles_text += f"""
ARTICLE {i}:
Title   : {a.get('title', '')}
URL     : {a.get('url', '')}
Content : {(a.get('raw_content') or a.get('content', ''))[:800]}
Tavily Score: {a.get('score', 0)}
---"""

    system_prompt = """You are a senior financial news analyst specializing in Indian equity markets.
Analyze news articles about Indian stocks with institutional-grade precision.
Always respond with valid JSON only — no preamble, no markdown fences."""

    user_prompt = f"""Analyze the following {len(articles)} news articles about {company} ({ticker}).

For EACH article return a JSON object in the array below.

VALID event_types: {json.dumps(EVENT_TYPES)}

Return ONLY this JSON structure:
{{
  "articles": [
    {{
      "article_id": <int starting from 1>,
      "source": "<domain name only e.g. moneycontrol.com>",
      "published_date": "<YYYY-MM-DD or null>",
      "sentiment": {{
        "label": "Positive | Negative | Neutral",
        "score": <float 0.0-1.0>,
        "reasoning": "<one sentence why>"
      }},
      "high_impact_events": [
        {{
          "event_type": "<from valid list above>",
          "description": "<one line>",
          "impact": "bullish | bearish | neutral",
          "severity": "high | medium | low"
        }}
      ],
      "one_line_summary": "<max 20 words, factual>"
    }}
  ],
  "top_5_article_ids": [<ids of top 5 most impactful articles, ranked>],
  "top_5_impact_reasons": {{
    "<article_id>": "<why this is top-5>"
  }}
}}

ARTICLES TO ANALYZE:
{articles_text}

Rules:
- high_impact_events must be an empty list [] if no significant event
- Be conservative with "high" severity — only genuine market-moving events
- For Indian PSU banks, regulatory_action includes RBI/SEBI notices
- published_date: extract from content if visible, else null
- source: domain only, no https or www
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response  = llm.invoke(messages)
    clean     = _clean_json_response(response.content)
    return json.loads(clean)


# ============================================================
# LLM: GENERATE FINAL NARRATIVE SUMMARY
# ============================================================

def _llm_generate_summary(
    llm:        ChatGroq,
    ticker:     str,
    company:    str,
    breakdown:  NewsSentimentBreakdown,
    top_stories: list[TopStory],
    events:     list[HighImpactEvent],
    signal:     NewsSignalScore,
) -> str:

    top_stories_text = "\n".join(
        f"  {s.rank}. [{s.sentiment_label}] {s.title} — {s.one_line_summary}"
        for s in top_stories
    )
    events_text = "\n".join(
        f"  - [{e.severity.upper()}] {e.event_type}: {e.description} ({e.impact})"
        for e in events
    ) or "  None detected"

    prompt = f"""Write a 4-sentence institutional-grade news analysis summary for {company} ({ticker}).

SENTIMENT: {breakdown.overall_label} — {breakdown.positive_percent}% positive, 
           {breakdown.negative_percent}% negative, {breakdown.neutral_percent}% neutral
           Score: {breakdown.overall_score}/10

TOP STORIES:
{top_stories_text}

HIGH IMPACT EVENTS:
{events_text}

SIGNAL SCORE: {signal.overall}/10

Rules:
- Sentence 1: Lead with the dominant news theme and its market implication
- Sentence 2: Highlight the single most important positive catalyst
- Sentence 3: State the key risk or negative signal
- Sentence 4: Overall news-based investment posture (bullish/cautious/neutral)
- No bullet points, no headers, no markdown
- Output ONLY the 4-sentence paragraph
"""

    messages = [
        SystemMessage(content=(
            "You are a senior equity research analyst at an institutional fund. "
            "Write crisp, data-driven, balanced analysis. No fluff."
        )),
        HumanMessage(content=prompt),
    ]
    return llm.invoke(messages).content.strip()


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def news_research_agent(
    state:        dict,
    llm:          ChatGroq,
) -> dict:
    """
    LangGraph-compatible News Research Agent.

    Reads  : state["ticker"]
    Writes : state["news_output"]

    Args:
        state        : LangGraph ResearchState dict
        llm          : ChatGroq instance
        tavily_tool  : Pre-configured TavilySearch tool.
                       If None, a default instance is created.
    Returns:
        Updated state dict with "news_output" key populated.
    """

    ticker:    str = state.get("ticker", "").strip().upper()
    timestamp: str = _now_iso()

    # ── Ensure NSE suffix ─────────────────────────────────────────
    if ticker and not ticker.endswith((".NS", ".BO")):
        ticker = ticker + ".NS"

    ticker_root = ticker.replace(".NS", "").replace(".BO", "")

    # ── Error builder ─────────────────────────────────────────────
    def _error_output(code: str, message: str) -> dict:
        output = NewsOutput(
            metadata={
                "ticker":             ticker,
                "articles_fetched":   0,
                "articles_analyzed":  0,
                "query_used":         "",
                "analysis_timestamp": timestamp,
                "data_source":        "tavily",
                "status":             "error",
                "error_code":         code,
                "error_message":      message,
            },
            analyzed_articles=[],
            rejected_articles=[],
            top_stories=[],
            sentiment_breakdown=NewsSentimentBreakdown(
                positive_count=0, negative_count=0, neutral_count=0,
                total_articles=0, positive_percent=None,
                negative_percent=None, neutral_percent=None,
                overall_label="insufficient_data", overall_score=None,
            ),
            all_high_impact_events=[],
            signal_score=NewsSignalScore(
                news_sentiment=None, event_severity=None,
                source_quality=None, recency=None, overall=None,
            ),
            llm_summary=(
                f"News analysis unavailable for {ticker}. "
                f"Reason: {message} Downstream agents should treat news signal as missing."
            ),
        )
        return output
        # return {**state, "news_output": output.model_dump()}

    # ── Get company name if available ───────────
    
    try:
        import yfinance as yf
        company_name = yf.Ticker(ticker).info['longName']
    except:
        company_name = ticker_root  # fallback

    # ── Setup Tavily ──────────────────────────────────────────────
    tavily_tool = TavilySearch(
        max_results=10,
        search_depth="advanced",
        include_domains=INDIAN_FINANCE_DOMAINS,
        include_answer=False,
    )

    # ============================================================
    # STEP 1 — FETCH WITH RETRY
    # ============================================================
    queries        = _build_queries(ticker, company_name)
    valid_articles = []
    rejected       = []
    query_used     = ""
    total_fetched  = 0

    for query in queries:
        if len(valid_articles) >= 5:   # enough good articles — stop fetching
            break

        try:
            raw_response = tavily_tool.invoke({"query": query})
        except Exception as e:
            return _error_output("TAVILY_FETCH_ERROR", str(e))

        results       = raw_response.get("results", [])
        total_fetched += len(results)
        query_used     = query

        for article in results:
            is_valid, reason = _is_valid_article(article, ticker_root)
            if is_valid:
                valid_articles.append(article)
            else:
                rejected.append(RejectedArticle(
                    url=article.get("url", ""),
                    title=article.get("title", ""),
                    rejection_reason=reason,
                    raw_score=article.get("score", 0),
                ))

    # ── Not enough valid articles ─────────────────────────────────
    if len(valid_articles) == 0:
        partial_output = _error_output(
            "INSUFFICIENT_VALID_ARTICLES",
            f"0 of {total_fetched} fetched articles passed content validation. "
            f"Likely junk content or wrong company. Tried {len(queries)} queries.",
        )
        # Still attach rejected articles for debugging
        partial_output["news_output"]["rejected_articles"] = [r.model_dump() for r in rejected]
        partial_output["news_output"]["metadata"]["articles_fetched"] = total_fetched
        partial_output["news_output"]["metadata"]["status"] = "insufficient_data"
        return partial_output

    # ============================================================
    # STEP 2 — LLM ANALYSIS (single batched call)
    # ============================================================
    try:
        llm_result   = _llm_analyze_articles(llm, valid_articles, ticker, company_name)
        article_data = llm_result.get("articles", [])
        top_5_ids    = llm_result.get("top_5_article_ids", [])
        top_5_reasons= llm_result.get("top_5_impact_reasons", {})
    except Exception as e:
        return _error_output("LLM_ANALYSIS_ERROR", f"Article analysis LLM call failed: {str(e)}")

    # ============================================================
    # STEP 3 — BUILD AnalyzedArticle objects
    # ============================================================
    analyzed: list[AnalyzedArticle] = []

    for i, (raw, llm_data) in enumerate(zip(valid_articles, article_data), 1):
        try:
            sentiment_data = llm_data.get("sentiment", {})
            events_data    = llm_data.get("high_impact_events", [])

            analyzed.append(AnalyzedArticle(
                article_id          = i,
                title               = raw.get("title", ""),
                url                 = raw.get("url", ""),
                source              = llm_data.get("source") or _extract_domain(raw.get("url", "")),
                published_date      = llm_data.get("published_date"),
                relevance_score     = round(raw.get("score", 0), 4),
                sentiment           = ArticleSentiment(
                    label     = sentiment_data.get("label", "Neutral"),
                    score     = float(sentiment_data.get("score", 0.5)),
                    reasoning = sentiment_data.get("reasoning", ""),
                ),
                high_impact_events  = [
                    HighImpactEvent(
                        event_type  = e.get("event_type", "none"),
                        description = e.get("description", ""),
                        impact      = e.get("impact", "neutral"),
                        severity    = e.get("severity", "low"),
                    )
                    for e in events_data
                ],
                one_line_summary    = llm_data.get("one_line_summary", ""),
            ))
        except Exception:
            # If one article fails to parse, skip it — don't crash the agent
            continue

    # ============================================================
    # STEP 4 — SENTIMENT BREAKDOWN
    # ============================================================
    pos_count = sum(1 for a in analyzed if a.sentiment.label == "Positive")
    neg_count = sum(1 for a in analyzed if a.sentiment.label == "Negative")
    neu_count = sum(1 for a in analyzed if a.sentiment.label == "Neutral")
    total     = len(analyzed)

    def _pct(n): return round(n / total * 100, 1) if total > 0 else None

    pos_pct = _pct(pos_count)
    neg_pct = _pct(neg_count)
    neu_pct = _pct(neu_count)

    if pos_pct is None:
        overall_label = "insufficient_data"
        overall_score = None
    elif pos_pct >= 60:
        overall_label = "Predominantly Positive"
        overall_score = round(5 + (pos_pct - 50) / 10, 1)
    elif neg_pct >= 60:
        overall_label = "Predominantly Negative"
        overall_score = round(5 - (neg_pct - 50) / 10, 1)
    elif abs(pos_pct - neg_pct) <= 15:
        overall_label = "Mixed"
        overall_score = 5.0
    else:
        overall_label = "Neutral"
        overall_score = 5.0

    sentiment_breakdown = NewsSentimentBreakdown(
        positive_count   = pos_count,
        negative_count   = neg_count,
        neutral_count    = neu_count,
        total_articles   = total,
        positive_percent = pos_pct,
        negative_percent = neg_pct,
        neutral_percent  = neu_pct,
        overall_label    = overall_label,
        overall_score    = overall_score,
    )

    # ============================================================
    # STEP 5 — ALL HIGH IMPACT EVENTS (deduplicated)
    # ============================================================
    all_events: list[HighImpactEvent] = []
    seen_descriptions = set()
    for article in analyzed:
        for event in article.high_impact_events:
            if event.event_type != "none" and event.description not in seen_descriptions:
                all_events.append(event)
                seen_descriptions.add(event.description)

    # Sort: high severity first, then bullish before bearish
    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_events.sort(key=lambda e: (severity_order.get(e.severity, 3), e.impact != "bullish"))

    # ============================================================
    # STEP 6 — TOP 5 STORIES
    # ============================================================
    article_map = {a.article_id: a for a in analyzed}
    top_stories: list[TopStory] = []

    for rank, aid in enumerate(top_5_ids[:5], 1):
        article = article_map.get(aid)
        if not article:
            continue
        top_stories.append(TopStory(
            rank             = rank,
            article_id       = aid,
            title            = article.title,
            url              = article.url,
            one_line_summary = article.one_line_summary,
            sentiment_label  = article.sentiment.label,
            impact_reason    = top_5_reasons.get(str(aid), "High relevance score"),
        ))

    # ============================================================
    # STEP 7 — SIGNAL SCORES
    # ============================================================
    news_sentiment_score = _sentiment_to_score(sentiment_breakdown)
    event_severity_score = _event_severity_score(all_events)
    source_quality_score = _source_quality_score(analyzed)
    recency_score        = _recency_score(analyzed)

    overall_signal = round(
        news_sentiment_score * 0.35 +
        event_severity_score * 0.30 +
        source_quality_score * 0.20 +
        recency_score        * 0.15,
        2,
    )

    signal_score = NewsSignalScore(
        news_sentiment = news_sentiment_score,
        event_severity = event_severity_score,
        source_quality = source_quality_score,
        recency        = recency_score,
        overall        = overall_signal,
    )

    # ============================================================
    # STEP 8 — LLM NARRATIVE SUMMARY
    # ============================================================
    try:
        llm_summary = _llm_generate_summary(
            llm         = llm,
            ticker      = ticker,
            company     = company_name,
            breakdown   = sentiment_breakdown,
            top_stories = top_stories,
            events      = all_events,
            signal      = signal_score,
        )
    except Exception as e:
        llm_summary = (
            f"Narrative summary generation failed ({str(e)}). "
            f"Sentiment: {overall_label}. Signal score: {overall_signal}/10."
        )

    # ============================================================
    # STEP 9 — ASSEMBLE FINAL OUTPUT
    # ============================================================
    news_output = NewsOutput(
        metadata={
            "ticker":             ticker,
            "company_name":       company_name,
            "articles_fetched":   total_fetched,
            "articles_analyzed":  len(analyzed),
            "articles_rejected":  len(rejected),
            "query_used":         query_used,
            "analysis_timestamp": timestamp,
            "data_source":        "tavily",
            "status":             "success" if len(analyzed) >= 5 else "partial",
            "warning": (
                f"Only {len(analyzed)} valid articles found — results may be thin."
                if len(analyzed) < 5 else None
            ),
        },
        analyzed_articles      = analyzed,
        rejected_articles      = rejected,
        top_stories            = top_stories,
        sentiment_breakdown    = sentiment_breakdown,
        all_high_impact_events = all_events,
        signal_score           = signal_score,
        llm_summary            = llm_summary,
    )

    return news_output
    # return {**state, "news_output": news_output.model_dump()}


# ============================================================
# QUICK TEST — python news_research_agent.py
# ============================================================
# if __name__ == "__main__":
#     llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

#     test_state = {
#         "ticker":              "SBIN",
#         "depth":               "standard",
#         "focus_area":          "news,fundamentals",
#         "fundamentals_output": None,
#     }

#     result = news_research_agent(test_state, llm=llm)
#     print(json.dumps(result["news_output"], indent=2, default=str))