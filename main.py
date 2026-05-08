"""
main.py
-------
FastAPI entry point for the Multi-Agent Stock Research System.

Routes:
  GET  /              → serves the HTML dashboard
  POST /analyze       → blocking analysis, returns full final_state JSON
  GET  /health        → health check
"""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.models import AnalyzeRequest, AnalyzeResponse, HealthResponse, ErrorResponse
from api.pipeline import run_research_pipeline
from api.config import get_settings

from api.auth import get_current_user, get_optional_user
from api.supabase_client import get_supabase
from api.models import ReportSummary, ReportDetail, SaveReportRequest
from fastapi import Depends


# ── Helpers ───────────────────────────────────────────────────────────

def _to_dict(obj) -> dict | None:
    """
    Coerce an agent output to a plain dict regardless of whether the
    pipeline returned a Pydantic model instance, a dataclass, or an
    already-plain dict.  Returns None if obj is None.
    """
    if obj is None:
        return None
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses
        return dataclasses.asdict(obj)
    # Already a dict
    if isinstance(obj, dict):
        return obj
    # Last resort: try __dict__
    return vars(obj) if hasattr(obj, "__dict__") else None

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stock_research.main")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Stock Research API starting up…")
    yield
    logger.info("👋 Stock Research API shutting down…")


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Agent Stock Research API",
    description="LangGraph-powered parallel research pipeline: Fundamentals · Technical · News → Synthesis → Critic",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS/JS if ever extracted)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing():
    """Serve the landing/hero page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Landing page not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the research dashboard (protected client-side by Supabase auth)."""
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard template not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Liveness probe."""
    return HealthResponse(status="ok", version=app.version)

_stock_cache: list[str] = []   # module-level cache

@app.get("/stocks", tags=["Meta"])
async def get_stocks():
    """Return all NSE stock codes — called once on page load."""
    global _stock_cache
    if _stock_cache:
        return JSONResponse(content=_stock_cache)
    try:
        from nsetools import Nse
        nse = Nse()
        all_stocks = nse.get_stock_codes()
        _stock_cache = sorted([t for t in all_stocks if t])
        return JSONResponse(content=_stock_cache)
    except Exception as exc:
        logger.warning("Failed to fetch stock codes: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not load stock list: {exc}")


@app.get(
    "/reports",
    tags=["Reports"],
    summary="List saved reports for the authenticated user",
)

async def list_reports(
    limit: int = 20,
    user: dict = Depends(get_current_user),
):
    """Return the N most recent reports for the logged-in user."""
    try:
        sb = get_supabase()
        result = (
            sb.table("reports")
            .select("id, ticker, company_name, verdict, elapsed_seconds, created_at")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return JSONResponse(content=result.data if result.data else [])
    except Exception as exc:
        logger.warning("Failed to fetch reports for user %s: %s", user["id"], exc)
        return JSONResponse(content=[])


@app.get(
    "/reports/{report_id}",
    response_model=ReportDetail,
    tags=["Reports"],
    summary="Fetch a single saved report by ID",
)
async def get_report(
    report_id: str,
    user: dict = Depends(get_current_user),
):
    """Fetch the full payload of a saved report. Enforces ownership."""
    sb = get_supabase()
    result = (
        sb.table("reports")
        .select("*")
        .eq("id", report_id)
        .eq("user_id", user["id"])   # ownership check
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found.")
    return result.data


@app.delete(
    "/reports/{report_id}",
    status_code=204,
    tags=["Reports"],
    summary="Delete a saved report",
)
async def delete_report(
    report_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a report. Only the owner can delete."""
    sb = get_supabase()
    sb.table("reports").delete().eq("id", report_id).eq("user_id", user["id"]).execute()

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Research"],
    summary="Run full multi-agent stock research",
)
async def analyze(request: AnalyzeRequest, user: dict = Depends(get_optional_user)):
    """
    Blocking endpoint: waits for all agents to complete before returning.
    Typical latency: 20–60 s depending on ticker and model.
    """

    ticker = request.ticker.strip().upper()
    logger.info("▶ /analyze  ticker=%s  depth=%s  user=%s", ticker, None, user)

    t0 = time.perf_counter()
    try:
        final_state = await run_research_pipeline(ticker=ticker, depth=None)
        elapsed = round(time.perf_counter() - t0, 2)
        # Auto-save report if user is authenticated
        if user:
            try:
                sb = get_supabase()
                synth  = final_state.get("synthesis_output")
                critic = final_state.get("critic_output")
                fund   = final_state.get("fundamentals_output")
                verdict = None
                if critic and hasattr(critic, 'critic_verdict'):
                    verdict = getattr(critic.critic_verdict, 'critic_adjusted_verdict', None)
                if not verdict and synth and hasattr(synth, 'recommendation'):
                    verdict = getattr(synth.recommendation, 'verdict', None)
                company = None

                if fund:
                    if hasattr(fund, 'metadata'):
                        metadata = fund.metadata
                        if isinstance(metadata, dict):
                            company = metadata.get('company_name')
                        else:
                            company = getattr(metadata, 'company_name', None)
                    elif isinstance(fund, dict):
                        company = (fund.get('metadata') or {}).get('company_name')

                print("Company_name:", company)
                sb.table("reports").insert({
                    "user_id":         user["id"],
                    "ticker":          ticker,
                    "company_name":    company,
                    "verdict":         verdict,
                    "elapsed_seconds": elapsed,
                    "payload":         AnalyzeResponse(
                        ticker=ticker, 
                        elapsed_seconds=elapsed,
                        fundamentals_output=_to_dict(final_state.get("fundamentals_output")),
                        technical_output=_to_dict(final_state.get("technical_output")),
                        news_output=_to_dict(final_state.get("news_output")),
                        synthesis_output=_to_dict(final_state.get("synthesis_output")),
                        critic_output=_to_dict(final_state.get("critic_output")),
                    ).model_dump(),
                }).execute()
            except Exception as save_exc:
                logger.warning("Failed to save report for user %s: %s", user['id'], save_exc)
    except ValueError as exc:
        logger.warning("Validation error for %s: %s", ticker, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Pipeline failed for %s", ticker)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    logger.info("✅ /analyze  ticker=%s  elapsed=%.2fs", ticker, elapsed)

    return AnalyzeResponse(
        ticker=ticker,
        elapsed_seconds=elapsed,
        fundamentals_output=_to_dict(final_state.get("fundamentals_output")),
        technical_output=_to_dict(final_state.get("technical_output")),
        news_output=_to_dict(final_state.get("news_output")),
        synthesis_output=_to_dict(final_state.get("synthesis_output")),
        critic_output=_to_dict(final_state.get("critic_output")),
    )