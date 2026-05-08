"""
api/auth.py
-----------
FastAPI dependency that verifies a Supabase JWT access token
sent in the Authorization: Bearer <token> header.

Usage in a route:
    @app.post("/analyze")
    async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
        ...

The dependency returns the decoded user dict on success,
or raises HTTP 401 if the token is missing / invalid.
"""

import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.supabase_client import get_supabase

logger = logging.getLogger("stock_research.auth")

_bearer = HTTPBearer(auto_error=False)   # auto_error=False → we return 401 manually


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """
    Verify Supabase JWT and return the user payload.
    Raises HTTP 401 if unauthenticated.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    sb    = get_supabase()

    try:
        # Supabase Python SDK v2 — verifies token server-side
        response = sb.auth.get_user(token)
        if response is None or response.user is None:
            raise ValueError("No user returned")
        return {
            "id":    response.user.id,
            "email": response.user.email,
            "meta":  response.user.user_metadata or {},
        }
    except Exception as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict | None:
    """
    Like get_current_user but returns None instead of raising 401.
    Use for endpoints that work both authenticated and unauthenticated.
    """
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
