"""
api/supabase_client.py
----------------------
Supabase admin (service-role) client — used server-side only.
Never expose the service key to the frontend.
"""

from functools import lru_cache
from supabase import create_client, Client
from api.config import get_settings


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Return a cached Supabase admin client."""
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_key)
