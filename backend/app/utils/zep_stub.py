"""
Zep stub module — provides no-op implementations when ZEP_API_KEY is not configured.
All Zep services check this flag and return empty/default data if Zep is unavailable.
"""

from ..config import Config
from .logger import get_logger

logger = get_logger('mirofish.zep_stub')

_warned = False

def is_zep_available() -> bool:
    """Check if Zep API key is configured."""
    global _warned
    available = bool(Config.ZEP_API_KEY)
    if not available and not _warned:
        logger.warning("ZEP_API_KEY not configured — all Zep features will return empty data. "
                       "To enable graph memory, set ZEP_API_KEY environment variable.")
        _warned = True
    return available
