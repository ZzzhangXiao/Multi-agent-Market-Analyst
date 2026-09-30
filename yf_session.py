"""
Shared yfinance transport.

Yahoo blocks default python-requests sessions via TLS fingerprinting.
A curl_cffi session impersonating Chrome gets past this. Every yfinance
call in the project (data_agent, fundamentals_analyst) should import
SESSION from here so the fix lives in exactly one place.
"""
import time
import random

try:
    from curl_cffi import requests as cffi_requests
    SESSION = cffi_requests.Session(impersonate="chrome")
except ImportError:
    # Fall back to yfinance's own default transport rather than crashing.
    print("  WARNING: curl_cffi not installed - yfinance will use its default session")
    SESSION = None


def jitter(lo: float = 1.0, hi: float = 2.5) -> None:
    """Randomised pause between Yahoo requests to avoid burst rate-limits."""
    time.sleep(random.uniform(lo, hi))
