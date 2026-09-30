from dotenv import load_dotenv
import os

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
# Only needed if LLM_PROVIDER is switched to "anthropic". Defined here so
# llm.py's anthropic branch no longer raises ImportError.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Switch this one line later to swap to Claude
LLM_PROVIDER = "groq"  # change to "anthropic" later

# ── SINGLE SOURCE OF TRUTH ──
# Every analyst (technical, fundamentals, news, sentiment) and main.py's
# DEBATE_TICKERS reads this list. Change coverage here only — no other
# file should hardcode a ticker list.
# (Previously defined twice in this file; the duplicate is removed.)
TICKERS = ["XOM", "NEE", "ULG.SI", "P52.SI", "DRAM"]
# Parked: "SPY", "QQQ", "TLT", "GLD", "0P00006G05"

FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi":            "CPIAUCSL",
    "unemployment":   "UNRATE",
    "10y_yield":      "GS10"
}
