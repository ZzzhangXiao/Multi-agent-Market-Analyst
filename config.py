from dotenv import load_dotenv
import os

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Switch this one line later to swap to Claude
LLM_PROVIDER = "groq"  # change to "anthropic" later

TICKERS = ["XOM", "NEE", "ULG.SI", "P52.SI"]
#     "SPY", "QQQ", "TLT", "GLD",
#     "XOM", "NEE",
#     "ULG.SI", "P52.SI",
#     "0P00006G05", "DRAM"
# ]

# Tickers approved for strategy and backtesting
# Must have clean continuous price history
STRATEGY_TICKERS = [
    "SPY", "QQQ", "TLT", "GLD",
    "XOM", "NEE", "P52.SI"
]

FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi":            "CPIAUCSL",
    "unemployment":   "UNRATE",
    "10y_yield":      "GS10"
}