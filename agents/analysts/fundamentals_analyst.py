import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import random
import yfinance as yf
from datetime import datetime
from llm import get_llm
from config import TICKERS

CACHE_PATH = "data/fundamentals_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60


def _load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _is_cache_fresh(entry: dict) -> bool:
    return time.time() - entry.get("timestamp", 0) < CACHE_TTL_SECONDS

ETF_QUOTE_TYPES = {"ETF", "MUTUALFUND", "INDEX"}


def asset_class_label(quote_type):
    if not quote_type:
        return "ASSET CLASS UNKNOWN"
    qt = str(quote_type).upper()
    if qt in ETF_QUOTE_TYPES:
        return f"{qt} — SKIP COMPANY-STYLE VALUATION/GROWTH ANALYSIS"
    if qt == "EQUITY":
        return "EQUITY"
    return f"OTHER ({qt})"

def _extract_fundamentals(info: dict) -> dict:
    return {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_book": info.get("priceToBook"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "roe": info.get("returnOnEquity"),
        "dividend_yield": info.get("dividendYield"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "market_cap": info.get("marketCap"),
        "quote_type": info.get("quoteType"),
        "long_name": info.get("longName"),
    }


def valuation_label(pe, forward_pe, peg, pb):
    if peg is not None and peg < 1:
        return "ATTRACTIVE VALUATION"
    if peg is not None and peg > 2:
        return "EXPENSIVE RELATIVE TO GROWTH"
    if pe is not None and forward_pe is not None and forward_pe < pe:
        return "EARNINGS EXPECTED TO IMPROVE"
    return "FAIR / INCONCLUSIVE VALUATION"


def balance_sheet_label(debt_to_equity, current_ratio):
    if current_ratio is None and debt_to_equity is None:
        return "BALANCE SHEET DATA UNAVAILABLE"

    if current_ratio is not None and current_ratio < 1:
        liquidity = "WEAK LIQUIDITY"
    elif current_ratio is not None and current_ratio >= 1.5:
        liquidity = "SOLID LIQUIDITY"
    elif current_ratio is not None:
        liquidity = "ADEQUATE LIQUIDITY"
    else:
        liquidity = "LIQUIDITY DATA UNAVAILABLE"

    if debt_to_equity is not None and debt_to_equity > 100:
        leverage = "HIGH LEVERAGE"
    elif debt_to_equity is not None and debt_to_equity > 50:
        leverage = "MODERATE LEVERAGE"
    elif debt_to_equity is not None:
        leverage = "LOW / MANAGEABLE LEVERAGE"
    else:
        leverage = "LEVERAGE DATA UNAVAILABLE"

    return f"{liquidity}; {leverage}"


def growth_label(revenue_growth, earnings_growth):
    if revenue_growth is None or earnings_growth is None:
        return "GROWTH DATA INCOMPLETE"
    if revenue_growth > 0 and earnings_growth > revenue_growth:
        return "EARNINGS GROWING FASTER THAN REVENUE"
    if revenue_growth > 0 and earnings_growth < 0:
        return "MARGIN PRESSURE / EARNINGS DETERIORATION"
    if revenue_growth < 0 and earnings_growth < 0:
        return "BROAD GROWTH DETERIORATION"
    return "MIXED GROWTH QUALITY"


def fetch_fundamentals_live(ticker: str, max_retries: int = 3) -> tuple:
    """
    Returns (data: dict, failure_reason: str | None).
    failure_reason is None on success.
    """
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if not info:
                return {}, "NO_INFO_RETURNED"

            f = _extract_fundamentals(info)

            has_useful_data = any(
                f.get(k) is not None
                for k in ["pe_ratio", "market_cap", "price_to_book", "dividend_yield"]
            )

            if has_useful_data:
                return f, None
            return {}, "NO_USEFUL_FIELDS"

        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                wait = (2 ** attempt) * 5 + random.uniform(0, 3)
                print(f"  Rate limited on {ticker}, backing off {wait:.1f}s...")
                time.sleep(wait)
                continue

            if "Too Many Requests" in str(e):
                print(f"  Rate limited on {ticker}, retries exhausted")
                return {}, "RATE_LIMITED_RETRIES_EXHAUSTED"

            print(f"  Fundamentals fetch failed for {ticker}: {e}")
            return {}, "FETCH_EXCEPTION"

    return {}, "RATE_LIMITED_RETRIES_EXHAUSTED"



def get_fundamentals(ticker: str, cache: dict, allow_live: bool = True) -> tuple:
    """
    Returns (data: dict, failure_reason: str | None).
    failure_reason is None whenever data came from somewhere usable
    (fresh cache, stale cache, or a successful live fetch).
    """
    entry = cache.get(ticker)

    if entry and _is_cache_fresh(entry):
        print(f"  Using fresh cache for {ticker}")
        return entry.get("data", {}), None

    if not allow_live:
        if entry:
            print(f"  Using stale cache for {ticker}")
            return entry.get("data", {}), None
        return {}, "CIRCUIT_BREAKER_SKIPPED_NO_CACHE"

    print(f"  Fetching fundamentals for {ticker}...")
    f, reason = fetch_fundamentals_live(ticker)

    if f:
        cache[ticker] = {
            "timestamp": time.time(),
            "date": datetime.today().strftime("%Y-%m-%d"),
            "data": f,
        }
        return f, None

    if entry:
        print(f"  Falling back to stale cache for {ticker}")
        return entry.get("data", {}), None

    return {}, reason


def pct(x):
    return f"{round(x * 100, 1)}%" if isinstance(x, (int, float)) else "n/a"


def num(x):
    return round(x, 2) if isinstance(x, (int, float)) else "n/a"


def div_yield_pct(x):
    if not isinstance(x, (int, float)):
        return "n/a"

    pct_value = x * 100 if x <= 1 else x
    return f"{round(pct_value, 2)}%"


FAILURE_REASON_MESSAGES = {
    "NO_INFO_RETURNED":
        "no data returned by yfinance for this ticker (likely delisted, "
        "wrong symbol, or unsupported exchange — NOT a rate-limit issue)",
    "NO_USEFUL_FIELDS":
        "yfinance returned a profile but none of the key fields "
        "(P/E, market cap, P/B, dividend yield) were populated — "
        "consistent with an ETF/fund/index that lacks company-style "
        "fundamentals, NOT a fetch failure",
    "RATE_LIMITED_RETRIES_EXHAUSTED":
        "yfinance rate-limited this request and all retries were "
        "exhausted — transient API throttling, not a data-availability issue",
    "FETCH_EXCEPTION":
        "an unexpected error occurred while fetching this ticker's data "
        "(see logs) — not a rate-limit or missing-data issue",
    "CIRCUIT_BREAKER_SKIPPED_NO_CACHE":
        "skipped live fetch because 2+ consecutive prior tickers were "
        "rate-limited (circuit breaker open), and no cached data exists "
        "for this ticker yet — try again later or on its own",
}


def build_fundamentals_summary(tickers: list) -> str:
    lines = []
    cache = _load_cache()
    consecutive_live_failures = 0

    for ticker in tickers:
        allow_live = consecutive_live_failures < 2

        if not allow_live:
            print(f"  Circuit breaker open — cache only for {ticker}")

        f, failure_reason = get_fundamentals(ticker, cache, allow_live=allow_live)

        if not f:
            consecutive_live_failures += 1
            reason_text = FAILURE_REASON_MESSAGES.get(
                failure_reason,
                "fundamentals unavailable (unknown reason)"
            )
            lines.append(f"{ticker}: FUNDAMENTALS UNAVAILABLE — {reason_text}")
            continue

        consecutive_live_failures = 0


def run(ticker: str = None) -> str:
    print("Fundamentals analyst running...")

    scope = [ticker] if ticker else TICKERS
    fundamentals_summary = build_fundamentals_summary(scope)

    llm = get_llm()
    prompt = f"""
You are a fundamentals analyst at a hedge fund.

Rules:
- Use ASSET CLASS LABEL verbatim.
- If ASSET CLASS LABEL is ETF or FUND, do not discuss company balance sheet, revenue growth, earnings growth, ROE, or leverage unless actual data is present.
- If BALANCE SHEET LABEL says DATA UNAVAILABLE, do not describe liquidity or leverage as good or bad.
- Use the precomputed VALUATION LABEL, BALANCE SHEET LABEL, and GROWTH LABEL verbatim.
- Do not say a company may struggle with liquidity unless BALANCE SHEET LABEL says WEAK LIQUIDITY.
- Do not say forward P/E below trailing P/E means overvaluation. It usually means earnings are expected to improve.
- Do not invent missing figures.
- For ETFs, funds, bond ETFs, gold ETFs, or thematic ETFs, do not analyze them like normal companies.
- Every conclusion must be tied to a number from the data.

For any ticker marked "FUNDAMENTALS UNAVAILABLE", the reason given is authoritative —
do not substitute your own explanation for why data is missing:
- If the reason mentions "no useful fields" or reads consistent with an ETF/fund,
  state plainly that this looks like an ETF/fund/index lacking company-style
  fundamentals. Do NOT describe this as a fetch failure or rate-limit issue.
- If the reason mentions "rate-limited", state plainly that this is a transient
  API throttling issue, not a reflection of the asset's actual data availability.
  Do NOT imply the company has no fundamentals data in general.
- If the reason mentions "no data returned", state plainly that yfinance had
  no profile at all for this symbol (possible delisting/wrong ticker/unsupported
  exchange). Do NOT attribute this to rate limits or to it being an ETF.
- If the reason mentions "circuit breaker", state plainly that the fetch was
  skipped this run due to prior consecutive failures, and no cached fallback
  existed — this says nothing about whether the ticker itself has fundamentals.
- If the reason mentions "unexpected error" (FETCH_EXCEPTION), say only that
  fetching failed for a technical reason and do not speculate further.
- Never merge these five reasons into a single generic "data unavailable" statement.

For each asset with company-level data, cover:
1. Valuation
2. Financial health
3. Growth quality
4. Profitability
5. Red flags

FUNDAMENTALS DATA:
{fundamentals_summary}
"""
    print("Calling LLM...")
    response = llm.invoke(prompt)
    analysis = response.content

    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs("reports", exist_ok=True)

    filepath = f"reports/fundamentals_analysis_{ticker or 'ALL'}_{today}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Fundamentals Analysis — {today}\n\n")
        f.write("## Deterministic Fundamentals Data\n\n")
        f.write(fundamentals_summary)
        f.write("\n\n## Analysis\n\n")
        f.write(analysis)

    print("\nFundamentals Analysis:")
    print(analysis)
    print(f"\nSaved to {filepath}")

    return analysis


if __name__ == "__main__":
    run()