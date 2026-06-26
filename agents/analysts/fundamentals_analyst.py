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


def fetch_fundamentals_live(ticker: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if not info:
                return {}

            f = _extract_fundamentals(info)

            has_useful_data = any(
                f.get(k) is not None
                for k in ["pe_ratio", "market_cap", "price_to_book", "dividend_yield"]
            )

            return f if has_useful_data else {}

        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                wait = (2 ** attempt) * 5 + random.uniform(0, 3)
                print(f"  Rate limited on {ticker}, backing off {wait:.1f}s...")
                time.sleep(wait)
                continue

            print(f"  Fundamentals fetch failed for {ticker}: {e}")
            return {}

    return {}


def get_fundamentals(ticker: str, cache: dict, allow_live: bool = True) -> dict:
    entry = cache.get(ticker)

    if entry and _is_cache_fresh(entry):
        print(f"  Using fresh cache for {ticker}")
        return entry.get("data", {})

    if not allow_live:
        if entry:
            print(f"  Using stale cache for {ticker}")
            return entry.get("data", {})
        return {}

    print(f"  Fetching fundamentals for {ticker}...")
    f = fetch_fundamentals_live(ticker)

    if f:
        cache[ticker] = {
            "timestamp": time.time(),
            "date": datetime.today().strftime("%Y-%m-%d"),
            "data": f,
        }
        return f

    if entry:
        print(f"  Falling back to stale cache for {ticker}")
        return entry.get("data", {})

    return {}


def pct(x):
    return f"{round(x * 100, 1)}%" if isinstance(x, (int, float)) else "n/a"


def num(x):
    return round(x, 2) if isinstance(x, (int, float)) else "n/a"


def div_yield_pct(x):
    if not isinstance(x, (int, float)):
        return "n/a"

    pct_value = x * 100 if x <= 1 else x
    return f"{round(pct_value, 2)}%"


def build_fundamentals_summary(tickers: list) -> str:
    lines = []
    cache = _load_cache()
    consecutive_live_failures = 0

    for ticker in tickers:
        allow_live = consecutive_live_failures < 2

        if not allow_live:
            print(f"  Circuit breaker open — cache only for {ticker}")

        f = get_fundamentals(ticker, cache, allow_live=allow_live)

        if not f:
            consecutive_live_failures += 1
            lines.append(
                f"{ticker}: fundamentals unavailable "
                f"(ETF/fund/limited coverage or rate-limited)"
            )
            continue

        consecutive_live_failures = 0

        valuation_view = valuation_label(
            f.get("pe_ratio"),
            f.get("forward_pe"),
            f.get("peg_ratio"),
            f.get("price_to_book"),
        )
        asset_class_view = asset_class_label(f.get("quote_type"))   # ADD THIS
        balance_view = balance_sheet_label(
            f.get("debt_to_equity"),
            f.get("current_ratio"),
        )

        growth_view = growth_label(
            f.get("revenue_growth"),
            f.get("earnings_growth"),
        )

        lines.append(
            f"{ticker}: "
            f"Name={f.get('long_name', 'n/a')} | Type={f.get('quote_type', 'n/a')} | "
            f"P/E={num(f.get('pe_ratio'))} | Fwd P/E={num(f.get('forward_pe'))} | "
            f"PEG={num(f.get('peg_ratio'))} | P/B={num(f.get('price_to_book'))} | "
            f"D/E={num(f.get('debt_to_equity'))} | Current Ratio={num(f.get('current_ratio'))} | "
            f"Profit Margin={pct(f.get('profit_margin'))} | "
            f"Revenue Growth={pct(f.get('revenue_growth'))} | "
            f"Earnings Growth={pct(f.get('earnings_growth'))} | "
            f"ROE={pct(f.get('roe'))} | "
            f"Div Yield={div_yield_pct(f.get('dividend_yield'))} | "
            f"52w Range=[{num(f.get('52w_low'))}, {num(f.get('52w_high'))}] | "
            f"ASSET CLASS LABEL={asset_class_view} | "
            f"VALUATION LABEL={valuation_view} | "
            f"BALANCE SHEET LABEL={balance_view} | "
            f"GROWTH LABEL={growth_view}"
        )

        time.sleep(0.5)

    _save_cache(cache)
    return "\n".join(lines)


def run() -> str:
    print("Fundamentals analyst running...")

    fundamentals_summary = build_fundamentals_summary(TICKERS)

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

    filepath = f"reports/fundamentals_analysis_{today}.md"
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