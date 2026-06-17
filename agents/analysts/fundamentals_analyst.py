import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import yfinance as yf
import pandas as pd
from datetime import datetime
from llm import get_llm
from config import TICKERS


def fetch_fundamentals(ticker: str) -> dict:
    """
    Pull fundamentals via yfinance — no new API key required.
    Returns {} if data is unavailable (common for SGX / mutual fund tickers).
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"  Fundamentals fetch failed for {ticker}: {e}")
        return {}

    # yfinance returns a near-empty dict for tickers with no fundamentals
    # (ETFs, mutual funds, some SGX listings) — detect and skip gracefully
    if not info or info.get("trailingPE") is None and info.get("marketCap") is None:
        return {}

    return {
        "market_cap":        info.get("marketCap"),
        "pe_ratio":           info.get("trailingPE"),
        "forward_pe":         info.get("forwardPE"),
        "peg_ratio":          info.get("pegRatio"),
        "price_to_book":      info.get("priceToBook"),
        "debt_to_equity":     info.get("debtToEquity"),
        "current_ratio":      info.get("currentRatio"),
        "profit_margin":      info.get("profitMargins"),
        "revenue_growth":     info.get("revenueGrowth"),
        "earnings_growth":    info.get("earningsGrowth"),
        "roe":                info.get("returnOnEquity"),
        "dividend_yield":     info.get("dividendYield"),
        "free_cashflow":      info.get("freeCashflow"),
        "52w_high":           info.get("fiftyTwoWeekHigh"),
        "52w_low":            info.get("fiftyTwoWeekLow"),
    }


def build_fundamentals_summary(tickers: list) -> str:
    lines = []

    for ticker in tickers:
        print(f"  Fetching fundamentals for {ticker}...")
        f = fetch_fundamentals(ticker)

        if not f:
            lines.append(f"{ticker}: fundamentals unavailable (ETF/fund/limited coverage)")
            continue

        def pct(x):
            return f"{round(x * 100, 1)}%" if isinstance(x, (int, float)) else "n/a"

        def num(x):
            return round(x, 2) if isinstance(x, (int, float)) else "n/a"

        lines.append(
            f"{ticker}: "
            f"P/E={num(f['pe_ratio'])} | Fwd P/E={num(f['forward_pe'])} | "
            f"PEG={num(f['peg_ratio'])} | P/B={num(f['price_to_book'])} | "
            f"D/E={num(f['debt_to_equity'])} | Current Ratio={num(f['current_ratio'])} | "
            f"Profit Margin={pct(f['profit_margin'])} | Revenue Growth={pct(f['revenue_growth'])} | "
            f"Earnings Growth={pct(f['earnings_growth'])} | ROE={pct(f['roe'])} | "
            f"Div Yield={pct(f['dividend_yield'])} | "
            f"52w Range=[{num(f['52w_low'])}, {num(f['52w_high'])}]"
        )

    return "\n".join(lines)


def run() -> str:
    print("Fundamentals analyst running...")

    fundamentals_summary = build_fundamentals_summary(TICKERS)

    llm    = get_llm()
    prompt = f"""
You are a fundamentals analyst at a hedge fund, focused on valuation and financial health.

Analyze the following fundamental data for each asset.
Be specific — reference exact P/E, PEG, margin, and growth figures.
Do not be generic. Every statement must reference a specific number.
Note explicitly when fundamentals are unavailable (ETFs, funds) and rely on
technical/macro context instead for those tickers.

For each asset with data, cover:
1. Valuation — is it cheap, fair, or expensive relative to growth (PEG) and book value
2. Financial health — debt/equity, current ratio, is the balance sheet solid
3. Growth quality — revenue growth vs earnings growth, are margins expanding or shrinking
4. Profitability — ROE and profit margin relative to typical sector levels
5. Red flags — anything suggesting deteriorating fundamentals

FUNDAMENTALS DATA:
{fundamentals_summary}
"""

    print("Calling LLM...")
    response = llm.invoke(prompt)
    analysis = response.content

    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/fundamentals_analysis_{today}.md"
    with open(filepath, "w") as f:
        f.write(f"# Fundamentals Analysis — {today}\n\n")
        f.write(analysis)

    print("\nFundamentals Analysis:")
    print(analysis)
    print(f"\nSaved to {filepath}")
    return analysis


if __name__ == "__main__":
    run()