import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import requests
from datetime import datetime, timedelta
from llm import get_llm
from config import TICKERS, FINNHUB_API_KEY

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"


def _is_direct_ticker_news(ticker: str, headline: str, summary: str) -> bool:
    text = f"{headline} {summary}".lower()
    ticker_lower = ticker.lower()

    company_aliases = {
        "XOM": ["exxon", "exxonmobil", "exxon mobil"],
        "NEE": ["nextera", "next era"],
        "SPY": ["s&p 500", "sp 500", "s&p"],
        "QQQ": ["nasdaq", "nasdaq 100"],
        "TLT": ["treasury", "20-year treasury", "long bond"],
        "GLD": ["gold", "gold etf"],
        "DRAM": ["dram", "memory chip", "memory chips", "micron", "sk hynix"],
        "P52.SI": ["pan-united", "pan united"],
        "ULG.SI": ["ulti", "ulti group"],
        "0P00006G05": [],
    }

    if ticker_lower in text:
        return True

    for alias in company_aliases.get(ticker, []):
        if alias.lower() in text:
            return True

    return False


def fetch_general_market_news(limit: int = 10) -> list:
    params = {
        "category": "general",
        "token": FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(FINNHUB_NEWS_URL, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"  General news failed: {resp.status_code}")
            return []
        return resp.json()[:limit]
    except Exception as e:
        print(f"  General news fetch failed: {e}")
        return []


def fetch_company_news(ticker: str, days_back: int = 7, limit: int = 5) -> list:
    end = datetime.today().date()
    start = end - timedelta(days=days_back)

    params = {
        "symbol": ticker,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "token": FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(FINNHUB_COMPANY_NEWS_URL, params=params, timeout=10)

        if resp.status_code != 200:
            print(f"  News fetch failed for {ticker}: {resp.status_code}")
            return []

        raw_news = resp.json()[:limit * 2]

        filtered = []
        for item in raw_news:
            headline = item.get("headline", "")
            summary = item.get("summary", "")

            if _is_direct_ticker_news(ticker, headline, summary):
                item["relevance"] = "DIRECT"
            else:
                item["relevance"] = "SECTOR_CONTEXT"

            filtered.append(item)

        return filtered[:limit]

    except Exception as e:
        print(f"  News fetch failed for {ticker}: {e}")
        return []


def build_news_summary(tickers: list) -> str:
    lines = []

    print("  Fetching general market news...")
    general_news = fetch_general_market_news(limit=10)

    lines.append("GENERAL MARKET NEWS:")
    if general_news:
        for item in general_news:
            source = item.get("source", "Unknown")
            headline = item.get("headline", "")
            lines.append(f"  [{source}] {headline}")
    else:
        lines.append("  No general market news found.")

    lines.append("\nPER-TICKER NEWS:")

    for ticker in tickers:
        print(f"  Fetching news for {ticker}...")
        news_items = fetch_company_news(ticker)

        lines.append(f"\n{ticker} NEWS:")

        if not news_items:
            lines.append("  No recent direct coverage found — MANUAL REVIEW required.")
            continue

        direct_count = 0

        for item in news_items:
            date = datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d")
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            source = item.get("source", "Unknown")
            relevance = item.get("relevance", "UNKNOWN")

            if relevance == "DIRECT":
                direct_count += 1

            lines.append(f"  [{date}] [{source}] [{relevance}] {headline}")
            if summary:
                lines.append(f"    {summary[:300]}")

        if direct_count == 0:
            lines.append("  TICKER SIGNAL QUALITY: NO DIRECT COVERAGE — use only as sector context.")

    return "\n".join(lines)


def run() -> str:
    print("News analyst running...")

    news_summary = build_news_summary(TICKERS)

    llm = get_llm()
    prompt = f"""
You are a market news analyst at a hedge fund.

Ticker universe:
{TICKERS}

Rules:
- Only discuss tickers in this universe: {TICKERS}
- Do not introduce new ticker-level recommendations for companies outside this universe.
- General market headlines may be mentioned only as macro or sector context.
- If a news item is not directly about a ticker, label it as sector context, not ticker-specific signal.
- For tickers with no direct coverage, write MANUAL REVIEW instead of NEUTRAL.
- Do not invent news.
- Do not convert sector-context headlines into strong ticker signals.
- Every per-ticker signal must be one of: POSITIVE, NEGATIVE, MIXED, MANUAL REVIEW.
- Do not mention JPM, GS, or any other non-universe ticker as a per-ticker signal unless they are in the ticker universe.

Required output:
1. Dominant macro themes
2. Sector-specific developments
3. Overall sentiment
4. Event risks
5. Per-ticker signal for each ticker in the ticker universe only

NEWS DATA:
{news_summary}
"""

    print("  Calling LLM with live news data...")
    response = llm.invoke(prompt)
    analysis = response.content

    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs("reports", exist_ok=True)

    filepath = f"reports/news_analysis_{today}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# News Analysis — {today}\n\n")
        f.write("## Raw News Data\n\n")
        f.write(news_summary)
        f.write("\n\n## Analysis\n\n")
        f.write(analysis)

    print("\nNews Analysis:")
    print(analysis)
    print(f"\nSaved to {filepath}")

    return analysis


if __name__ == "__main__":
    run()