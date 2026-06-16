import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import requests
import pandas as pd
from datetime import datetime, timedelta
from llm import get_llm
from config import FINNHUB_API_KEY, TICKERS


def fetch_company_news(ticker: str, days_back: int = 7) -> list:
    """
    Fetch real news headlines from Finnhub for a given ticker.
    Note: SGX tickers (ending in .SI) have limited Finnhub coverage.
    """
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Finnhub uses different format for some tickers
    # SGX tickers not well supported — skip gracefully
    if ".SI" in ticker:
        return []

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from":   start_date,
        "to":     end_date,
        "token":  FINNHUB_API_KEY
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            articles = resp.json()
            # Return top 5 most recent
            return [
                {
                    "headline": a.get("headline", ""),
                    "summary":  a.get("summary",  "")[:200],
                    "source":   a.get("source",   ""),
                    "datetime": datetime.fromtimestamp(
                        a.get("datetime", 0)
                    ).strftime("%Y-%m-%d")
                }
                for a in articles[:5]
            ]
    except Exception as e:
        print(f"  News fetch failed for {ticker}: {e}")

    return []


def fetch_market_news(days_back: int = 3) -> list:
    """
    Fetch general market news — not ticker specific.
    Good for macro themes and sentiment.
    """
    url    = "https://finnhub.io/api/v1/news"
    params = {
        "category": "general",
        "token":    FINNHUB_API_KEY
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            articles = resp.json()
            return [
                {
                    "headline": a.get("headline", ""),
                    "summary":  a.get("summary",  "")[:200],
                    "source":   a.get("source",   ""),
                }
                for a in articles[:10]
            ]
    except Exception as e:
        print(f"  Market news fetch failed: {e}")

    return []


def build_news_summary(tickers: list) -> str:
    lines = []

    # General market news
    print("  Fetching general market news...")
    market_news = fetch_market_news()
    if market_news:
        lines.append("GENERAL MARKET NEWS:")
        for n in market_news:
            lines.append(f"  [{n['source']}] {n['headline']}")
    else:
        lines.append("GENERAL MARKET NEWS: unavailable")

    lines.append("")

    # Per-ticker news
    for ticker in tickers:
        print(f"  Fetching news for {ticker}...")
        news = fetch_company_news(ticker)
        if news:
            lines.append(f"{ticker} NEWS:")
            for n in news:
                lines.append(f"  [{n['datetime']}] {n['headline']}")
                if n["summary"]:
                    lines.append(f"    {n['summary']}")
        else:
            lines.append(f"{ticker} NEWS: no recent coverage found")
        lines.append("")

    return "\n".join(lines)


def run() -> str:
    print("News analyst running...")

    if not FINNHUB_API_KEY:
        print("  No Finnhub API key found — using LLM knowledge only")
        return _run_llm_only()

    news_summary = build_news_summary(TICKERS)

    llm    = get_llm()
    today  = datetime.today().strftime("%Y-%m-%d")

    prompt = f"""
You are a financial news analyst at a hedge fund.
Today's date is {today}.

Analyze the following REAL current news headlines and produce a news brief.

Cover:
1. Dominant macro themes from today's headlines
2. Sector specific developments — what is moving each asset class
3. Overall sentiment — risk-on or risk-off based on current headlines
4. Event risks — anything in the news suggesting upcoming volatility
5. Per-ticker signal — for each ticker flag POSITIVE / NEGATIVE / NEUTRAL
   with one specific reason from the actual headlines

Be specific. Reference actual headlines. Do not use generic statements.
Note where news coverage is absent and flag those tickers for manual review.

REAL NEWS DATA ({today}):
{news_summary}
"""

    print("  Calling LLM with live news data...")
    response = llm.invoke(prompt)
    analysis = response.content

    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/news_analysis_{today}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# News Analysis — {today}\n\n")
        f.write(f"*Live data from Finnhub as of {today}*\n\n")
        f.write("## Raw Headlines\n\n")
        f.write(news_summary)
        f.write("\n\n## Analysis\n\n")
        f.write(analysis)

    print("\nNews Analysis:")
    print(analysis)
    print(f"\nSaved to reports/news_analysis_{today}.md")
    return analysis


def _run_llm_only() -> str:
    """Fallback if no Finnhub key."""
    llm    = get_llm()
    prompt = f"""
You are a financial news analyst.
Note clearly that you have no real-time news access.
Provide general market themes based on your training data.
Flag explicitly that human review of current news is required.

Tickers: {', '.join(TICKERS)}
"""
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    run()