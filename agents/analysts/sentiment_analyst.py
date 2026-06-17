import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import requests
import pandas as pd
from datetime import datetime
from llm import get_llm
from config import FINNHUB_API_KEY, TICKERS


def fetch_sentiment(ticker: str) -> dict:
    """
    Finnhub news-sentiment endpoint — aggregate bullish/bearish scoring
    derived from recent news and social mentions. Reuses the existing
    Finnhub key instead of adding a Reddit/StockTwits credential.
    Note: SGX tickers (.SI) and some funds have no coverage.
    """
    if ".SI" in ticker or not FINNHUB_API_KEY:
        return {}

    url = "https://finnhub.io/api/v1/news-sentiment"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if not data or "sentiment" not in data:
                return {}
            return {
                "bullish_pct":      data.get("sentiment", {}).get("bullishPercent"),
                "bearish_pct":      data.get("sentiment", {}).get("bearishPercent"),
                "buzz_articles":    data.get("buzz", {}).get("articlesInLastWeek"),
                "buzz_weekly_avg":  data.get("buzz", {}).get("weeklyAverage"),
                "sector_avg_score": data.get("sectorAverageBullishPercent"),
            }
    except Exception as e:
        print(f"  Sentiment fetch failed for {ticker}: {e}")

    return {}


def build_sentiment_summary(tickers: list) -> str:
    lines = []

    for ticker in tickers:
        print(f"  Fetching sentiment for {ticker}...")
        s = fetch_sentiment(ticker)

        if not s or s.get("bullish_pct") is None:
            lines.append(f"{ticker}: sentiment data unavailable (no coverage)")
            continue

        buzz_ratio = "n/a"
        if s["buzz_articles"] is not None and s["buzz_weekly_avg"]:
            buzz_ratio = round(s["buzz_articles"] / max(s["buzz_weekly_avg"], 1), 2)

        lines.append(
            f"{ticker}: bullish={round(s['bullish_pct'] * 100, 1)}% | "
            f"bearish={round(s['bearish_pct'] * 100, 1)}% | "
            f"articles_this_week={s['buzz_articles']} | "
            f"buzz_vs_normal={buzz_ratio}x | "
            f"sector_avg_bullish={round((s['sector_avg_score'] or 0) * 100, 1)}%"
        )

    return "\n".join(lines)


def run() -> str:
    print("Sentiment analyst running...")

    if not FINNHUB_API_KEY:
        print("  No Finnhub API key found — sentiment analysis skipped")
        return _run_llm_only()

    sentiment_summary = build_sentiment_summary(TICKERS)

    llm    = get_llm()
    prompt = f"""
You are a market sentiment analyst at a hedge fund, tracking news-derived
bullish/bearish scoring and media attention ("buzz") for each asset.

Analyze the following sentiment data.
Be specific — reference exact bullish/bearish percentages and buzz ratios.
Note explicitly when sentiment data is unavailable and rely on the news
and technical analysts' context instead for those tickers.

For each asset with data, cover:
1. Sentiment skew — net bullish or bearish, and how strongly
2. Attention level — is buzz elevated vs normal (buzz_vs_normal > 1.5x is notable)
3. Divergence — does sentiment disagree with the stock's own sector average,
   which can signal an idiosyncratic story (not just sector-wide mood)
4. Crowding risk — extreme bullish sentiment (>70%) combined with high buzz
   can signal a crowded trade vulnerable to reversal

SENTIMENT DATA:
{sentiment_summary}
"""

    print("Calling LLM...")
    response = llm.invoke(prompt)
    analysis = response.content

    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/sentiment_analysis_{today}.md"
    with open(filepath, "w") as f:
        f.write(f"# Sentiment Analysis — {today}\n\n")
        f.write(sentiment_summary)
        f.write("\n\n## Analysis\n\n")
        f.write(analysis)

    print("\nSentiment Analysis:")
    print(analysis)
    print(f"\nSaved to {filepath}")
    return analysis


def _run_llm_only() -> str:
    """Fallback if no Finnhub key."""
    llm    = get_llm()
    prompt = f"""
You are a market sentiment analyst.
Note clearly that you have no real-time sentiment data access.
Flag explicitly that human review of social/news sentiment is required.

Tickers: {', '.join(TICKERS)}
"""
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    run()