import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import requests
from datetime import datetime
from llm import get_llm
from config import TICKERS

CACHE_PATH = "data/sentiment_cache.json"
CACHE_TTL_SECONDS = 6 * 60 * 60

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"


def _has_stocktwits_coverage(ticker: str) -> bool:
    return ".SI" not in ticker and not ticker.startswith("0P")


def confidence_tier(n_labeled: int) -> str:
    if n_labeled < 15:
        return "LOW CONFIDENCE (n<15, treat as noise)"
    elif n_labeled < 30:
        return "MODERATE CONFIDENCE (n<30, directional only)"
    return "HIGH CONFIDENCE"


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


def fetch_sentiment(ticker: str, cache: dict) -> dict:
    cached = cache.get(ticker)

    if cached and (time.time() - cached.get("_fetched_at", 0)) < CACHE_TTL_SECONDS:
        print(f"  Using cached sentiment for {ticker}")
        result = {k: v for k, v in cached.items() if k != "_fetched_at"}

        if "sample_confidence" not in result:
            result["sample_confidence"] = confidence_tier(result.get("labeled_messages", 0))

        return result

    if not _has_stocktwits_coverage(ticker):
        return {}

    url = STOCKTWITS_URL.format(ticker=ticker)

    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "trading-agents-research/1.0"},
        )
    except Exception as e:
        print(f"  Sentiment fetch failed for {ticker}: {e}")
        if cached:
            print(f"  Falling back to stale cache for {ticker}")
            return {k: v for k, v in cached.items() if k != "_fetched_at"}
        return {}

    if resp.status_code == 404:
        return {}

    if resp.status_code != 200:
        print(f"  StockTwits returned {resp.status_code} for {ticker}")
        if cached:
            print(f"  Falling back to stale cache for {ticker}")
            return {k: v for k, v in cached.items() if k != "_fetched_at"}
        return {}

    try:
        data = resp.json()
    except ValueError:
        return {}

    messages = data.get("messages", [])
    if not messages:
        return {}

    bullish = 0
    bearish = 0
    follower_weighted_bullish = 0
    follower_weighted_total = 0

    for msg in messages:
        sentiment = (msg.get("entities") or {}).get("sentiment")
        label = (sentiment or {}).get("basic") if sentiment else None
        followers = ((msg.get("user") or {}).get("followers")) or 1

        if label == "Bullish":
            bullish += 1
            follower_weighted_bullish += followers
            follower_weighted_total += followers
        elif label == "Bearish":
            bearish += 1
            follower_weighted_total += followers

    labeled = bullish + bearish

    if labeled == 0:
        result = {
            "total_messages": len(messages),
            "labeled_messages": 0,
            "bullish_pct": None,
            "bearish_pct": None,
            "sample_confidence": "NO DIRECTIONAL SIGNAL",
        }
    else:
        result = {
            "total_messages": len(messages),
            "labeled_messages": labeled,
            "bullish_pct": round(bullish / labeled, 4),
            "bearish_pct": round(bearish / labeled, 4),
            "follower_weighted_bullish_pct": (
                round(follower_weighted_bullish / follower_weighted_total, 4)
                if follower_weighted_total else None
            ),
            "sample_confidence": confidence_tier(labeled),
        }

    cache[ticker] = {**result, "_fetched_at": time.time()}
    return result


def build_sentiment_summary(tickers: list) -> str:
    lines = []
    cache = _load_cache()

    for ticker in tickers:
        print(f"  Fetching sentiment for {ticker}...")

        if not _has_stocktwits_coverage(ticker):
            lines.append(
                f"{ticker}: sentiment data unavailable "
                f"(no StockTwits coverage for SGX/fund tickers)"
            )
            continue

        s = fetch_sentiment(ticker, cache)
        time.sleep(0.5)

        if not s or not s.get("total_messages"):
            lines.append(
                f"{ticker}: sentiment data unavailable "
                f"(no recent StockTwits activity)"
            )
            continue

        if s.get("bullish_pct") is None:
            lines.append(
                f"{ticker}: {s['total_messages']} messages this stream, "
                f"but none carried a bullish/bearish tag — volume only, "
                f"no directional signal | sample_confidence=NO DIRECTIONAL SIGNAL"
            )
            continue

        weighted = s.get("follower_weighted_bullish_pct")
        weighted_str = f"{round(weighted * 100, 1)}%" if weighted is not None else "n/a"

        lines.append(
            f"{ticker}: bullish={round(s['bullish_pct'] * 100, 1)}% | "
            f"bearish={round(s['bearish_pct'] * 100, 1)}% | "
            f"labeled_messages={s['labeled_messages']}/{s['total_messages']} | "
            f"sample_confidence={s.get('sample_confidence', confidence_tier(s.get('labeled_messages', 0)))} | "
            f"follower_weighted_bullish={weighted_str}"
        )

    _save_cache(cache)
    return "\n".join(lines)


def run() -> str:
    print("Sentiment analyst running...")

    sentiment_summary = build_sentiment_summary(TICKERS)

    llm = get_llm()
    prompt = f"""
You are a market sentiment analyst at a hedge fund, tracking retail-trader
sentiment from StockTwits message streams for each asset.

Analyze the following sentiment data.

Rules:
- Be specific and reference exact bullish/bearish percentages and message counts.
- Sentiment marked LOW CONFIDENCE must be reported as inconclusive, not bullish or bearish.
- Sentiment marked MODERATE CONFIDENCE may be described as directional only, not strong evidence.
- Only HIGH CONFIDENCE samples can support strong sentiment conclusions.
- Do not call a trade crowded unless sentiment is extreme and sample confidence is HIGH.
- If sentiment data is unavailable, say so clearly and rely on news/technical context instead.
- Do not invent sentiment data for SGX listings, mutual funds, or low-activity tickers.

For each asset with data, cover:
1. Sentiment skew
2. Sample confidence
3. Follower-weighted versus raw bullish percentage
4. Crowding risk

SENTIMENT DATA:
{sentiment_summary}
"""

    print("Calling LLM...")
    response = llm.invoke(prompt)
    analysis = response.content

    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs("reports", exist_ok=True)

    filepath = f"reports/sentiment_analysis_{today}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Sentiment Analysis — {today}\n\n")
        f.write(sentiment_summary)
        f.write("\n\n## Analysis\n\n")
        f.write(analysis)

    print("\nSentiment Analysis:")
    print(analysis)
    print(f"\nSaved to {filepath}")

    return analysis


if __name__ == "__main__":
    run()