import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime
from llm import get_llm


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta  = series.diff()
    gain   = delta.where(delta > 0, 0).rolling(period).mean()
    loss   = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs     = gain / (loss + 1e-8)
    rsi    = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)


def compute_macd(series: pd.Series):
    ema12  = series.ewm(span=12).mean()
    ema26  = series.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist   = macd - signal
    return round(macd.iloc[-1], 4), round(signal.iloc[-1], 4), round(hist.iloc[-1], 4)


def compute_bollinger(series: pd.Series, period: int = 20):
    ma    = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    pct_b = (series - lower) / (upper - lower + 1e-8)
    return round(upper.iloc[-1], 2), round(lower.iloc[-1], 2), round(pct_b.iloc[-1], 3)


def compute_atr(prices: pd.DataFrame, ticker: str, period: int = 14) -> float:
    high  = prices[ticker].rolling(2).max()
    low   = prices[ticker].rolling(2).min()
    close = prices[ticker]
    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return round(tr.rolling(period).mean().iloc[-1], 4)


def build_technical_summary(prices: pd.DataFrame) -> str:
    lines = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 50:
            continue

        price   = round(series.iloc[-1], 2)
        ret_1d  = round(series.pct_change(1).iloc[-1] * 100, 2)
        ret_5d  = round(series.pct_change(5).iloc[-1] * 100, 2)
        ret_21d = round(series.pct_change(21).iloc[-1] * 100, 2)

        ma20    = round(series.rolling(20).mean().iloc[-1], 2)
        ma50    = round(series.rolling(50).mean().iloc[-1], 2)
        ma200   = round(series.rolling(min(200, len(series))).mean().iloc[-1], 2)

        rsi             = compute_rsi(series)
        macd, sig, hist = compute_macd(series)
        bb_upper, bb_lower, pct_b = compute_bollinger(series)
        atr             = compute_atr(prices, ticker)

        trend = "uptrend" if price > ma50 > ma200 else \
                "downtrend" if price < ma50 < ma200 else "mixed"

        rsi_signal = "overbought" if rsi > 70 else \
                     "oversold"   if rsi < 30 else "neutral"

        macd_signal = "bullish crossover" if hist > 0 else "bearish crossover"

        lines.append(f"""
{ticker}:
  Price: {price} | 1d: {ret_1d}% | 5d: {ret_5d}% | 21d: {ret_21d}%
  Trend: {trend} | MA20: {ma20} | MA50: {ma50} | MA200: {ma200}
  RSI({14}): {rsi} [{rsi_signal}]
  MACD: {macd} | Signal: {sig} | Hist: {hist} [{macd_signal}]
  Bollinger: upper={bb_upper} lower={bb_lower} %B={pct_b}
  ATR(14): {atr}""")

    return "\n".join(lines)


def run() -> str:
    print("Technical analyst running...")

    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True
    )

    technical_summary = build_technical_summary(prices)

    llm    = get_llm()
    prompt = f"""
You are a professional technical analyst at a hedge fund.

Analyze the following technical indicators for each asset.
Be specific — reference exact RSI levels, MACD crossovers, Bollinger band positions.
Do not be generic. Every statement must reference a specific number.

For each asset cover:
1. Trend structure (is price above or below key MAs, what does that mean)
2. Momentum (RSI level and direction, overbought/oversold risk)
3. MACD signal (bullish or bearish crossover, histogram direction)
4. Volatility (ATR, Bollinger band position — is price extended?)
5. Short term outlook (1-4 weeks) with specific price levels to watch

Flag any assets showing strong technical setups or breakdown risk.

TECHNICAL DATA:
{technical_summary}
"""

    print("Calling LLM...")
    response = llm.invoke(prompt)
    analysis = response.content

    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/technical_analysis_{today}.md"
    with open(filepath, "w") as f:
        f.write(f"# Technical Analysis — {today}\n\n")
        f.write(analysis)

    print("\nTechnical Analysis:")
    print(analysis)
    print(f"\nSaved to {filepath}")
    return analysis


if __name__ == "__main__":
    run()