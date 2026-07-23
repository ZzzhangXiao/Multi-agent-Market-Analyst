import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datetime import datetime
from llm import get_llm
from config import TICKERS


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return round(macd.iloc[-1], 4), round(signal.iloc[-1], 4), round(hist.iloc[-1], 4)


def compute_bollinger(series: pd.Series, period: int = 20):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    pct_b = (series - lower) / (upper - lower + 1e-8)
    return round(upper.iloc[-1], 2), round(lower.iloc[-1], 2), round(pct_b.iloc[-1], 3)


def compute_close_range_atr_proxy(prices: pd.DataFrame, ticker: str, period: int = 14) -> float:
    """
    NOTE: This is a close-price-only proxy for ATR, not true ATR.
    True ATR requires intraday high/low; we only have daily close prices,
    so this approximates volatility from close-to-close ranges instead.
    """
    high = prices[ticker].rolling(2).max()
    low = prices[ticker].rolling(2).min()
    close = prices[ticker]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return round(tr.rolling(period).mean().iloc[-1], 4)


def ma_relation(price: float, ma: float) -> str:
    if price > ma:
        return "above"
    elif price < ma:
        return "below"
    return "equal to"


def classify_setup(rsi: float, hist: float, trend: str) -> str:
    bullish_macd = hist > 0

    if rsi < 30 and bullish_macd:
        return "BULLISH REVERSAL SETUP (oversold + bullish MACD confirmation)"

    if rsi > 70 and not bullish_macd:
        return "BEARISH REVERSAL SETUP (overbought + bearish MACD confirmation)"

    if rsi < 30 and not bullish_macd:
        return "WEAK BOUNCE CANDIDATE (oversold but MACD not confirming — trend risk remains)"

    if rsi > 70 and bullish_macd:
        return "EXTENDED UPTREND (overbought but momentum still confirming — pullback risk)"

    if trend == "uptrend" and bullish_macd and 40 <= rsi <= 70:
        return "STRONG TREND CONTINUATION"

    if trend == "downtrend" and not bullish_macd and 30 <= rsi <= 60:
        return "STRONG TREND CONTINUATION (bearish)"

    return "NO CLEAR SETUP (mixed signals)"


def build_technical_summary(prices: pd.DataFrame, tickers: list) -> str:
    lines = []

    for ticker in tickers:
        if ticker not in prices.columns:
            lines.append(f"\n{ticker}: NO PRICE DATA AVAILABLE (not in prices.csv) — skipped")
            continue

        series = prices[ticker].dropna()

        if len(series) < 50:
            lines.append(f"\n{ticker}: INSUFFICIENT HISTORY (<50 rows) — skipped")
            continue

        price = round(series.iloc[-1], 2)

        ret_1d = round(series.pct_change(1).iloc[-1] * 100, 2)
        ret_5d = round(series.pct_change(5).iloc[-1] * 100, 2)
        ret_21d = round(series.pct_change(21).iloc[-1] * 100, 2)

        ma20 = round(series.rolling(20).mean().iloc[-1], 2)
        ma50 = round(series.rolling(50).mean().iloc[-1], 2)
        ma200 = round(series.rolling(min(200, len(series))).mean().iloc[-1], 2)

        rsi = compute_rsi(series)
        macd, sig, hist = compute_macd(series)
        bb_upper, bb_lower, pct_b = compute_bollinger(series)
        atr = compute_close_range_atr_proxy(prices, ticker)   # renamed call

        trend = (
            "uptrend" if price > ma50 > ma200
            else "downtrend" if price < ma50 < ma200
            else "mixed"
        )

        rsi_signal = (
            "overbought" if rsi > 70
            else "oversold" if rsi < 30
            else "neutral"
        )

        macd_signal = "bullish crossover" if hist > 0 else "bearish crossover"

        vs_ma20 = ma_relation(price, ma20)
        vs_ma50 = ma_relation(price, ma50)
        vs_ma200 = ma_relation(price, ma200)

        setup_label = classify_setup(rsi, hist, trend)

        lines.append(f"""
{ticker}:
  Price: {price} | 1d: {ret_1d}% | 5d: {ret_5d}% | 21d: {ret_21d}%
  Trend: {trend}
  Price vs MA20: {vs_ma20} ({ma20})
  Price vs MA50: {vs_ma50} ({ma50})
  Price vs MA200: {vs_ma200} ({ma200})
  RSI(14): {rsi} [{rsi_signal}]
  MACD: {macd} | Signal: {sig} | Hist: {hist} [{macd_signal}]
  SETUP LABEL: {setup_label}
  Bollinger: upper={bb_upper} lower={bb_lower} %B={pct_b}
  Close-range volatility proxy(14): {atr}""")

    return "\n".join(lines)


def run(ticker: str = None) -> str:
    print("Technical analyst running...")

    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True
    )

    scope = [ticker] if ticker else TICKERS
    technical_summary = build_technical_summary(prices, scope)
    llm = get_llm()
    prompt = f"""
You are a professional technical analyst at a hedge fund.

Analyze the following technical indicators for each asset.

Rules:
- Be specific and - Reference the exact Close-Range Volatility Proxy value (an ATR-style measure
  computed from close prices only, not true intraday ATR — do not call it
  "true ATR" or compare it to published ATR figures elsewhere).
- Use the precomputed Price vs MA20/MA50/MA200 labels verbatim.
- Do not infer price-vs-MA relationships yourself.
- Use the precomputed SETUP LABEL verbatim.
- A WEAK BOUNCE CANDIDATE is not a STRONG setup.
- Do not call an asset a strong setup unless the SETUP LABEL says STRONG or BULLISH/BEARISH REVERSAL SETUP.
- Do not invent support/resistance levels unless they are directly implied by the moving averages or Bollinger bands.
- Every statement must reference a specific number from the technical data.

For each asset, cover:
1. Trend structure
2. Momentum
3. MACD signal
4. Volatility
5. Setup label and short-term outlook

TECHNICAL DATA:
{technical_summary}
"""

    print("Calling LLM...")
    response = llm.invoke(prompt)
    analysis = response.content

    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs("reports", exist_ok=True)

    filepath = f"reports/technical_analysis_{ticker or 'ALL'}_{today}.md"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Technical Analysis — {today}\n\n")
        f.write("## Deterministic Technical Data\n\n")
        f.write(technical_summary)
        f.write("\n\n## Analysis\n\n")
        f.write(analysis)

    print("\nTechnical Analysis:")
    print(analysis)
    print(f"\nSaved to {filepath}")

    return analysis


if __name__ == "__main__":
    run()