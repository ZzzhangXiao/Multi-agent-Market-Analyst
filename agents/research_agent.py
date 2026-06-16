import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from llm import get_llm

def load_latest_data():
    prices   = pd.read_csv("data/prices.csv",   index_col=0, parse_dates=True)
    macro    = pd.read_csv("data/macro.csv",     index_col=0, parse_dates=True)
    features = pd.read_csv("data/features.csv",  index_col=0, parse_dates=True)
    return prices, macro, features

def summarize_macro(macro):
    lines = []
    for col in macro.columns:
        series = macro[col].dropna()
        if len(series) < 2:
            continue
        current  = round(series.iloc[-1], 2)
        m3_ago   = round(series.iloc[-13], 2) if len(series) > 13 else "n/a"
        m6_ago   = round(series.iloc[-26], 2) if len(series) > 26 else "n/a"
        trend    = "rising" if series.iloc[-1] > series.iloc[-3] else "falling"
        lines.append(f"{col}: current={current}, 3m_ago={m3_ago}, 6m_ago={m6_ago}, trend={trend}")
    return "\n".join(lines)

def summarize_prices(prices, features):
    lines = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 20:
            continue
        current = round(series.iloc[-1], 2)
        ret_1m  = round((series.iloc[-1] / series.iloc[max(-21, -len(series))] - 1) * 100, 2)
        ret_3m  = round((series.iloc[-1] / series.iloc[max(-63, -len(series))] - 1) * 100, 2)
        ma20    = round(series.rolling(20).mean().iloc[-1], 2)
        ma50    = round(series.rolling(50).mean().iloc[-1], 2) if len(series) >= 50 else "n/a"
        vol     = round(series.pct_change().rolling(20).std().iloc[-1] * 100, 2)
        above_ma50 = "yes" if ma50 != "n/a" and current > ma50 else "n/a"
        lines.append(
            f"{ticker}: price={current}, 1m_return={ret_1m}%, 3m_return={ret_3m}%, "
            f"ma20={ma20}, ma50={ma50}, above_ma50={above_ma50}, vol20={vol}%"
        )
    return "\n".join(lines)

def run():
    print("🔍 Research agent starting...")
    prices, macro, features = load_latest_data()

    macro_summary  = summarize_macro(macro)
    prices_summary = summarize_prices(prices, features)

    llm = get_llm()

    prompt = f"""
You are a professional market research analyst.
Based on the following data, write a concise market brief.

Cover:
1. Market conditions — price trends, momentum, which assets are strong/weak
2. Macro environment — direction of rates, inflation, unemployment and what it means
3. Key risks
4. Overall sentiment: Bullish / Neutral / Bearish and why

Be specific. Reference actual numbers. Under 300 words.

PRICE DATA:
{prices_summary}

MACRO DATA:
{macro_summary}
"""

    print("💬 Calling LLM...")
    response = llm.invoke(prompt)
    brief = response.content

    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/market_brief_{today}.md"
    with open(filepath, "w") as f:
        f.write(f"# Market Brief — {today}\n\n")
        f.write(brief)

    print("\n📝 Market Brief:")
    print(brief)
    print(f"\n✅ Saved to {filepath}")
    return brief

if __name__ == "__main__":
    run()