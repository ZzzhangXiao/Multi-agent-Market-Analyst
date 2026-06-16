import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from llm import get_llm

def run(market_brief: str):
    print("📡 Signal agent starting...")

    prices   = pd.read_csv("data/prices.csv",  index_col=0, parse_dates=True)
    features = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)

    llm = get_llm()

    prompt = f"""
You are a systematic trading signal generator.

Based on the market brief and data below, generate a signal for each asset.

For each asset output exactly this format:
ASSET: [ticker]
SIGNAL: [BUY / HOLD / REDUCE / AVOID]
CONFIDENCE: [High / Medium / Low]
REASON: [one sentence]
RISK: [one sentence]

Assets to cover: "ULG.SI", "XOM", "P52.SI", "NEE", 
           "SPY", "QQQ", "TLT", "GLD", "0P00006G05","DRAM"

MARKET BRIEF:
{market_brief}

LATEST PRICES:
{prices.tail(5).to_string()}

LATEST FEATURES:
{features.tail(3).to_string()}
"""

    print("💬 Calling LLM...")
    response = llm.invoke(prompt)
    signals  = response.content

    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/signals_{today}.md"
    with open(filepath, "w") as f:
        f.write(f"# Trading Signals — {today}\n\n")
        f.write(signals)

    print("\n📡 Signals:")
    print(signals)
    print(f"\n✅ Saved to {filepath}")
    return signals

if __name__ == "__main__":
    # For standalone testing, load today's brief
    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/market_brief_{today}.md"
    with open(filepath, "r") as f:
        brief = f.read()
    run(brief)