from agents.data_agent                  import run as run_data_agent
from agents.data_agent                  import process_prices, calculate_features
from agents.analysts.technical_analyst  import run as run_technical_analyst
from agents.analysts.macro_analyst      import run as run_macro_analyst
from agents.analysts.news_analyst       import run as run_news_analyst
from agents.trader                      import run as run_trader
from config import TICKERS
import pandas as pd

TEST_MODE = True

# Focus debate on clean tickers only
DEBATE_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "XOM", "NEE"]

def main():
    print("Starting trading agent pipeline...")
    print("=" * 50)

    # Step 1 — Data
    if TEST_MODE:
        print("TEST MODE — skipping fetch, running processing only\n")
        raw_prices = pd.read_csv("data/prices_raw.csv",
                                  index_col=0, parse_dates=True)
        macro      = pd.read_csv("data/macro.csv",
                                  index_col=0, parse_dates=True)
        prices     = process_prices(raw_prices)
        features   = calculate_features(prices)
        data       = {"prices": prices, "macro": macro, "features": features}
    else:
        data = run_data_agent()

    print(f"  prices:   {data['prices'].shape}")
    print(f"  macro:    {data['macro'].shape}")
    print(f"  features: {data['features'].shape}")

    # Step 2 — Analyst team
    print("\n" + "=" * 50)
    technical_report = run_technical_analyst()

    print("\n" + "=" * 50)
    macro_report = run_macro_analyst()

    print("\n" + "=" * 50)
    news_report = run_news_analyst()

    # Step 3 — Combine analyst reports
    analyst_reports = f"""
TECHNICAL ANALYSIS:
{technical_report}

MACRO ANALYSIS:
{macro_report}

NEWS ANALYSIS:
{news_report}
"""

    # Step 4 — Bull/Bear debate + Trader decision
    print("\n" + "=" * 50)
    decisions = run_trader(analyst_reports, DEBATE_TICKERS)

    print("\n" + "=" * 50)
    print("Pipeline complete. Reports saved to /reports")

if __name__ == "__main__":
    main()