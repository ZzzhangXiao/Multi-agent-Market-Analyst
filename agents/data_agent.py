import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
from config import FRED_API_KEY, TICKERS, FRED_SERIES

# ─────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────

def fetch_market_data() -> pd.DataFrame:
    print("📈 Fetching market data...")
    raw = yf.download(TICKERS, period="2y", interval="1d", progress=False)

    if raw.empty:
        raise ConnectionError("❌ Yahoo Finance rate limited. Set TEST_MODE=True in main.py.")

    prices = raw["Close"]
    prices.to_csv("data/prices_raw.csv")
    print(f"✅ Raw prices saved — shape {prices.shape}")
    return prices


def fetch_macro_data() -> pd.DataFrame:
    print("\n🏦 Fetching macro data...")
    fred = Fred(api_key=FRED_API_KEY)
    macro = {}
    for name, series_id in FRED_SERIES.items():
        macro[name] = fred.get_series(series_id, observation_start="2020-01-01")
    df = pd.DataFrame(macro)
    df.to_csv("data/macro.csv")
    print(f"✅ Macro saved — shape {df.shape}")
    return df


# ─────────────────────────────────────────
# PROCESS
# ─────────────────────────────────────────

def process_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Central data processing function.
    Add all cleaning, filling, and feature steps here as project grows.
    """
    print("\n⚙️  Processing prices...")

    # ── Step 1: NaN report before filling ──
    nan_before = prices.isnull().sum()

    # ── Step 2: Forward fill then back fill ──
    prices = prices.ffill()
    prices = prices.bfill()

    # ── Step 3: NaN report after filling ──
    nan_after = prices.isnull().sum()

    quality = pd.DataFrame({
        "total_rows":    len(prices),
        "nan_before":    nan_before,
        "filled":        nan_before - nan_after,
        "nan_remaining": nan_after,
        "coverage_pct":  ((len(prices) - nan_after) / len(prices) * 100).round(1)
    })
    print("\n📋 Data Quality Report:")
    print(quality.to_string())

    # ── Step 4: Flag low quality tickers ──
    bad_tickers = quality[quality["coverage_pct"] < 70].index.tolist()
    if bad_tickers:
        print(f"\n  Low coverage tickers (< 70%): {bad_tickers}")
        print("   These will be skipped in strategy calculations.")

    # ── Step 5: Save processed prices ──
    prices.to_csv("data/prices.csv")
    print(f"\n✅ Processed prices saved — shape {prices.shape}")
    return prices


def calculate_features(prices: pd.DataFrame) -> pd.DataFrame:
    print("\n Calculating features...")
    features = pd.DataFrame(index=prices.index)

    for ticker in prices.columns:
        features[f"{ticker}_return_1d"] = prices[ticker].pct_change(1)
        features[f"{ticker}_return_5d"] = prices[ticker].pct_change(5)
        features[f"{ticker}_ma20"]      = prices[ticker].rolling(20).mean()
        features[f"{ticker}_ma50"]      = prices[ticker].rolling(50).mean()
        features[f"{ticker}_vol20"]     = prices[ticker].pct_change().rolling(20).std()

    features.dropna(inplace=True)
    features.to_csv("data/features.csv")
    print(f" Features saved — shape {features.shape}")
    return features


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

def run() -> dict:
    print(" Data agent starting...\n")

    # Fetch
    raw_prices = fetch_market_data()
    macro      = fetch_macro_data()

    # Process
    prices   = process_prices(raw_prices)
    features = calculate_features(prices)

    print("\n Data agent complete.")
    return {
        "prices":   prices,
        "macro":    macro,
        "features": features
    }


if __name__ == "__main__":
    run()