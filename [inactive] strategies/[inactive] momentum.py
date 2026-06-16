import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_momentum_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-period momentum scoring.
    Combines 1m, 3m, 6m returns into a single score per asset.
    This is the core of most momentum strategies at quant funds.
    """
    scores = pd.DataFrame(index=prices.index)

    for ticker in prices.columns:
        series = prices[ticker].dropna()

        # Raw momentum — returns over multiple periods
        mom_1m = series.pct_change(21)   # 1 month
        mom_3m = series.pct_change(63)   # 3 months
        mom_6m = series.pct_change(126)  # 6 months

        # Weighted composite score — more recent = more weight
        scores[f"{ticker}_mom_1m"] = mom_1m
        scores[f"{ticker}_mom_3m"] = mom_3m
        scores[f"{ticker}_mom_6m"] = mom_6m
        scores[f"{ticker}_score"]  = (
            mom_1m * 0.4 +
            mom_3m * 0.25 +
            mom_6m * 0.15
        )

    return scores.dropna()


def calculate_trend_filters(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Trend filters — only take momentum signals when trend confirms.
    Avoids buying falling knives.
    """
    filters = pd.DataFrame(index=prices.index)

    for ticker in prices.columns:
        series = prices[ticker].dropna()

        ma50  = series.rolling(50).mean()
        ma200 = series.rolling(200).mean()

        # Price above MA50 = short term uptrend
        filters[f"{ticker}_above_ma50"]  = (series > ma50).astype(int)
        # MA50 above MA200 = long term uptrend (golden cross)
        filters[f"{ticker}_golden_cross"] = (ma50 > ma200).astype(int)
        # Trend score: 0, 1, or 2
        filters[f"{ticker}_trend_score"]  = (
            filters[f"{ticker}_above_ma50"] +
            filters[f"{ticker}_golden_cross"]
        )

    return filters.dropna()


def calculate_volatility_adjusted_score(
    scores: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Volatility-adjust the momentum score.
    A 10% return with 5% vol is better than 10% return with 20% vol.
    This is the Sharpe-like adjustment quants use.
    """
    adjusted = pd.DataFrame(index=scores.index)

    for ticker in prices.columns:
        series  = prices[ticker].dropna()
        vol     = series.pct_change().rolling(21).std()
        vol     = vol.reindex(scores.index)

        raw_score = scores[f"{ticker}_score"]
        # Avoid division by zero
        adjusted[f"{ticker}_adj_score"] = raw_score / (vol + 1e-8)

    return adjusted.dropna()


def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — combines everything into final signals.
    Returns a clean signal DataFrame with BUY/HOLD/REDUCE/AVOID.
    """
    # Guard — skip tickers without enough history
    valid_tickers = [
        t for t in prices.columns
        if prices[t].dropna().shape[0] >= 130
    ]

    if not valid_tickers:
        raise ValueError("No tickers have enough history. Need at least 130 rows.")

    prices = prices[valid_tickers].copy()

    scores   = calculate_momentum_scores(prices)
    filters  = calculate_trend_filters(prices)
    adjusted = calculate_volatility_adjusted_score(scores, prices)

    # Align all dataframes on common index
    common_idx = scores.index \
        .intersection(filters.index) \
        .intersection(adjusted.index)

    scores   = scores.loc[common_idx]
    filters  = filters.loc[common_idx]
    adjusted = adjusted.loc[common_idx]

    # Build final signal table — latest row only
    results = []
    for ticker in valid_tickers:
        try:
            adj_score   = adjusted[f"{ticker}_adj_score"].iloc[-1]
            trend_score = filters[f"{ticker}_trend_score"].iloc[-1]
            mom_1m      = scores[f"{ticker}_mom_1m"].iloc[-1]
            mom_3m      = scores[f"{ticker}_mom_3m"].iloc[-1]
            mom_6m      = scores[f"{ticker}_mom_6m"].iloc[-1]

            if adj_score > 0.5 and trend_score == 2:
                signal, confidence = "BUY", "High"
            elif adj_score > 0.2 and trend_score >= 1:
                signal, confidence = "BUY", "Medium"
            elif adj_score < -0.5 or trend_score == 0:
                signal     = "AVOID"
                confidence = "High" if adj_score < -0.5 else "Medium"
            elif adj_score < -0.2:
                signal, confidence = "REDUCE", "Medium"
            else:
                signal, confidence = "HOLD", "Low"

            results.append({
                "ticker":      ticker,
                "signal":      signal,
                "confidence":  confidence,
                "adj_score":   round(adj_score, 4),
                "trend_score": int(trend_score),
                "mom_1m_pct":  round(mom_1m * 100, 2),
                "mom_3m_pct":  round(mom_3m * 100, 2),
                "mom_6m_pct":  round(mom_6m * 100, 2),
            })

        except Exception as e:
            print(f"  Skipping {ticker}: {e}")
            continue

    if not results:
        raise ValueError("All tickers skipped — check data quality.")

    return pd.DataFrame(results).set_index("ticker")
def run():
    print("📊 Momentum strategy running...")
    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True
    )

    signals = generate_signals(prices)

    # Save
    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"data/momentum_signals_{today}.csv"
    signals.to_csv(filepath)

    print("\n🎯 Momentum Signals:")
    print(signals.to_string())
    print(f"\n✅ Saved to {filepath}")
    return signals


if __name__ == "__main__":
    run()