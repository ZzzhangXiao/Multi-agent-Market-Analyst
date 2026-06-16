import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from strategies.momentum import generate_signals


def run_backtest(
    prices: pd.DataFrame,
    lookback_days: int = 130,
    rebalance_freq: int = 21,
    top_n: int = 3,
    long_only: bool = True,
) -> dict:
    """
    Walk-forward backtest of momentum strategy.

    Logic:
    - Every rebalance_freq days, generate momentum signals
    - Go long top_n assets by adj_score
    - Equal weight across held positions
    - Track daily portfolio returns

    Parameters:
        lookback_days   : minimum history before first signal
        rebalance_freq  : how often to rebalance in trading days
        top_n           : number of assets to hold at once
        long_only       : if True, only take BUY signals
    """
    print("Running backtest...")
    print(f"  Rebalance every {rebalance_freq} days | Hold top {top_n} assets")

    # Filter to valid tickers only
    valid_tickers = [
        t for t in prices.columns
        if prices[t].dropna().shape[0] >= 256
    ]

    if not valid_tickers:
        raise ValueError("No tickers passed the data quality filter.")

    prices = prices[valid_tickers].copy()
    print(f"  Valid tickers: {valid_tickers}")

    daily_returns   = prices.pct_change()
    portfolio_rets  = []
    holdings_log    = []
    current_weights = pd.Series(0.0, index=valid_tickers)

    for i in range(lookback_days, len(prices)):
        date        = prices.index[i]
        price_slice = prices.iloc[:i]

        # Rebalance on schedule
        if (i - lookback_days) % rebalance_freq == 0:
            signals = generate_signals(price_slice)

            if long_only:
                buy_signals = signals[signals["signal"].isin(["BUY"])]
            else:
                buy_signals = signals[signals["signal"].isin(["BUY", "HOLD"])]

            # Rank by adj_score, take top N
            top_assets = (
                buy_signals
                .sort_values("adj_score", ascending=False)
                .head(top_n)
                .index.tolist()
            )

            # Equal weight
            current_weights = pd.Series(0.0, index=valid_tickers)
            if top_assets:
                weight = 1.0 / len(top_assets)
                for asset in top_assets:
                    current_weights[asset] = weight

            holdings_log.append({
                "date":     date,
                "holdings": top_assets,
                "weights":  current_weights[top_assets].to_dict()
            })

        # Daily portfolio return
        todays_returns = daily_returns.loc[date]
        port_ret       = (current_weights * todays_returns).sum()
        portfolio_rets.append({"date": date, "portfolio_return": port_ret})

    # Build equity curve
    results_df           = pd.DataFrame(portfolio_rets).set_index("date")
    results_df["equity"] = (1 + results_df["portfolio_return"]).cumprod()

    # Strategy metrics
    metrics = calculate_metrics(results_df["portfolio_return"])

    # Save equity curve
    today    = datetime.today().strftime("%Y-%m-%d")
    results_df.to_csv(f"data/backtest_equity_{today}.csv")

    print("\nBacktest Results:")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")

    # Benchmark comparison
    print("\n" + "-" * 40)
    benchmark = calculate_benchmark(prices, ticker="SPY")

    print("\nStrategy vs Benchmark:")
    print(f"  {'Metric':<25} {'Strategy':>10} {'SPY B&H':>10}")
    print(f"  {'-' * 47}")
    for k in metrics:
        s_val = metrics[k]
        b_val = benchmark.get(k, "n/a")
        print(f"  {k:<25} {str(s_val):>10} {str(b_val):>10}")

    return {
        "equity_curve": results_df,
        "metrics":      metrics,
        "benchmark":    benchmark,
        "holdings_log": holdings_log,
    }


def calculate_metrics(returns: pd.Series) -> dict:
    """
    Core quant performance metrics.
    These are what every hedge fund looks at.
    """
    returns      = returns.dropna()
    trading_days = 252

    # Annualised return
    total_return = (1 + returns).prod() - 1
    n_years      = len(returns) / trading_days
    ann_return   = (1 + total_return) ** (1 / n_years) - 1

    # Annualised volatility
    ann_vol = returns.std() * np.sqrt(trading_days)

    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    equity       = (1 + returns).cumprod()
    rolling_max  = equity.cummax()
    drawdown     = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    # Sortino ratio
    downside_vol = returns[returns < 0].std() * np.sqrt(trading_days)
    sortino      = ann_return / downside_vol if downside_vol > 0 else 0

    return {
        "total_return_pct":   round(total_return * 100, 2),
        "ann_return_pct":     round(ann_return * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio":       round(sharpe, 3),
        "sortino_ratio":      round(sortino, 3),
        "calmar_ratio":       round(calmar, 3),
        "max_drawdown_pct":   round(max_drawdown * 100, 2),
        "win_rate_pct":       round(win_rate * 100, 2),
    }


def calculate_benchmark(prices: pd.DataFrame, ticker: str = "SPY") -> dict:
    """
    SPY buy and hold benchmark.
    Every backtest result must be compared against this.
    If your strategy cannot beat SPY, just buy SPY.
    """
    if ticker not in prices.columns:
        print(f"Benchmark ticker {ticker} not found in prices.")
        return {}

    series  = prices[ticker].pct_change().dropna()
    metrics = calculate_metrics(series)

    print(f"\nBenchmark ({ticker} Buy and Hold):")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")

    return metrics


def run():
    from config import STRATEGY_TICKERS

    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True
    )

    # Only use approved clean tickers
    prices = prices[
        [t for t in STRATEGY_TICKERS if t in prices.columns]
    ]

    print(f"Running backtest on: {prices.columns.tolist()}")
    results = run_backtest(prices)
    return results


if __name__ == "__main__":
    run()