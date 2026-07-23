from agents.data_agent import run as run_data_agent
from agents.data_agent import process_prices, calculate_features
from agents.trader import run as run_trader
from graph import run_analyst_graph
from config import TICKERS
import pandas as pd

TEST_MODE = True
USE_LLM_SUPERVISOR = False   # flip to True once you trust the LLM routing decisions

# ── Cost controls ──
# Debate + trader is the most expensive stage: each ticker runs
# (2 + 2*(rounds-1)) debate calls + 1 trader call. With the default
# rounds=2 and 6 tickers that's already ~30 Groq calls per run.
# Use these knobs to iterate cheaply on the analyst graph before
# spending tokens on the full debate.
RUN_DEBATE = False          # set True to run debate + trader stage at all
DEBATE_ROUNDS = 1            # was hardcoded to 2 in trader.py — now overridable here
DEBATE_TICKERS_LIMIT = None  # e.g. 2 to debate only the first 2 tickers while testing
PER_TICKER_ANALYSTS = False   # False reverts to the old single portfolio-wide pass

# Focus debate on clean tickers only
DEBATE_TICKERS = TICKERS.copy()
if DEBATE_TICKERS_LIMIT:
    DEBATE_TICKERS = DEBATE_TICKERS[:DEBATE_TICKERS_LIMIT]
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

# Step 2 — Analyst team, run once per ticker so each analyst is
    # properly scoped (see graph.py node wrappers). Macro stays
    # portfolio-wide since it's not ticker-specific.
    print("\n" + "=" * 50)
    print("Running analyst team via dynamic supervisor graph (per ticker)...")

    per_ticker_states = {}
    for ticker in TICKERS:
        print(f"\n--- Analyzing {ticker} ---")
        analyst_state = run_analyst_graph(
            ticker=ticker,
            prices=data["prices"],
            macro=data["macro"],
            features=data["features"],
            use_llm_supervisor=USE_LLM_SUPERVISOR,
        )
        per_ticker_states[ticker] = analyst_state

        print(f"  Supervisor routing trace for {ticker}:")
        for line in analyst_state["routing_log"]:
            print(f"    {line}")

    # Step 3 — Combine reports across all tickers. Macro is identical
    # across tickers (portfolio-wide), so only include it once.
    report_sections = []
    macro_included = False
    for ticker, analyst_state in per_ticker_states.items():
        for label, key in [
            ("TECHNICAL ANALYSIS",   "technical_report"),
            ("NEWS ANALYSIS",         "news_report"),
            ("FUNDAMENTALS ANALYSIS", "fundamentals_report"),
            ("SENTIMENT ANALYSIS",    "sentiment_report"),
        ]:
            if analyst_state.get(key):
                report_sections.append(f"{label} [{ticker}]:\n{analyst_state[key]}")

        if not macro_included and analyst_state.get("macro_report"):
            report_sections.append(f"MACRO ANALYSIS (portfolio-wide):\n{analyst_state['macro_report']}")
            macro_included = True

    analyst_reports = "\n\n".join(report_sections)

    # Step 4 — Bull/Bear debate + Trader decision
    print("\n" + "=" * 50)
    if RUN_DEBATE:
        decisions = run_trader(analyst_reports, DEBATE_TICKERS, rounds=DEBATE_ROUNDS)
    else:
        print(f"RUN_DEBATE=False — skipping debate + trader stage "
              f"(would have run {len(DEBATE_TICKERS)} tickers x debate calls).")
        print("Analyst reports above are still saved to /reports individually.")
        decisions = None

    print("\n" + "=" * 50)
    print("Pipeline complete. Reports saved to /reports")


if __name__ == "__main__":
    main()