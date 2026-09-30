from agents.data_agent import run as run_data_agent
from agents.data_agent import process_prices, calculate_features
from agents.analysts.macro_analyst import run as run_macro
from agents.trader import run as run_trader
from graph import run_analyst_graph
from config import TICKERS
import pandas as pd

TEST_MODE = False
USE_LLM_SUPERVISOR = False   # flip to True once you trust the LLM routing decisions

# ── Cost controls ──
# Debate + trader is the most expensive stage: each ticker runs
# (2 + 2*(rounds-1)) debate calls + 1 trader call.
# Use these knobs to iterate cheaply on the analyst graph before
# spending tokens on the full debate.
RUN_TRADER = True            # set True to run the trader stage (and log decisions)
DEBATE_ROUNDS = 0            # 0 = trader decides straight from analyst reports (no debate,
                             #     1 trader call per ticker — cheapest way to log decisions)
DEBATE_TICKERS_LIMIT = 2     # e.g. 2 to debate only the first 2 tickers while testing
# (PER_TICKER_ANALYSTS removed: it was never read — the graph is always per-ticker now.)

DEBATE_TICKERS = TICKERS.copy()
if DEBATE_TICKERS_LIMIT:
    DEBATE_TICKERS = DEBATE_TICKERS[:DEBATE_TICKERS_LIMIT]

REPORT_KEYS = [
    ("TECHNICAL ANALYSIS",    "technical_report"),
    ("NEWS ANALYSIS",         "news_report"),
    ("FUNDAMENTALS ANALYSIS", "fundamentals_report"),
    ("SENTIMENT ANALYSIS",    "sentiment_report"),
]


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
        data       = {"prices": prices, "prices_raw": raw_prices,
                      "macro": macro, "features": features}
    else:
        data = run_data_agent()

    print(f"  prices:   {data['prices'].shape}")
    print(f"  macro:    {data['macro'].shape}")
    print(f"  features: {data['features'].shape}")

    # Step 2a — Macro ONCE (portfolio-wide). BUGFIX: previously the graph
    # re-ran macro inside every per-ticker invocation.
    print("\n" + "=" * 50)
    macro_report = run_macro()

    # Step 2b — Per-ticker analyst graph (macro seeded, so it is skipped)
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
            macro_report=macro_report,
        )
        per_ticker_states[ticker] = analyst_state

        print(f"  Supervisor routing trace for {ticker}:")
        for line in analyst_state["routing_log"]:
            print(f"    {line}")

    # Step 3 — Build a report bundle PER TICKER (that ticker's four reports
    # + the shared macro report). BUGFIX: previously every ticker's debate
    # received all tickers' reports concatenated — ~5x the tokens per call
    # (likely over Groq's per-minute token limit for the 8b model) and a
    # cross-contamination risk where the bull for NEE cites XOM's numbers.
    reports_by_ticker = {}
    for ticker, analyst_state in per_ticker_states.items():
        sections = [
            f"{label} [{ticker}]:\n{analyst_state[key]}"
            for label, key in REPORT_KEYS
            if analyst_state.get(key)
        ]
        if macro_report:
            sections.append(f"MACRO ANALYSIS (portfolio-wide):\n{macro_report}")
        reports_by_ticker[ticker] = "\n\n".join(sections)

    # Step 4 — Bull/Bear debate + Trader decision
    print("\n" + "=" * 50)
    if RUN_TRADER:
        # prices passed through so each logged decision records the exact
        # close and date it saw (the anchor for forward-return evaluation).
        # BUGFIX: pass the UNFILLED prices. The ffilled frame copies the last
        # close into rows a market has not closed yet, so a run before the US
        # open logged data_as_of = today with yesterday's price, and stale_data
        # could never trigger for a ticker that stopped updating.
        decisions = run_trader(reports_by_ticker, DEBATE_TICKERS,
                               rounds=DEBATE_ROUNDS, prices=data["prices_raw"])
    else:
        print(f"RUN_TRADER=False — skipping debate + trader stage "
              f"(would have run {len(DEBATE_TICKERS)} tickers).")
        print("Analyst reports above are still saved to /reports individually.")
        decisions = None

    print("\n" + "=" * 50)
    print("Pipeline complete. Reports saved to /reports")
    return decisions


if __name__ == "__main__":
    main()
