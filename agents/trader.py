import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from llm import get_llm
from agents.researchers.debate import run_debate
from decisions.schema import TradeDecision, bands_for_prompt
from decisions.log import log_decision, new_run_id

# Evaluation horizon the score refers to. Recorded on every decision so the
# scoring script (step 2) knows which forward return to compare against.
HORIZON_DAYS = 21
MAX_ATTEMPTS = 2   # 1 try + 1 retry with the validation error fed back


def _reports_for(analyst_reports, ticker: str) -> str:
    """Accepts {ticker: bundle} (current main.py) or one combined string."""
    if isinstance(analyst_reports, dict):
        return analyst_reports.get(ticker, "")
    return analyst_reports


def _load_prices_fallback():
    """Used only when trader.run() is called without a prices DataFrame."""
    try:
        import pandas as pd
        return pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    except Exception:
        return None


def _write_debate_log(debates: list, today: str) -> str:
    """Writes the transcript from debates that already ran (no re-run)."""
    filepath = f"reports/debate_log_{today}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Debate Log — {today}\n\n")
        for debate in debates:
            f.write(f"## {debate['ticker']}\n\n")
            for round_data in debate["debate_log"]:
                f.write(f"### Round {round_data['round']}\n\n")
                f.write(f"**BULL:**\n{round_data['bull']}\n\n")
                f.write(f"**BEAR:**\n{round_data['bear']}\n\n")
    return filepath


def _build_prompt(ticker: str, ticker_reports: str, debate: dict | None) -> str:
    rules = f"""
Return your decision by calling the TradeDecision tool. Rules:
- score is your view of {ticker}'s return relative to its benchmark over the
  next ~{HORIZON_DAYS} trading days, from -1 (strongly negative) to +1 (strongly positive).
  0 means no edge. Use the full range only when the evidence is strong and consistent.
- signal MUST match score: {bands_for_prompt()}
- key_evidence: 1-4 items, each containing a number copied EXACTLY from the
  material above. Do not round, convert or invent numbers.
- If data is missing or signals conflict, move score toward 0 and lower confidence.
"""
    if debate is None:
        # DEBATE_ROUNDS = 0: decide directly from analyst reports.
        return f"""
You are the head trader at a hedge fund. Make the final trading decision for {ticker}
using only the analyst reports below. There was no bull/bear debate, so set
winning_side to "none".

ANALYST REPORTS:
{ticker_reports}
{rules}"""
    return f"""
You are the head trader at a hedge fund.
You have just observed a structured debate between a bull and bear researcher.
Make the final trading decision for {ticker}.
Review both sides fairly. The best argument wins, not the loudest.

FINAL BULL ARGUMENT:
{debate['final_bull']}

FINAL BEAR ARGUMENT:
{debate['final_bear']}
{rules}"""


def _decide(structured_llm, prompt: str, ticker: str):
    """
    Calls the LLM with tool-calling structured output. On a validation
    failure (e.g. BUY with a negative score), retries once with the error
    message appended. Returns (decision | None, status, attempts, error).
    """
    last_error = None
    current_prompt = prompt
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = structured_llm.invoke(current_prompt)
        except Exception as e:
            # Groq sometimes returns 400 tool_use_failed for malformed calls.
            last_error = f"{type(e).__name__}: {e}"
            status = "llm_error"
        else:
            parsed = result.get("parsed")
            if parsed is not None:
                if parsed.ticker.upper() != ticker.upper():
                    # Don't trust a decision labelled with the wrong asset.
                    last_error = f"ticker mismatch: got {parsed.ticker}, expected {ticker}"
                    status = "parse_failed"
                else:
                    return parsed, "ok", attempt, None
            else:
                last_error = f"{type(result.get('parsing_error')).__name__}: {result.get('parsing_error')}"
                status = "parse_failed"

        print(f"  Attempt {attempt} for {ticker} failed ({status}): {last_error[:200]}")
        current_prompt = (
            prompt
            + f"\n\nYOUR PREVIOUS RESPONSE WAS REJECTED: {last_error[:500]}\n"
              f"Call the TradeDecision tool again with ticker=\"{ticker}\" and fix this."
        )
    return None, status, MAX_ATTEMPTS, last_error


def _render(decision: TradeDecision | None, ticker: str, status: str, error: str | None) -> str:
    """Human-readable markdown, same spirit as the old free-text output."""
    if decision is None:
        return f"ASSET: {ticker}\nNO DECISION ({status}): {error}"
    evidence = "\n".join(f"  - {e}" for e in decision.key_evidence)
    return (
        f"ASSET: {ticker}\n"
        f"FINAL SIGNAL: {decision.signal} (score {decision.score:+.2f})\n"
        f"CONFIDENCE: {decision.confidence}\n"
        f"WINNING ARGUMENT: {decision.winning_side}\n"
        f"KEY EVIDENCE:\n{evidence}\n"
        f"REASONING: {decision.reasoning}\n"
        f"RISK: {decision.key_risk}"
    )


def run(analyst_reports, tickers: list, rounds: int = 2,
        prices=None, run_id: str | None = None) -> list:
    """
    Returns a list of log records (dicts), one per ticker, including failures.
    rounds = 0 skips the debate and decides straight from analyst reports.
    """
    print("\nTrader agent starting...")
    llm = get_llm(lite=False)
    # include_raw=True: validation errors come back as parsing_error instead
    # of raising, so we can log them and retry.
    structured_llm = llm.with_structured_output(TradeDecision, include_raw=True)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    temperature = getattr(llm, "temperature", None)

    prices = prices if prices is not None else _load_prices_fallback()
    run_id = run_id or new_run_id()

    records, rendered, debates = [], [], []

    for ticker in tickers:
        ticker_reports = _reports_for(analyst_reports, ticker)
        if not ticker_reports:
            print(f"  No analyst reports for {ticker} — logging as no_inputs")
            records.append(log_decision(
                run_id=run_id, ticker=ticker, decision=None, status="no_inputs",
                prompt="", inputs_text="", prices=prices, model=model_name,
                temperature=temperature, debate_rounds=rounds,
                horizon_days=HORIZON_DAYS, attempts=0, error="no analyst reports",
            ))
            continue

        debate = None
        if rounds > 0:
            debate = run_debate(ticker_reports, ticker, rounds=rounds)
            debates.append(debate)

        prompt = _build_prompt(ticker, ticker_reports, debate)
        decision, status, attempts, error = _decide(structured_llm, prompt, ticker)

        # inputs_text = everything the decision could legitimately draw on;
        # the grounding check looks for cited numbers in here.
        inputs_text = ticker_reports if debate is None else (
            ticker_reports + "\n\n" + debate["final_bull"] + "\n\n" + debate["final_bear"])

        record = log_decision(
            run_id=run_id, ticker=ticker, decision=decision, status=status,
            prompt=prompt, inputs_text=inputs_text, prices=prices,
            model=model_name, temperature=temperature, debate_rounds=rounds,
            horizon_days=HORIZON_DAYS, attempts=attempts, error=error,
        )
        records.append(record)

        text = _render(decision, ticker, status, error)
        rendered.append(text)
        print(f"\n  Decision for {ticker}:\n{text}")
        if record.get("evidence_grounded_frac") is not None and record["evidence_grounded_frac"] < 1.0:
            print(f"  WARNING: only {record['evidence_grounded_frac']:.0%} of cited numbers "
                  f"appear in the inputs for {ticker}")

    divider = "\n\n" + "=" * 40 + "\n\n"
    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs("reports", exist_ok=True)
    filepath = f"reports/trader_decisions_{today}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Trader Decisions — {today} (run {run_id})\n\n")
        f.write(divider.join(rendered))
    print(f"\nSaved to {filepath}")
    if debates:
        print(f"Debate transcript saved to {_write_debate_log(debates, today)}")
    ok = sum(r["status"] == "ok" for r in records)
    print(f"Logged {len(records)} decisions ({ok} ok) to logs/decisions.jsonl")
    return records
