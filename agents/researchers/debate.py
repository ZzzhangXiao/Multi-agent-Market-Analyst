import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from llm import get_llm
from agents.researchers.bull_researcher import run as bull_run
from agents.researchers.bear_researcher import run as bear_run


def run_debate(
    analyst_reports: str,
    ticker: str,
    rounds: int = 1
) -> dict:
    """
    analyst_reports should be the report bundle for THIS ticker only
    (built per ticker in main.py).
    """
    print(f"\nStarting debate for {ticker} ({rounds} rounds)...")

    # BUGFIX (cost): rebuttals were using get_llm() = llama-3.3-70b, while
    # openings used the 8b model. Debate rounds are meant to run on the
    # cheap model; 70b is reserved for the trader's final decision.
    llm = get_llm(lite=True)
    debate_log = []

    # Round 1 — opening arguments
    bull_case = bull_run(analyst_reports, ticker)
    bear_case = bear_run(analyst_reports, ticker)

    debate_log.append({"round": 1, "bull": bull_case, "bear": bear_case})
    print("  Round 1 complete")

    # Rounds 2+ — rebuttal
    for r in range(2, rounds + 1):
        # BUGFIX: rebuttal prompts never named the ticker, so the model could
        # drift onto whichever asset dominated the reports.
        bull_rebuttal_prompt = f"""
You are the bull researcher arguing the LONG case for {ticker}.
The bear has made the following argument about {ticker}:

BEAR CASE:
{bear_case}

Rebut their strongest points using specific data about {ticker} from the
original reports. Do not discuss any other ticker.
Stay focused. 3-4 sentences maximum per point.

ORIGINAL ANALYST REPORTS:
{analyst_reports}
"""
        bear_rebuttal_prompt = f"""
You are the bear researcher arguing the SHORT/AVOID case for {ticker}.
The bull has made the following argument about {ticker}:

BULL CASE:
{bull_case}

Rebut their strongest points using specific data about {ticker} from the
original reports. Do not discuss any other ticker.
Stay focused. 3-4 sentences maximum per point.

ORIGINAL ANALYST REPORTS:
{analyst_reports}
"""
        # Both prompts are built from the PREVIOUS round's cases before either
        # is overwritten, so each side rebuts the same round.
        bull_case = llm.invoke(bull_rebuttal_prompt).content
        bear_case = llm.invoke(bear_rebuttal_prompt).content

        debate_log.append({"round": r, "bull": bull_case, "bear": bear_case})
        print(f"  Round {r} complete")

    return {
        "ticker":    ticker,
        "debate_log": debate_log,
        "final_bull": bull_case,
        "final_bear": bear_case,
    }
