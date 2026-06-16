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
    print(f"\nStarting debate for {ticker} ({rounds} rounds)...")

    llm = get_llm()
    debate_log = []

    # Round 1 — opening arguments
    bull_case = bull_run(analyst_reports, ticker)
    bear_case = bear_run(analyst_reports, ticker)

    debate_log.append({"round": 1, "bull": bull_case, "bear": bear_case})
    print(f"  Round 1 complete")

    # Rounds 2+ — rebuttal
    for r in range(2, rounds + 1):
        bull_rebuttal_prompt = f"""
You are the bull researcher. The bear has made the following argument:

BEAR CASE:
{bear_case}

Rebut their strongest points using specific data from the original reports.
Stay focused. 3-4 sentences maximum per point.

ORIGINAL ANALYST REPORTS:
{analyst_reports}
"""
        bear_rebuttal_prompt = f"""
You are the bear researcher. The bull has made the following argument:

BULL CASE:
{bull_case}

Rebut their strongest points using specific data from the original reports.
Stay focused. 3-4 sentences maximum per point.

ORIGINAL ANALYST REPORTS:
{analyst_reports}
"""
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