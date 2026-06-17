import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from llm import get_llm
from agents.researchers.debate import run_debate


def run(analyst_reports: str, tickers: list, rounds: int = 2) -> str:
    print("\nTrader agent starting...")
    llm = get_llm(lite=False)

    all_decisions = []

    for ticker in tickers:
        debate = run_debate(analyst_reports, ticker, rounds=rounds)

        prompt = f"""
You are the head trader at a hedge fund.
You have just observed a structured debate between a bull and bear researcher.

Your job is to make the final trading decision for {ticker}.

Review both sides fairly. The best argument wins, not the loudest.
Your decision must be data-driven and reference specific numbers.

FINAL BULL ARGUMENT:
{debate['final_bull']}

FINAL BEAR ARGUMENT:
{debate['final_bear']}

Output exactly:
ASSET: {ticker}
FINAL SIGNAL: [BUY / HOLD / REDUCE / AVOID]
CONFIDENCE: [High / Medium / Low]
WINNING ARGUMENT: [bull/bear] and why in one sentence
REASONING: [2-3 sentences referencing specific numbers]
RISK: [biggest risk to this call]
"""
        response = llm.invoke(prompt)
        decision = response.content
        all_decisions.append(decision)
        print(f"\n  Decision for {ticker}:\n{decision}")

    full_output = "\n\n" + "=" * 40 + "\n\n".join(all_decisions)

    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/trader_decisions_{today}.md"
    with open(filepath, "w") as f:
        f.write(f"# Trader Decisions — {today}\n\n")
        f.write(full_output)

    print(f"\nSaved to {filepath}")
    return full_output


def run_debate_log(analyst_reports: str, tickers: list, rounds: int = 2) -> None:
    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/debate_log_{today}.md"
    with open(filepath, "w") as f:
        f.write(f"# Debate Log — {today}\n\n")
        for ticker in tickers:
            debate = run_debate(analyst_reports, ticker, rounds=rounds)
            f.write(f"## {ticker}\n\n")
            for round_data in debate["debate_log"]:
                f.write(f"### Round {round_data['round']}\n\n")
                f.write(f"**BULL:**\n{round_data['bull']}\n\n")
                f.write(f"**BEAR:**\n{round_data['bear']}\n\n")