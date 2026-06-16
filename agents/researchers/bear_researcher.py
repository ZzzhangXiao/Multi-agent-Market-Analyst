import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from llm import get_llm


def run(analyst_reports: str, ticker: str) -> str:
    print(f"  Bear researcher analyzing {ticker}...")
    llm = get_llm(lite=True)


    prompt = f"""
You are a bear-side researcher at a hedge fund.
Your job is to construct the strongest possible SHORT or AVOID case for {ticker}.

You must:
- Find every bearish signal in the data, even weak ones
- Argue why bullish signals are misleading or unsustainable
- Reference specific numbers from the analyst reports
- Be persuasive but intellectually honest
- Conclude with a clear REDUCE or AVOID recommendation

Do not be balanced. You are arguing one side.

ANALYST REPORTS:
{analyst_reports}

Structure your response as:
BEAR CASE FOR {ticker}:
1. [strongest bearish argument with specific numbers]
2. [second argument]
3. [counter to most obvious bull argument]
RECOMMENDATION: [REDUCE/AVOID] | DOWNSIDE TARGET: [price range] | TIMEFRAME: [weeks/months]
"""

    response = llm.invoke(prompt)
    return response.content