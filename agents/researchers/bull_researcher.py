import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from llm import get_llm


def run(analyst_reports: str, ticker: str) -> str:
    print(f"  Bull researcher analyzing {ticker}...")
    llm = get_llm(lite=True)


    prompt = f"""
You are a bull-side researcher at a hedge fund.
Your job is to construct the strongest possible LONG case for {ticker}.

You must:
- Find every bullish signal in the data, even weak ones
- Argue why bearish signals are temporary or overstated
- Reference specific numbers from the analyst reports
- Be persuasive but intellectually honest
- Conclude with a clear BUY or HOLD recommendation and target price range

Do not be balanced. You are arguing one side.

ANALYST REPORTS:
{analyst_reports}

Structure your response as:
BULL CASE FOR {ticker}:
1. [strongest bullish argument with specific numbers]
2. [second argument]
3. [counter to most obvious bear argument]
RECOMMENDATION: [BUY/HOLD] | TARGET: [price range] | TIMEFRAME: [weeks/months]
"""

    response = llm.invoke(prompt)
    return response.content