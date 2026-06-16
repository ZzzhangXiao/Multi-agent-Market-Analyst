import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datetime import datetime
from llm import get_llm


def build_macro_summary(macro: pd.DataFrame) -> str:
    lines = []
    for col in macro.columns:
        series = macro[col].dropna()
        if len(series) < 4:
            continue

        current  = round(series.iloc[-1], 3)
        prev_1m  = round(series.iloc[-2], 3)
        prev_3m  = round(series.iloc[-4], 3) if len(series) >= 4 else "n/a"
        prev_6m  = round(series.iloc[-7], 3) if len(series) >= 7 else "n/a"
        chg_1m   = round(current - prev_1m, 3)
        trend    = "rising" if current > prev_1m else "falling"

        lines.append(
            f"{col}: current={current} | 1m_change={chg_1m:+.3f} "
            f"| 3m_ago={prev_3m} | 6m_ago={prev_6m} | trend={trend}"
        )
    return "\n".join(lines)


def classify_regime(macro: pd.DataFrame) -> str:
    regimes = []

    fed = macro["fed_funds_rate"].dropna()
    cpi = macro["cpi"].dropna()
    une = macro["unemployment"].dropna()
    tsy = macro["10y_yield"].dropna()

    # Rate regime
    if len(fed) >= 3:
        if fed.iloc[-1] < fed.iloc[-3]:
            regimes.append("EASING: Fed cutting rates — tailwind for equities and bonds")
        else:
            regimes.append("TIGHTENING: Fed holding or hiking — headwind for duration assets")

    # Inflation regime
    if len(cpi) >= 3:
        cpi_chg = cpi.iloc[-1] - cpi.iloc[-3]
        if cpi_chg > 1:
            regimes.append("INFLATION RISING: headwind for bonds, tailwind for commodities/GLD")
        elif cpi_chg < -1:
            regimes.append("INFLATION FALLING: tailwind for bonds, neutral for equities")
        else:
            regimes.append("INFLATION STABLE: neutral macro backdrop")

    # Labor market
    if len(une) >= 3:
        if une.iloc[-1] < une.iloc[-3]:
            regimes.append("LABOR: tightening — supports consumer spending, mild inflation risk")
        else:
            regimes.append("LABOR: loosening — recession risk rising, defensive posture warranted")

    # Yield curve
    if len(tsy) >= 2:
        if tsy.iloc[-1] > 4.5:
            regimes.append("YIELDS ELEVATED: pressure on growth stocks and real estate")
        elif tsy.iloc[-1] < 3.5:
            regimes.append("YIELDS LOW: supportive for equities and long duration bonds")
        else:
            regimes.append(f"YIELDS NEUTRAL: 10y at {round(tsy.iloc[-1], 2)}%")

    return "\n".join(regimes)


def run() -> str:
    print("Macro analyst running...")

    macro = pd.read_csv(
        "data/macro.csv",
        index_col=0,
        parse_dates=True
    )

    macro_summary  = build_macro_summary(macro)
    regime_summary = classify_regime(macro)

    llm    = get_llm()
    prompt = f"""
You are a senior macro economist at a global macro hedge fund.

Analyze the following macroeconomic data and produce a rigorous macro outlook.

Cover:
1. Rate environment — what is the Fed doing and what does it mean for asset classes
2. Inflation dynamics — trajectory, implications for real returns and asset allocation
3. Labor market — what does unemployment trend signal about economic cycle
4. Yield curve — what does the 10y level signal about growth expectations
5. Macro regime classification — which of these best describes the current environment:
   - Goldilocks (low inflation, growth)
   - Stagflation (high inflation, low growth)
   - Reflation (rising inflation, rising growth)
   - Deflationary bust (falling inflation, falling growth)
6. Asset class implications — given this regime, what should be overweight/underweight

Be specific. Reference exact numbers. No generic statements.

MACRO DATA:
{macro_summary}

REGIME SIGNALS:
{regime_summary}
"""

    print("Calling LLM...")
    response = llm.invoke(prompt)
    analysis = response.content

    today    = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/macro_analysis_{today}.md"
    with open(filepath, "w") as f:
        f.write(f"# Macro Analysis — {today}\n\n")
        f.write(analysis)

    print("\nMacro Analysis:")
    print(analysis)
    print(f"\nSaved to reports/macro_analysis_{today}.md")
    return analysis


if __name__ == "__main__":
    run()