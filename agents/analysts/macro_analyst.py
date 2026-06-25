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

        current = round(series.iloc[-1], 3)
        prev_1m = round(series.iloc[-2], 3)
        prev_3m = round(series.iloc[-4], 3)
        prev_6m = round(series.iloc[-7], 3) if len(series) >= 7 else "n/a"

        chg_1m = round(current - prev_1m, 3)
        chg_3m = round(current - prev_3m, 3)
        chg_6m = round(current - prev_6m, 3) if isinstance(prev_6m, float) else "n/a"

        trend = "rising" if chg_1m > 0 else "falling" if chg_1m < 0 else "flat"

        lines.append(
            f"{col}: current={current} | 1m_change={chg_1m:+.3f} "
            f"| 3m_change={chg_3m:+.3f} | 6m_change={chg_6m} "
            f"| trend={trend}"
        )

    return "\n".join(lines)


def _confirmed_trend(series: pd.Series, higher_is_rising: bool = True) -> str:
    """
    Uses 2-of-3 recent monthly moves to reduce one-month noise.
    """
    series = series.dropna()

    if len(series) < 4:
        return "insufficient data"

    moves = [
        series.iloc[-1] - series.iloc[-2],
        series.iloc[-2] - series.iloc[-3],
        series.iloc[-3] - series.iloc[-4],
    ]

    positive = sum(m > 0 for m in moves)
    negative = sum(m < 0 for m in moves)

    if positive >= 2:
        return "rising" if higher_is_rising else "falling"
    if negative >= 2:
        return "falling" if higher_is_rising else "rising"

    return "mixed"


def classify_regime(macro: pd.DataFrame) -> dict:
    fed = macro["fed_funds_rate"].dropna()
    cpi = macro["cpi"].dropna()
    une = macro["unemployment"].dropna()
    tsy = macro["10y_yield"].dropna()

    signals = []

    # Rate signal
    fed_trend = _confirmed_trend(fed)
    if fed_trend == "falling":
        rate_label = "EASING"
        signals.append("EASING: Fed rates are falling — tailwind for equities and bonds")
    elif fed_trend == "rising":
        rate_label = "TIGHTENING"
        signals.append("TIGHTENING: Fed rates are rising — headwind for duration assets")
    else:
        rate_label = "MIXED"
        signals.append("RATES MIXED: Fed signal is unclear over recent months")

    # Inflation signal
    cpi_trend = _confirmed_trend(cpi)
    if cpi_trend == "rising":
        inflation_label = "RISING"
        signals.append("INFLATION RISING: headwind for bonds, tailwind for commodities/GLD")
    elif cpi_trend == "falling":
        inflation_label = "FALLING"
        signals.append("INFLATION FALLING: tailwind for bonds, easing pressure on equities")
    else:
        inflation_label = "STABLE/MIXED"
        signals.append("INFLATION MIXED: no clean inflation trend")

    # Labor signal
    unemployment_trend = _confirmed_trend(une)

    if unemployment_trend == "falling":
        labor_label = "TIGHTENING"
        growth_label = "IMPROVING"
        signals.append("LABOR: tightening — unemployment is falling, supporting consumer demand")
    elif unemployment_trend == "rising":
        labor_label = "LOOSENING"
        growth_label = "WEAKENING"
        signals.append("LABOR: loosening — unemployment is rising, recession risk increasing")
    else:
        labor_label = "MIXED"
        growth_label = "MIXED"
        signals.append("LABOR MIXED: unemployment signal is not decisive")

    # Yield signal
    current_10y = tsy.iloc[-1] if len(tsy) else None
    tsy_trend = _confirmed_trend(tsy)

    if current_10y is not None:
        if current_10y > 4.5:
            yield_label = "ELEVATED"
            signals.append(f"YIELDS ELEVATED: 10y at {round(current_10y, 2)}%, pressure on growth stocks and real estate")
        elif current_10y < 3.5:
            yield_label = "LOW"
            signals.append(f"YIELDS LOW: 10y at {round(current_10y, 2)}%, supportive for duration assets")
        else:
            yield_label = "NEUTRAL"
            signals.append(f"YIELDS NEUTRAL: 10y at {round(current_10y, 2)}%")
    else:
        yield_label = "UNKNOWN"
        signals.append("YIELDS UNKNOWN: insufficient 10y yield data")

    # Regime classification
    support = {
        "Goldilocks": 0,
        "Stagflation": 0,
        "Reflation": 0,
        "Deflationary bust": 0,
    }

    if inflation_label == "RISING" and growth_label == "IMPROVING":
        support["Reflation"] += 2
    if inflation_label == "RISING" and growth_label == "WEAKENING":
        support["Stagflation"] += 2
    if inflation_label == "FALLING" and growth_label == "IMPROVING":
        support["Goldilocks"] += 2
    if inflation_label == "FALLING" and growth_label == "WEAKENING":
        support["Deflationary bust"] += 2

    if rate_label == "EASING":
        support["Goldilocks"] += 1
        support["Deflationary bust"] += 1
    elif rate_label == "TIGHTENING":
        support["Stagflation"] += 1

    if tsy_trend == "rising":
        support["Reflation"] += 1
        support["Stagflation"] += 1
    elif tsy_trend == "falling":
        support["Goldilocks"] += 1
        support["Deflationary bust"] += 1

    top_regime = max(support, key=support.get)
    top_score = support[top_regime]
    total_score = sum(support.values())

    confidence = "low"
    if total_score > 0:
        confidence_ratio = top_score / total_score
        if confidence_ratio >= 0.6:
            confidence = "high"
        elif confidence_ratio >= 0.45:
            confidence = "medium"

    regime_sentence = (
        f"AUTHORITATIVE REGIME CLASSIFICATION: {top_regime} "
        f"with {confidence} confidence. "
        f"Support scores: {support}."
    )

    return {
        "signals": "\n".join(signals),
        "regime": regime_sentence,
        "rate_label": rate_label,
        "inflation_label": inflation_label,
        "labor_label": labor_label,
        "growth_label": growth_label,
        "yield_label": yield_label,
        "confidence": confidence,
    }


def run() -> str:
    print("Macro analyst running...")

    macro = pd.read_csv(
        "data/macro.csv",
        index_col=0,
        parse_dates=True
    )

    macro_summary = build_macro_summary(macro)
    regime_info = classify_regime(macro)

    deterministic_header = f"""
## Deterministic Macro Signals

{regime_info["signals"]}

{regime_info["regime"]}

Key labels:
- Rates: {regime_info["rate_label"]}
- Inflation: {regime_info["inflation_label"]}
- Labor: {regime_info["labor_label"]}
- Growth proxy: {regime_info["growth_label"]}
- Yields: {regime_info["yield_label"]}
"""

    llm = get_llm()
    prompt = f"""
You are a senior macro economist at a global macro hedge fund.

The deterministic macro signals below are authoritative.
You must not contradict them.
Do not reinterpret tightening as loosening, or loosening as tightening.
Use the exact labels provided.

Your job is only to explain implications for asset classes.

Required output:
1. Rate environment
2. Inflation dynamics
3. Labor market
4. Yield environment
5. Macro regime implications
6. Asset class implications

Rules:
- Reference exact numbers from MACRO DATA.
- Reference the exact labels from DETERMINISTIC SIGNALS.
- Do not invent data.
- If the regime confidence is low or medium, explicitly say the regime is qualified rather than certain.

MACRO DATA:
{macro_summary}

DETERMINISTIC SIGNALS:
{deterministic_header}
"""

    print("Calling LLM...")
    response = llm.invoke(prompt)
    llm_analysis = response.content

    analysis = deterministic_header + "\n\n## Macro Interpretation\n\n" + llm_analysis

    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs("reports", exist_ok=True)

    filepath = f"reports/macro_analysis_{today}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Macro Analysis — {today}\n\n")
        f.write(analysis)

    print("\nMacro Analysis:")
    print(analysis)
    print(f"\nSaved to {filepath}")

    return analysis


if __name__ == "__main__":
    run()