"""
smoke_check.py — verifies the 2026-09-30 bugfixes against YOUR real data.
Makes ZERO Groq calls. Run from the repo root:

    python smoke_check.py           # offline checks only (uses data/*.csv)
    python smoke_check.py --live    # also hits Yahoo once to test the fundamentals cache

Each check prints PASS / FAIL / INFO. Nothing here writes to reports/.
Note: process_prices() rewrites data/prices.csv — that is intended, it
regenerates the processed file without bfill.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import TICKERS

LIVE = "--live" in sys.argv
results = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))


def info(msg):
    print(f"  [INFO] {msg}")


# ───────────────────────────────────────── 1. prices: bfill removed
print("\n1. Price processing (bfill regression)")
from agents.data_agent import process_prices, calculate_features

raw = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
prices = process_prices(raw.copy())

for t in TICKERS:
    if t not in raw.columns:
        info(f"{t}: not in prices_raw.csv")
        continue
    first_real = raw[t].first_valid_index()
    leading_raw = raw.loc[:first_real, t].isna().sum()
    leading_proc = prices.loc[:first_real, t].isna().sum()
    # Leading NaNs must survive processing (bfill would have filled them).
    check(f"{t}: pre-listing rows left as NaN",
          leading_raw == leading_proc,
          f"first real price {first_real.date()}, {leading_raw} leading NaN rows")

# ───────────────────────────────────────── 2. features not truncated
print("\n2. Feature matrix (global dropna)")
features = calculate_features(prices)
# Old code: len(features) ~= rows after the LATEST listing date only.
check("features keep history beyond the latest listing",
      len(features) >= len(prices) - 60,
      f"{len(features)} feature rows vs {len(prices)} price rows")

# ───────────────────────────────────────── 3. technical labels
print("\n3. Technical summary (MACD / MA200 / volatility proxy)")
from agents.analysts.technical_analyst import build_technical_summary

summary = build_technical_summary(prices, TICKERS)
check("no NaN values in technical summary", "nan" not in summary.lower())
check("no bare 'bullish crossover' / 'bearish crossover' labels",
      "[bullish crossover]" not in summary and "[bearish crossover]" not in summary)
for t in TICKERS:
    n = prices[t].dropna().shape[0] if t in prices.columns else 0
    block = summary.split(f"\n{t}:")[-1].split("\n\n")[0]
    ma200_line = next((l.strip() for l in block.splitlines() if "MA200" in l), "missing")
    if n < 200:
        check(f"{t}: MA200 reported UNAVAILABLE ({n} rows)", "UNAVAILABLE" in ma200_line, ma200_line)
    else:
        info(f"{t}: {ma200_line}")

# ───────────────────────────────────────── 4. macro: CPI YoY
print("\n4. Macro regime (CPI level vs YoY)")
from agents.analysts.macro_analyst import classify_regime, _cpi_yoy, _confirmed_trend

macro = pd.read_csv("data/macro.csv", index_col=0, parse_dates=True)
yoy = _cpi_yoy(macro)
old_label = _confirmed_trend(macro["cpi"])      # what the old code used
new = classify_regime(macro)
info(f"CPI YoY last 4 months: {[round(v, 2) for v in yoy.tail(4)]}")
info(f"old (level-based) trend: {old_label}  ->  new inflation label: {new['inflation_label']}")
info(new["regime"])
check("inflation label derived from YoY rate",
      new["inflation_label"] in {"RISING", "FALLING", "STABLE/MIXED"}
      and "YoY" in new["signals"])

# ───────────────────────────────────────── 5. news matching
print("\n5. News DIRECT matching (word boundaries)")
from agents.analysts.news_analyst import _is_direct_ticker_news as direct

cases = [
    ("NEE", "Utilities need more grid capex", False),
    ("NEE", "NEE vs. CEG: which energy stock?", True),
    ("ULG.SI", "Multi-year results for SGX mid-caps", False),
    ("DRAM", "Awards-season drama continues", False),
    ("XOM", "ExxonMobil beats estimates", True),
]
for t, headline, expected in cases:
    check(f"{t} / '{headline}' -> {expected}", direct(t, headline, "") == expected)

# ───────────────────────────────────────── 6. fundamentals cache (optional, hits Yahoo)
print("\n6. Fundamentals cache + dividend yield")
if not LIVE:
    info("skipped — run with --live to test (makes Yahoo calls, no Groq calls)")
else:
    from agents.analysts import fundamentals_analyst as fa

    t0 = time.time()
    s1 = fa.build_fundamentals_summary(TICKERS)
    first = time.time() - t0
    check("cache file written", os.path.exists(fa.CACHE_PATH), fa.CACHE_PATH)

    t0 = time.time()
    s2 = fa.build_fundamentals_summary(TICKERS)
    second = time.time() - t0
    # Second pass should be all cache hits: no jitter sleeps, no network.
    check("second run served from cache", second < 1.0,
          f"first {first:.1f}s, second {second:.2f}s")

    for line in s1.splitlines():
        if "Dividend Yield" in line or "UNAVAILABLE" in line:
            info(line.strip())
    yields = [float(l.split(":")[1].strip().rstrip("%"))
              for l in s1.splitlines()
              if "Dividend Yield:" in l and "n/a" not in l]
    check("all dividend yields plausible (<25%)", all(y < 25 for y in yields), str(yields))

# ─────────────────────────────────────────
print("\n" + "=" * 50)
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
