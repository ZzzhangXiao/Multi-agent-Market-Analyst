import sys, os, types, json, time
sys.path.insert(0, ".")
os.makedirs("_testrun", exist_ok=True); os.chdir("_testrun"); sys.path.insert(0, "..")
import numpy as np, pandas as pd

# ---- stub LLM: record every call (model + prompt) ----
CALLS = []
class FakeLLM:
    def __init__(self, lite): self.lite = lite
    def invoke(self, prompt):
        CALLS.append(("8b" if self.lite else "70b", prompt))
        return types.SimpleNamespace(content=f"stub-{'8b' if self.lite else '70b'} -> ok")
    def with_structured_output(self, schema, include_raw=False):
        # Trader now uses structured output; return a valid decision per ticker.
        outer = self
        class _S:
            def invoke(self, prompt):
                CALLS.append(("70b", prompt))
                t = prompt.split("decision for ")[1].split()[0].rstrip(".")
                d = schema(ticker=t, signal="HOLD", score=0.0, confidence="Low", winning_side="bull",
                           key_evidence=["stub 12.5"], reasoning="r", key_risk="k")
                return {"raw": None, "parsed": d, "parsing_error": None}
        return _S()
llm_mod = types.ModuleType("llm"); llm_mod.get_llm = lambda lite=False: FakeLLM(lite)
sys.modules["llm"] = llm_mod

# ---- synthetic data: DRAM lists late (only 120 rows) ----
os.makedirs("data", exist_ok=True)
idx = pd.bdate_range("2024-10-01", periods=400)
rng = np.random.default_rng(0)
raw = pd.DataFrame({t: 100*np.exp(np.cumsum(rng.normal(0,0.01,400))) for t in ["XOM","NEE","ULG.SI","P52.SI","DRAM"]}, index=idx)
raw.loc[raw.index[:280], "DRAM"] = np.nan
raw.to_csv("data/prices_raw.csv")
midx = pd.date_range("2024-01-01", periods=24, freq="MS")
yoy = np.linspace(3.5, 2.0, 24)
macro = pd.DataFrame({"fed_funds_rate": np.linspace(5.3,4.0,24), "cpi": 300*np.cumprod(1+yoy/1200),
                      "unemployment": np.linspace(3.9,4.3,24), "10y_yield": np.linspace(4.6,4.1,24)}, index=midx)
macro.to_csv("data/macro.csv")

from agents.data_agent import process_prices, calculate_features
prices = process_prices(pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True))
assert prices["DRAM"].iloc[:280].isna().all(), "bfill regression"
feats = calculate_features(prices)
assert len(feats) > 300, f"features truncated to {len(feats)}"
print("[1] bfill removed; features rows:", len(feats))

from agents.analysts.technical_analyst import build_technical_summary
ts = build_technical_summary(prices, ["XOM", "DRAM"])
assert "UNAVAILABLE (only 120 rows" in ts and "nan" not in ts.lower()
print("[2] technical summary (DRAM excerpt):", [l.strip() for l in ts.splitlines() if "MA200" in l or "MACD" in l or "Trend" in l][-3:])

from agents.analysts.macro_analyst import classify_regime, build_macro_summary
r = classify_regime(macro)
assert r["inflation_label"] == "FALLING", r["inflation_label"]
print("[3] inflation label on disinflation data:", r["inflation_label"], "|", r["regime"][:80])

import agents.analysts.fundamentals_analyst as fa
fa.jitter = lambda *a, **k: None
infos = {"XOM": {"trailingPE": 14, "marketCap": 4e11, "dividendYield": 3.4, "dividendRate": 3.96, "currentPrice": 116.5, "quoteType": "EQUITY"},
         "LOWY": {"trailingPE": 30, "marketCap": 1e11, "dividendYield": 0.5, "dividendRate": 1.0, "currentPrice": 200.0, "quoteType": "EQUITY"},
         "ETF1": {"quoteType": "ETF"}, "ETF2": {"quoteType": "ETF"}}
class FakeTicker:
    def __init__(self, t, session=None): self.info = infos[t]
fa.yf.Ticker = FakeTicker
s = fa.build_fundamentals_summary(["ETF1", "ETF2", "XOM", "LOWY"])
assert os.path.exists("data/fundamentals_cache.json"), "cache not saved"
assert "Circuit breaker" not in s and "LOWY" in s and "Dividend Yield: 0.5%" in s and "Dividend Yield: 3.4%" in s
print("[4] cache saved; 2 ETFs did not trip breaker; yields:", [l.strip() for l in s.splitlines() if "Dividend" in l])

# ---- graph + main with stubbed network analysts ----
import agents.analysts.news_analyst as na, agents.analysts.sentiment_analyst as sa
na.build_news_summary = lambda scope: "news data"
sa.build_sentiment_summary = lambda scope: "sentiment data"
fa.build_fundamentals_summary = lambda scope: "fund data"
import main as M
M.RUN_TRADER = True
CALLS.clear()
out = M.main()
macro_calls = [c for c in CALLS if "senior macro economist" in c[1]]
assert len(macro_calls) == 1, len(macro_calls)
debate_calls = [c for c in CALLS if "researcher" in c[1] and "head trader" not in c[1]]
assert all(m == "8b" for m, _ in debate_calls)
nee_bull = [p for m, p in CALLS if "LONG case for NEE" in p][0]
assert "[XOM]" not in nee_bull and "[NEE]" in nee_bull
md = open(sorted(p for p in os.listdir("reports") if p.startswith("trader_decisions"))[-1].join(["reports/", ""]), encoding="utf-8").read()
assert md.count("=" * 40) == 4 and len(out) == 5 and all(r["status"] == "ok" for r in out)
print(f"[5] macro LLM calls: {len(macro_calls)} (was 5); debate calls all 8b: {len(debate_calls)}; NEE debate has no XOM reports; dividers: {md.count('='*40)}; decisions logged: {len(out)}")
print("reports written:", sorted(os.listdir("reports"))[-3:])
print("ALL TESTS PASSED")
