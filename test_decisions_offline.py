"""
Offline test for the structured decision + logging layer.
Uses the REAL langchain_groq structured-output parser; only the HTTP
call to Groq is faked. Run from the repo root:  python test_decisions_offline.py
"""
import sys, os, json, shutil, types
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.makedirs(os.path.join(REPO, "_testrun"), exist_ok=True)
os.chdir(os.path.join(REPO, "_testrun"))
shutil.rmtree("logs", ignore_errors=True)

import numpy as np, pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# Scripted tool-call responses, consumed in order.
SCRIPT = []

class FakeGroq(ChatGroq):
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        args = SCRIPT.pop(0)
        if isinstance(args, Exception):
            raise args
        msg = AIMessage(content="", tool_calls=[{"name": "TradeDecision", "args": args, "id": "c1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])

llm_mod = types.ModuleType("llm")
from config import GROQ_MODEL
llm_mod.get_llm = lambda lite=False: FakeGroq(api_key="x", model=GROQ_MODEL, temperature=0.2)
sys.modules["llm"] = llm_mod

from agents import trader
from decisions.log import load_decisions, evidence_grounding

idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=60)
prices = pd.DataFrame({"XOM": np.linspace(100, 110, 60), "NEE": np.linspace(80, 75, 60),
                       "DRAM": np.linspace(20, 25, 60)}, index=idx)
reports = {
    "XOM": "TECHNICAL [XOM]: RSI(14) 58.2, price 110.0 above MA50 104.9",
    "NEE": "TECHNICAL [NEE]: RSI(14) 31.7, Current Ratio 0.54, Debt/Equity 156.69",
    "DRAM": "TECHNICAL [DRAM]: RSI(14) 71.9",
}
good = lambda t, s, sc, ev: {"ticker": t, "signal": s, "score": sc, "confidence": "Medium",
    "winning_side": "none", "key_evidence": ev, "reasoning": "r", "key_risk": "k"}

SCRIPT[:] = [
    good("XOM", "BUY", 0.45, ["RSI(14) 58.2", "price 110.0 above MA50 104.9"]),   # clean
    good("NEE", "BUY", -0.3, ["Current Ratio 0.54"]),                              # inconsistent -> retry
    good("NEE", "REDUCE", -0.3, ["Current Ratio 0.54", "Debt/Equity 150.0"]),       # fixed; 150.0 is invented
    RuntimeError("tool_use_failed"),                                                # DRAM fails twice
    RuntimeError("tool_use_failed"),
]
recs = trader.run(reports, ["XOM", "NEE", "DRAM"], rounds=0, prices=prices)

by = {r["ticker"]: r for r in load_decisions()}
assert len(by) == 3 and len(SCRIPT) == 0
x, n, d = by["XOM"], by["NEE"], by["DRAM"]
assert x["status"] == "ok" and x["signal"] == "BUY" and x["attempts"] == 1
assert x["price_at_decision"] == 110.0 and x["stale_data"] is False and x["evidence_grounded_frac"] == 1.0
print("[1] clean decision logged:", {k: x[k] for k in ["signal", "score", "data_as_of", "price_at_decision"]})
assert n["status"] == "ok" and n["signal"] == "REDUCE" and n["attempts"] == 2
assert n["evidence_grounded_frac"] == 0.5
print("[2] BUY with score -0.3 rejected, retried to REDUCE; invented number caught: grounded =", n["evidence_grounded_frac"])
assert d["status"] == "llm_error" and d["score"] is None and d["attempts"] == 2
print("[3] failure still logged:", d["status"], "|", d["error"][:40])
assert os.path.exists(os.path.join("logs", "snapshots")) and x["prompt_sha256"] and x["model"] == GROQ_MODEL
print("[4] snapshots + model/temperature/git metadata recorded:", x["model"], x["temperature"], x["git_commit"])

# stale-data flag: prices ending 30 days ago
old = prices.copy(); old.index = old.index - pd.Timedelta(days=30)
SCRIPT[:] = [good("XOM", "HOLD", 0.0, ["RSI(14) 58.2"])]
trader.run({"XOM": reports["XOM"]}, ["XOM"], rounds=0, prices=old)
assert load_decisions()[-1]["stale_data"] is True
print("[5] stale input data flagged")

print("\nSummary view:")
from decisions.log import summarize; summarize()
print("\nALL DECISION TESTS PASSED")
