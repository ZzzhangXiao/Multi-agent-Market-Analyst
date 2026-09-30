"""
Append-only decision log.

Every trader decision — including failures — becomes one JSON line in
logs/decisions.jsonl, plus content-addressed snapshots of the exact prompt
and inputs in logs/snapshots/. This is the raw material for step 2
(forward returns, IC, hit rate) and step 3 (baselines/ablations).

Why log failures too: if only successful parses are kept, the evaluated
sample is silently filtered by "cases where the LLM behaved", which biases
every downstream metric.

Why commit logs/ to git: commit timestamps are independent evidence that a
decision existed before its outcome was known — useful credibility for a
forward (paper-trading) test.

Usage:
    python -m decisions.log              # summary of the last 20 decisions
    python -m decisions.log --last 50
"""
import os
import re
import sys
import json
import uuid
import hashlib
import subprocess
from datetime import datetime, timezone

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "decisions.jsonl")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
RECORD_SCHEMA_VERSION = 1

# A decision is flagged stale if the latest price it saw is this many
# calendar days older than the decision time (e.g. TEST_MODE on old CSVs).
STALE_DATA_DAYS = 5


# ───────────────────────────── run-level metadata

def new_run_id() -> str:
    """One id per pipeline run; groups the decisions made together."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def git_state() -> dict:
    """Commit hash + dirty flag, so every decision is tied to the exact code."""
    def _git(*args):
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None
    commit = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


# ───────────────────────────── snapshots

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_snapshot(text: str) -> str:
    """Store text once under its hash; identical inputs are never duplicated."""
    h = sha256(text)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"{h[:16]}.txt")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return h


# ───────────────────────────── market context at decision time

def price_context(prices, ticker: str, decided_at: datetime) -> dict:
    """
    Last real close the decision could have seen, and its date. Step 2
    measures forward returns FROM this point, so it must be recorded now,
    not reconstructed later from a CSV that may have been revised.
    """
    if prices is None or ticker not in getattr(prices, "columns", []):
        return {"data_as_of": None, "price_at_decision": None, "stale_data": None}
    series = prices[ticker].dropna()
    if series.empty:
        return {"data_as_of": None, "price_at_decision": None, "stale_data": None}
    as_of = series.index[-1]
    age_days = (decided_at.date() - as_of.date()).days
    return {
        "data_as_of": as_of.strftime("%Y-%m-%d"),
        "price_at_decision": float(series.iloc[-1]),
        "stale_data": age_days > STALE_DATA_DAYS,
    }


# ───────────────────────────── grounding check

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def evidence_grounding(evidence: list, source_text: str) -> float | None:
    """
    Fraction of numbers cited in key_evidence that literally appear in the
    reports the trader was given. A cheap deterministic hallucination check:
    1.0 = every cited number exists in the inputs. None = nothing to check.
    """
    nums = [n for item in (evidence or []) for n in _NUM.findall(item)]
    # Ignore trivial integers like "1", "2" (list numbering, "2-3 sentences").
    nums = [n for n in nums if len(n.lstrip("-").replace(".", "")) >= 2]
    if not nums:
        return None
    found = sum(1 for n in nums if n in source_text)
    return round(found / len(nums), 3)


# ───────────────────────────── write

def log_decision(
    *,
    run_id: str,
    ticker: str,
    decision,                 # TradeDecision or None on failure
    status: str,              # "ok" | "parse_failed" | "llm_error" | "no_inputs"
    prompt: str,
    inputs_text: str,
    prices,
    model: str,
    temperature,
    debate_rounds: int,
    horizon_days: int,
    attempts: int,
    error: str | None = None,
    strategy: str = "llm_trader",
    extra: dict | None = None,
) -> dict:
    decided_at = datetime.now(timezone.utc)
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_id": uuid.uuid4().hex,
        "run_id": run_id,
        "strategy": strategy,           # later: "rule_baseline", "llm_no_debate", ...
        "decided_at_utc": decided_at.isoformat(timespec="seconds"),
        "ticker": ticker,
        **price_context(prices, ticker, decided_at),
        "horizon_days": horizon_days,
        "status": status,
        "signal": None, "score": None, "confidence": None, "winning_side": None,
        "key_evidence": None, "reasoning": None, "key_risk": None,
        "evidence_grounded_frac": None,
        "model": model,
        "temperature": temperature,
        "debate_rounds": debate_rounds,
        "attempts": attempts,
        "error": error,
        "prompt_sha256": save_snapshot(prompt) if prompt else None,
        "inputs_sha256": save_snapshot(inputs_text) if inputs_text else None,
        **git_state(),
    }
    if decision is not None:
        record.update(decision.model_dump(exclude={"ticker"}))
        record["evidence_grounded_frac"] = evidence_grounding(decision.key_evidence, inputs_text)
    if extra:
        record.update(extra)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# ───────────────────────────── read

def load_decisions(path: str = LOG_PATH) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: skipping corrupt line {i} in {path}")
    return out


def summarize(last: int = 20) -> None:
    recs = load_decisions()
    if not recs:
        print(f"No decisions logged yet ({LOG_PATH}).")
        return
    ok = [r for r in recs if r["status"] == "ok"]
    print(f"{len(recs)} records ({len(ok)} ok, {len(recs) - len(ok)} failed) in {LOG_PATH}\n")
    print(f"{'decided (UTC)':20} {'ticker':8} {'status':13} {'signal':7} {'score':>6} "
          f"{'conf':6} {'as_of':10} {'price':>9} {'ground':>6} {'stale':5}")
    for r in recs[-last:]:
        score = f"{r['score']:+.2f}" if r.get("score") is not None else "-"
        price = f"{r['price_at_decision']:.2f}" if r.get("price_at_decision") is not None else "-"
        g = r.get("evidence_grounded_frac")
        print(f"{r['decided_at_utc'][:19]:20} {r['ticker']:8} {r['status']:13} "
              f"{(r.get('signal') or '-'):7} {score:>6} {(r.get('confidence') or '-'):6} "
              f"{(r.get('data_as_of') or '-'):10} {price:>9} "
              f"{(f'{g:.2f}' if g is not None else '-'):>6} {str(r.get('stale_data')):5}")


if __name__ == "__main__":
    n = 20
    if "--last" in sys.argv:
        n = int(sys.argv[sys.argv.index("--last") + 1])
    summarize(n)
