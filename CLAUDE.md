# CLAUDE.md

Multi-agent trading research pipeline. Python fetches data and computes every
signal and label; LLM analysts (LangGraph + Groq) write reports around those
labels; a bull/bear debate and a trader produce a structured, logged decision.

## Environment

- Windows. Use the conda env `ml` (`conda activate ml`, or `conda run -n ml python ...`).
- Always run scripts from the repo root. Modules read and write relative paths
  (`data/`, `reports/`, `logs/`) and assume that working directory.
- Never read, print, cat, or log `.env`. API keys (`GROQ_API_KEY`, `FRED_API_KEY`,
  `FINNHUB_API_KEY`, `ANTHROPIC_API_KEY`) are loaded by `config.py` through
  `python-dotenv`. Refer to them by name only.

## Layout

- `main.py` is the pipeline entry point. It fetches data, runs macro once for the
  whole portfolio, runs the per-ticker analyst graph, then (if `RUN_TRADER`)
  runs the debate and trader.
  Cost knobs at the top: `TEST_MODE`, `USE_LLM_SUPERVISOR`, `RUN_TRADER`,
  `DEBATE_ROUNDS`, `DEBATE_TICKERS_LIMIT`.
- `config.py` is the single source of truth for `TICKERS`, `LLM_PROVIDER`, and
  FRED series. No other file hardcodes a ticker list.
- `llm.py` has `get_llm(lite=False)`: `lite=True` returns `GROQ_MODEL_LITE` and the
  default returns `GROQ_MODEL`. Both are set in `config.py`.
- `graph.py` is the LangGraph supervisor. Its routing is rule-based by default,
  and every analyst node loops back to the supervisor.
- `agents/data_agent.py` fetches prices and macro data and computes features
  (`data/*.csv`).
- `agents/analysts/` holds `technical`, `macro`, `news`, `fundamentals`, and
  `sentiment`. Each has `run(ticker=None)` and a `__main__` block, and writes
  `reports/<name>_analysis_<TICKER>_<date>.md`.
- `agents/researchers/` holds the bull/bear researchers and `debate.py`
  (lite model).
- `agents/trader.py` makes the final decision through structured output
  (`TradeDecision`).
- `decisions/schema.py` defines the `TradeDecision` model and `SIGNAL_BANDS`, the
  single score-to-signal mapping, which a validator enforces.
- `decisions/log.py` is the append-only `logs/decisions.jsonl` plus prompt/input
  snapshots. It logs failures too. Run `python -m decisions.log` for a summary.
- `yf_session.py` provides the shared curl_cffi session for all yfinance calls.
- `[inactive] backtester/` and `[inactive] strategies/` are parked. Do not wire
  them in.
- `_testrun/` is scratch space for the offline tests (gitignored).

## Design philosophy: Python computes, LLMs narrate

- Every label is computed deterministically in Python: MA position, MACD
  crossover vs. momentum, setup label, macro regime, signal bands, and so on.
  The LLM receives those labels and must repeat them verbatim. It never infers
  them.
- New signals go into Python code with an explicit label string. The prompt then
  tells the model to "use the X label verbatim". Do not ask the LLM to compute or
  classify from raw numbers.
- Prompts require every claim to cite a number from the supplied data. The trader's
  `key_evidence` numbers must be copied exactly, and `evidence_grounding` checks
  them.
- Consistency rules (such as signal vs. score) live in Python validators, not only
  in prompt text.
- Reports save the deterministic data block alongside the LLM narrative so each can
  be audited against the other.

## LLM models and token cost

- Groq model IDs (`GROQ_MODEL`, `GROQ_MODEL_LITE`, `GROQ_REASONING_EFFORT`) are set
  in `config.py` only. Do not hardcode model names in `llm.py`, agents, graph, or
  tests. Agents call `get_llm()` / `get_llm(lite=True)`.
- Token cost and Groq rate limits are real constraints:
  - Prefer `lite=True` unless the task needs the full model.
  - The gpt-oss models are reasoning models. Hidden reasoning tokens count toward
    Groq limits, so keep `GROQ_REASONING_EFFORT` at `low` unless there is a reason
    to raise it.
  - Keep prompts per-ticker. Never concatenate every ticker's reports into one call.
  - Macro runs once per pipeline, not once per ticker.
  - Do not add LLM calls where Python can decide (routing stays rule-based unless
    `USE_LLM_SUPERVISOR`).
  - Do not run the full pipeline or the trader stage casually. Use `RUN_TRADER=False`,
    `DEBATE_ROUNDS=0`, or `DEBATE_TICKERS_LIMIT` while iterating.

## Code conventions

- No emoji in `print()` output or in written reports. Plain ASCII markers only
  (e.g. `PASS`/`FAIL`, `WARNING:`), because the Windows console encoding breaks
  on emoji.
- Always open files with `encoding="utf-8"`, for both reading and writing, including
  JSON caches.
- Keep the existing style: `BUGFIX:` / `NOTE:` comments that explain *why*, and
  section banners (`# ── ... ──`).

## Testing workflow

1. Test each analyst standalone before running `main.py`:
   ```
   python agents/analysts/technical_analyst.py
   python agents/analysts/macro_analyst.py
   python agents/analysts/news_analyst.py
   python agents/analysts/fundamentals_analyst.py
   python agents/analysts/sentiment_analyst.py
   ```
   These make real LLM calls, so run only the analyst you changed.
2. After any change, run both offline suites. They stub the LLM and make zero Groq
   calls:
   ```
   python test_bugfixes_offline.py
   python test_decisions_offline.py
   ```
   Both must pass before the change is considered done.
3. Optional: `python smoke_check.py` runs offline checks against real `data/*.csv`.
   `--live` also hits Yahoo once. It rewrites `data/prices.csv` by design.
4. Only then run `python main.py`.
