import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Optional, Literal
import pandas as pd
from langgraph.graph import StateGraph, END
from llm import get_llm


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────

class TradingState(TypedDict):
    # Data layer — filled by data agent
    prices:   pd.DataFrame
    macro:    pd.DataFrame
    features: pd.DataFrame

    # Per-ticker context for this run
    ticker: str

    # Analyst reports — start empty, filled in as supervisor routes to each
    technical_report:    Optional[str]
    macro_report:         Optional[str]
    news_report:          Optional[str]
    fundamentals_report:  Optional[str]
    sentiment_report:     Optional[str]

    # Bookkeeping for the supervisor
    completed_analysts: list[str]   # which analysts have already run
    routing_log:        list[str]   # supervisor's own trace, useful for debugging/demo
    next_step:          str         # supervisor's routing decision, read by conditional edge

    # Downstream report layer
    research_report:    Optional[str]
    portfolio_strategy:  Optional[str]
    signals:             Optional[str]


ANALYST_NAMES = ["technical", "macro", "news", "fundamentals", "sentiment"]


# ─────────────────────────────────────────
# SUPERVISOR — the dynamic routing node
# ─────────────────────────────────────────

def supervisor_node(state: TradingState) -> TradingState:
    """
    Decides which analyst (if any) should run next, based on what's
    already in state. This is what makes the graph dynamic rather than
    a fixed pipeline: the supervisor can skip analysts it judges
    unnecessary, or call one again with more context later if a future
    extension supports re-runs.

    For now the policy is rule-based and cheap (no LLM call) because the
    state we're routing on — which analysts have completed — is fully
    known and doesn't need interpretation. The LLM gets used inside each
    analyst node, where the actual reasoning happens. See
    `llm_supervisor_node` below for the alternative where routing itself
    is an LLM decision based on early analyst findings (e.g. skip
    fundamentals if technical+macro already show a clear breakdown).
    """
    completed = state.get("completed_analysts", [])
    remaining = [a for a in ANALYST_NAMES if a not in completed]

    if not remaining:
        next_step = "done"
    else:
        next_step = remaining[0]

    log = state.get("routing_log", [])
    log.append(f"supervisor: completed={completed} -> next={next_step}")

    return {**state, "routing_log": log, "next_step": next_step}


def llm_supervisor_node(state: TradingState) -> TradingState:
    """
    Optional upgrade to supervisor_node: asks the LLM which analyst to
    run next (or to stop early) based on the reports gathered so far.
    Swap this in for supervisor_node in build_graph() once you want true
    judgment-based skipping rather than fixed ordering.

    Example judgment calls this enables:
      - skip fundamentals entirely for an ETF/index ticker (SPY, QQQ)
      - skip sentiment if news_report already says "no coverage found"
      - stop after technical + macro if both already strongly agree,
        treating news/fundamentals/sentiment as confirmation-only
    """
    completed = state.get("completed_analysts", [])
    remaining = [a for a in ANALYST_NAMES if a not in completed]

    if not remaining:
        return {**state, "next_step": "done"}

    llm = get_llm(lite=True)
    reports_so_far = "\n\n".join(
        f"{name.upper()}:\n{state.get(f'{name}_report') or '(not run)'}"
        for name in ANALYST_NAMES if name in completed
    ) or "(no analysts have run yet)"

    prompt = f"""
You are the routing supervisor for a multi-agent trading analysis pipeline.
Ticker under analysis: {state['ticker']}

Analysts already completed: {completed or 'none'}
Analysts still available: {remaining}

REPORTS SO FAR:
{reports_so_far}

Decide the single most valuable next analyst to run, or "done" if the
existing reports already give enough signal to proceed to trading decision.
Skip an analyst only if you have a specific reason (e.g. ticker is an ETF
with no fundamentals, or news/sentiment already flagged as unavailable).

Respond with exactly one word: one of {remaining + ['done']}
"""
    response = llm.invoke(prompt).content.strip().lower()
    next_step = response if response in remaining else (remaining[0] if remaining else "done")

    log = state.get("routing_log", [])
    log.append(f"llm_supervisor: completed={completed} -> next={next_step} (raw='{response}')")

    return {**state, "routing_log": log, "next_step": next_step}


def route_from_supervisor(state: TradingState) -> str:
    """Reads the supervisor's decision and tells LangGraph where to go."""
    return state.get("next_step", "done")


# ─────────────────────────────────────────
# ANALYST NODE WRAPPERS
# ─────────────────────────────────────────
# Each wraps the existing run() function from agents/analysts/*.py so the
# graph can call them as nodes without changing the analyst modules at all.

def _mark_done(state: TradingState, name: str, report_key: str, report: str) -> TradingState:
    completed = state.get("completed_analysts", [])
    return {
        **state,
        report_key: report,
        "completed_analysts": completed + [name],
    }


def technical_node(state: TradingState) -> TradingState:
    from agents.analysts.technical_analyst import run as run_technical
    ticker = state["ticker"] if state["ticker"] != "PORTFOLIO" else None
    return _mark_done(state, "technical", "technical_report", run_technical(ticker))


def macro_node(state: TradingState) -> TradingState:
    from agents.analysts.macro_analyst import run as run_macro
    return _mark_done(state, "macro", "macro_report", run_macro())


def news_node(state: TradingState) -> TradingState:
    from agents.analysts.news_analyst import run as run_news
    ticker = state["ticker"] if state["ticker"] != "PORTFOLIO" else None
    return _mark_done(state, "news", "news_report", run_news(ticker))


def fundamentals_node(state: TradingState) -> TradingState:
    from agents.analysts.fundamentals_analyst import run as run_fundamentals
    ticker = state["ticker"] if state["ticker"] != "PORTFOLIO" else None
    return _mark_done(state, "fundamentals", "fundamentals_report", run_fundamentals(ticker))


def sentiment_node(state: TradingState) -> TradingState:
    from agents.analysts.sentiment_analyst import run as run_sentiment
    ticker = state["ticker"] if state["ticker"] != "PORTFOLIO" else None
    return _mark_done(state, "sentiment", "sentiment_report", run_sentiment(ticker))
# ─────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────

def build_graph(use_llm_supervisor: bool = False):
    """
    Builds the StateGraph. Every analyst node routes back through the
    supervisor rather than to a fixed next node — that loop-back is what
    lets the supervisor re-evaluate after each step instead of committing
    to a full plan upfront.
    """
    graph = StateGraph(TradingState)

    graph.add_node("supervisor", llm_supervisor_node if use_llm_supervisor else supervisor_node)
    graph.add_node("technical", technical_node)
    graph.add_node("macro", macro_node)
    graph.add_node("news", news_node)
    graph.add_node("fundamentals", fundamentals_node)
    graph.add_node("sentiment", sentiment_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "technical":    "technical",
            "macro":         "macro",
            "news":          "news",
            "fundamentals":  "fundamentals",
            "sentiment":     "sentiment",
            "done":          END,
        },
    )

    # Every analyst returns control to the supervisor for re-evaluation
    for name in ["technical", "macro", "news", "fundamentals", "sentiment"]:
        graph.add_edge(name, "supervisor")

    return graph.compile()


def run_analyst_graph(ticker: str, prices, macro, features, use_llm_supervisor: bool = False) -> TradingState:
    """
    Entry point for main.py. Runs the full analyst stage for one ticker
    and returns the final state, including every report produced and the
    routing_log (handy to print for a demo — it's the visible evidence
    the graph made decisions rather than following a fixed script).
    """
    app = build_graph(use_llm_supervisor=use_llm_supervisor)

    initial_state: TradingState = {
        "prices": prices,
        "macro": macro,
        "features": features,
        "ticker": ticker,
        "technical_report": None,
        "macro_report": None,
        "news_report": None,
        "fundamentals_report": None,
        "sentiment_report": None,
        "completed_analysts": [],
        "routing_log": [],
        "next_step": "",
        "research_report": None,
        "portfolio_strategy": None,
        "signals": None,
    }

    return app.invoke(initial_state)