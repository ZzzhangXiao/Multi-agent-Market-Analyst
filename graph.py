from typing import TypedDict
import pandas as pd

class TradingState(TypedDict):
    # Data layer — filled by data agent
    prices: pd.DataFrame
    macro: pd.DataFrame
    features: pd.DataFrame

    # Report layer — filled by future agents
    market_brief: str
    research_report: str
    portfolio_strategy: str
    signals: str