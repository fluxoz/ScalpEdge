"""ScalpEdge — Intraday 5-minute candle scalping backtester."""

from .data import DataManager
from .ta_indicators import add_all_indicators
from .probabilities import MonteCarlo, MarkovChain
from .options import BlackScholes
from .ml import MLEngine
from .backtester import Backtester
from .strategies import HybridStrategy

__version__ = "0.1.0"
__all__ = [
    "DataManager",
    "add_all_indicators",
    "MonteCarlo",
    "MarkovChain",
    "BlackScholes",
    "MLEngine",
    "Backtester",
    "HybridStrategy",
]
