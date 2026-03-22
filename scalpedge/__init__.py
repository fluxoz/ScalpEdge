"""ScalpEdge — Intraday 5-minute candle scalping backtester.

Import individual modules directly rather than via this package namespace
to avoid loading the entire dependency chain on import::

    from scalpedge.data import DataManager
    from scalpedge.ta_indicators import add_all_indicators
    from scalpedge.options import BlackScholes
    from scalpedge.live_engine import LiveSignalEngine, SignalEvent
    # etc.
"""

__version__ = "0.1.0"

from scalpedge.live_engine import LiveSignalEngine, SignalEvent

__all__ = ["LiveSignalEngine", "SignalEvent", "__version__"]
