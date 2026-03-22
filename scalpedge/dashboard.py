"""ScalpEdge — Textual TUI Dashboard with live charts, tabs, and ticker search.

Features
--------
- Per-ticker tabs with candlestick, volume, RSI, MACD, and ADX charts
- Overview tab showing a compact status panel for every monitored ticker
- SPY market regime banner (Bullish / Bearish / Neutral)
- Scrolling signal log (always visible at the bottom)
- Ticker search: press ``/`` to focus the search bar, type a symbol, Enter to jump
- Session statistics (total signals, uptime, last bar time)
- Graceful shutdown on ``Ctrl+C``

Requires ``pip install scalpedge[tui]``.
"""

from __future__ import annotations

import collections
import logging
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual_plotext import PlotextPlot

if TYPE_CHECKING:
    from scalpedge.live_engine import LiveSignalEngine, SignalEvent

logger = logging.getLogger("scalpedge.dashboard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGIME_LABELS = {1: "🟢  BULLISH", -1: "🔴  BEARISH", 0: "⚪  NEUTRAL"}
_SIGNAL_LABELS = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪ FLAT"}
_SIGNAL_COLORS = {1: "green", -1: "red", 0: "bright_black"}

# Max bars kept in the per-ticker history ring for chart rendering.
_CHART_HISTORY = 120

# How many x-axis tick labels to show on charts.
_MAX_XTICKS = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ticker_id(ticker: str) -> str:
    """Return a DOM/CSS-safe ID fragment for *ticker* (e.g. ``BRK.B`` → ``brk_b``)."""
    return re.sub(r"[^a-z0-9]", "_", ticker.lower())


def _fmt(value: object, spec: str, fallback: str = "N/A") -> str:
    """Format *value* with *spec*, returning *fallback* on NaN / Inf / None."""
    if value is None:
        return fallback
    try:
        f = float(value)  # type: ignore[arg-type]
        if math.isnan(f) or math.isinf(f):
            return fallback
        return format(f, spec)
    except (TypeError, ValueError):
        return fallback


def _time_label(bar: dict) -> str:
    """Extract an ``HH:MM`` time label from a bar dict's ``datetime`` field."""
    dt = bar.get("datetime")
    if dt is None:
        return ""
    try:
        import pandas as pd

        if isinstance(dt, pd.Timestamp):
            return dt.strftime("%H:%M")
        s = str(dt)
        # ISO-8601 strings: "2024-01-02 09:30:00+00:00" → "09:30"
        parts = s.replace("T", " ").split(" ")
        if len(parts) >= 2:
            return parts[1][:5]
    except Exception:
        pass
    return str(dt)[:5]


def _build_xticks(labels: list[str], max_ticks: int = _MAX_XTICKS) -> tuple[list[int], list[str]]:
    """Evenly-spaced x-tick positions and labels from a list of *labels*."""
    n = len(labels)
    if n == 0:
        return [], []
    step = max(1, n // max_ticks)
    positions = list(range(0, n, step))
    return positions, [labels[i] for i in positions]


def _compute_regime_from_bar(bar: dict) -> int:
    """Estimate SPY market regime from a single bar's indicator snapshot.

    Uses a two-factor vote — price vs. VWAP and price vs. EMA-21.

    Returns
    -------
    int
        ``1`` (bullish), ``-1`` (bearish), or ``0`` (neutral).
    """
    score = 0
    try:
        c = float(bar.get("close") or 0)
        v = float(bar.get("vwap") or 0)
        if v != 0 and not (math.isnan(c) or math.isnan(v)):
            score += 1 if c > v else -1
    except (TypeError, ValueError):
        pass
    try:
        c = float(bar.get("close") or 0)
        e = float(bar.get("ema_21") or 0)
        if e != 0 and not (math.isnan(c) or math.isnan(e)):
            score += 1 if c > e else -1
    except (TypeError, ValueError):
        pass
    return 1 if score > 0 else (-1 if score < 0 else 0)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class BarUpdated(Message):
    """Posted each time a processed bar arrives for a ticker."""

    def __init__(self, ticker: str, bar: dict, last_signal: int) -> None:
        super().__init__()
        self.ticker = ticker
        self.bar = bar
        self.last_signal = last_signal


class SignalFired(Message):
    """Posted when :class:`~scalpedge.live_engine.LiveSignalEngine` fires a signal."""

    def __init__(self, event: "SignalEvent") -> None:
        super().__init__()
        self.event = event


# ---------------------------------------------------------------------------
# Chart widgets (PlotextPlot subclasses)
# ---------------------------------------------------------------------------

class CandleChart(PlotextPlot):
    """Candlestick + price EMA overlay for one ticker."""

    DEFAULT_CSS = """
    CandleChart {
        height: 1fr;
        border: round $primary;
    }
    """

    def __init__(self, ticker: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ticker = ticker
        self.border_title = f" {ticker} — Candles (5m) "
        self._bars: list[dict] = []

    def update_bars(self, bars: list[dict]) -> None:
        self._bars = bars
        self.refresh()

    def render(self):  # type: ignore[override]
        plt = self.plt
        plt.clf()
        plt.title(f"{self.ticker} — Candlestick")

        bars = self._bars
        if len(bars) < 2:
            plt.text("Waiting for bar data…", 1, 1)
            return super().render()

        opens  = [float(b.get("open")  or 0) for b in bars]
        closes = [float(b.get("close") or 0) for b in bars]
        highs  = [float(b.get("high")  or 0) for b in bars]
        lows   = [float(b.get("low")   or 0) for b in bars]
        labels = [_time_label(b) for b in bars]
        xs = list(range(len(bars)))

        plt.candlestick(xs, {"Open": opens, "Close": closes,
                             "High": highs, "Low": lows})

        # EMA-9 overlay
        ema9 = [b.get("ema_9") for b in bars]
        if any(v is not None for v in ema9):
            clean_xs = [x for x, v in zip(xs, ema9) if v is not None]
            clean_v  = [float(v) for v in ema9 if v is not None]
            plt.plot(clean_xs, clean_v, color="cyan", label="EMA9")

        tick_pos, tick_lbl = _build_xticks(labels)
        if tick_pos:
            plt.xticks(tick_pos, tick_lbl)

        return super().render()


class VolumeChart(PlotextPlot):
    """Volume bar chart for one ticker."""

    DEFAULT_CSS = """
    VolumeChart {
        height: 8;
        border: round $primary;
    }
    """

    def __init__(self, ticker: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ticker = ticker
        self.border_title = f" {ticker} — Volume "
        self._bars: list[dict] = []

    def update_bars(self, bars: list[dict]) -> None:
        self._bars = bars
        self.refresh()

    def render(self):  # type: ignore[override]
        plt = self.plt
        plt.clf()
        plt.title("Volume")

        bars = self._bars
        if not bars:
            plt.text("Waiting…", 1, 1)
            return super().render()

        vols   = [float(b.get("volume") or 0) for b in bars]
        labels = [_time_label(b) for b in bars]
        xs = list(range(len(bars)))

        closes = [float(b.get("close") or 0) for b in bars]
        opens  = [float(b.get("open")  or 0) for b in bars]
        colors = ["green" if c >= o else "red" for c, o in zip(closes, opens)]

        plt.bar(xs, vols, color=colors, orientation="vertical")

        tick_pos, tick_lbl = _build_xticks(labels)
        if tick_pos:
            plt.xticks(tick_pos, tick_lbl)

        return super().render()


class RSIChart(PlotextPlot):
    """RSI line chart with overbought / oversold reference lines."""

    DEFAULT_CSS = """
    RSIChart {
        width: 1fr;
        height: 1fr;
        border: round $primary;
    }
    """

    def __init__(self, ticker: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ticker = ticker
        self.border_title = f" {ticker} — RSI (14) "
        self._bars: list[dict] = []

    def update_bars(self, bars: list[dict]) -> None:
        self._bars = bars
        self.refresh()

    def render(self):  # type: ignore[override]
        plt = self.plt
        plt.clf()
        plt.title("RSI (14)")

        bars = self._bars
        vals = [b.get("rsi_14") for b in bars]
        xs_all = list(range(len(bars)))
        xs = [x for x, v in zip(xs_all, vals) if v is not None]
        ys = [float(v) for v in vals if v is not None]

        if len(ys) < 2:
            plt.text("Waiting…", 1, 1)
            return super().render()

        plt.plot(xs, ys, color="yellow", label="RSI")
        plt.hline(70, "red")
        plt.hline(30, "green")
        plt.ylim(0, 100)

        labels = [_time_label(b) for b in bars]
        tick_pos, tick_lbl = _build_xticks(labels)
        if tick_pos:
            plt.xticks(tick_pos, tick_lbl)

        return super().render()


class MACDChart(PlotextPlot):
    """MACD histogram (green above zero, red below)."""

    DEFAULT_CSS = """
    MACDChart {
        width: 1fr;
        height: 1fr;
        border: round $primary;
    }
    """

    def __init__(self, ticker: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ticker = ticker
        self.border_title = f" {ticker} — MACD (12,26,9) "
        self._bars: list[dict] = []

    def update_bars(self, bars: list[dict]) -> None:
        self._bars = bars
        self.refresh()

    def render(self):  # type: ignore[override]
        plt = self.plt
        plt.clf()
        plt.title("MACD")

        bars = self._bars
        vals = [b.get("macd") for b in bars]
        xs_all = list(range(len(bars)))
        xs = [x for x, v in zip(xs_all, vals) if v is not None]
        ys = [float(v) for v in vals if v is not None]

        if len(ys) < 2:
            plt.text("Waiting…", 1, 1)
            return super().render()

        colors = ["green" if v >= 0 else "red" for v in ys]
        plt.bar(xs, ys, color=colors, orientation="vertical")
        plt.hline(0, "white")

        labels = [_time_label(b) for b in bars]
        tick_pos, tick_lbl = _build_xticks(labels)
        if tick_pos:
            plt.xticks(tick_pos, tick_lbl)

        return super().render()


class ADXChart(PlotextPlot):
    """ADX + DI+/DI- line chart."""

    DEFAULT_CSS = """
    ADXChart {
        width: 1fr;
        height: 1fr;
        border: round $primary;
    }
    """

    def __init__(self, ticker: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ticker = ticker
        self.border_title = f" {ticker} — ADX (14) "
        self._bars: list[dict] = []

    def update_bars(self, bars: list[dict]) -> None:
        self._bars = bars
        self.refresh()

    def render(self):  # type: ignore[override]
        plt = self.plt
        plt.clf()
        plt.title("ADX / DI")

        bars = self._bars
        adx_vals  = [b.get("adx_14") for b in bars]
        dip_vals  = [b.get("dmp_14") for b in bars]
        dim_vals  = [b.get("dmn_14") for b in bars]
        xs_all    = list(range(len(bars)))

        xs_adx = [x for x, v in zip(xs_all, adx_vals) if v is not None]
        ys_adx = [float(v) for v in adx_vals if v is not None]

        if len(ys_adx) < 2:
            plt.text("Waiting…", 1, 1)
            return super().render()

        plt.plot(xs_adx, ys_adx, color="white", label="ADX")
        plt.hline(25, "dim")

        xs_dip = [x for x, v in zip(xs_all, dip_vals) if v is not None]
        ys_dip = [float(v) for v in dip_vals if v is not None]
        if len(ys_dip) >= 2:
            plt.plot(xs_dip, ys_dip, color="green", label="DI+")

        xs_dim = [x for x, v in zip(xs_all, dim_vals) if v is not None]
        ys_dim = [float(v) for v in dim_vals if v is not None]
        if len(ys_dim) >= 2:
            plt.plot(xs_dim, ys_dim, color="red", label="DI−")

        labels = [_time_label(b) for b in bars]
        tick_pos, tick_lbl = _build_xticks(labels)
        if tick_pos:
            plt.xticks(tick_pos, tick_lbl)

        return super().render()


# ---------------------------------------------------------------------------
# Stats panel (text)
# ---------------------------------------------------------------------------

class TickerStats(Static):
    """Compact key-indicator summary panel for one ticker."""

    DEFAULT_CSS = """
    TickerStats {
        height: 9;
        border: round $primary;
        padding: 0 1;
    }
    """

    def __init__(self, ticker: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ticker = ticker
        self.border_title = f" {ticker} — Stats "
        self._bar: dict = {}
        self._signal = 0
        self._signal_count = 0

    def update_bar(self, bar: dict, signal: int) -> None:
        self._bar = bar
        self._signal = signal
        self._refresh_display()

    def bump_signal_count(self) -> None:
        self._signal_count += 1
        self._refresh_display()

    def _refresh_display(self) -> None:
        bar = self._bar
        sig = self._signal

        close   = _fmt(bar.get("close"),  ",.2f", "—")
        rsi_str = _fmt(bar.get("rsi_14"), ".1f")
        macd_v  = bar.get("macd")
        macd_str = _fmt(macd_v, "+.4f")
        adx_str = _fmt(bar.get("adx_14"), ".1f")

        close_f = bar.get("close") or 0.0
        vwap    = bar.get("vwap")  or 0.0
        vwap_dev: str = "N/A"
        try:
            c, v = float(close_f), float(vwap)
            if v != 0 and not (math.isnan(c) or math.isnan(v)):
                vwap_dev = f"{(c - v) / v * 100:+.2f}%"
        except (TypeError, ValueError):
            pass

        atr_str  = _fmt(bar.get("atr_14"), ".4f")
        ema9_str = _fmt(bar.get("ema_9"),  ",.2f")
        ema21_str = _fmt(bar.get("ema_21"), ",.2f")

        color     = _SIGNAL_COLORS.get(sig, "bright_black")
        sig_label = _SIGNAL_LABELS.get(sig, "⚪ FLAT")

        self.update(
            f"[bold {color}]{sig_label}[/]    Price: [bold]${close}[/]\n"
            f"RSI: {rsi_str}    MACD: {macd_str}    ADX: {adx_str}\n"
            f"VWAP Δ: {vwap_dev}    ATR: {atr_str}\n"
            f"EMA9: {ema9_str}    EMA21: {ema21_str}\n"
            f"Signals fired: [bold]{self._signal_count}[/]"
        )


# ---------------------------------------------------------------------------
# Compact overview panel (one per ticker, shown in Overview tab)
# ---------------------------------------------------------------------------

class TickerOverviewPanel(Static):
    """Compact per-ticker widget used inside the Overview tab grid."""

    DEFAULT_CSS = """
    TickerOverviewPanel {
        width: 1fr;
        min-width: 22;
        height: 16;
        border: round $primary;
        padding: 0 1;
        margin: 0 1;
    }
    """

    def __init__(self, ticker: str) -> None:
        super().__init__("⏳  Waiting for data…", id=f"overview-{_ticker_id(ticker)}")
        self.ticker = ticker
        self.border_title = f" {ticker} "
        self._bar: dict = {}
        self._signal = 0
        self._signal_count = 0

    def update_bar(self, bar: dict, signal: int) -> None:
        self._bar = bar
        self._signal = signal
        self._refresh_display()

    def bump_signal_count(self) -> None:
        self._signal_count += 1
        self._refresh_display()

    def _refresh_display(self) -> None:
        bar = self._bar
        sig = self._signal

        close   = _fmt(bar.get("close"),  ",.2f", "—")
        rsi_str = _fmt(bar.get("rsi_14"), ".1f")
        macd_str = _fmt(bar.get("macd"), "+.4f")
        adx_str = _fmt(bar.get("adx_14"), ".1f")

        close_f = bar.get("close") or 0.0
        vwap    = bar.get("vwap")  or 0.0
        vwap_str = "N/A"
        try:
            c, v = float(close_f), float(vwap)
            if v != 0 and not (math.isnan(c) or math.isnan(v)):
                vwap_str = f"{(c - v) / v * 100:+.2f}%"
        except (TypeError, ValueError):
            pass

        color     = _SIGNAL_COLORS.get(sig, "bright_black")
        sig_label = _SIGNAL_LABELS.get(sig, "⚪ FLAT")

        open_str  = _fmt(bar.get("open"),   ".2f")
        high_str  = _fmt(bar.get("high"),   ".2f")
        low_str   = _fmt(bar.get("low"),    ".2f")
        vol       = bar.get("volume") or 0.0
        try:
            vf = float(vol)
            vol_str = f"{vf / 1_000_000:.1f}M" if vf >= 1_000_000 else f"{vf / 1_000:.0f}K"
        except (TypeError, ValueError):
            vol_str = "N/A"

        self.update(
            f"[bold {color}]{sig_label}[/]\n"
            f"Price:   [bold]${close}[/]\n"
            f"[dim]──────────────────[/]\n"
            f"RSI:     {rsi_str}\n"
            f"MACD:    {macd_str}\n"
            f"ADX:     {adx_str}\n"
            f"VWAP Δ:  {vwap_str}\n"
            f"[dim]──────────────────[/]\n"
            f"O:{open_str}  H:{high_str}\n"
            f"L:{low_str}  V:{vol_str}\n"
            f"[dim]──────────────────[/]\n"
            f"Signals: [bold]{self._signal_count}[/]"
        )


# ---------------------------------------------------------------------------
# Regime banner
# ---------------------------------------------------------------------------

class RegimeBar(Static):
    """Top-of-screen banner showing the current SPY market regime."""

    DEFAULT_CSS = """
    RegimeBar {
        height: 3;
        content-align: center middle;
        text-style: bold;
        border: solid $primary;
    }
    RegimeBar.bullish { background: $success 20%; color: $success; }
    RegimeBar.bearish { background: $error 20%;   color: $error;   }
    RegimeBar.neutral { background: $surface;     color: $text-muted; }
    """

    regime: reactive[int] = reactive(0)

    def on_mount(self) -> None:
        self._apply(self.regime)

    def watch_regime(self, value: int) -> None:
        self._apply(value)

    def _apply(self, value: int) -> None:
        self.remove_class("bullish", "bearish", "neutral")
        label = _REGIME_LABELS.get(value, "⚪  NEUTRAL")
        cls   = {1: "bullish", -1: "bearish"}.get(value, "neutral")
        self.add_class(cls)
        self.update(f"  SPY MARKET REGIME  ▸  {label}  ")


# ---------------------------------------------------------------------------
# Session stats bar
# ---------------------------------------------------------------------------

class SessionBar(Static):
    """One-line footer showing total signal count and session uptime."""

    DEFAULT_CSS = """
    SessionBar {
        height: 1;
        background: $surface;
        color: $text-muted;
        content-align: center middle;
    }
    """

    total_signals: reactive[int] = reactive(0)
    last_update:   reactive[str] = reactive("—")

    def __init__(self, start_time: datetime, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._start = start_time

    def on_mount(self) -> None:
        self._refresh()

    def watch_total_signals(self, _: int) -> None:
        self._refresh()

    def watch_last_update(self, _: str) -> None:
        self._refresh()

    def _refresh(self) -> None:
        elapsed = datetime.now(tz=timezone.utc) - self._start
        h, rem  = divmod(int(elapsed.total_seconds()), 3600)
        m, s    = divmod(rem, 60)
        self.update(
            f" Signals: [bold]{self.total_signals}[/]"
            f"  │  Uptime: {h:02d}:{m:02d}:{s:02d}"
            f"  │  Last bar: {self.last_update}"
            f"  │  [dim]/: search  Ctrl+C: quit[/] "
        )


# ---------------------------------------------------------------------------
# Per-ticker detail view (charts + stats)
# ---------------------------------------------------------------------------

class TickerDetailView(Vertical):
    """Full per-ticker layout: candle chart, volume, indicator row, stats."""

    DEFAULT_CSS = """
    TickerDetailView {
        layout: vertical;
    }
    #chart-row {
        height: 1fr;
    }
    #indicator-row {
        height: 12;
    }
    """

    def __init__(self, ticker: str) -> None:
        super().__init__(id=f"detail-{_ticker_id(ticker)}")
        self.ticker = ticker

    def compose(self) -> ComposeResult:
        t = self.ticker
        tid = _ticker_id(t)
        with Vertical(id="chart-row"):
            yield CandleChart(t, id=f"candle-{tid}")
            yield VolumeChart(t, id=f"volume-{tid}")
        with Horizontal(id="indicator-row"):
            yield RSIChart(t,  id=f"rsi-{tid}")
            yield MACDChart(t, id=f"macd-{tid}")
            yield ADXChart(t,  id=f"adx-{tid}")
        yield TickerStats(t, id=f"stats-{tid}")

    def update_bars(self, bars: list[dict], signal: int) -> None:
        tid = _ticker_id(self.ticker)
        try:
            self.query_one(f"#candle-{tid}", CandleChart).update_bars(bars)
        except NoMatches:
            pass
        try:
            self.query_one(f"#volume-{tid}", VolumeChart).update_bars(bars)
        except NoMatches:
            pass
        try:
            self.query_one(f"#rsi-{tid}", RSIChart).update_bars(bars)
        except NoMatches:
            pass
        try:
            self.query_one(f"#macd-{tid}", MACDChart).update_bars(bars)
        except NoMatches:
            pass
        try:
            self.query_one(f"#adx-{tid}", ADXChart).update_bars(bars)
        except NoMatches:
            pass
        try:
            self.query_one(f"#stats-{tid}", TickerStats).update_bar(bars[-1] if bars else {}, signal)
        except NoMatches:
            pass

    def bump_signal_count(self) -> None:
        tid = _ticker_id(self.ticker)
        try:
            self.query_one(f"#stats-{tid}", TickerStats).bump_signal_count()
        except NoMatches:
            pass


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class ScalpEdgeDashboard(App[None]):
    """Real-time Textual TUI that drives :class:`~scalpedge.live_engine.LiveSignalEngine`.

    Parameters
    ----------
    tickers:
        List of ticker symbols being monitored.
    engine:
        A fully initialised and seeded (but not yet running)
        :class:`~scalpedge.live_engine.LiveSignalEngine` instance.
    """

    TITLE    = "ScalpEdge — Live Signal Monitor"
    SUB_TITLE = "Real-time intraday signal dashboard"

    BINDINGS = [
        Binding("ctrl+c", "quit",        "Quit"),
        Binding("/",      "focus_search","Search ticker"),
        Binding("escape", "blur_search", "Close search", show=False),
    ]

    CSS = """
    Screen { layout: vertical; }

    #regime-bar { }

    #search-row {
        height: 3;
        align: center middle;
        background: $surface;
        border: solid $primary;
    }
    #search-label { width: auto; margin: 0 1; }
    #search-input { width: 24; }
    #search-hint  { width: auto; color: $text-muted; margin: 0 1; }

    #tabs { height: 1fr; }

    /* Overview grid: fixed height matches TickerOverviewPanel */
    #overview-grid { height: 16; }

    #log-header {
        height: 1;
        background: $primary;
        color: $background;
        content-align: center middle;
        text-style: bold;
    }
    #signal-log {
        height: 12;
        border: round $primary;
        margin: 0 1 0 1;
    }
    """

    def __init__(self, tickers: list[str], engine: "LiveSignalEngine") -> None:
        super().__init__()
        self.tickers = tickers
        self.engine  = engine
        self._session_start  = datetime.now(tz=timezone.utc)
        self._total_signals = 0
        # Rolling per-ticker bar history for chart rendering
        self._history: dict[str, collections.deque[dict]] = {
            t: collections.deque(maxlen=_CHART_HISTORY) for t in tickers
        }

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield RegimeBar(id="regime-bar")

        with Horizontal(id="search-row"):
            yield Label("🔍 Ticker:", id="search-label")
            yield Input(placeholder="SPY, TSLA …", id="search-input")
            yield Label(
                "Enter to jump · Tab to cycle · Esc to dismiss",
                id="search-hint",
            )

        with TabbedContent(id="tabs"):
            with TabPane("Overview", id="tab-overview"):
                with VerticalScroll():
                    with Horizontal(id="overview-grid"):
                        for ticker in self.tickers:
                            yield TickerOverviewPanel(ticker)

            for ticker in self.tickers:
                with TabPane(ticker, id=f"tab-{_ticker_id(ticker)}"):
                    yield TickerDetailView(ticker)

        yield Static(" 📋  Signal Log ", id="log-header")
        yield RichLog(id="signal-log", highlight=True, markup=True, auto_scroll=True)
        yield SessionBar(self._session_start, id="session-bar")
        yield Footer()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Seed chart history from the engine buffer, wire callbacks, start stream."""
        for ticker in self.tickers:
            buf_df = self.engine.get_buffer(ticker)
            if not buf_df.empty:
                for row in buf_df.tail(_CHART_HISTORY).to_dict("records"):
                    self._history[ticker].append(row)

        self.engine.on_signal     = self._handle_signal
        self.engine.on_bar_update = self._handle_bar_update
        self.run_worker(
            self._run_engine(),
            exclusive=True,
            group="live-engine",
            exit_on_error=False,
        )

    # ------------------------------------------------------------------
    # Engine callbacks (run inside Textual's event loop via worker)
    # ------------------------------------------------------------------

    async def _handle_signal(self, event: "SignalEvent") -> None:
        self.post_message(SignalFired(event))

    async def _handle_bar_update(self, ticker: str, bar: dict, last_signal: int) -> None:
        self.post_message(BarUpdated(ticker, bar, last_signal))

    async def _run_engine(self) -> None:
        """Run the live engine; surface errors to the signal log on failure."""
        try:
            await self.engine.run()
        except Exception as exc:  # noqa: BLE001
            logger.error("Live engine terminated with error: %s", exc, exc_info=True)
            try:
                self.query_one("#signal-log", RichLog).write(
                    f"[bold red]⚠  Engine error:[/]  {exc}\n"
                    "Verify [bold]POLYGON_API_KEY[/] is set and the network is reachable."
                )
            except NoMatches:
                pass

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def on_bar_updated(self, message: BarUpdated) -> None:
        """Append the bar to history, refresh charts + overview, update regime."""
        ticker = message.ticker
        bar    = message.bar
        sig    = message.last_signal

        # Accumulate history
        if ticker not in self._history:
            self._history[ticker] = collections.deque(maxlen=_CHART_HISTORY)
        self._history[ticker].append(bar)
        bars = list(self._history[ticker])

        # Update detail view
        tid = _ticker_id(ticker)
        try:
            self.query_one(f"#detail-{tid}", TickerDetailView).update_bars(bars, sig)
        except NoMatches:
            pass

        # Update overview panel
        try:
            self.query_one(
                f"#overview-{tid}", TickerOverviewPanel
            ).update_bar(bar, sig)
        except NoMatches:
            pass

        # Regime from SPY
        if ticker == "SPY":
            try:
                self.query_one("#regime-bar", RegimeBar).regime = (
                    _compute_regime_from_bar(bar)
                )
            except NoMatches:
                pass

        # Session-bar last-update timestamp
        try:
            self.query_one("#session-bar", SessionBar).last_update = (
                datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
            )
        except NoMatches:
            pass

    def on_signal_fired(self, message: SignalFired) -> None:
        """Log signal, bump counters."""
        event = message.event
        self._total_signals += 1

        try:
            self.query_one("#session-bar", SessionBar).total_signals = self._total_signals
        except NoMatches:
            pass

        tid = _ticker_id(event.ticker)
        try:
            self.query_one(f"#detail-{tid}", TickerDetailView).bump_signal_count()
        except NoMatches:
            pass
        try:
            self.query_one(f"#overview-{tid}", TickerOverviewPanel).bump_signal_count()
        except NoMatches:
            pass

        # Append to signal log
        try:
            log = self.query_one("#signal-log", RichLog)
        except NoMatches:
            return
        try:
            ts  = event.bar_time.strftime("%H:%M:%S") if event.bar_time else "--:--:--"
            ind = event.indicators
            rsi_str  = _fmt(ind.get("rsi_14"),  ".1f")
            macd_str = _fmt(ind.get("macd"),    "+.4f")
            adx_str  = _fmt(ind.get("adx_14"),  ".1f")
            vwap_v   = ind.get("vwap")
            price    = event.price
            vwap_str = "N/A"
            try:
                vf = float(vwap_v or 0)
                if vf != 0:
                    vwap_str = f"{(price - vf) / vf * 100:+.2f}%"
            except (TypeError, ValueError):
                pass
            sig   = event.signal
            color = _SIGNAL_COLORS.get(sig, "bright_black")
            lbl   = _SIGNAL_LABELS.get(sig, "⚪ FLAT")
            log.write(
                f"[dim]{ts}[/]  "
                f"[bold {color}]{lbl}[/]  "
                f"[bold]{event.ticker:<6}[/]  "
                f"@ [bold]${price:,.2f}[/]  "
                f"RSI:[cyan]{rsi_str}[/]  "
                f"MACD:[yellow]{macd_str}[/]  "
                f"ADX:[magenta]{adx_str}[/]  "
                f"VWAP Δ:[white]{vwap_str}[/]"
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to write signal to log", exc_info=True)

    # ------------------------------------------------------------------
    # Search / navigation
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Jump to the entered ticker's tab (case-insensitive, fuzzy prefix match)."""
        if event.input.id != "search-input":
            return
        query = event.value.strip().upper()
        event.input.clear()
        self.action_blur_search()

        if not query:
            return

        # Exact match first, then prefix match
        candidates = [t for t in self.tickers if t == query]
        if not candidates:
            candidates = [t for t in self.tickers if t.startswith(query)]
        if not candidates:
            candidates = [t for t in self.tickers if query in t]

        if candidates:
            tab_id = f"tab-{_ticker_id(candidates[0])}"
            try:
                self.query_one("#tabs", TabbedContent).active = tab_id
            except NoMatches:
                pass
        else:
            try:
                self.query_one("#signal-log", RichLog).write(
                    f"[yellow]⚠  Ticker [bold]{query}[/] is not being monitored.[/]"
                )
            except NoMatches:
                pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_focus_search(self) -> None:
        """Focus the search input."""
        try:
            self.query_one("#search-input", Input).focus()
        except NoMatches:
            pass

    def action_blur_search(self) -> None:
        """Return focus from the search input."""
        try:
            self.query_one("#search-input", Input).blur()
        except NoMatches:
            pass

    def action_quit(self) -> None:
        """Graceful shutdown."""
        self.exit()
