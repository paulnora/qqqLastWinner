"""
webapp/signals.py — Compute Iter31 signal breakdown for a list of tickers.
Called by app.py; can also be run standalone for debugging.
"""

import os
import sys
import json
import time
import urllib.request
from datetime import datetime, date

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np

AUTORESEARCH_DIR = os.path.join(ROOT_DIR, "autoresearch")
TICKERS = ["QQQ"]

PERIOD1 = 631152000  # 1990-01-01


# ── Data refresh ──────────────────────────────────────────────────────────────

def _fetch_ohlcv(ticker: str) -> pd.DataFrame:
    period2 = int((datetime.now()).timestamp()) + 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={PERIOD1}&period2={period2}&interval=1d&events=adjsplit"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    result = data["chart"]["result"][0]
    ts     = result["timestamp"]
    quotes = result["indicators"]["quote"][0]
    adjc   = result["indicators"].get("adjclose", [{}])[0].get("adjclose", None)

    dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
    df = pd.DataFrame({
        "open": quotes["open"], "high": quotes["high"],
        "low":  quotes["low"],  "close": quotes["close"],
        "volume": quotes["volume"],
    }, index=dates)
    df.index.name = "Date"

    if adjc is not None:
        ratio = pd.Series(adjc, index=dates) / df["close"]
        for col in ["open", "high", "low", "close"]:
            df[f"{col}_adj"] = (df[col] * ratio).round(6)
    else:
        for col in ["open", "high", "low", "close"]:
            df[f"{col}_adj"] = df[col].round(6)

    return df[["open_adj", "high_adj", "low_adj", "close_adj", "volume"]].dropna(subset=["close_adj"])


def refresh_ticker(ticker: str) -> bool:
    """Fetch latest OHLCV for ticker and MERGE with existing history.

    Yahoo only returns QQQ from its 1999-03-10 inception, so a naive overwrite
    destroys any pre-1999 NDX-proxy history that extends the backtest to 1990.
    Instead we preserve rows older than the fetch and re-anchor them onto the
    fresh adjusted-close basis (Yahoo's back-adjustment drifts as dividends
    accrue), keeping the 1999 join seamless across refreshes. Returns True on
    success.
    """
    path = os.path.join(AUTORESEARCH_DIR, f"{ticker.lower()}_ohlcv.csv")
    try:
        fresh = _fetch_ohlcv(ticker)
        if os.path.exists(path):
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            existing.index = pd.to_datetime(existing.index).tz_localize(None).normalize()
            join = fresh.index.min()
            older = existing.loc[existing.index < join].copy()
            if len(older) and join in existing.index and existing.loc[join, "close_adj"]:
                # Re-anchor preserved history to the new basis so the boundary
                # doesn't drift each refresh.
                drift = float(fresh.loc[join, "close_adj"]) / float(existing.loc[join, "close_adj"])
                for col in ("open_adj", "high_adj", "low_adj", "close_adj"):
                    if col in older.columns:
                        older[col] = (older[col] * drift).round(6)
                older = older[fresh.columns]  # align column order/set
                combined = pd.concat([older, fresh])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            else:
                combined = fresh
        else:
            combined = fresh
        combined.to_csv(path)
        return True
    except Exception:
        return False


# ── Signal computation ────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)


def compute_signal(ticker: str) -> dict:
    from autoresearch.experimental_strategy import _load_ohlcv, _compute_mfi_cmf, _compute_adx, _OHLCV_CACHE
    from autoresearch.prepare import load_prices

    os.environ["STRATEGY_TICKER"] = ticker.upper()
    _OHLCV_CACHE.clear()

    prices = load_prices(ticker)
    close  = prices["close"].astype(float)

    # ── Indicators ────────────────────────────────────────────────────────────
    sma50   = close.rolling(50).mean()
    sma200  = close.rolling(200).mean()
    sma27   = close.rolling(27).mean()
    ema5    = close.ewm(span=5,  adjust=False).mean()
    ema8    = close.ewm(span=8,  adjust=False).mean()
    ema13   = close.ewm(span=13, adjust=False).mean()
    ema17   = close.ewm(span=17, adjust=False).mean()
    ema34   = close.ewm(span=34, adjust=False).mean()
    roll252 = close.rolling(252).max()

    rsi2  = _rsi(close, 2)
    rsi3  = _rsi(close, 3)
    pv    = close / sma200.replace(0, np.nan)
    mom5  = close / close.shift(5)  - 1
    mom20 = close / close.shift(20) - 1
    dd_yr = (close - roll252) / roll252 * 100

    ohlcv = _load_ohlcv(ticker)
    mfi, cmf = _compute_mfi_cmf(ohlcv, close.index)
    adx, plus_di, minus_di = _compute_adx(ohlcv, close.index)

    # ── Regime ────────────────────────────────────────────────────────────────
    in_bull      = sma50 > sma200
    genuine_bear = ~in_bull & (dd_yr < -15)
    bear_short   = genuine_bear & (ema13 < ema34) & (mom20 <= 0) & (cmf <= 0) & (rsi3 < 85)
    shallow_rec  = (~in_bull) & (~genuine_bear) & (ema5 > ema8) & (mom5 > 0)

    def g(s): return float(s.iloc[-1])

    # ── Sub-strategy votes ────────────────────────────────────────────────────
    strong_up = bool((g(adx) > 25) and (g(plus_di) > g(minus_di)))
    rsi3_thr  = 85 if g(pv) >= 1.15 else 90

    # Sub-A
    if not g(in_bull):
        subA = -1 if g(bear_short) else 0
        subA_reason = "Not in bull — bear rules"
    elif g(rsi3) > rsi3_thr:
        subA = 0
        subA_reason = f"RSI3={g(rsi3):.1f} > {rsi3_thr} — overbought exit"
    else:
        subA = 1
        subA_reason = f"RSI3={g(rsi3):.1f} ≤ {rsi3_thr} — no exit signal"

    # Sub-B / Sub-D
    subB = 1 if g(in_bull) else (-1 if g(bear_short) else 0)
    subB_reason = "SMA50 > SMA200" if g(in_bull) else "Not in bull"
    subD, subD_reason = subB, subB_reason

    # Sub-C
    recent_rsi2 = (rsi2.iloc[-12:-1] < 10) & in_bull.iloc[-12:-1]
    recent_mom5 = (mom5.iloc[-11:-1] < -0.05) & in_bull.iloc[-11:-1]
    subC_hold   = bool(recent_rsi2.any() or recent_mom5.any())
    if not g(in_bull):
        subC = -1 if g(bear_short) else 0
        subC_reason = "Not in bull"
    elif subC_hold:
        subC = 1
        trigger = "RSI2<10" if recent_rsi2.any() else "Mom5<-5%"
        subC_reason = f"Hold active ({trigger} triggered within 11 days)"
    else:
        subC = 0
        subC_reason = "No recent dip trigger"

    # Sub-E
    if g(in_bull):
        if g(pv) >= 1.10:
            subE = 1 if g(close) > g(sma27) else 0
            subE_reason = f"pv={g(pv):.3f}≥1.10 → SMA27 branch; close {'>' if subE else '≤'} SMA27"
        else:
            subE = 1 if g(close) > g(ema17) else 0
            subE_reason = f"pv={g(pv):.3f}<1.10 → EMA17 branch; close {'>' if subE else '≤'} EMA17"
    else:
        subE = -1 if g(bear_short) else 0
        subE_reason = "Not in bull"

    # Sub-F
    MFI_OB = 63
    if g(in_bull):
        if g(pv) < 1.06:
            subF = 1
            subF_reason = f"pv={g(pv):.3f}<1.06 → MFI bypass (near SMA200 support)"
        else:
            mfi_ok = g(mfi) <= MFI_OB
            subF = 1 if mfi_ok else 0
            subF_reason = f"pv={g(pv):.3f}≥1.06 → MFI={g(mfi):.1f} {'≤63 OK' if mfi_ok else '>63 overbought exit'}"
    else:
        subF = -1 if g(bear_short) else 0
        subF_reason = "Not in bull"

    votes = [subA, subB, subC, subD, subE, subF]
    vote_long  = sum(1 for v in votes if v == 1)
    vote_short = sum(1 for v in votes if v == -1)

    if vote_long >= 5:
        final = "LONG"
    elif vote_short >= 2:
        final = "SHORT"
    else:
        final = "CASH"

    prev_close = float(close.iloc[-2]) if len(close) > 1 else g(close)
    day_chg    = (g(close) - prev_close) / prev_close * 100

    return {
        "ticker":   ticker,
        "as_of":    str(close.index[-1].date()),
        "close":    round(g(close), 2),
        "day_chg":  round(day_chg, 2),
        "final":    final,
        # Regime
        "in_bull":       bool(g(in_bull)),
        "genuine_bear":  bool(g(genuine_bear)),
        "bear_short":    bool(g(bear_short)),
        "shallow_rec":   bool(g(shallow_rec)),
        # Key indicators
        "sma50":     round(g(sma50), 2),
        "sma200":    round(g(sma200), 2),
        "pv":        round(g(pv), 4),
        "dd_yr":     round(g(dd_yr), 2),
        "rsi2":      round(g(rsi2), 1),
        "rsi3":      round(g(rsi3), 1),
        "mfi":       round(g(mfi), 1),
        "cmf":       round(g(cmf), 4),
        "adx":       round(g(adx), 1),
        "plus_di":   round(g(plus_di), 1),
        "minus_di":  round(g(minus_di), 1),
        "mom5":      round(g(mom5) * 100, 2),
        "mom20":     round(g(mom20) * 100, 2),
        "ema13_gt_ema34": bool(g(ema13) > g(ema34)),
        "strong_up": strong_up,
        "rsi3_thr":  rsi3_thr,
        # Votes
        "votes": {
            "A": {"vote": subA, "reason": subA_reason, "label": "Adaptive Exit"},
            "B": {"vote": subB, "reason": subB_reason, "label": "Always Long"},
            "C": {"vote": subC, "reason": subC_reason, "label": "Dip Buyer"},
            "D": {"vote": subD, "reason": subD_reason, "label": "Always Long #2"},
            "E": {"vote": subE, "reason": subE_reason, "label": "Adaptive MA"},
            "F": {"vote": subF, "reason": subF_reason, "label": "MFI / Volume"},
        },
        "vote_long":  vote_long,
        "vote_short": vote_short,
        # Chart payload — equity curve + markers from 2007-present.
        "chart":      _compute_iter31_chart(ticker, close, prices.index[-1]),
    }


def _compute_iter31_chart(ticker: str, close, as_of):
    """Run Iter31 (ExperimentalStrategy) once to get the position series, then
    delegate to the shared chart builder."""
    from autoresearch.experimental_strategy import ExperimentalStrategy
    from autoresearch.run_experiment import load_strategy
    strat = load_strategy("autoresearch.experimental_strategy", "ExperimentalStrategy")
    result_df = strat.execute("2007-01-03", as_of.strftime("%Y-%m-%d"), {})
    return build_strategy_chart(ticker, result_df, close, as_of)


def build_strategy_chart(ticker: str, result_df, close, as_of, INITIAL=10000.0) -> dict:
    """Strategy-agnostic chart data builder. Takes the result_df from
    strategy.execute() and produces DAILY equity + B&H curves plus paired
    entry/exit markers from 2007 to as_of.

    Daily (not weekly) resolution so the user can zoom into a single month
    and still see every transition at its true date. Markers come in
    matched pairs: every entry-long/entry-short is followed by exactly
    one exit marker carrying the realized P&L of that round trip."""
    import pandas as pd
    chart_start = pd.Timestamp("2007-01-01")
    chart_idx = close.index[(close.index >= chart_start) & (close.index <= as_of)]
    # Position series from result_df (-1 / 0 / 1 per day)
    pos = pd.Series(0, index=result_df.index, dtype=int)
    pos[result_df["longPositionPct"] >= 1.0] = 1
    pos[result_df["shortPositionPct"] >= 1.0] = -1
    chart_pos = pos.reindex(chart_idx, method="ffill").fillna(0).astype(int)
    chart_close = close.reindex(chart_idx).astype(float)

    # Daily equity curve (mark-to-market on close prices, prior day's position)
    strat_equity = [INITIAL]
    prev_close = float(chart_close.iloc[0])
    for i in range(1, len(chart_close)):
        cur = float(chart_close.iloc[i])
        ret = cur / prev_close - 1
        prev_pos = int(chart_pos.iloc[i - 1])
        if prev_pos == 1:
            strat_equity.append(strat_equity[-1] * (1 + ret))
        elif prev_pos == -1:
            strat_equity.append(strat_equity[-1] * (1 - ret))
        else:
            strat_equity.append(strat_equity[-1])
        prev_close = cur
    bh_equity = (chart_close / float(chart_close.iloc[0]) * INITIAL).tolist()

    # Build paired entry/exit markers. Walk daily positions; whenever the
    # position changes, emit appropriate markers and (on exit) carry the
    # realized P&L vs the matched entry.
    markers = []
    open_trade = None   # dict | None
    prev_pos = 0
    date_strs = [d.strftime("%Y-%m-%d") for d in chart_idx]

    def _emit_exit(i_close, exit_kind_suffix):
        nonlocal open_trade
        if open_trade is None:
            return
        entry_eq = open_trade["entry_equity"]
        exit_eq  = strat_equity[i_close]
        pnl_pct  = (exit_eq / entry_eq - 1) * 100.0
        markers.append({
            "x": date_strs[i_close],
            "y": round(exit_eq, 2),
            "kind": "exit",
            "side": open_trade["side"],         # "long" / "short" — which trade closed
            "entry_date": open_trade["entry_date"],
            "exit_date": date_strs[i_close],
            "entry_price": round(open_trade["entry_price"], 2),
            "exit_price": round(float(chart_close.iloc[i_close]), 2),
            "entry_equity": round(entry_eq, 2),
            "exit_equity": round(exit_eq, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_abs": round(exit_eq - entry_eq, 2),
            "duration": i_close - open_trade["entry_index"],
            "profitable": exit_eq > entry_eq,
        })
        open_trade = None

    def _emit_entry(i, side):
        nonlocal open_trade
        kind = "entry-long" if side == "long" else "entry-short"
        markers.append({
            "x": date_strs[i],
            "y": round(strat_equity[i], 2),
            "kind": kind,
            "side": side,
            "entry_date": date_strs[i],
            "entry_price": round(float(chart_close.iloc[i]), 2),
            "entry_equity": round(strat_equity[i], 2),
        })
        open_trade = {
            "side": side,
            "entry_date": date_strs[i],
            "entry_price": float(chart_close.iloc[i]),
            "entry_equity": strat_equity[i],
            "entry_index": i,
        }

    for i in range(len(chart_idx)):
        p = int(chart_pos.iloc[i])
        if p == prev_pos:
            continue
        # Position changed. If we had an open trade, close it first.
        if prev_pos != 0:
            _emit_exit(i, "long" if prev_pos == 1 else "short")
        # Then if the new position is non-flat, open a new trade.
        if p == 1:
            _emit_entry(i, "long")
        elif p == -1:
            _emit_entry(i, "short")
        prev_pos = p

    # If a trade is still open at as_of, force-close it on the last bar so
    # every entry has a matching exit visible on the chart.
    if open_trade is not None:
        _emit_exit(len(chart_idx) - 1, "long" if prev_pos == 1 else "short")

    return {
        "dates":        date_strs,
        "close":        [round(v, 2) for v in chart_close.tolist()],
        "positions":    chart_pos.tolist(),
        "strat_equity": [round(v, 2) for v in strat_equity],
        "bh_equity":    [round(v, 2) for v in bh_equity],
        "markers":      markers,
        "initial":      INITIAL,
    }


def compute_signal_for_strategy(ticker: str, strategy_spec: dict) -> dict:
    """Generic signal computation for any strategy (non-Iter31). Returns the
    minimal shape plus a strategy-detail payload (recent position transitions,
    param diffs vs iter31, regime snapshot, and strategy metrics) so the page
    doesn't look empty for non-Iter31 strategies."""
    from autoresearch.prepare import load_prices
    from webapp.strategy_registry import instantiate_strategy
    from autoresearch.harness.parametric_strategy import ITER31_DEFAULTS
    import pandas as pd
    import numpy as np

    os.environ["STRATEGY_TICKER"] = ticker.upper()
    prices = load_prices(ticker)
    close = prices["close"].astype(float)
    as_of = prices.index[-1]
    # Long-horizon view: 2007-01-01 through latest available data.
    start_dt = "2007-01-01"
    end_dt = as_of.strftime("%Y-%m-%d")

    strat = instantiate_strategy(strategy_spec, ticker=ticker)
    result_df = strat.execute(start_dt, end_dt, {})

    last = result_df.iloc[-1]
    long_pct = float(last.get("longPositionPct", 0.0))
    short_pct = float(last.get("shortPositionPct", 0.0))
    if long_pct >= 1.0:
        final = "LONG"
    elif short_pct >= 1.0:
        final = "SHORT"
    else:
        final = "CASH"

    # Build a "position" series from result_df: +1 LONG / -1 SHORT / 0 CASH
    pos = pd.Series(0, index=result_df.index, dtype=int)
    pos[result_df["longPositionPct"] >= 1.0] = 1
    pos[result_df["shortPositionPct"] >= 1.0] = -1

    # Extract recent transitions (last 8 changes) for the detail panel
    transitions = []
    prev = 0
    for date, p in pos.items():
        if p != prev:
            transitions.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "from": {1: "LONG", -1: "SHORT", 0: "CASH"}[prev],
                "to": {1: "LONG", -1: "SHORT", 0: "CASH"}[int(p)],
                "close": round(float(close.reindex([date]).iloc[-1]), 2)
                         if date in close.index else None,
            })
            prev = int(p)
    recent_transitions = transitions[-8:][::-1]  # most recent first

    # Days in current position (since last transition)
    last_change_idx = None
    for i in range(len(pos) - 1, 0, -1):
        if pos.iloc[i] != pos.iloc[i - 1]:
            last_change_idx = i
            break
    days_in_position = (pos.index[-1] - pos.index[last_change_idx]).days if last_change_idx else 0

    # Regime snapshot — pure price-data indicators, strategy-agnostic
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    in_bull = bool(sma50 > sma200) if not pd.isna(sma50) and not pd.isna(sma200) else None

    def _rsi_local(s: pd.Series, period: int) -> float:
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
        return float(rsi.iloc[-1])

    rsi2 = _rsi_local(close, 2)
    rsi14 = _rsi_local(close, 14)
    mom5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0.0
    mom20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0.0
    dd_yr = float((close.iloc[-1] / close.rolling(252).max().iloc[-1] - 1) * 100) if len(close) > 252 else 0.0

    regime = {
        "in_bull": in_bull,
        "sma50": float(sma50) if not pd.isna(sma50) else None,
        "sma200": float(sma200) if not pd.isna(sma200) else None,
        "rsi2": round(rsi2, 1),
        "rsi14": round(rsi14, 1),
        "mom5": round(mom5, 2),
        "mom20": round(mom20, 2),
        "dd_yr": round(dd_yr, 2),
    }

    # Param diffs vs Iter31 — only meaningful when params dict is provided
    diffs = []
    params = strategy_spec.get("params") or {}
    for k, v in sorted(params.items()):
        if ITER31_DEFAULTS.get(k) != v:
            diffs.append({"key": k, "iter31": ITER31_DEFAULTS.get(k), "current": v})

    # Active subs row
    active_subs = ["A", "B", "C", "D", "E", "F"]
    for ch in "GHIJKL":
        if int(params.get(f"enable_{ch}", 0)) == 1:
            active_subs.append(ch)

    # Position stats over the eval window
    long_days = int((pos == 1).sum())
    short_days = int((pos == -1).sum())
    cash_days = int((pos == 0).sum())
    total_days = max(len(pos), 1)

    # Chart data — full 2007-onwards window via shared helper.
    chart = build_strategy_chart(ticker, result_df, close, as_of)

    return {
        "ticker": ticker,
        "close": float(close.iloc[-1]),
        "day_chg": float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0.0,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "final": final,
        "vote_long": None,
        "vote_short": None,
        "sub_votes": {},
        "is_generic": True,
        "strategy_label": strategy_spec.get("label", strategy_spec.get("id", "?")),
        # Strategy-detail payload for the rendered panel:
        "recent_transitions": recent_transitions,
        "days_in_position": days_in_position,
        "regime": regime,
        "diffs": diffs,
        "n_diffs": len(diffs),
        "active_subs": active_subs,
        "position_mix": {
            "long_pct": round(long_days / total_days * 100, 1),
            "short_pct": round(short_days / total_days * 100, 1),
            "cash_pct": round(cash_days / total_days * 100, 1),
            "window_years": round((pos.index[-1] - pos.index[0]).days / 365.25, 1),
        },
        "chart": chart,
        "metrics": strategy_spec.get("metrics"),  # composite/sharpe/etc from registry
    }


def get_all_signals(refresh_data: bool = False, strategy_id: str = "iter31") -> dict:
    from webapp.strategy_registry import find_strategy

    if refresh_data:
        for ticker in TICKERS:
            refresh_ticker(ticker)
            time.sleep(0.5)

    strategy_spec = find_strategy(strategy_id)
    is_iter31 = strategy_spec["source"] == "iter31"

    results = {}
    for ticker in TICKERS:
        try:
            results[ticker] = compute_signal(ticker) if is_iter31 \
                else compute_signal_for_strategy(ticker, strategy_spec)
        except Exception as e:
            results[ticker] = {"ticker": ticker, "error": str(e), "final": "ERROR"}

    return {
        "tickers": results,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": {
            "id": strategy_spec["id"], "label": strategy_spec["label"],
            "source": strategy_spec["source"], "description": strategy_spec["description"],
        },
    }
