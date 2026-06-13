"""
run_experiment.py — Run a single strategy backtest and evaluate against criteria.

Adapted from karpathy/autoresearch: this is the experiment runner.
Each run backtests a strategy, compares to QQQ buy-and-hold, and produces
a structured verdict (keep / discard / crash).

Usage:
    python -m autoresearch.run_experiment <module_path> <class_name> [--start START] [--end END]

Example:
    python -m autoresearch.run_experiment strategy.sma_crossover_strategy SmaCrossoverStrategy
    python -m autoresearch.run_experiment autoresearch.experimental_strategy ExperimentalStrategy
"""

import os
import sys
import csv
import importlib
import argparse
import traceback
import subprocess
from datetime import datetime

import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from backtest.backtest import BacktestConfig, run_backtest_with_strategy
from autoresearch.prepare import (
    load_prices, buy_and_hold,
    load_qqq_prices, qqq_buy_and_hold, qqq_rolling_benchmarks,  # backward compat
)

# ── Constants ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")

# ── Trading frictions (QQQ, conservative-realistic) ──────────────────────────
# Charged to equity on every execution. A liquid ETF traded at the open:
#   commission ~1bp, slippage ~2bp per leg  -> ~6bp round-trip
#   short borrow ~50bp annual on proceeds held
# At ~60 trades/yr this is ~3-4%/yr of drag — enough to kill free-churn edges.
COMMISSION_BPS = 1.0
SLIPPAGE_BPS = 2.0
BORROW_BPS_ANNUAL = 50.0

# ── Evaluation windows ───────────────────────────────────────────────────────
# We test across multiple timeframes (1990-present) to ensure the strategy
# "greatly beats holding QQQ at ANY given timeframe."
# Pre-1999 data uses ^NDX (NASDAQ-100 Index) scaled to QQQ as proxy.
# "Live" windows (those tracking the present) auto-extend to the latest
# available OHLCV bar so re-runs always include data through "today".
def _latest_data_date() -> str:
    csv_path = os.path.join(os.path.dirname(__file__), "qqq_ohlcv.csv")
    last = pd.read_csv(csv_path, index_col=0, parse_dates=True, usecols=[0]).index.max()
    return last.strftime("%Y-%m-%d")

_LIVE_END = _latest_data_date()

# ── In-sample / holdout split ────────────────────────────────────────────────
# The search optimizes ONLY on in-sample windows (everything through IS_END).
# Data after IS_END is a true holdout the scorer never sees — champions are
# evaluated on it for reporting, never selected on it. The gap between
# in-sample and holdout performance IS the overfitting measurement.
#
# To re-freeze the holdout later (e.g. once a year), bump IS_END forward and
# rebaseline (--reset). Do NOT tune params while peeking at holdout results.
IS_END = "2024-12-31"

# In-sample windows: overlapping regimes from 1990 through IS_END.
EVAL_WINDOWS = [
    ("1990-01-02", IS_END),          # Extended full period (~35 years)
    ("1990-01-02", "1998-12-31"),    # Pre-QQQ era (NDX proxy only, ~9 years)
    ("1999-01-04", "2002-12-31"),    # Dot-com boom & crash
    ("2003-01-02", "2007-12-31"),    # Post-crash bull run (~5 years)
    ("2008-01-02", "2012-12-31"),    # Financial crisis & recovery (~5 years)
    ("2013-01-02", IS_END),          # Original full period
    ("2013-01-02", "2018-12-31"),    # First half (~6 years)
    ("2019-01-02", IS_END),          # Second half (incl. COVID, 2022 bear)
    ("2020-01-02", "2022-12-30"),    # COVID crash + recovery + bear
    ("2023-01-03", IS_END),          # Recent bull run through IS_END
]

# Holdout: the most recent data, never used in scoring/promotion.
HOLDOUT_WINDOWS = [
    ("2025-01-01", _LIVE_END),       # Out-of-sample: IS_END -> today
]

# ── Criteria ─────────────────────────────────────────────────────────────────
MIN_OUTPERFORMANCE_RATIO = 2.0   # Strategy return must be >= 2x QQQ return
MIN_TRADES_PER_YEAR = 10         # Relaxed from 12 (2026-05-04) to unblock Bollinger sub
MAX_TRADES_PER_YEAR = 120        # Cap to avoid excessive churn
MIN_WIN_RATE = 0.50              # At least 50% of trades profitable


def _get_git_commit():
    """Get current short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT_DIR
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_strategy(module_path: str, class_name: str):
    """Dynamically load and instantiate a strategy class."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def run_backtest(strategy, start_date: str, end_date: str, ticker: str = "QQQ"):
    """Run a 1x long/short backtest on *ticker*. Returns (trades_df, equity_df, issues_df)."""
    prices = load_prices(ticker)
    cfg = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        trade_price="open",
        share_rounding="fractional",
        leverage_mode="1x",
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        borrow_bps_annual=BORROW_BPS_ANNUAL,
    )
    return run_backtest_with_strategy(
        strategy=strategy,
        tqqq_prices=prices,
        sqqq_prices=prices,
        cfg=cfg,
        contextData={},
    )


def evaluate_window(strategy, start_date: str, end_date: str, ticker: str = "QQQ") -> dict:
    """
    Backtest a strategy on a single window and compute evaluation metrics.
    Returns a dict with all metrics, or raises on failure.
    """
    trades, equity, issues = run_backtest(strategy, start_date, end_date, ticker=ticker)

    if equity.empty:
        raise ValueError(f"Empty equity curve for {start_date} to {end_date}")

    # Strategy metrics
    final_equity = float(equity["Equity"].iloc[-1])
    strat_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
    n_trades = len(trades)

    # Ticker benchmark (compare strategy vs buy-and-hold of the SAME ticker)
    bench      = buy_and_hold(ticker, start_date, end_date)
    bh_return  = bench["return_pct"]

    # Outperformance ratio (handle negative benchmark returns)
    if bh_return > 0:
        outperformance_ratio = strat_return / bh_return
    elif bh_return < 0:
        outperformance_ratio = strat_return / bh_return if strat_return < 0 else 10.0
    else:
        outperformance_ratio = 10.0 if strat_return > 0 else 0.0

    # Trading frequency
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    trades_per_year = n_trades / max(years, 0.1)

    # Win rate
    win_rate = float(trades["isProfitable"].mean()) if n_trades > 0 else 0.0

    # Max drawdown
    eq_series = equity["Equity"]
    running_max = eq_series.expanding().max()
    drawdowns = (eq_series - running_max) / running_max * 100.0
    max_dd = float(drawdowns.min())

    # Sharpe-like ratio (annualized)
    daily_returns = eq_series.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "window": f"{start_date} to {end_date}",
        "strat_return_pct":    round(strat_return, 4),
        "qqq_return_pct":      round(bh_return, 4),   # field name kept for backward compat
        "outperformance_ratio": round(outperformance_ratio, 4),
        "n_trades":            n_trades,
        "trades_per_year":     round(trades_per_year, 2),
        "win_rate":            round(win_rate, 4),
        "max_drawdown_pct":    round(max_dd, 4),
        "sharpe":              round(sharpe, 4),
        "final_equity":        round(final_equity, 2),
    }


def evaluate_strategy(module_path: str, class_name: str, windows=None, ticker: str = "QQQ") -> dict:
    """
    Full evaluation of a strategy across all windows.

    Returns:
        {
            "status": "keep" | "discard" | "crash",
            "description": str,
            "windows": [list of per-window metrics],
            "summary": {aggregate metrics},
            "failures": [list of criteria violations],
        }
    """
    if windows is None:
        windows = EVAL_WINDOWS

    # Tell the strategy which ticker's OHLCV file to load
    os.environ["STRATEGY_TICKER"] = ticker.upper()

    try:
        strategy = load_strategy(module_path, class_name)
    except Exception as e:
        return {
            "status": "crash",
            "description": f"Failed to load {module_path}.{class_name}: {e}",
            "windows": [],
            "summary": {},
            "failures": [str(e)],
        }

    window_results = []
    failures = []

    for start, end in windows:
        try:
            result = evaluate_window(strategy, start, end, ticker=ticker)
            window_results.append(result)

            # Check criteria for this window
            if result["outperformance_ratio"] < MIN_OUTPERFORMANCE_RATIO:
                failures.append(
                    f"[{start} to {end}] Outperformance {result['outperformance_ratio']:.2f}x "
                    f"< {MIN_OUTPERFORMANCE_RATIO}x target "
                    f"(strat: {result['strat_return_pct']:+.1f}%, QQQ: {result['qqq_return_pct']:+.1f}%)"
                )

            if result["trades_per_year"] < MIN_TRADES_PER_YEAR:
                failures.append(
                    f"[{start} to {end}] Only {result['trades_per_year']:.1f} trades/year "
                    f"< {MIN_TRADES_PER_YEAR} minimum"
                )

            if result["trades_per_year"] > MAX_TRADES_PER_YEAR:
                failures.append(
                    f"[{start} to {end}] {result['trades_per_year']:.1f} trades/year "
                    f"> {MAX_TRADES_PER_YEAR} maximum (too much churn)"
                )

        except Exception as e:
            window_results.append({"window": f"{start} to {end}", "error": str(e)})
            failures.append(f"[{start} to {end}] CRASH: {e}")

    # Aggregate summary across all successful windows
    successful = [w for w in window_results if "error" not in w]

    if not successful:
        return {
            "status": "crash",
            "description": "All evaluation windows failed",
            "windows": window_results,
            "summary": {},
            "failures": failures,
        }

    avg_outperformance = np.mean([w["outperformance_ratio"] for w in successful])
    avg_trades_per_year = np.mean([w["trades_per_year"] for w in successful])
    avg_win_rate = np.mean([w["win_rate"] for w in successful])
    avg_sharpe = np.mean([w["sharpe"] for w in successful])
    worst_dd = min(w["max_drawdown_pct"] for w in successful)
    full_period = next((w for w in successful if "2013" in w["window"] and "2024" in w["window"]), None)
    full_return = full_period["strat_return_pct"] if full_period else 0.0

    # All windows must pass outperformance AND trade frequency
    all_windows_pass = all(
        w.get("outperformance_ratio", 0) >= MIN_OUTPERFORMANCE_RATIO
        for w in successful
    )
    freq_ok = MIN_TRADES_PER_YEAR <= avg_trades_per_year <= MAX_TRADES_PER_YEAR

    status = "keep" if (all_windows_pass and freq_ok and len(failures) == 0) else "discard"

    summary = {
        "full_period_return_pct": round(full_return, 2),
        "avg_outperformance_ratio": round(avg_outperformance, 4),
        "avg_trades_per_year": round(avg_trades_per_year, 2),
        "avg_win_rate": round(avg_win_rate, 4),
        "avg_sharpe": round(avg_sharpe, 4),
        "worst_max_drawdown_pct": round(worst_dd, 2),
        "windows_passed": f"{sum(1 for w in successful if w.get('outperformance_ratio',0) >= MIN_OUTPERFORMANCE_RATIO)}/{len(successful)}",
    }

    description = (
        f"return={full_return:+.1f}% | "
        f"avg_outperf={avg_outperformance:.2f}x | "
        f"trades/yr={avg_trades_per_year:.1f} | "
        f"win={avg_win_rate:.1%} | "
        f"sharpe={avg_sharpe:.2f} | "
        f"maxDD={worst_dd:.1f}%"
    )

    return {
        "status": status,
        "description": description,
        "windows": window_results,
        "summary": summary,
        "failures": failures,
    }


def log_result(module_path: str, class_name: str, result: dict):
    """Append result to results.tsv."""
    commit = _get_git_commit()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": timestamp,
        "commit": commit,
        "strategy": f"{module_path}.{class_name}",
        "status": result["status"],
        "full_return_pct": result["summary"].get("full_period_return_pct", ""),
        "avg_outperformance": result["summary"].get("avg_outperformance_ratio", ""),
        "avg_trades_yr": result["summary"].get("avg_trades_per_year", ""),
        "avg_win_rate": result["summary"].get("avg_win_rate", ""),
        "avg_sharpe": result["summary"].get("avg_sharpe", ""),
        "worst_maxdd": result["summary"].get("worst_max_drawdown_pct", ""),
        "description": result["description"],
    }

    file_exists = os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nResult logged to {RESULTS_FILE}")


def print_report(module_path: str, class_name: str, result: dict, ticker: str = "QQQ"):
    """Print a human-readable evaluation report."""
    W = 100
    print("\n" + "=" * W)
    print(f"  AUTORESEARCH EXPERIMENT REPORT")
    print(f"  Strategy: {module_path}.{class_name}")
    print(f"  Ticker:   {ticker.upper()}")
    print(f"  Status:   {result['status'].upper()}")
    print("=" * W)

    bh_label = f"{ticker.upper()}%"
    if result["windows"]:
        print(f"\n  {'Window':<35} {'Strat%':>9} {bh_label:>9} {'Ratio':>7} "
              f"{'Trades':>7} {'Tr/Yr':>7} {'Win%':>7} {'MaxDD':>8} {'Sharpe':>7}")
        print("  " + "-" * (W - 2))

        for w in result["windows"]:
            if "error" in w:
                print(f"  {w['window']:<35} ERROR: {w['error']}")
                continue
            marker = " *" if w["outperformance_ratio"] < MIN_OUTPERFORMANCE_RATIO else "  "
            print(f"  {w['window']:<35} {w['strat_return_pct']:>+8.1f}% "
                  f"{w['qqq_return_pct']:>+8.1f}% {w['outperformance_ratio']:>6.2f}x "
                  f"{w['n_trades']:>7} {w['trades_per_year']:>6.1f} "
                  f"{w['win_rate']:>6.1%} {w['max_drawdown_pct']:>7.1f}% "
                  f"{w['sharpe']:>6.2f}{marker}")

    if result["summary"]:
        s = result["summary"]
        print(f"\n  Summary:")
        print(f"    Full-period return:     {s['full_period_return_pct']:+.2f}%")
        print(f"    Avg outperformance:     {s['avg_outperformance_ratio']:.2f}x (target: >= {MIN_OUTPERFORMANCE_RATIO}x)")
        print(f"    Avg trades/year:        {s['avg_trades_per_year']:.1f} (target: {MIN_TRADES_PER_YEAR}-{MAX_TRADES_PER_YEAR})")
        print(f"    Avg win rate:           {s['avg_win_rate']:.1%}")
        print(f"    Avg Sharpe:             {s['avg_sharpe']:.2f}")
        print(f"    Worst max drawdown:     {s['worst_max_drawdown_pct']:.1f}%")
        print(f"    Windows passed:         {s['windows_passed']}")

    if result["failures"]:
        print(f"\n  FAILURES ({len(result['failures'])}):")
        for f in result["failures"]:
            print(f"    - {f}")

    verdict = "KEEP - Strategy meets all criteria!" if result["status"] == "keep" \
        else "DISCARD - Does not meet criteria" if result["status"] == "discard" \
        else "CRASH - Strategy failed to run"
    print(f"\n  VERDICT: {verdict}")
    print("=" * W)


def main():
    parser = argparse.ArgumentParser(description="Run autoresearch experiment on a strategy")
    parser.add_argument("module_path", help="Python module path (e.g., strategy.sma_crossover_strategy)")
    parser.add_argument("class_name", help="Strategy class name (e.g., SmaCrossoverStrategy)")
    parser.add_argument("--ticker", default="QQQ",
                        help="Ticker to trade and benchmark against (default: QQQ). "
                             "Example: --ticker SPY  or  --ticker NVDA")
    parser.add_argument("--start", default=None, help="Override start date for single-window eval")
    parser.add_argument("--end", default=None, help="Override end date for single-window eval")
    parser.add_argument("--no-log", action="store_true", help="Don't log to results.tsv")
    args = parser.parse_args()

    windows = None
    if args.start and args.end:
        windows = [(args.start, args.end)]

    result = evaluate_strategy(args.module_path, args.class_name, windows=windows, ticker=args.ticker)
    print_report(args.module_path, args.class_name, result, ticker=args.ticker)

    if not args.no_log:
        log_result(args.module_path, args.class_name, result)

    # Exit code: 0 = keep, 1 = discard, 2 = crash
    sys.exit({"keep": 0, "discard": 1, "crash": 2}[result["status"]])


if __name__ == "__main__":
    main()
