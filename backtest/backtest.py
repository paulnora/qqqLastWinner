import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Tuple

PriceField = Literal["open", "close"]
Rounding = Literal["floor", "nearest", "fractional"]
LeverageMode = Literal["3x", "1x"]

# -------------------------
# Configuration
# -------------------------
@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    start_date: str
    end_date: str
    initial_capital: float = 10_000.0
    trade_price: PriceField = "open"          # Execute at t+1 open or close
    share_rounding: Rounding = "floor"        # How to round share quantities
    leverage_mode: LeverageMode = "3x"        # "3x" = TQQQ/SQQQ, "1x" = QQQ long/short
    # Trading frictions — charged to cash (and therefore equity) at every
    # execution. Default 0.0 preserves legacy frictionless behavior; callers
    # that want realistic results (e.g. the research harness) set these.
    commission_bps: float = 0.0               # per-leg commission, basis points of notional
    slippage_bps: float = 0.0                 # per-leg slippage, basis points of notional
    borrow_bps_annual: float = 0.0            # annual short-borrow cost, bps of short proceeds

# -------------------------
# Helper Functions
# -------------------------
def _round_shares(x: float, mode: Rounding) -> float:
    """Round share quantities based on specified mode."""
    if mode == "fractional":
        return float(x)
    if mode == "nearest":
        return float(np.round(x))
    return float(np.floor(max(x, 0.0)))  # Non-negative floor


def _txn_cost(notional: float, cfg: "BacktestConfig") -> float:
    """Per-leg transaction cost (commission + slippage) on a traded notional."""
    return abs(notional) * (cfg.commission_bps + cfg.slippage_bps) / 10_000.0


def _borrow_cost(proceeds: float, entry_date, exit_date, cfg: "BacktestConfig") -> float:
    """Short-borrow cost accrued over the holding period of a short position."""
    if cfg.borrow_bps_annual <= 0:
        return 0.0
    days = max((exit_date - entry_date).days, 0)
    return abs(proceeds) * cfg.borrow_bps_annual / 10_000.0 * days / 365.25

def _build_trade_record(
    symbol: str,
    direction: str,
    is_short_sale: bool,
    trade: dict,
    exit_date: pd.Timestamp,
    exit_price: float,
    profit: float,
    profit_pct: float,
    exec_dates: pd.DatetimeIndex,
    share_rounding: str,
) -> dict:
    """Build the standard trade log entry dict."""
    return {
        "Symbol": symbol,
        "Direction": direction,
        "StartDate": trade["entry_date"],
        "EndDate": exit_date,
        "Duration": exec_dates.get_loc(exit_date) - exec_dates.get_loc(trade["entry_date"]),
        "BuyPrice": round(trade["entry_price"], 6),
        "SellPrice": round(exit_price, 6),
        "ShareSize": int(trade["shares"]) if share_rounding != "fractional" else trade["shares"],
        "Profit": round(profit, 6),
        "ProfitPercent": round(profit_pct, 6),
        "isProfitable": profit > 0,
        "isShortSale": is_short_sale,
    }


def _prepare_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare price data for backtesting.
    Expects columns: 'open', 'close' with DatetimeIndex.
    """
    out = df[["open", "close"]].copy()
    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)
    
    if out.isna().any().any():
        raise ValueError("Price data contains NaNs. Clean your data first.")
    
    return out

# -------------------------
# Strategy Output Processing
# -------------------------
def normalize_strategy_output(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Normalize strategy output to standard format.
    
    Input: DataFrame with 'date', 'longPositionPct', 'shortPositionPct', 'error'
    Output: (positions_df, errors_series)
    
    Uses generic column names w_long / w_short so the engine is instrument-agnostic.
    """
    out = df.copy()
    
    # Handle different error column names
    err_col = "error" if "error" in out.columns else ("Error" if "Error" in out.columns else None)
    
    # Set date as index
    if "date" in out.columns:
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)
    
    # Validate required columns
    if "longPositionPct" not in out.columns or "shortPositionPct" not in out.columns:
        raise ValueError("Strategy must provide 'longPositionPct' and 'shortPositionPct'")
    
    # Clean and validate position percentages
    out["longPositionPct"] = pd.to_numeric(out["longPositionPct"], errors="coerce").clip(0.0, 1.0)
    out["shortPositionPct"] = pd.to_numeric(out["shortPositionPct"], errors="coerce").clip(0.0, 1.0)
    
    # Create positions DataFrame — instrument-agnostic names
    positions = pd.DataFrame({
        "w_long": out["longPositionPct"],
        "w_short": out["shortPositionPct"]
    }, index=out.index)
    
    errors = out[err_col] if err_col else pd.Series(index=out.index, dtype="object")
    return positions, errors

# -------------------------
# Data Alignment and Execution Timing
# -------------------------
def align_data_and_shift_positions(
    positions: pd.DataFrame,
    long_prices: pd.DataFrame,
    short_prices: pd.DataFrame,
    cfg: BacktestConfig
) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align data to ETF trading calendar and shift positions for t+1 execution.
    
    This implements the core requirement: decisions made on day t are executed
    at the next trading day's open/close (t+1).
    
    In 3x mode: long_prices = TQQQ, short_prices = SQQQ
    In 1x mode: long_prices = QQQ,  short_prices = QQQ  (short-sell mechanics)
    """
    # Prepare price data
    long_px = _prepare_price_data(long_prices)
    short_px = _prepare_price_data(short_prices)
    
    # Define date range
    start, end = pd.to_datetime(cfg.start_date), pd.to_datetime(cfg.end_date)
    
    # Get common trading days for both instruments
    etf_calendar = long_px.index.intersection(short_px.index)
    etf_calendar = etf_calendar[(etf_calendar >= start) & (etf_calendar <= end)]
    
    # Filter price data to trading calendar
    long_px = long_px.loc[etf_calendar]
    short_px = short_px.loc[etf_calendar]
    
    # Align positions to ETF calendar (forward-fill to persist last target)
    aligned_positions = positions.reindex(etf_calendar).ffill().fillna(0.0)
    
    # Shift positions by +1 day for t+1 execution
    shifted_positions = aligned_positions.shift(1)
    execution_dates = etf_calendar[1:]  # First day has no prior signal
    
    # Filter to execution dates
    shifted_positions = shifted_positions.loc[execution_dates]
    long_px = long_px.loc[execution_dates]
    short_px = short_px.loc[execution_dates]
    
    # Ensure position weights are valid
    shifted_positions["w_long"] = shifted_positions["w_long"].clip(0.0, 1.0)
    shifted_positions["w_short"] = shifted_positions["w_short"].clip(0.0, 1.0)
    
    return execution_dates, shifted_positions, long_px, short_px

# -------------------------
# Core Backtesting Engine
# -------------------------
def run_backtest_with_strategy(
    strategy,
    tqqq_prices: pd.DataFrame,
    sqqq_prices: pd.DataFrame,
    cfg: BacktestConfig,
    contextData: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run backtest with the given strategy.
    
    Supports two leverage modes via cfg.leverage_mode:
      - "3x": Buys TQQQ for long, SQQQ for short (default, legacy behaviour)
      - "1x": Buys QQQ for long, short-sells QQQ for short
    
    For 1x mode, pass QQQ prices as *both* tqqq_prices and sqqq_prices.
    
    Returns:
        trades_df: Complete trade log with entry/exit details
        equity_df: Daily equity curve
        issues_df: Strategy errors and warnings
    """
    is_1x = cfg.leverage_mode == "1x"
    long_sym = "QQQ" if is_1x else "TQQQ"
    short_sym = "QQQ_SHORT" if is_1x else "SQQQ"

    # 1. Get strategy signals
    strategy_output = strategy.execute(
        startDate=cfg.start_date, 
        endDate=cfg.end_date, 
        contextData=contextData or {}
    )
    positions, errors = normalize_strategy_output(strategy_output)
    
    # 2. Align data and shift for t+1 execution
    exec_dates, positions, long_px, short_px = align_data_and_shift_positions(
        positions, tqqq_prices, sqqq_prices, cfg
    )
    
    # 3. Collect strategy errors
    issues_df = errors.reindex(exec_dates).dropna().to_frame(name="error").reset_index()
    issues_df.rename(columns={"index": "Date"}, inplace=True)
    
    # 4. Initialize backtest state
    cash = float(cfg.initial_capital)
    shares = {long_sym: 0.0, short_sym: 0.0}
    open_trades = {long_sym: None, short_sym: None}
    trade_log = []
    equity_log = []
    
    def calculate_equity(date: pd.Timestamp) -> float:
        """Calculate total portfolio equity at close prices."""
        eq = cash
        # Long position: value = shares * price
        eq += shares[long_sym] * float(long_px.loc[date, "close"])
        if is_1x:
            # Short position: value = short_proceeds - shares * current_price
            # shares[short_sym] stores the number of shares sold short.
            # open_trades[short_sym]["short_proceeds"] stores cash received at entry.
            if shares[short_sym] > 0 and open_trades[short_sym] is not None:
                short_value = (open_trades[short_sym]["short_proceeds"]
                               - shares[short_sym] * float(short_px.loc[date, "close"]))
                eq += short_value
        else:
            # 3x mode: SQQQ is a normal long position
            eq += shares[short_sym] * float(short_px.loc[date, "close"])
        return eq
    
    # 5. Execute trades only on signal changes
    prev_w_long = None
    prev_w_short = None
    
    for date in exec_dates:
        # Get target allocations and prices
        w_long = float(positions.loc[date, "w_long"])
        w_short = float(positions.loc[date, "w_short"])
        long_price = float(long_px.loc[date, cfg.trade_price])
        short_price = float(short_px.loc[date, cfg.trade_price])
        
        # Only trade if signal has changed
        if prev_w_long is not None and (w_long == prev_w_long and w_short == prev_w_short):
            # No signal change, just record equity
            equity_log.append({
                "Date": date,
                "Equity": calculate_equity(date),
                "Cash": cash,
                f"{long_sym}_Shares": shares[long_sym],
                f"{short_sym}_Shares": shares[short_sym]
            })
            continue
        
        # Signal has changed - execute trades
        # First, close all current positions
        # --- Close long position ---
        if shares[long_sym] > 0:
            sell_price = float(long_px.loc[date, cfg.trade_price])
            sell_value = shares[long_sym] * sell_price
            cash += sell_value
            cash -= _txn_cost(sell_value, cfg)

            if open_trades[long_sym] is not None:
                trade = open_trades[long_sym]
                profit = (sell_price - trade["entry_price"]) * trade["shares"]
                profit_pct = profit / (trade["entry_price"] * trade["shares"])
                
                trade_log.append(_build_trade_record(
                    long_sym, "LONG", False, trade, date, sell_price,
                    profit, profit_pct, exec_dates, cfg.share_rounding
                ))
                open_trades[long_sym] = None
            shares[long_sym] = 0.0

        # --- Close short position ---
        if shares[short_sym] > 0:
            cover_price = float(short_px.loc[date, cfg.trade_price])
            if is_1x:
                # Cover short: buy back shares, profit = proceeds - cover_cost
                cover_cost = shares[short_sym] * cover_price
                if open_trades[short_sym] is not None:
                    trade = open_trades[short_sym]
                    profit = trade["short_proceeds"] - cover_cost
                    profit_pct = profit / trade["short_proceeds"]
                    # Cash increases by the profit (proceeds were already accounted)
                    # At entry we had: cash += short_proceeds (selling shares)
                    # At exit we spend: cash -= cover_cost
                    # Net P&L = short_proceeds - cover_cost = profit
                    cash += trade["short_proceeds"]  # return the proceeds
                    cash -= cover_cost               # pay to buy back shares
                    cash -= _txn_cost(cover_cost, cfg)
                    cash -= _borrow_cost(trade["short_proceeds"], trade["entry_date"], date, cfg)

                    trade_log.append(_build_trade_record(
                        "QQQ", "SHORT", True, trade, date, cover_price,
                        profit, profit_pct, exec_dates, cfg.share_rounding
                    ))
                    open_trades[short_sym] = None
            else:
                # 3x mode: SQQQ is a normal long position, sell it
                sell_value = shares[short_sym] * cover_price
                cash += sell_value
                cash -= _txn_cost(sell_value, cfg)
                if open_trades[short_sym] is not None:
                    trade = open_trades[short_sym]
                    profit = (cover_price - trade["entry_price"]) * trade["shares"]
                    profit_pct = profit / (trade["entry_price"] * trade["shares"])

                    trade_log.append(_build_trade_record(
                        short_sym, "LONG", False, trade, date, cover_price,
                        profit, profit_pct, exec_dates, cfg.share_rounding
                    ))
                    open_trades[short_sym] = None
            shares[short_sym] = 0.0
        
        # Now open new positions based on target allocation
        current_equity = calculate_equity(date)
        
        # --- Open long position ---
        target_long_value = w_long * current_equity
        target_long_shares = _round_shares(target_long_value / max(long_price, 1e-9), cfg.share_rounding)
        
        if target_long_shares > 0:
            # Cap to what cash can actually buy
            affordable = cash / max(long_price, 1e-9)
            actual_shares = _round_shares(min(target_long_shares, affordable), cfg.share_rounding)
            if actual_shares > 0:
                trade_value = long_price * actual_shares
                cash -= trade_value
                cash -= _txn_cost(trade_value, cfg)
                shares[long_sym] = actual_shares
                open_trades[long_sym] = {
                    "symbol": long_sym,
                    "entry_date": date,
                    "entry_price": long_price,
                    "shares": actual_shares
                }
        
        # --- Open short position ---
        target_short_value = w_short * current_equity
        target_short_shares = _round_shares(target_short_value / max(short_price, 1e-9), cfg.share_rounding)
        
        if target_short_shares > 0:
            if is_1x:
                # Short-sell: we sell shares we don't own, receiving cash as proceeds.
                # We hold the proceeds separately and must buy back later.
                short_proceeds = short_price * target_short_shares
                # Don't add proceeds to cash yet — track them in the trade record.
                # Equity = cash + short_proceeds - current_short_value
                cash -= _txn_cost(short_proceeds, cfg)
                shares[short_sym] = target_short_shares
                open_trades[short_sym] = {
                    "symbol": short_sym,
                    "entry_date": date,
                    "entry_price": short_price,
                    "shares": target_short_shares,
                    "short_proceeds": short_proceeds  # value received from selling short
                }
            else:
                # 3x mode: buy SQQQ — cap to what cash can afford
                affordable = cash / max(short_price, 1e-9)
                actual_shares = _round_shares(min(target_short_shares, affordable), cfg.share_rounding)
                if actual_shares > 0:
                    trade_value = short_price * actual_shares
                    cash -= trade_value
                    cash -= _txn_cost(trade_value, cfg)
                    shares[short_sym] = actual_shares
                    open_trades[short_sym] = {
                        "symbol": short_sym,
                        "entry_date": date,
                        "entry_price": short_price,
                        "shares": actual_shares
                    }
        
        # Update previous positions for next iteration
        prev_w_long = w_long
        prev_w_short = w_short
        
        # Record daily equity
        equity_log.append({
            "Date": date,
            "Equity": calculate_equity(date),
            "Cash": cash,
            f"{long_sym}_Shares": shares[long_sym],
            f"{short_sym}_Shares": shares[short_sym]
        })
    
    # 6. Force-close remaining positions on final day
    final_date = exec_dates[-1]
    
    # Close long
    if shares[long_sym] > 0:
        final_price = float(long_px.loc[final_date, cfg.trade_price])
        final_value = final_price * shares[long_sym]
        cash += final_value
        cash -= _txn_cost(final_value, cfg)
        if open_trades[long_sym] is not None:
            trade = open_trades[long_sym]
            profit = (final_price - trade["entry_price"]) * trade["shares"]
            profit_pct = profit / (trade["entry_price"] * trade["shares"])
            trade_log.append(_build_trade_record(
                long_sym, "LONG", False, trade, final_date, final_price,
                profit, profit_pct, exec_dates, cfg.share_rounding
            ))
        shares[long_sym] = 0.0
    
    # Close short
    if shares[short_sym] > 0:
        final_price = float(short_px.loc[final_date, cfg.trade_price])
        if is_1x:
            if open_trades[short_sym] is not None:
                trade = open_trades[short_sym]
                cover_cost = shares[short_sym] * final_price
                profit = trade["short_proceeds"] - cover_cost
                profit_pct = profit / trade["short_proceeds"]
                cash += trade["short_proceeds"]
                cash -= cover_cost
                cash -= _txn_cost(cover_cost, cfg)
                cash -= _borrow_cost(trade["short_proceeds"], trade["entry_date"], final_date, cfg)
                trade_log.append(_build_trade_record(
                    "QQQ", "SHORT", True, trade, final_date, final_price,
                    profit, profit_pct, exec_dates, cfg.share_rounding
                ))
        else:
            final_value = final_price * shares[short_sym]
            cash += final_value
            cash -= _txn_cost(final_value, cfg)
            if open_trades[short_sym] is not None:
                trade = open_trades[short_sym]
                profit = (final_price - trade["entry_price"]) * trade["shares"]
                profit_pct = profit / (trade["entry_price"] * trade["shares"])
                trade_log.append(_build_trade_record(
                    short_sym, "LONG", False, trade, final_date, final_price,
                    profit, profit_pct, exec_dates, cfg.share_rounding
                ))
        shares[short_sym] = 0.0
    
    # 7. Create output DataFrames
    trades_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity_log).set_index("Date").sort_index()
    
    return trades_df, equity_df, issues_df
