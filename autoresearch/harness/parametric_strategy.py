"""
parametric_strategy.py — Extended ensemble with all knobs exposed.

Original 6-sub Iter31 ensemble (Sub-A..F) PLUS 5 new opt-in subs (Sub-G..K)
covering MACD, KDJ, Bollinger, Williams %R + OBV, and Donchian + Aroon.

Each new sub is multi-condition (memory rule: ensemble-only — no
single-indicator rules), and is gated by an enable_X flag defaulting to 0.
With all new subs disabled, the ensemble produces identical signals to the
preserved Iter31/gen3 baseline. The harness mutates enable_X flags and
threshold knobs to discover whether activating a new sub improves the
strict-Pareto MinRatio across the 10 evaluation windows.

Reads parameters from env-var HARNESS_PARAMS_FILE (JSON). Missing keys fall
back to defaults below.
"""

import os
import json
import pandas as pd
import numpy as np
from strategy.base_strategy import BaseStrategy
from autoresearch.harness import indicators as ind


_OHLCV_PATH = os.path.join(os.path.dirname(__file__), "..", "qqq_ohlcv.csv")
_OHLCV_CACHE: pd.DataFrame = None


ITER31_DEFAULTS = {
    "vote_long": 5,
    "vote_short": 2,
    "mfi_ob": 63,
    "subF_pv_bypass": 1.06,
    "subE_pv_split": 1.10,
    "subE_sma_period": 27,
    "subE_ema_period": 17,
    "subA_pv_high": 1.20,
    "subA_pv_mid": 1.10,
    "subA_pv_rsi_low": 1.15,
    "subA_rsi3_low": 85,
    "subA_rsi3_high": 90,
    "subA_strong_adx": 25,
    "subA_cash_high": 2,
    "subA_cash_mid": 1,
    "subA_cash_low": 3,
    "subC_rsi2_thresh": 10,
    "subC_rsi2_hold": 11,
    "subC_mom5_thresh": -0.05,
    "subC_mom5_hold": 10,
    "regime_dd_thresh": -15,
    "bear_short_rsi3": 85,
    "shallow_rsi3_long": 10,
    "sma200_period": 200,
    "sma50_period": 50,

    # --- Sub-G: MACD trend confirmation (multi-condition) -----------------
    # LONG when in_bull AND MACD > signal AND histogram > hist_thresh
    "enable_G": 0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal_period": 9,
    "macd_hist_thresh": 0.0,

    # --- Sub-H: KDJ momentum (multi-condition: oversold-dip OR midzone-up) -
    # LONG when in_bull AND ((J < kdj_oversold) OR (K>D AND J>kdj_neutral_low AND J<kdj_overbought))
    "enable_H": 0,
    "kdj_period": 9,
    "kdj_oversold": 20,
    "kdj_neutral_low": 30,
    "kdj_overbought": 80,

    # --- Sub-I: Bollinger middle-band trend (multi-condition) ---------------
    # LONG when in_bull AND close>mid AND close<upper AND bandwidth>bw_min
    "enable_I": 0,
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_bandwidth_min": 0.04,

    # --- Sub-J: Williams %R + OBV (multi-condition) -------------------------
    # LONG when in_bull AND wr>wr_low AND wr<wr_high AND OBV>OBV_SMA
    "enable_J": 0,
    "wr_period": 14,
    "wr_low": -80,
    "wr_high": -20,
    "obv_sma_period": 20,

    # --- Sub-K: Donchian + Aroon breakout (multi-condition) -----------------
    # LONG when in_bull AND close > don_low + don_pos*don_range AND aroon_up>aroon_down
    "enable_K": 0,
    "don_period": 20,
    "don_pos": 0.5,
    "aroon_period": 25,

    # --- Sub-L: 252d Z-score mean-reversion (kept hypothesis) ----------------
    # Hypothesis (BH+sign-agreement kept, holdout rho=-0.21, p=0.001):
    #   high zscore_252 of close predicts negative forward 63d return.
    # Strategy interpretation: vote LONG in bull regime when zscore is not
    # statistically overheated; go to CASH (vote 0) when zscore > threshold.
    # Optional oversold-boost: vote LONG even outside bull when zscore is
    # deeply negative (mean-revert from extreme drawdown).
    "enable_L": 0,
    "subL_zscore_period": 252,
    "subL_zscore_high": 2.0,
    "subL_zscore_low": -1.5,

    # --- Sub-M: long-horizon return reversal (kept hypothesis: STRONGEST family) -
    # ~12 independent papers (Fama-French 1988, Poterba-Summers 1988, Balvers/Wu,
    # long-run reversal literature): the multi-year trailing log return NEGATIVELY
    # predicts forward 63d return (holdout rho -0.28 to -0.32, p=0.001). Nothing in
    # subs A-L looks past 252 days, so this entire signal family was unrepresented.
    # Strategy interpretation (multi-condition):
    #   overextended = (long-horizon log-return > ret_high) AND (long z-score > z_high)
    #     -> in bull, abstain to CASH (anticipate reversion of the multi-year run-up)
    #   depressed = long-horizon log-return < ret_low
    #     -> vote LONG even outside bull (anticipate post-washout rebound)
    "enable_M": 0,
    "subM_lookback": 504,          # ~2yr trailing log-return window (strongest)
    "subM_ret_high": 0.80,         # log-return above this = overextended
    "subM_ret_low": -0.30,         # log-return below this = deeply depressed
    "subM_z_period": 504,          # z-score window for overextension confirmation
    "subM_z_high": 1.5,            # z-score threshold confirming overextension
}


def load_params() -> dict:
    path = os.environ.get(
        "HARNESS_PARAMS_FILE",
        os.path.join(os.path.dirname(__file__), "current_params.json"),
    )
    params = dict(ITER31_DEFAULTS)
    try:
        with open(path) as f:
            user = json.load(f)
        params.update({k: v for k, v in user.items() if k in ITER31_DEFAULTS})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return params


def _load_ohlcv() -> pd.DataFrame:
    global _OHLCV_CACHE
    if _OHLCV_CACHE is None:
        df = pd.read_csv(_OHLCV_PATH, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        _OHLCV_CACHE = df
    return _OHLCV_CACHE


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)


def _compute_mfi_cmf(ohlcv, idx):
    H = ohlcv["high_adj"].astype(float)
    L = ohlcv["low_adj"].astype(float)
    C = ohlcv["close_adj"].astype(float)
    V = ohlcv["volume"].astype(float)
    TP = (H + L + C) / 3.0
    raw = TP * V
    pos = raw.where(TP > TP.shift(1), 0.0).rolling(14).sum()
    neg = raw.where(TP < TP.shift(1), 0.0).rolling(14).sum()
    mfi = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).fillna(50)
    rng = (H - L).replace(0, np.nan)
    mfm = ((C - L) - (H - C)) / rng
    cmf = ((mfm * V).rolling(20).sum() / V.rolling(20).sum()).fillna(0)
    return mfi.reindex(idx, method="ffill").fillna(50), cmf.reindex(idx, method="ffill").fillna(0)


def _compute_adx(ohlcv, idx, period=14):
    H = ohlcv["high_adj"].astype(float)
    L = ohlcv["low_adj"].astype(float)
    C = ohlcv["close_adj"].astype(float)
    tr = pd.concat([H - L, (H - C.shift(1)).abs(), (L - C.shift(1)).abs()], axis=1).max(axis=1)
    up, dn = H - H.shift(1), L.shift(1) - L
    p_dm = up.where((up > dn) & (up > 0), 0.0)
    n_dm = dn.where((dn > up) & (dn > 0), 0.0)
    a = 1.0 / period
    atr = tr.ewm(alpha=a, adjust=False).mean()
    sp = p_dm.ewm(alpha=a, adjust=False).mean()
    sn = n_dm.ewm(alpha=a, adjust=False).mean()
    pdi = 100.0 * sp / atr.replace(0, np.nan)
    ndi = 100.0 * sn / atr.replace(0, np.nan)
    dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(alpha=a, adjust=False).mean()
    return (
        adx.reindex(idx, method="ffill").fillna(20.0),
        pdi.reindex(idx, method="ffill").fillna(25.0),
        ndi.reindex(idx, method="ffill").fillna(25.0),
    )


class ParametricStrategy(BaseStrategy):
    """6-sub ensemble (Iter31 architecture) with externalized parameters."""

    def __init__(self):
        super().__init__(allow_short=True)
        self._ohlcv_data = None
        self.P = load_params()

    def _load_data(self) -> None:
        """Override base: load from autoresearch OHLCV (up-to-date) instead of
        data/QQQ.csv. Mirrors ExperimentalStrategy._load_data so refresh on the
        frontend (which writes autoresearch/qqq_ohlcv.csv) actually feeds every
        non-Iter31 strategy too."""
        ohlcv = _load_ohlcv()
        col_map = {}
        if "close_adj" in ohlcv.columns:
            col_map["close_adj"] = "close"
        if "open_adj" in ohlcv.columns:
            col_map["open_adj"] = "open"
        if "high_adj" in ohlcv.columns:
            col_map["high_adj"] = "high"
        if "low_adj" in ohlcv.columns:
            col_map["low_adj"] = "low"
        self._data = ohlcv.rename(columns=col_map).copy()
        for col in ["open", "close", "high", "low"]:
            if col in self._data.columns and self._data[col].dtype == "object":
                self._data[col] = pd.to_numeric(
                    self._data[col].astype(str).str.replace(",", ""), errors="coerce"
                )

    def _get_ohlcv(self):
        if self._ohlcv_data is None:
            self._ohlcv_data = _load_ohlcv()
        return self._ohlcv_data

    def _calculate_positions(self, data, contextData=None):
        P = self.P
        result_df = self._make_result_df(data)
        full = self._data["close"].astype(float)

        sma50 = full.rolling(int(P["sma50_period"])).mean().reindex(data.index)
        sma200 = full.rolling(int(P["sma200_period"])).mean().reindex(data.index)
        sma_e = full.rolling(int(P["subE_sma_period"])).mean().reindex(data.index)
        ema5 = full.ewm(span=5, adjust=False).mean().reindex(data.index)
        ema8 = full.ewm(span=8, adjust=False).mean().reindex(data.index)
        ema13 = full.ewm(span=13, adjust=False).mean().reindex(data.index)
        ema_e = full.ewm(span=int(P["subE_ema_period"]), adjust=False).mean().reindex(data.index)
        ema34 = full.ewm(span=34, adjust=False).mean().reindex(data.index)
        roll252 = full.rolling(252).max().reindex(data.index)

        close = data["close"].astype(float)
        dd_yr = (close - roll252) / roll252 * 100
        rsi2 = _rsi(close, 2)
        rsi3 = _rsi(close, 3)
        pv = close / sma200.replace(0, np.nan)
        mom5 = close / close.shift(5) - 1
        mom20 = close / close.shift(20) - 1

        try:
            ohlcv = self._get_ohlcv()
            mfi, cmf = _compute_mfi_cmf(ohlcv, data.index)
            adx, pdi, ndi = _compute_adx(ohlcv, data.index)
        except Exception:
            mfi = (100 - _rsi(close, 14))
            cmf = pd.Series(0.0, index=data.index)
            adx = pd.Series(20.0, index=data.index)
            pdi = pd.Series(25.0, index=data.index)
            ndi = pd.Series(25.0, index=data.index)

        in_bull = sma50 > sma200
        genuine_bear = ~in_bull & (dd_yr < float(P["regime_dd_thresh"]))

        bear_short = (
            genuine_bear & (ema13 < ema34) & (mom20 <= 0) & (cmf <= 0)
            & (rsi3 < float(P["bear_short_rsi3"]))
        )
        bear_cash = genuine_bear & (ema13 < ema34) & ~bear_short
        shallow_rec = (~in_bull) & (~genuine_bear) & (ema5 > ema8) & (mom5 > 0)

        def apply_bear(s):
            s[bear_short] = -1
            s[bear_cash] = 0
            s[genuine_bear & (ema13 >= ema34)] = 0
            s[~in_bull & ~genuine_bear] = 0
            s[rsi3 < float(P["shallow_rsi3_long"])] = 1
            s[shallow_rec] = 1
            return s

        # Sub-A
        sigA = pd.Series(0, index=data.index)
        bull_pos = pd.Series(0, index=data.index)
        cash_days = 0
        for i in range(len(data)):
            if in_bull.iloc[i]:
                if cash_days > 0:
                    cash_days -= 1
                else:
                    strong = (adx.iloc[i] > float(P["subA_strong_adx"])) and (pdi.iloc[i] > ndi.iloc[i])
                    rsi3_th = float(P["subA_rsi3_low"]) if pv.iloc[i] >= float(P["subA_pv_rsi_low"]) else float(P["subA_rsi3_high"])
                    if strong:
                        if rsi3.iloc[i] > rsi3_th:
                            cash_days = int(P["subA_cash_high"]) if pv.iloc[i] >= float(P["subA_pv_high"]) else int(P["subA_cash_mid"])
                        else:
                            bull_pos.iloc[i] = 1
                    else:
                        if rsi3.iloc[i] > rsi3_th:
                            if pv.iloc[i] >= float(P["subA_pv_high"]):
                                cash_days = int(P["subA_cash_high"])
                            elif pv.iloc[i] >= float(P["subA_pv_mid"]):
                                cash_days = int(P["subA_cash_mid"])
                            else:
                                cash_days = int(P["subA_cash_low"])
                        else:
                            bull_pos.iloc[i] = 1
            else:
                cash_days = 0
        sigA[bull_pos == 1] = 1
        sigA = apply_bear(sigA)

        # Sub-B / Sub-D
        sigB = pd.Series(0, index=data.index)
        sigB[in_bull] = 1
        sigB = apply_bear(sigB)
        sigD = sigB.copy()

        # Sub-C
        sigC = pd.Series(0, index=data.index)
        in_hold = False
        hold_count = 0
        for i in range(len(data)):
            if in_bull.iloc[i]:
                if rsi2.iloc[i] < float(P["subC_rsi2_thresh"]):
                    in_hold, hold_count = True, int(P["subC_rsi2_hold"])
                elif mom5.iloc[i] < float(P["subC_mom5_thresh"]):
                    in_hold, hold_count = True, int(P["subC_mom5_hold"])
                if in_hold:
                    sigC.iloc[i] = 1
                    hold_count -= 1
                    if hold_count <= 0:
                        in_hold = False
            else:
                in_hold = False
        sigC = apply_bear(sigC)

        # Sub-E
        sigE = pd.Series(0, index=data.index)
        sigE[in_bull & (pv >= float(P["subE_pv_split"])) & (close > sma_e)] = 1
        sigE[in_bull & (pv < float(P["subE_pv_split"])) & (close > ema_e)] = 1
        sigE = apply_bear(sigE)

        # Sub-F
        sigF = pd.Series(0, index=data.index)
        sigF[in_bull & (pv >= float(P["subF_pv_bypass"])) & (mfi <= float(P["mfi_ob"]))] = 1
        sigF[in_bull & (pv < float(P["subF_pv_bypass"]))] = 1
        sigF = apply_bear(sigF)

        # ─── New subs (G-K): all default DISABLED (enable_X=0 → vote 0) ──────
        # When disabled they contribute nothing; baseline equals Iter31.
        ohlcv_for_ind = None
        try:
            ohlcv_for_ind = self._get_ohlcv()
        except Exception:
            pass

        # Sub-G: MACD trend confirmation (multi-condition)
        sigG = pd.Series(0, index=data.index)
        if int(P["enable_G"]) == 1 and ohlcv_for_ind is not None:
            try:
                macd_line, macd_sig, macd_hist = ind.macd(
                    ohlcv_for_ind, data.index,
                    fast=int(P["macd_fast"]),
                    slow=int(P["macd_slow"]),
                    signal=int(P["macd_signal_period"]),
                )
                sigG[in_bull & (macd_line > macd_sig) & (macd_hist > float(P["macd_hist_thresh"]))] = 1
                sigG = apply_bear(sigG)
            except Exception:
                sigG = pd.Series(0, index=data.index)

        # Sub-H: KDJ momentum (oversold-dip OR midzone-uptrend)
        sigH = pd.Series(0, index=data.index)
        if int(P["enable_H"]) == 1 and ohlcv_for_ind is not None:
            try:
                K_, D_, J_ = ind.kdj(
                    ohlcv_for_ind, data.index, period=int(P["kdj_period"])
                )
                cond_dip = J_ < float(P["kdj_oversold"])
                cond_mid = (K_ > D_) & (J_ > float(P["kdj_neutral_low"])) & (J_ < float(P["kdj_overbought"]))
                sigH[in_bull & (cond_dip | cond_mid)] = 1
                sigH = apply_bear(sigH)
            except Exception:
                sigH = pd.Series(0, index=data.index)

        # Sub-I: Bollinger mid-band trend (in middle, not breakout, with vol expansion)
        sigI = pd.Series(0, index=data.index)
        if int(P["enable_I"]) == 1 and ohlcv_for_ind is not None:
            try:
                bb_mid, bb_up, bb_lo, bb_bw = ind.bollinger(
                    ohlcv_for_ind, data.index,
                    period=int(P["bb_period"]),
                    n_std=float(P["bb_std"]),
                )
                sigI[in_bull & (close > bb_mid) & (close < bb_up) & (bb_bw > float(P["bb_bandwidth_min"]))] = 1
                sigI = apply_bear(sigI)
            except Exception:
                sigI = pd.Series(0, index=data.index)

        # Sub-J: Williams %R + OBV volume confirmation
        sigJ = pd.Series(0, index=data.index)
        if int(P["enable_J"]) == 1 and ohlcv_for_ind is not None:
            try:
                wr = ind.williams_r(ohlcv_for_ind, data.index, period=int(P["wr_period"]))
                obv_s, obv_sma = ind.obv(ohlcv_for_ind, data.index, sma_period=int(P["obv_sma_period"]))
                sigJ[in_bull & (wr > float(P["wr_low"])) & (wr < float(P["wr_high"])) & (obv_s > obv_sma)] = 1
                sigJ = apply_bear(sigJ)
            except Exception:
                sigJ = pd.Series(0, index=data.index)

        # Sub-K: Donchian midpoint + Aroon trend
        sigK = pd.Series(0, index=data.index)
        if int(P["enable_K"]) == 1 and ohlcv_for_ind is not None:
            try:
                don_hi, don_lo, _ = ind.donchian(
                    ohlcv_for_ind, data.index, period=int(P["don_period"])
                )
                aroon_up, aroon_dn = ind.aroon(
                    ohlcv_for_ind, data.index, period=int(P["aroon_period"])
                )
                threshold = don_lo + float(P["don_pos"]) * (don_hi - don_lo)
                sigK[in_bull & (close > threshold) & (aroon_up > aroon_dn)] = 1
                sigK = apply_bear(sigK)
            except Exception:
                sigK = pd.Series(0, index=data.index)

        # Sub-L: 252d z-score mean-reversion (from kept hypothesis)
        # When zscore_252 > high: overheated -> cash. When zscore_252 < low: deep
        # oversold -> long even outside bull regime.
        sigL = pd.Series(0, index=data.index)
        if int(P["enable_L"]) == 1:
            try:
                z_period = int(P["subL_zscore_period"])
                z_mean = full.rolling(z_period).mean().reindex(data.index)
                z_std = full.rolling(z_period).std().reindex(data.index)
                zs = ((close - z_mean) / z_std.replace(0, np.nan)).fillna(0)
                sigL[in_bull & (zs < float(P["subL_zscore_high"]))] = 1
                sigL[zs < float(P["subL_zscore_low"])] = 1  # oversold boost
                sigL = apply_bear(sigL)
            except Exception:
                sigL = pd.Series(0, index=data.index)

        # Sub-M: long-horizon return reversal (from kept hypotheses, strongest family)
        # Overextended multi-year run-up -> cash (reversion); deeply depressed
        # multi-year return -> long (post-washout rebound).
        sigM = pd.Series(0, index=data.index)
        if int(P["enable_M"]) == 1:
            try:
                lb = int(P["subM_lookback"])
                logc = np.log(full.clip(lower=1e-9))
                lhret = (logc - logc.shift(lb)).reindex(data.index)
                zp = int(P["subM_z_period"])
                z_mean = full.rolling(zp).mean().reindex(data.index)
                z_std = full.rolling(zp).std().reindex(data.index)
                zlong = ((close - z_mean) / z_std.replace(0, np.nan)).fillna(0)
                overext = (lhret > float(P["subM_ret_high"])) & (zlong > float(P["subM_z_high"]))
                depressed = lhret < float(P["subM_ret_low"])
                sigM[in_bull & ~overext.fillna(False)] = 1     # bull, not stretched -> long
                sigM[depressed.fillna(False)] = 1              # multi-year washout -> long
                sigM = apply_bear(sigM)
            except Exception:
                sigM = pd.Series(0, index=data.index)

        # ─── Ensemble vote (13 subs total; disabled subs contribute 0) ────────
        vote_long = ((sigA == 1).astype(int) + (sigB == 1).astype(int)
                     + (sigC == 1).astype(int) + (sigD == 1).astype(int)
                     + (sigE == 1).astype(int) + (sigF == 1).astype(int)
                     + (sigG == 1).astype(int) + (sigH == 1).astype(int)
                     + (sigI == 1).astype(int) + (sigJ == 1).astype(int)
                     + (sigK == 1).astype(int) + (sigL == 1).astype(int)
                     + (sigM == 1).astype(int))
        vote_short = ((sigA == -1).astype(int) + (sigB == -1).astype(int)
                      + (sigC == -1).astype(int) + (sigD == -1).astype(int)
                      + (sigE == -1).astype(int) + (sigF == -1).astype(int)
                      + (sigG == -1).astype(int) + (sigH == -1).astype(int)
                      + (sigI == -1).astype(int) + (sigJ == -1).astype(int)
                      + (sigK == -1).astype(int) + (sigL == -1).astype(int)
                      + (sigM == -1).astype(int))

        signals = pd.Series(0, index=data.index)
        signals[vote_long >= int(P["vote_long"])] = 1
        signals[vote_short >= int(P["vote_short"])] = -1
        signals.iloc[:5] = 0

        return self.apply_signals(result_df, signals)
