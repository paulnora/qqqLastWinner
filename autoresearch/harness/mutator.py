"""
mutator.py — Generates parameter mutations for the parametric ensemble.

Each parameter has a (min, max, step) spec. A mutation:
  - Picks 1-3 parameters at random (weighted toward 1).
  - Perturbs each by +/- N steps (small step bias).
  - Clamps to spec bounds.

Hard rules (memory: ensemble-only):
  - vote_long >= 4 always (never collapse the ensemble below 4 of 6 votes).
  - vote_short >= 1.
  - Hold counts >= 1.

The mutator is stateless. Caller passes the current best params, gets a
candidate dict back. Identity mutations are filtered out.
"""

import random
import copy

# Parameter spec: (min, max, step). Steps tuned to be small but meaningful.
SPEC = {
    "vote_long":          (4,    9,    1),
    "vote_short":         (1,    4,    1),
    "mfi_ob":             (55,   72,   1),
    "subF_pv_bypass":     (1.03, 1.15, 0.005),
    "subE_pv_split":      (1.05, 1.20, 0.005),
    "subE_sma_period":    (20,   40,   1),
    "subE_ema_period":    (10,   25,   1),
    "subA_pv_high":       (1.12, 1.30, 0.01),
    "subA_pv_mid":        (1.05, 1.18, 0.01),
    "subA_pv_rsi_low":    (1.08, 1.22, 0.01),
    "subA_rsi3_low":      (78,   92,   1),
    "subA_rsi3_high":     (82,   95,   1),
    "subA_strong_adx":    (18,   32,   1),
    "subA_cash_high":     (1,    4,    1),
    "subA_cash_mid":      (0,    3,    1),
    "subA_cash_low":      (1,    5,    1),
    "subC_rsi2_thresh":   (5,    18,   1),
    "subC_rsi2_hold":     (5,    16,   1),
    "subC_mom5_thresh":   (-0.10, -0.02, 0.005),
    "subC_mom5_hold":     (5,    14,   1),
    "regime_dd_thresh":   (-25,  -10,  1),
    "bear_short_rsi3":    (75,   92,   1),
    "shallow_rsi3_long":  (5,    18,   1),
    "sma200_period":      (180,  220,  5),
    "sma50_period":       (40,   65,   1),

    # --- Sub-G: MACD ---
    "enable_G":            (0,    1,    1),
    "macd_fast":           (8,    16,   1),
    "macd_slow":           (20,   34,   1),
    "macd_signal_period":  (5,    14,   1),
    "macd_hist_thresh":    (-0.5, 0.5,  0.05),

    # --- Sub-H: KDJ ---
    "enable_H":            (0,    1,    1),
    "kdj_period":          (5,    21,   1),
    "kdj_oversold":        (10,   30,   1),
    "kdj_neutral_low":     (20,   45,   1),
    "kdj_overbought":      (70,   90,   1),

    # --- Sub-I: Bollinger ---
    "enable_I":            (0,    1,    1),
    "bb_period":           (12,   30,   1),
    "bb_std":              (1.5,  2.8,  0.1),
    "bb_bandwidth_min":    (0.0,  0.10, 0.005),

    # --- Sub-J: Williams %R + OBV ---
    "enable_J":            (0,    1,    1),
    "wr_period":           (8,    21,   1),
    "wr_low":              (-95,  -60,  1),
    "wr_high":             (-30,  -5,   1),
    "obv_sma_period":      (10,   40,   1),

    # --- Sub-K: Donchian + Aroon ---
    "enable_K":            (0,    1,    1),
    "don_period":          (10,   40,   1),
    "don_pos":             (0.0,  0.9,  0.05),
    "aroon_period":        (14,   40,   1),

    # --- Sub-L: 252d Z-score mean-reversion (kept hypothesis: rho=-0.21, p=0.001)
    "enable_L":            (0,    1,    1),
    "subL_zscore_period":  (126,  378,  21),
    "subL_zscore_high":    (1.0,  3.0,  0.25),
    "subL_zscore_low":     (-2.5, 0.0,  0.25),

    # --- Sub-M: long-horizon return reversal (kept hypotheses: rho up to -0.32)
    "enable_M":            (0,    1,    1),
    "subM_lookback":       (252,  1260, 63),    # 1yr .. 5yr trailing log-return
    "subM_ret_high":       (0.4,  1.2,  0.05),
    "subM_ret_low":        (-0.6, 0.0,  0.05),
    "subM_z_period":       (252,  756,  63),
    "subM_z_high":         (1.0,  3.0,  0.25),
}


def _round(val, step):
    """Round a float to the nearest step."""
    if step >= 1:
        return int(round(val / step) * step)
    return round(round(val / step) * step, 6)


def clamp(name, value):
    lo, hi, step = SPEC[name]
    if step >= 1:
        value = int(round(value))
    value = max(lo, min(hi, value))
    return _round(value, step)


def random_perturb(name, current):
    """Perturb a single param by 1-2 steps in a random direction."""
    lo, hi, step = SPEC[name]
    n_steps = random.choices([1, 2, 3], weights=[6, 3, 1])[0]
    direction = random.choice([-1, 1])
    delta = n_steps * step * direction
    return clamp(name, current + delta)


def mutate(params: dict, n_changes: int = None) -> dict:
    """Return a perturbed copy of params. Picks n_changes params (default 1-3 weighted)."""
    if n_changes is None:
        n_changes = random.choices([1, 2, 3], weights=[7, 2, 1])[0]

    keys = list(SPEC.keys())
    candidate = copy.deepcopy(params)
    chosen = random.sample(keys, k=min(n_changes, len(keys)))
    for k in chosen:
        if k not in candidate:
            candidate[k] = SPEC[k][0]
        candidate[k] = random_perturb(k, candidate[k])

    # Hard constraints
    candidate["vote_long"] = max(4, min(6, int(candidate.get("vote_long", 5))))
    candidate["vote_short"] = max(1, min(3, int(candidate.get("vote_short", 2))))

    # Diff vs original — record which keys changed
    diff = {k: (params.get(k), candidate[k]) for k in chosen if params.get(k) != candidate[k]}
    return candidate, diff


def is_identity(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in SPEC.keys())


def random_value(name):
    """Sample a uniform random value from a param's (lo, hi, step) range, snapped to step."""
    lo, hi, step = SPEC[name]
    if step >= 1:
        n_steps = int((hi - lo) // step)
        return clamp(name, lo + random.randint(0, n_steps) * step)
    n_steps = int(round((hi - lo) / step))
    return clamp(name, lo + random.randint(0, n_steps) * step)


# When the random sampler flips an enable_X flag from 0 to 1, force-sample 2-3 of
# that sub's threshold params too. This addresses the "enable-flips die because
# default thresholds aren't aligned with the rest of the ensemble" failure mode.
SUB_THRESHOLD_PARAMS = {
    "enable_G": ["macd_fast", "macd_slow", "macd_signal_period", "macd_hist_thresh"],
    "enable_H": ["kdj_period", "kdj_oversold", "kdj_neutral_low", "kdj_overbought"],
    "enable_I": ["bb_period", "bb_std", "bb_bandwidth_min"],
    "enable_J": ["wr_period", "wr_low", "wr_high", "obv_sma_period"],
    "enable_K": ["don_period", "don_pos", "aroon_period"],
    "enable_L": ["subL_zscore_period", "subL_zscore_high", "subL_zscore_low"],
    "enable_M": ["subM_lookback", "subM_ret_high", "subM_ret_low", "subM_z_period", "subM_z_high"],
}


def random_sample(defaults: dict, max_k: int = 8, base: dict | None = None):
    """
    Build a candidate by picking k params (k uniform in [1, max_k]) and assigning
    each a random value within its SPEC range. Unpicked params inherit `base` if
    given, else `defaults`. Coordinated enable-flag sampling: if an enable_X
    transitions 0->1, also re-sample 2-3 of that sub's threshold params.

    Returns (candidate, picks) where picks is {name: (prior_value, sampled_value)}.
    """
    keys = list(SPEC.keys())
    k = random.randint(1, min(max_k, len(keys)))
    chosen = list(random.sample(keys, k=k))

    src = base if base is not None else defaults
    candidate = copy.deepcopy(src)
    for name in chosen:
        candidate[name] = random_value(name)

    # Coordinated enable-flag sampling: any flag flipped 0->1 in this draw
    # triggers re-sampling of 2-3 of that sub's threshold knobs, even if those
    # knobs weren't originally picked. Greatly raises the probability that an
    # enable-on candidate produces a coherent (vs noisy) sub.
    for flag, threshold_params in SUB_THRESHOLD_PARAMS.items():
        if flag in chosen and int(candidate.get(flag, 0)) == 1 and int(src.get(flag, 0)) == 0:
            n_extra = random.randint(2, min(3, len(threshold_params)))
            extra = random.sample(threshold_params, k=n_extra)
            for t in extra:
                candidate[t] = random_value(t)
                if t not in chosen:
                    chosen.append(t)

    # Hard floors
    candidate["vote_long"] = max(4, min(6, int(candidate.get("vote_long", 5))))
    candidate["vote_short"] = max(1, min(3, int(candidate.get("vote_short", 2))))

    picks = {n: (src.get(n), candidate[n]) for n in chosen if src.get(n) != candidate[n]}
    return candidate, picks
