"""
loop.py — Random-sampling ensemble strategy harness (returns-dominant scoring).

Lifecycle (each iteration):
  1. Load best-ever state (state.json) — baseline = Iter31 if state is empty.
  2. If baseline score isn't cached, evaluate Iter31 once and cache.
  3. Build a candidate by random sampling: pick k params (k uniform in [1, MAX_K])
     and assign each a fresh random value from its SPEC range. Unpicked params
     stay at Iter31 defaults — so candidates are independent draws, not local
     mutations of the current best.
  4. Evaluate candidate across all 10 EVAL_WINDOWS via evaluate_strategy.
  5. Compute composite score (returns-dominant 70/20/10):
         score = 0.70 * mean(strat_return_pct/100)
               + 0.20 * mean(sharpe)
               + 0.10 * mean(min(outperformance_ratio, 10))
               - 0.05 * abs(mean(max_drawdown_pct)/100)
  6. Trade-frequency gate: every window must be in [12, 120] trades/yr.
  7. If candidate.score > best.score: PROMOTE — write state.json, snapshot.
  8. Append every iteration to log.jsonl. Update heartbeat.
  9. Loop forever. Catch any exception per iteration — never exit.

Stop signal: create file STOP in the harness dir, or kill PID.

State persistence: state.json is atomically replaced after every iteration so
the best-ever record survives crashes and process death.

Usage:
    python -m autoresearch.harness.loop                  # run forever
    python -m autoresearch.harness.loop --max-iters 1    # one iter (smoke test)
    python -m autoresearch.harness.loop --reset          # rebaseline & re-eval
"""

import os
import sys
import json
import time
import random
import argparse
import traceback
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from autoresearch.harness.parametric_strategy import ITER31_DEFAULTS
from autoresearch.harness.mutator import random_sample
from autoresearch.harness.tested_set import TestedSet, total_combo_space
from autoresearch.run_experiment import MIN_TRADES_PER_YEAR, MAX_TRADES_PER_YEAR


HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HARNESS_DIR, "state.json")
CANDIDATE_FILE = os.path.join(HARNESS_DIR, "current_params.json")
LOG_FILE = os.path.join(HARNESS_DIR, "log.jsonl")
HEARTBEAT_FILE = os.path.join(HARNESS_DIR, "heartbeat.txt")
STOP_FILE = os.path.join(HARNESS_DIR, "STOP")
SNAPSHOT_DIR = os.path.join(HARNESS_DIR, "snapshots")

MAX_K = 8                     # cap on # of params perturbed per draw
COLLISION_RETRIES = 100       # per iter — try this many fresh draws before counting a collision-block
SATURATION_BLOCKS = 50        # # of consecutive collision-blocks before declaring sampler-saturation
MAX_CHAMPIONS = 5             # multi-champion (island) search: keep N diverse top candidates
MIN_CHAMPION_DISTANCE = 4     # Hamming distance: a candidate may not be promoted if it's this close to any non-target champion
P_CROSSOVER = 0.5             # probability of mixing base-champion params with another champion's params before mutation
CROSSOVER_INHERIT_FRACTION = 0.25  # fraction of params drawn from the second champion during crossover
CHAMPION_FOCUS_DIMS = ["score", "min_ratio", "sharpe", "outperf", "diverse"]
# "diverse" focus: optimize composite score but the champion is held open for any architecturally novel candidate
W_RETURN, W_SHARPE, W_OUTPERF, W_DD = 0.70, 0.20, 0.10, 0.05
OUTPERF_CLIP = 10.0           # cap outperformance ratio so a bear-window win can't dominate
TRADES_GATE_MIN = 12          # tighter than MIN_TRADES_PER_YEAR — keep ensemble busy
TRADES_GATE_MAX = MAX_TRADES_PER_YEAR

os.environ["HARNESS_PARAMS_FILE"] = CANDIDATE_FILE
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def heartbeat(msg: str):
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            f.write(f"{_now()} | {msg}\n")
    except Exception:
        pass


def log_event(rec: dict):
    rec = {"ts": _now(), **rec}
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def load_state() -> dict:
    """Load state, migrating single-best -> multi-champion schema if needed."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            if isinstance(s.get("champions"), list) and s["champions"]:
                return s
            # Legacy single-best schema -> migrate to champion #0 (focus=score).
            if isinstance(s.get("metrics"), dict) and "score" in s["metrics"]:
                champion = {
                    "label": s.get("label", "migrated_champion"),
                    "focus": "score",
                    "params": s["params"],
                    "metrics": s["metrics"],
                    "born_iter": s.get("iterations", 0),
                }
                return {
                    "champions": [champion],
                    "iterations": s.get("iterations", 0),
                    "promotions": s.get("promotions", 0),
                }
        except Exception:
            pass
    return {
        "champions": [],
        "iterations": 0,
        "promotions": 0,
    }


OUTPERF_MIN_RATIO_FLOOR = 1.0  # outperf champion must at least match QQQ in worst window


def focus_metric(metrics: dict, focus: str) -> float:
    if not metrics or not metrics.get("ok"):
        return float("-inf")
    if focus == "min_ratio":
        return metrics.get("min_ratio", 0.0)
    if focus == "sharpe":
        return metrics.get("mean_sharpe", 0.0)
    if focus == "outperf":
        # Disqualify Pareto-corner freaks: high mean_outperf paired with
        # min_ratio < 1.0 (i.e. losing to QQQ in the worst eval window) is
        # not a usable strategy — it's just leverage on the bull windows.
        # Returning -inf for such candidates makes any qualifying balanced
        # strategy strictly preferable on this focus dim.
        if metrics.get("min_ratio", 0.0) < OUTPERF_MIN_RATIO_FLOOR:
            return float("-inf")
        return metrics.get("mean_outperf", 0.0)
    # "score" or "diverse" both use composite score
    return metrics.get("score", 0.0)


def hamming(p1: dict, p2: dict) -> int:
    keys = set(p1.keys()) | set(p2.keys())
    return sum(1 for k in keys if p1.get(k) != p2.get(k))


def behaviorally_identical(m1: dict, m2: dict, tol: float = 1e-4) -> bool:
    """Return True if two metric dicts produce indistinguishable strategy behavior.
    Catches the case where two candidates have a few different params but those
    params are all dormant (e.g., enable_X=0 sub thresholds) so the strategy
    produces identical signals and identical aggregate metrics."""
    if not (m1 and m2):
        return False
    keys = ["score", "mean_return_pct", "mean_sharpe", "mean_outperf", "mean_max_dd_pct"]
    for k in keys:
        if abs(float(m1.get(k, 0.0)) - float(m2.get(k, 0.0))) > tol:
            return False
    return True


def best_champion(champions: list) -> dict | None:
    return max(champions, key=lambda c: c["metrics"].get("score", float("-inf"))) if champions else None


def try_promote_champion(champions: list, cand_params: dict, cand_metrics: dict,
                          iter_id: int) -> tuple:
    """
    Attempt to insert candidate into the champion list. Returns
    (replaced_index, new_champ, reason). If replaced_index is None, candidate
    was not accepted; reason explains why.

    Algorithm:
      - For each champion C: if candidate beats C on C's focus AND candidate is
        sufficiently distant (Hamming) from each OTHER champion, this slot is
        replaceable. Pick the replacement that gives the largest improvement
        on its focus dim.
      - If no slot is replaceable, candidate is discarded.
    """
    candidates = []
    for i, C in enumerate(champions):
        focus = C.get("focus", "score")
        c_score = focus_metric(C["metrics"], focus)
        cand_score = focus_metric(cand_metrics, focus)
        if cand_score <= c_score:
            continue
        # Architectural constraint check. If this champion is pinned to a
        # specific architecture (e.g. enable_L=1 for the "diverse-subL" champion),
        # the candidate MUST satisfy that constraint to replace this slot.
        # Prevents the seeded architectural diversity from eroding as the
        # harness finds candidates that happen to beat the focus metric but
        # have dropped the load-bearing arch flag.
        arch = C.get("arch_constraint") or {}
        if arch:
            mismatch = [k for k, v in arch.items() if cand_params.get(k) != v]
            if mismatch:
                continue
        # Distance + behavioral-identity check against other champions.
        # A candidate cannot replace champion #i if it would leave the pool
        # with a structurally OR behaviorally duplicate champion.
        too_close_to = None
        for j, other in enumerate(champions):
            if j == i:
                continue
            d = hamming(cand_params, other["params"])
            if d < MIN_CHAMPION_DISTANCE:
                too_close_to = (j, other, d, "param-hamming")
                break
            if behaviorally_identical(cand_metrics, other["metrics"]):
                too_close_to = (j, other, d, "behavioral-identity")
                break
        if too_close_to is not None:
            continue
        improvement = cand_score - c_score
        candidates.append((improvement, i, C))

    if not candidates:
        # Diagnostic: explain why
        best_focus_gap = None
        for i, C in enumerate(champions):
            focus = C.get("focus", "score")
            gap = focus_metric(cand_metrics, focus) - focus_metric(C["metrics"], focus)
            if best_focus_gap is None or gap > best_focus_gap[0]:
                best_focus_gap = (gap, i, focus)
        if best_focus_gap and best_focus_gap[0] <= 0:
            return None, None, f"no champion beat (best focus gap {best_focus_gap[0]:+.3f} on champ#{best_focus_gap[1]} {best_focus_gap[2]})"
        return None, None, f"beats champion(s) on focus but blocked by diversity (Hamming<{MIN_CHAMPION_DISTANCE} or behavioral-identity)"

    candidates.sort(key=lambda x: -x[0])
    improvement, idx, target = candidates[0]
    new_label = f"{target['focus']}_p{idx}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    new_champ = {
        "label": new_label,
        "focus": target["focus"],
        "params": cand_params,
        "metrics": cand_metrics,
        "born_iter": iter_id,
    }
    # Carry the arch_constraint forward so the slot's pinned architecture
    # survives across generations.
    if target.get("arch_constraint"):
        new_champ["arch_constraint"] = dict(target["arch_constraint"])
    return idx, new_champ, f"replaces #{idx} ({target['label']}) on focus={target['focus']}, +{improvement:.3f}"


def save_state(state: dict):
    # Mirror the best champion at top-level for backward-compat with
    # launch_detached --status / monitor task that read params/metrics/label
    # directly without knowing about champion lists.
    best = best_champion(state.get("champions") or [])
    if best is not None:
        state["params"] = best["params"]
        state["metrics"] = best["metrics"]
        state["label"] = best["label"]
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def write_candidate_params(params: dict):
    tmp = CANDIDATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(params, f, indent=2)
    os.replace(tmp, CANDIDATE_FILE)


def evaluate(params: dict) -> dict:
    """Evaluate a parameter set across all EVAL_WINDOWS. Returns scored metrics dict."""
    write_candidate_params(params)

    if "autoresearch.harness.parametric_strategy" in sys.modules:
        del sys.modules["autoresearch.harness.parametric_strategy"]
    from autoresearch.run_experiment import evaluate_strategy

    result = evaluate_strategy(
        "autoresearch.harness.parametric_strategy",
        "ParametricStrategy",
    )

    successful = [w for w in result["windows"] if "error" not in w]
    if not successful:
        return {"ok": False, "reason": "all-windows-crashed"}

    rets = {w["window"]: w["strat_return_pct"] for w in successful}
    qqq_rets = {w["window"]: w["qqq_return_pct"] for w in successful}
    ratios = {w["window"]: w["outperformance_ratio"] for w in successful}
    trades_yr = {w["window"]: w["trades_per_year"] for w in successful}
    win_rate = {w["window"]: w["win_rate"] for w in successful}
    sharpe = {w["window"]: w["sharpe"] for w in successful}
    max_dd = {w["window"]: w["max_drawdown_pct"] for w in successful}

    mean_ret = sum(rets.values()) / len(rets) / 100.0
    mean_sharpe = sum(sharpe.values()) / len(sharpe)
    mean_outperf = sum(min(r, OUTPERF_CLIP) for r in ratios.values()) / len(ratios)
    mean_dd = sum(max_dd.values()) / len(max_dd) / 100.0

    score = (
        W_RETURN * mean_ret
        + W_SHARPE * mean_sharpe
        + W_OUTPERF * mean_outperf
        - W_DD * abs(mean_dd)
    )

    return {
        "ok": True,
        "score": score,
        "mean_return_pct": round(mean_ret * 100, 4),
        "mean_sharpe": round(mean_sharpe, 4),
        "mean_outperf": round(mean_outperf, 4),
        "mean_max_dd_pct": round(mean_dd * 100, 4),
        "min_ratio": min(ratios.values()),
        "avg_ratio": sum(ratios.values()) / len(ratios),
        "rets": rets,
        "qqq_rets": qqq_rets,
        "ratios": ratios,
        "trades_yr": trades_yr,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "summary": result.get("summary", {}),
    }


def passes_gates(cand: dict) -> tuple:
    """Return (ok, reason). Only trade-frequency is gated; score handles the rest."""
    if not cand["ok"]:
        return False, cand.get("reason", "eval-failed")
    for win, t in cand["trades_yr"].items():
        if t < TRADES_GATE_MIN or t > TRADES_GATE_MAX:
            return False, f"trades/yr out-of-range on {win}: {t:.1f}"
    return True, "ok"


def is_better(cand: dict, base: dict, eps: float = 1e-6) -> tuple:
    if not cand["ok"] or not base["ok"]:
        return False, "eval-failed"
    if cand["score"] <= base["score"] + eps:
        return False, f"score not improved: {cand['score']:.4f} <= {base['score']:.4f}"
    return True, f"score {base['score']:.4f} -> {cand['score']:.4f}"


def snapshot_baseline(state: dict, label: str):
    path = os.path.join(SNAPSHOT_DIR, f"{label}.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def stop_requested() -> bool:
    return os.path.exists(STOP_FILE)


def _seed_initial_champions(defaults: dict) -> list:
    """Create N starter champions with distinct focus dims + architectural seeds.
    Each seed may include an `arch_constraint` — a dict of params that any
    candidate replacing this champion MUST keep at the seeded value. Prevents
    the harness from drifting these slots back to a uniform architecture."""
    seeds = []

    seeds.append({"_seed_params": dict(defaults), "focus": "score", "_seed_label": "score-iter31"})

    bear_seed = dict(defaults)
    bear_seed["regime_dd_thresh"] = -12
    bear_seed["bear_short_rsi3"] = 82
    seeds.append({"_seed_params": bear_seed, "focus": "min_ratio", "_seed_label": "minratio-bear-protect"})

    sharpe_seed = dict(defaults)
    sharpe_seed["subA_rsi3_low"] = 80
    sharpe_seed["subA_pv_high"] = 1.18
    seeds.append({"_seed_params": sharpe_seed, "focus": "sharpe", "_seed_label": "sharpe-tighter-subA"})

    subL_seed = dict(defaults)
    subL_seed["enable_L"] = 1
    subL_seed["subL_zscore_period"] = 252
    subL_seed["subL_zscore_high"] = 2.0
    subL_seed["subL_zscore_low"] = -1.5
    subL_seed["vote_long"] = 4
    seeds.append({
        "_seed_params": subL_seed, "focus": "diverse", "_seed_label": "diverse-subL-on",
        "arch_constraint": {"enable_L": 1},
    })

    subG_seed = dict(defaults)
    subG_seed["enable_G"] = 1
    subG_seed["macd_fast"] = 12
    subG_seed["macd_slow"] = 26
    subG_seed["vote_long"] = 4
    seeds.append({
        "_seed_params": subG_seed, "focus": "outperf", "_seed_label": "outperf-subG-on",
        "arch_constraint": {"enable_G": 1},
    })

    return seeds


def run_loop(max_iters: int = None, reset: bool = False):
    state = load_state()
    defaults = dict(ITER31_DEFAULTS)

    if reset:
        state["champions"] = []
        save_state(state)
        log_event({"event": "reset"})

    print(f"[{_now()}] Harness starting. Champions: {len(state.get('champions', []))}")
    print(f"[{_now()}] Iterations so far: {state.get('iterations', 0)}, promotions: {state.get('promotions', 0)}")

    # Seed champion pool if missing
    if not state.get("champions"):
        print(f"[{_now()}] No champions yet — seeding initial pool of {MAX_CHAMPIONS}...")
        seeds = _seed_initial_champions(defaults)[:MAX_CHAMPIONS]
        champions = []
        for s in seeds:
            heartbeat(f"evaluating seed {s['_seed_label']}")
            try:
                m = evaluate(s["_seed_params"])
            except Exception as e:
                print(f"[{_now()}] Seed {s['_seed_label']} eval crashed: {e}")
                continue
            if not m["ok"]:
                print(f"[{_now()}] Seed {s['_seed_label']} failed: {m.get('reason')}")
                continue
            champ = {
                "label": s["_seed_label"],
                "focus": s["focus"],
                "params": s["_seed_params"],
                "metrics": m,
                "born_iter": 0,
            }
            if s.get("arch_constraint"):
                champ["arch_constraint"] = dict(s["arch_constraint"])
            champions.append(champ)
            print(f"[{_now()}]   {s['_seed_label']:<28} focus={s['focus']:<10} score={m['score']:.2f} ret={m['mean_return_pct']:.0f}%")
        state["champions"] = champions
        save_state(state)
        for c in champions:
            snapshot_baseline({"params": c["params"], "metrics": c["metrics"], "label": c["label"]},
                              f"seed_{c['label']}_{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        log_event({"event": "champion-seed", "n": len(champions),
                   "labels": [c["label"] for c in champions]})

    if not state["champions"]:
        print(f"[{_now()}] All seeds failed; cannot start. Exiting.")
        return

    iters_done = 0
    tested = TestedSet()
    for c in state["champions"]:
        tested.add(c["params"])
    total_space = total_combo_space()
    print(f"[{_now()}] Tested-set: {len(tested)} entries loaded. Total space: {total_space:.3e}")
    consecutive_blocks = 0

    while True:
        if stop_requested():
            print(f"[{_now()}] STOP file detected. Exiting.")
            log_event({"event": "stop"})
            return
        if max_iters is not None and iters_done >= max_iters:
            print(f"[{_now()}] Reached max_iters={max_iters}. Exiting.")
            return

        iters_done += 1
        state["iterations"] = state.get("iterations", 0) + 1
        iter_id = state["iterations"]

        try:
            heartbeat(f"iter {iter_id} — sampling")
            base_champ = random.choice(state["champions"])

            # Optional crossover: with probability P_CROSSOVER, mix base champion's
            # params with another champion's. This lets discoveries on one island
            # (e.g. tuned MACD on outperf champion) flow into other islands.
            # Preserve the BASE champion's arch_constraint params during crossover
            # so a Sub-L-pinned slot doesn't accidentally inherit enable_L=0.
            crossover_mate = None
            base_params = dict(base_champ["params"])
            arch_keys = set((base_champ.get("arch_constraint") or {}).keys())
            if len(state["champions"]) >= 2 and random.random() < P_CROSSOVER:
                others = [c for c in state["champions"] if c["label"] != base_champ["label"]]
                if others:
                    crossover_mate = random.choice(others)
                    mate_params = crossover_mate["params"]
                    candidate_keys = [k for k in mate_params if k not in arch_keys]
                    n_inherit = max(1, int(len(candidate_keys) * CROSSOVER_INHERIT_FRACTION))
                    inherited = random.sample(candidate_keys, k=min(n_inherit, len(candidate_keys)))
                    for k in inherited:
                        base_params[k] = mate_params[k]

            cand_params, picks = None, None
            for attempt in range(COLLISION_RETRIES):
                trial, trial_picks = random_sample(defaults, max_k=MAX_K, base=base_params)
                if not tested.has(trial):
                    cand_params, picks = trial, trial_picks
                    break

            if cand_params is None:
                consecutive_blocks += 1
                log_event({"event": "collision-block", "iter": iter_id,
                           "tested": len(tested), "consecutive_blocks": consecutive_blocks})
                print(f"[{_now()}] iter {iter_id} collision-block consec={consecutive_blocks}/{SATURATION_BLOCKS}")
                save_state(state)
                if consecutive_blocks >= SATURATION_BLOCKS:
                    print(f"[{_now()}] SAMPLER SATURATED — {SATURATION_BLOCKS} consecutive collision blocks.")
                    log_event({"event": "saturated", "tested": len(tested), "total_space": str(total_space)})
                    return
                heartbeat(f"iter {iter_id} collision-block ({consecutive_blocks}/{SATURATION_BLOCKS})")
                continue

            consecutive_blocks = 0
            tested.add(cand_params)
            k = len(picks)

            cx_tag = f" x {crossover_mate['label']}" if crossover_mate else ""
            heartbeat(f"iter {iter_id} — evaluating k={k} base={base_champ['label']}{cx_tag}")
            t0 = time.time()
            cand_metrics = evaluate(cand_params)
            elapsed = time.time() - t0

            if not cand_metrics["ok"]:
                log_event({"event": "crash", "iter": iter_id, "k": k, "picks": picks,
                           "base": base_champ["label"], "reason": cand_metrics.get("reason"),
                           "elapsed_s": round(elapsed, 1)})
                print(f"[{_now()}] iter {iter_id} CRASH ({cand_metrics.get('reason')}) k={k} base={base_champ['label']}")
                save_state(state)
                continue

            gate_ok, gate_reason = passes_gates(cand_metrics)
            if not gate_ok:
                log_event({"event": "discard", "iter": iter_id, "k": k, "picks": picks,
                           "base": base_champ["label"], "score": cand_metrics["score"],
                           "reason": gate_reason, "elapsed_s": round(elapsed, 1)})
                print(f"[{_now()}] iter {iter_id} discard ({gate_reason}) k={k} base={base_champ['label']}")
                save_state(state)
                continue

            replaced_idx, new_champ, reason = try_promote_champion(
                state["champions"], cand_params, cand_metrics, iter_id)

            rec = {
                "event": "promote" if replaced_idx is not None else "discard",
                "iter": iter_id, "k": k, "picks": picks,
                "base": base_champ["label"],
                "crossover_mate": crossover_mate["label"] if crossover_mate else None,
                "score": cand_metrics["score"],
                "mean_return_pct": cand_metrics["mean_return_pct"],
                "mean_sharpe": cand_metrics["mean_sharpe"],
                "mean_outperf": cand_metrics["mean_outperf"],
                "mean_max_dd_pct": cand_metrics["mean_max_dd_pct"],
                "min_ratio": cand_metrics["min_ratio"],
                "reason": reason,
                "elapsed_s": round(elapsed, 1),
            }
            log_event(rec)

            if replaced_idx is not None:
                old = state["champions"][replaced_idx]
                state["champions"][replaced_idx] = new_champ
                state["promotions"] = state.get("promotions", 0) + 1
                save_state(state)
                snapshot_baseline(
                    {"params": new_champ["params"], "metrics": new_champ["metrics"], "label": new_champ["label"]},
                    new_champ["label"])
                print(f"[{_now()}] iter {iter_id} ✅ PROMOTE  {reason}  base={base_champ['label']}")
            else:
                save_state(state)
                print(f"[{_now()}] iter {iter_id} discard ({reason}) base={base_champ['label']} score={cand_metrics['score']:.3f}")

        except KeyboardInterrupt:
            print(f"[{_now()}] KeyboardInterrupt — exiting cleanly.")
            log_event({"event": "interrupt"})
            return
        except Exception as e:
            tb = traceback.format_exc()
            log_event({"event": "iter-error", "iter": iter_id, "error": str(e), "trace": tb})
            print(f"[{_now()}] iter {iter_id} ERROR: {e}")
            time.sleep(2)

        heartbeat(f"iter {iter_id} done; promotions={state['promotions']} champs={len(state['champions'])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()

    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    run_loop(max_iters=args.max_iters, reset=args.reset)


if __name__ == "__main__":
    main()
