"""
webapp/research.py — Data loaders for the Research dashboard.

Aggregates state from:
  - autoresearch/harness/state.json (champion pool)
  - autoresearch/harness/heartbeat.txt, harness.pid (liveness)
  - adversarial/ledger.jsonl, adversarial/candidates/ (regime hunter)
  - hypothesis_search/ledger.jsonl, promoted_features.md (hypothesis search)
  - monitoring/latest.md, monitoring/history.jsonl (monitor)
"""
import os
import re
import json
import subprocess
import sys
from datetime import datetime
from collections import Counter
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATE_FILE = os.path.join(ROOT, "autoresearch", "harness", "state.json")
HEARTBEAT_FILE = os.path.join(ROOT, "autoresearch", "harness", "heartbeat.txt")
PID_FILE = os.path.join(ROOT, "autoresearch", "harness", "harness.pid")
TESTED_FILE = os.path.join(ROOT, "autoresearch", "harness", "tested.hashes")

ADV_LEDGER = os.path.join(ROOT, "adversarial", "ledger.jsonl")
ADV_CANDIDATES = os.path.join(ROOT, "adversarial", "candidates")

HYP_LEDGER = os.path.join(ROOT, "hypothesis_search", "ledger.jsonl")
HYP_PROMOTED = os.path.join(ROOT, "hypothesis_search", "promoted_features.md")
HYP_SEEN = os.path.join(ROOT, "hypothesis_search", "seen_papers.txt")

MONITOR_LATEST = os.path.join(ROOT, "monitoring", "latest.md")
MONITOR_HISTORY = os.path.join(ROOT, "monitoring", "history.jsonl")


def _is_alive(pid: int) -> bool:
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=3
            )
            return str(pid) in r.stdout
        os.kill(pid, 0)
        return True
    except Exception:
        return False


_LABEL_TS_RE = re.compile(r"_(\d{8})T(\d{6})$")


def _age_from_label(label: str) -> dict:
    """Parse a champion label like 'score_p0_20260510T210014' into:
    {'promoted_at': '2026-05-10 21:00', 'age_days': 22, 'bucket': 'fresh'|'recent'|'stale'}.
    Returns empty dict if label doesn't match the expected suffix pattern."""
    if not label:
        return {}
    m = _LABEL_TS_RE.search(label)
    if not m:
        return {}
    try:
        ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return {}
    age = (datetime.now() - ts).total_seconds() / 86400
    if age < 7:
        bucket = "fresh"
    elif age < 30:
        bucket = "recent"
    else:
        bucket = "stale"
    return {
        "promoted_at": ts.strftime("%Y-%m-%d %H:%M"),
        "age_days": int(age),
        "bucket": bucket,
    }


def _load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def harness_status() -> dict:
    out = {
        "alive": False,
        "pid": None,
        "heartbeat": None,
        "iterations": 0,
        "promotions": 0,
        "tested_combos": 0,
        "champions": [],
    }
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            out["pid"] = pid
            out["alive"] = _is_alive(pid)
        except Exception:
            pass
    if os.path.exists(HEARTBEAT_FILE):
        try:
            with open(HEARTBEAT_FILE) as f:
                out["heartbeat"] = f.read().strip()
        except Exception:
            pass
    if os.path.exists(TESTED_FILE):
        try:
            with open(TESTED_FILE) as f:
                out["tested_combos"] = sum(1 for _ in f)
        except Exception:
            pass
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                s = json.load(f)
            out["iterations"] = s.get("iterations", 0)
            out["promotions"] = s.get("promotions", 0)
            champs = s.get("champions") or []
            # Compute iter31 diffs and active subs for each champion
            try:
                from autoresearch.harness.parametric_strategy import ITER31_DEFAULTS
            except Exception:
                ITER31_DEFAULTS = {}
            for c in champs:
                p = c.get("params", {})
                active = ["A", "B", "C", "D", "E", "F"]
                for ch in "GHIJKL":
                    if int(p.get(f"enable_{ch}", 0)) == 1:
                        active.append(ch)
                diffs = {k: {"iter31": ITER31_DEFAULTS.get(k), "current": v}
                         for k, v in p.items() if ITER31_DEFAULTS.get(k) != v}
                c["_active_subs"] = active
                c["_diffs"] = diffs
                c["_n_diffs"] = len(diffs)
                c["_age"] = _age_from_label(c.get("label", ""))
                # Generalization gap: in-sample outperf vs true holdout outperf.
                # A large positive gap (IS >> OOS) flags overfitting.
                cm = c.get("metrics") or {}
                ho = cm.get("holdout") or {}
                if ho.get("ok"):
                    is_outperf = cm.get("mean_outperf", 0.0)
                    oos_outperf = ho.get("mean_outperf", 0.0)
                    c["_holdout"] = {
                        "ok": True,
                        "outperf": oos_outperf,
                        "ret": ho.get("mean_return_pct", 0.0),
                        "sharpe": ho.get("mean_sharpe", 0.0),
                        "min_ratio": ho.get("min_ratio", 0.0),
                        "gap": round(is_outperf - oos_outperf, 3),
                        # holds up if OOS still beats QQQ (ratio >= 1) and the
                        # gap isn't catastrophic
                        "holds": oos_outperf >= 1.0,
                    }
                else:
                    c["_holdout"] = {"ok": False}
                # Headline metric — what this champion is actually being judged on
                focus = c.get("focus", "score")
                m = c.get("metrics", {})
                if focus == "min_ratio":
                    c["_headline"] = {"label": "Min Ratio (worst window)", "value": m.get("min_ratio", 0), "fmt": "{:.3f}"}
                elif focus == "sharpe":
                    c["_headline"] = {"label": "Mean Sharpe", "value": m.get("mean_sharpe", 0), "fmt": "{:.2f}"}
                elif focus == "outperf":
                    c["_headline"] = {"label": "Mean Outperf vs QQQ", "value": m.get("mean_outperf", 0), "fmt": "{:.2f}x"}
                else:
                    c["_headline"] = {"label": "Composite Score", "value": m.get("score", 0), "fmt": "{:.1f}"}
                # Arch constraint badge
                arch = c.get("arch_constraint") or {}
                c["_arch_badge"] = None
                if arch:
                    parts = []
                    for k, v in arch.items():
                        if k.startswith("enable_") and v == 1:
                            parts.append(f"Sub-{k[-1]} pinned")
                        else:
                            parts.append(f"{k}={v} pinned")
                    c["_arch_badge"] = " · ".join(parts)
            out["champions"] = champs
        except Exception:
            pass
    return out


def champion_diversity(champs: list) -> list:
    """Pairwise Hamming distance between champion param sets."""
    if not champs:
        return []
    n = len(champs)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            p1 = champs[i].get("params", {})
            p2 = champs[j].get("params", {})
            keys = set(p1.keys()) | set(p2.keys())
            d = sum(1 for k in keys if p1.get(k) != p2.get(k))
            matrix[i][j] = matrix[j][i] = d
    return matrix


def adversarial_summary() -> dict:
    rows = _load_jsonl(ADV_LEDGER)
    verdicts = Counter(r.get("verdict") for r in rows if r.get("verdict"))
    recs = Counter(r.get("merge_recommendation") for r in rows if r.get("merge_recommendation"))
    generalizing = [r for r in rows if r.get("verdict") == "generalizes"]
    candidates = []
    if os.path.isdir(ADV_CANDIDATES):
        for fn in sorted(os.listdir(ADV_CANDIDATES)):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(ADV_CANDIDATES, fn), encoding="utf-8") as f:
                        candidates.append(json.load(f))
                except Exception:
                    continue
    return {
        "total_cycles": len(rows),
        "verdicts": dict(verdicts),
        "recommendations": dict(recs),
        "generalizing": generalizing,
        "candidates": candidates,
        "recent": rows[-10:][::-1],
    }


def hypothesis_summary() -> dict:
    rows = _load_jsonl(HYP_LEDGER)
    statuses = Counter(r.get("status") for r in rows if r.get("status"))
    kept = [r for r in rows if r.get("status") == "kept"]
    seen = 0
    if os.path.exists(HYP_SEEN):
        try:
            with open(HYP_SEEN, encoding="utf-8") as f:
                seen = sum(1 for line in f if line.strip())
        except Exception:
            pass
    promoted_md = ""
    if os.path.exists(HYP_PROMOTED):
        try:
            with open(HYP_PROMOTED, encoding="utf-8") as f:
                promoted_md = f.read()
        except Exception:
            pass
    return {
        "total_hypotheses": len(rows),
        "statuses": dict(statuses),
        "seen_papers": seen,
        "kept": kept,
        "promoted_md": promoted_md,
        "recent": rows[-10:][::-1],
    }


def monitor_summary() -> dict:
    latest = ""
    if os.path.exists(MONITOR_LATEST):
        try:
            with open(MONITOR_LATEST, encoding="utf-8") as f:
                latest = f.read()
        except Exception:
            pass
    history = _load_jsonl(MONITOR_HISTORY)
    return {
        "latest_md": latest,
        "history_rows": len(history),
        "recent_history": history[-20:],
    }


def get_research_data() -> dict:
    return {
        "harness": harness_status(),
        "diversity": champion_diversity((harness_status().get("champions") or [])),
        "adversarial": adversarial_summary(),
        "hypothesis": hypothesis_summary(),
        "monitor": monitor_summary(),
    }
