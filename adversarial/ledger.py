"""
ledger.py — Append-only JSONL ledger for adversarial findings.

Each row records one cycle:
  - regime: vol/trend/dispersion/rate tags
  - failure_window: (start, end) where baseline underperformed
  - baseline_score: per-window score on that window
  - patch: SPEC param diff proposed by Patcher
  - validation: per-window scores under patch on (failure_window, same-regime
    sample, different-regime sample, full holdout)
  - generalize: True/False/insufficient
  - notes: short Patcher/Validator reasoning text
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional


HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
LEDGER_FILE = os.path.join(HARNESS_DIR, "ledger.jsonl")
CANDIDATES_DIR = os.path.join(HARNESS_DIR, "candidates")
os.makedirs(CANDIDATES_DIR, exist_ok=True)


def append(rec: dict):
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), **rec}
    with open(LEDGER_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_all() -> list[dict]:
    if not os.path.exists(LEDGER_FILE):
        return []
    out = []
    with open(LEDGER_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_candidate(label: str, payload: dict) -> str:
    """Save a generalizing patch as a candidate snapshot for human review."""
    safe = label.replace("/", "_").replace(":", "-").replace(" ", "_")
    path = os.path.join(CANDIDATES_DIR, f"{safe}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def attempted_failure_windows(limit: Optional[int] = None) -> set[str]:
    """Set of windows the Breaker has already exploited; helps avoid repeats.

    Normalizes ``failure_window`` to the canonical ``"<start> to <end>"``
    string form. Older rows stored it as a string; newer rows store a
    ``[start, end]`` list, which is unhashable and must be joined.
    """
    rows = load_all()
    if limit:
        rows = rows[-limit:]
    out: set[str] = set()
    for r in rows:
        fw = r.get("failure_window")
        if not fw:
            continue
        if isinstance(fw, (list, tuple)):
            out.add(f"{fw[0]} to {fw[1]}")
        else:
            out.add(fw)
    return out
