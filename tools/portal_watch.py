"""Poll the portal and log every new tournament: our rank, capital, top 3.

  uv run python tools/portal_watch.py --after-sub <old_submission_id> --count 5
Stops after `count` tournaments played with a tight submission different from
--after-sub (i.e. the new upload), or after --minutes.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

URL = "https://pricing-hackaton.duckdns.org/api/leaderboard"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="tight")
    ap.add_argument("--after-sub", default="")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--every", type=float, default=20)
    a = ap.parse_args()
    seen, hits = set(), 0
    end = time.time() + a.minutes * 60
    while time.time() < end and hits < a.count:
        try:
            with urllib.request.urlopen(URL, timeout=20) as resp:
                data = json.load(resp)
        except Exception as exc:  # network hiccup: retry
            print("error", exc, flush=True)
            time.sleep(a.every)
            continue
        tid = data["tournament_id"]
        if tid not in seen:
            seen.add(tid)
            sc = data["scenarios"][0]
            rows = sc["rankings"]
            me = next((r for r in rows if r["username"] == a.team), None)
            hist = {s["username"]: s["capital"] for s in sc["capital_history"]["series"]}
            new = me is not None and me["submission_id"] != a.after_sub
            if new:
                hits += 1
            c = hist.get(a.team, [0])
            early = c[min(10, len(c) - 1)] - c[0]
            top = ", ".join(f"{r['username']} {r['final_capital']:,.0f}" for r in rows[:3])
            print(f"#{tid} {'NEW' if new else 'old'} sub={me and me['submission_id']} rank={me and me['rank']}/{len(rows)} "
                  f"cap={me and round(me['final_capital'])} early={early:+.0f} | top: {top}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
