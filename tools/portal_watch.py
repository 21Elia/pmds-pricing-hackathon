"""Poll the portal and log every new tournament: our rank per scenario and aggregate.

  uv run python tools/portal_watch.py --after-sub <old_submission_id> --count 5
Stops after `count` tournaments played with a tight submission different from
--after-sub (i.e. a new upload), or after --minutes. Logs the submission id so
uploads by teammates on the same account can be told apart.
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
            with urllib.request.urlopen(URL, timeout=60) as resp:
                data = json.load(resp)
        except Exception as exc:  # network hiccup: retry
            print("error", exc, flush=True)
            time.sleep(a.every)
            continue
        tid = data["tournament_id"]
        if tid not in seen:
            seen.add(tid)
            agg = data.get("aggregate", [])
            me_agg = next((r for r in agg if r["username"] == a.team), None)
            sub = me_agg and me_agg["submission_id"]
            new = sub is not None and sub != a.after_sub
            hits += new
            parts = []
            for sc in data["scenarios"]:
                rows = sc["rankings"]
                me = next((r for r in rows if r["username"] == a.team), None)
                lead = rows[0]
                parts.append(f"{sc['name']}: {me and me['rank']}/{len(rows)} cap={me and round(me['final_capital']):,} "
                             f"(1st {lead['username']} {round(lead['final_capital']):,})")
            agg_txt = f"agg rank {me_agg and me_agg['rank']} (avg {me_agg and me_agg['average_rank']})"
            print(f"#{tid} {'NEW' if new else 'old'} sub={sub} {agg_txt} | " + " | ".join(parts), flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
