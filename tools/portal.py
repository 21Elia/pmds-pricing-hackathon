"""Show the latest public portal tournament: ranking and per-turn capital changes.

  uv run python tools/portal.py            # top 10 + tight
  uv run python tools/portal.py gabriele   # also show this team's deltas
"""
from __future__ import annotations

import json
import sys
import urllib.request

URL = "https://pricing-hackaton.duckdns.org/api/leaderboard"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=20) as resp:
        data = json.load(resp)
    show = {"tight", *sys.argv[1:]}
    for sc in data["scenarios"]:
        hist = {s["username"]: s["capital"] for s in sc["capital_history"]["series"]}
        print(f"tournament {data['tournament_id']}  {sc['name']} seed={sc['seed']} turns={sc['turns']} teams={len(sc['rankings'])}")
        for r in sc["rankings"]:
            u = r["username"]
            if r["rank"] <= 10 or u in show:
                c = hist[u]
                early = c[min(10, len(c) - 1)] - c[0]
                print(f"  {r['rank']:>2} {u:<22} final={c[-1]:>9,.0f}  gain turns 1-10={early:>+7,.0f}  after={c[-1] - c[min(10, len(c) - 1)]:>+7,.0f}")
        for u in show:
            if u in hist:
                c = hist[u]
                print(f"  {u} deltas:", [round(c[i] - c[i - 1]) for i in range(1, len(c))])


if __name__ == "__main__":
    main()
