"""Smoke/fuzz test for the submission bundle on the official engine.

Checks, over the 4 official scenarios and many random markets (with crowds and
copies of our agent): every decision is valid per the official validator, no
exception escapes, and decisions stay far below the 100 ms deadline.

  uv run python tools/test_bundle.py [bundle_path]   (default: src/dme_simulator/agents/tight)
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from dme_simulator.runner import _validate_decision  # noqa: E402
from dme_simulator.simulation import DMESimulator  # noqa: E402

import evaluate as ev  # noqa: E402

BUNDLE = str(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "src/dme_simulator/agents/tight"))


def run_checked(raw, n_ours, n_rivals, seed):
    cfg = ev.build_config(raw)
    rng = random.Random(seed)
    agents = {f"us{i}": ev.make_agent(BUNDLE, {}) for i in range(n_ours)}
    for i in range(n_rivals):
        agents[f"r{i}"] = ev.make_agent(str(ev.RIVAL), {"style": rng.choice(ev.STYLES), "seed": i})
    sim = DMESimulator(list(agents), cfg, seed=seed)
    worst = 0.0
    for _ in range(cfg.turns):
        decisions = {}
        for tid, agent in agents.items():
            if sim.teams_state[tid].bankrupt:
                continue
            state = sim.get_team_state(tid)
            t0 = time.perf_counter()
            # ours: call _decide so internal exceptions are NOT hidden by the fallback
            inner = getattr(agent, "_decide", None) if tid.startswith("us") else None
            d = inner(state, cfg.goods) if inner else agent.decide(state, cfg.goods)
            dt = (time.perf_counter() - t0) * 1000
            if tid.startswith("us"):
                worst = max(worst, dt)
                _validate_decision(d, state, cfg.goods)   # raises if invalid
            decisions[tid] = d
        sim.run_turn(decisions)
    return worst


def main():
    worst = 0.0
    n = 0
    for name in ev.OFFICIAL:
        worst = max(worst, run_checked(ev.load_raw(name), 2, 4, 1))
        n += 1
    rng = random.Random(7)
    for i in range(40):
        worst = max(worst, run_checked(ev.random_market(rng), rng.randint(1, 4), rng.randint(0, 25), i))
        n += 1
    print(f"OK: {n} markets, all decisions valid, worst decision time {worst:.2f} ms")
    assert worst < 30, "decision too slow"


if __name__ == "__main__":
    main()
