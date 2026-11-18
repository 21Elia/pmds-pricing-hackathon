"""Local arena: run many seeds/scenarios and report the official metric (mean rank).

Usage (from repo root):
  uv run python tools/arena.py --config config/default.yaml --config config/scaled_market.yaml \
      --seeds 8 --extra sparring
Every bundle under src/dme_simulator/agents plays; --extra adds every bundle
found in the given folders (e.g. sparring copies of our agent).
"""
from __future__ import annotations

import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dme_simulator.agents.loader import discover_agent_bundles, load_simulation_agents, resolve_agent_directories
from dme_simulator.config import load_config
from dme_simulator.runner import run_simulation


def _job(args):
    config_path, seed, extras, turns = args
    config = load_config(config_path)
    dirs = resolve_agent_directories(Path(config_path), config.agents.directory, None)
    agents = load_simulation_agents(dirs, config.agents.include, extras)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        sim = run_simulation(config, agents, seed_override=seed, turns_override=turns, quiet=True)
    return config_path, seed, {t: (s.capital if not s.bankrupt else -1.0) for t, s in sim.teams_state.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--turns", type=int, default=None)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    configs = a.config or ["config/default.yaml", "config/scaled_market.yaml"]
    extras = [p for root in a.extra for p in discover_agent_bundles([Path(root)])]
    jobs = [(c, a.seed0 + i, extras, a.turns) for c in configs for i in range(a.seeds)]
    ranks: dict[str, list[int]] = {}
    caps: dict[str, list[float]] = {}
    per_cfg: dict[tuple[str, str], list[int]] = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for cfg, seed, res in ex.map(_job, jobs):
            order = sorted(res, key=lambda t: -res[t])
            for r, t in enumerate(order, 1):
                ranks.setdefault(t, []).append(r)
                caps.setdefault(t, []).append(res[t])
                per_cfg.setdefault((Path(cfg).stem, t), []).append(r)
    cfg_names = [Path(c).stem for c in configs]
    head = f"{'team':<22}{'avg rank':>9}{'avg cash':>12}" + "".join(f"{n[:14]:>16}" for n in cfg_names)
    print(head)
    for t in sorted(ranks, key=lambda t: sum(ranks[t]) / len(ranks[t])):
        row = f"{t:<22}{sum(ranks[t]) / len(ranks[t]):>9.2f}{sum(caps[t]) / len(caps[t]):>12,.0f}"
        for n in cfg_names:
            r = per_cfg.get((n, t), [])
            row += f"{(sum(r) / len(r)) if r else float('nan'):>16.2f}"
        print(row)


if __name__ == "__main__":
    main()
