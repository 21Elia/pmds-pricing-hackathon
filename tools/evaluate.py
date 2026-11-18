"""Evaluate agent bundles on the OFFICIAL DME engine with the OFFICIAL score.

Official score: in each match agents are ranked by final capital (1 = best); the
overall ranking is the mean rank across matches (lower is better). Inventory is
worth nothing.

Every match puts ALL candidates in the same market (so they are compared
head-to-head), plus a lineup of opponents.

Examples (from repo root):
  uv run python tools/evaluate.py                              # tight, all suites
  uv run python tools/evaluate.py --cand "tight=src/dme_simulator/agents/tight" \
      --cand "v3=src/dme_simulator/agents/davide_v3" --isolated
  uv run python tools/evaluate.py --suite self --n 60 --cand "v3=src/dme_simulator/agents/davide_v3"
  Candidate kwargs (constructor args): --cand "name=path:k=v,k=v"
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import random
import statistics
import sys
from pathlib import Path

import yaml

from dme_simulator.agents.loader import load_agent_bundle_class
from dme_simulator.config import EnvironmentConfig, _from_mapping
from dme_simulator.runner import run_simulation

import dme_simulator

ROOT = Path(__file__).resolve().parents[1]          # repo root
SCEN = ROOT / "config"
OFFICIAL = ["default", "scaled_market", "tutorial_pricing", "tutorial_auction"]
BUILTIN = Path(dme_simulator.__file__).parent / "agents" / "builtin_bundles"
RIVAL = ROOT / "tools/opponents/Rival"
STYLES = ["fixed", "undercut", "highprice", "overbid", "hoarder", "dumper",
          "bandit", "valuebid", "trembler"]


# ---------------------------------------------------------------- scenarios
def load_raw(name: str) -> dict:
    with open(SCEN / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(raw: dict) -> EnvironmentConfig:
    raw = copy.deepcopy(raw)
    raw["output"] = {**raw.get("output", {}), "render_plots": False, "dashboard": "none"}
    cfg = _from_mapping(EnvironmentConfig, raw)
    cfg.validate()
    return cfg


def random_market(rng: random.Random) -> dict:
    """Perturb a base scenario: price scale, WTP, supply, demand, beta, horizon, capital."""
    raw = load_raw(rng.choice(["default", "default", "scaled_market"]))
    price_scale = math.exp(rng.uniform(math.log(0.3), math.log(3.0)))
    wtp_scale = price_scale * math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
    supply = math.exp(rng.uniform(math.log(0.3), math.log(3.0)))
    demand = math.exp(rng.uniform(math.log(0.3), math.log(3.0)))
    raw["turns"] = rng.choice([8, 12, 20, 30, 40, 60])
    raw["initial_state"]["starting_capital"] *= math.exp(rng.uniform(math.log(0.3), math.log(3)))
    for s in raw["suppliers"].values():
        s["base_price"] *= price_scale * rng.uniform(0.7, 1.4)
        s["production_min"] = max(1, round(s["production_min"] * supply))
        s["production_max"] = max(s["production_min"], round(s["production_max"] * supply))
        if s.get("stock_capacity"):
            s["stock_capacity"] = max(s["production_max"], round(s["stock_capacity"] * supply))
    cm = raw["consumer_market"]
    cm["base_customers"] = max(1, round(cm["base_customers"] * demand))
    cm["amplitude"] = min(cm["base_customers"], round(cm.get("amplitude", 0) * demand))
    cm["price_sensitivity_beta"] = rng.choice([0.002, 0.005, 0.01, 0.02, 0.05]) / price_scale
    for seg in cm["segments"].values():
        k = rng.uniform(0.7, 1.4)
        seg["willingness_to_pay"] = {g: [lo * wtp_scale * k, hi * wtp_scale * k]
                                     for g, (lo, hi) in seg["willingness_to_pay"].items()}
    if rng.random() < 0.3:
        raw["production"]["capacity_per_turn"] = rng.choice([5, 10, 20, 50])
    raw["agents"]["observe_all_transactions"] = rng.random() < 0.4
    raw["seed"] = rng.randrange(10**6)
    return raw


# ---------------------------------------------------------------- agents
def parse_cand(spec: str):
    """'name=path[:k=v,k=v]' -> (name, path, kwargs)."""
    name, rest = spec.split("=", 1)
    path, _, kw = rest.partition(":")
    kwargs = {}
    for item in filter(None, kw.split(",")):
        k, v = item.split("=", 1)
        try:
            v = float(v) if any(c in v for c in ".e") else int(v)
        except ValueError:
            pass
        kwargs[k] = v
    return name, str(ROOT / path) if not Path(path).is_absolute() else path, kwargs


def lineup(rng: random.Random, kind: str, n_crowd: int, cands):
    """Opponents as (team_id, bundle_path, kwargs)."""
    opp = []
    if kind == "builtin":
        for b in ["DummyAgent", "PatientAgent", "PriceWarAgent"]:
            opp.append((b, str(BUILTIN / b), {}))
    elif kind == "league":
        for i, st in enumerate(rng.sample(STYLES, rng.randint(3, 6))):
            opp.append((f"R_{st}", str(RIVAL), {"style": st, "seed": rng.randrange(999)}))
        opp.append(("PatientAgent", str(BUILTIN / "PatientAgent"), {}))
    elif kind == "crowd":
        # crowd: rivals + extra copies of our candidates (other good teams look like us)
        n_opp = max(0, n_crowd - len(cands))
        for i in range(n_opp):
            if rng.random() < 0.3 and cands:
                name, path, kw = rng.choice(cands)
                opp.append((f"clone{i}_{name}", path, kw))
            else:
                st = rng.choice(STYLES)
                opp.append((f"R{i}_{st}", str(RIVAL), {"style": st, "seed": rng.randrange(999)}))
    return opp


_CLASS_CACHE = {}


def make_agent(path, kwargs):
    if path not in _CLASS_CACHE:
        _CLASS_CACHE[path] = load_agent_bundle_class(path)
    return _CLASS_CACHE[path](**kwargs)


# ---------------------------------------------------------------- match
def play(job):
    raw, opp, cands, seed = job
    cfg = build_config(raw)
    teams = {}
    order = [(f"C_{n}", p, kw) for n, p, kw in cands] + list(opp)
    random.Random(seed).shuffle(order)
    for tid, path, kw in order:
        teams[tid] = make_agent(path, kw)
    sim = run_simulation(cfg, teams, quiet=True, seed_override=seed)
    cap = {t: s.capital for t, s in sim.teams_state.items()}
    ranked = sorted(cap, key=lambda t: -cap[t])
    ranks = {}
    for t in cap:  # ties share the best rank
        ranks[t] = 1 + sum(1 for u in cap if cap[u] > cap[t])
    start = cfg.initial_state.starting_capital
    return {n: (ranks[f"C_{n}"], cap[f"C_{n}"] / start, len(cap)) for n, _, _ in cands}, \
        ranked[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", default=[],
                    help="name=bundle_path[:k=v,...] (repeatable)")
    ap.add_argument("--suite", default="all", help="official|random|crowd|self|all (comma-separated)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per official scenario x lineup")
    ap.add_argument("--n", type=int, default=60, help="matches for random/crowd suites")
    ap.add_argument("--crowd", default="10,25", help="crowd sizes")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--isolated", action="store_true",
                    help="each candidate plays ALONE in identical markets (same seeds/opponents)")
    ap.add_argument("--clones", action="append", default=[],
                    help="extra clone types for crowd opponents: name=path[:k=v] (repeatable)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    cands = [parse_cand(c) for c in args.cand] or [parse_cand("tight=src/dme_simulator/agents/tight")]
    clone_pool = [parse_cand(c) for c in args.clones] or cands
    rng = random.Random(args.seed0)
    jobs = []  # (suite, job)
    suites = ["official", "random", "crowd"] if args.suite == "all" else args.suite.split(",")
    if "official" in suites:
        for name in OFFICIAL:
            for kind in ["builtin", "league"]:
                for s in range(args.seeds):
                    jobs.append((f"official:{name}", (load_raw(name), lineup(rng, kind, 0, cands),
                                                      cands, args.seed0 + s)))
    if "random" in suites:
        for i in range(args.n):
            jobs.append(("random", (random_market(rng), lineup(rng, "league", 0, cands),
                                    cands, rng.randrange(10**6))))
    if "crowd" in suites:
        sizes = [int(x) for x in args.crowd.split(",")]
        for i in range(args.n):
            raw = load_raw(rng.choice(["default", "scaled_market"])) if i % 2 else random_market(rng)
            jobs.append(("crowd", (raw, lineup(rng, "crowd", rng.choice(sizes), clone_pool),
                                   cands, rng.randrange(10**6))))

    self_jobs = []  # self-play: copies of ONE candidate + PatientAgent (does it lose money vs itself?)
    if "self" in suites:
        for i in range(args.n):
            raw = load_raw(rng.choice(OFFICIAL[:2])) if i % 3 == 0 else random_market(rng)
            k = rng.choice([3, 5, 8])
            seed = rng.randrange(10**6)
            for c in cands:
                opp = [(f"clone{j}_{c[0]}", c[1], c[2]) for j in range(k - 1)]
                opp.append(("PatientAgent", str(BUILTIN / "PatientAgent"), {}))
                self_jobs.append((c[0], (raw, opp, [c], seed)))

    if args.isolated:
        flat = [(i, (raw, opp, [c], seed)) for i, (_, (raw, opp, cs, seed)) in enumerate(jobs) for c in cs]
    with mp.get_context("fork").Pool(args.workers) as pool:
        if args.isolated:
            out = pool.map(play, [j for _, j in flat], chunksize=1)
            merged = [({}, None) for _ in jobs]
            for (i, _), (res, w) in zip(flat, out):
                merged[i][0].update(res)
                merged[i] = (merged[i][0], w)
            results = merged
        else:
            results = pool.map(play, [j for _, j in jobs], chunksize=1)
        self_results = pool.map(play, [j for _, j in self_jobs], chunksize=1)

    if self_jobs:
        print("\nSELF-PLAY (copies of one version + PatientAgent):")
        agg = {}
        for (name, job), (res, winner) in zip(self_jobs, self_results):
            agg.setdefault(name, []).append((res[name][1], winner == "PatientAgent"))
        for name, rows in agg.items():
            below = sum(1 for c, _ in rows if c < 1.0) / len(rows)
            patient = sum(1 for _, w in rows if w) / len(rows)
            med = statistics.median(c for c, _ in rows)
            print(f"  {name:20s} n={len(rows):3d}  median cap/x0 {med:5.2f}  "
                  f"below start {below:4.0%}  PatientAgent wins {patient:4.0%}")
    if not jobs:
        return

    by_suite, winners = {}, {}
    for (suite, _), (res, winner) in zip(jobs, results):
        key = suite.split(":")[0]
        for k in dict.fromkeys((suite, key, "ALL")):
            by_suite.setdefault(k, []).append(res)
        winners.setdefault(key, []).append(winner)

    names = [c[0] for c in cands]
    print(f"\n{'suite':28s} {'n':>4s} " + " ".join(f"{n:>22s}" for n in names))
    print(f"{'':28s} {'':>4s} " + " ".join(f"{'rank  pct1  cap/x0':>22s}" for _ in names))
    for suite in sorted(by_suite, key=lambda s: (s == "ALL", s)):
        rows = by_suite[suite]
        line = f"{suite:28s} {len(rows):4d} "
        for n in names:
            r = [x[n][0] for x in rows]
            c = [x[n][1] for x in rows]
            first = sum(1 for v in r if v == 1) / len(r)
            line += f"{statistics.mean(r):8.2f} {first:5.0%} {statistics.median(c):7.2f}"
        print(line)
    from collections import Counter
    for key, ws in winners.items():
        top = Counter(w.split("_", 1)[-1] if w.startswith(("R", "clone")) else w for w in ws)
        print(f"  winners {key}: " + ", ".join(f"{k} {v}" for k, v in top.most_common(6)))
    # loss check: how often we end below starting capital
    rows = by_suite["ALL"]
    for n in names:
        below = sum(1 for x in rows if x[n][1] < 1.0) / len(rows)
        print(f"  {n}: below starting capital in {below:.0%} of matches")


if __name__ == "__main__":
    sys.exit(main())
