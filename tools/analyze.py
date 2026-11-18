"""Turn-by-turn behaviour of ONE agent in a full match (default: tight_v3, 40 turns).

Writes a CSV per match and prints a compact summary: where the money goes
(material spend vs revenue), sell-through per good, price ladder vs what sold,
auction fill/overpay vs reserve, stock left at the end (worth 0).

  uv run python tools/analyze.py                                   # default.yaml, crowd of 12
  uv run python tools/analyze.py --config config/scaled_market.yaml --field small
  uv run python tools/analyze.py --agent src/dme_simulator/agents/tight_v3 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from dme_simulator.config import load_config  # noqa: E402
from dme_simulator.simulation import DMESimulator  # noqa: E402

import evaluate as ev  # noqa: E402


def field(kind, agent_path):
    A = ROOT / "src/dme_simulator/agents"
    if kind == "small":     # the local default roster
        return [("Dummy", str(ev.BUILTIN / "DummyAgent"), {}), ("Patient", str(ev.BUILTIN / "PatientAgent"), {}),
                ("PriceWar", str(ev.BUILTIN / "PriceWarAgent"), {})]
    # crowd: copies of the same agent (other strong teams) + other strategies
    out = [(f"copy{i}", agent_path, {}) for i in range(3)]
    out += [("davide_v3", str(A / "davide_v3"), {}), ("tight_v2", str(A / "tight_v2"), {})]
    for i, st in enumerate(["valuebid", "overbid", "undercut", "fixed", "bandit", "hoarder"]):
        out.append((f"R_{st}", str(ev.RIVAL), {"style": st, "seed": i}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=str(ROOT / "src/dme_simulator/agents/tight_v3"))
    ap.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    ap.add_argument("--field", default="crowd", help="crowd|small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    agents = {"ME": ev.make_agent(a.agent, {})}
    for tid, p, kw in field(a.field, a.agent):
        agents[tid] = ev.make_agent(p, kw)
    sim = DMESimulator(list(agents), cfg, seed=a.seed)
    gc = cfg.goods
    rows = []
    for turn in range(1, cfg.turns + 1):
        decisions = {}
        st_me = None
        for tid, ag in agents.items():
            if sim.teams_state[tid].bankrupt:
                continue
            st = sim.get_team_state(tid)
            if tid == "ME":
                st_me = st
            decisions[tid] = ag.decide(st, gc)
        lots = {l.good: l for l in st_me.auction_lots}
        d = decisions["ME"]
        cap0 = sim.teams_state["ME"].capital
        rep = sim.run_turn(decisions)
        row = {"turn": turn, "capital_start": round(cap0, 1),
               "capital_end": round(sim.teams_state["ME"].capital, 1),
               "spend": round(rep.auction_spend.get("ME", 0.0), 1),
               "revenue": round(rep.revenue.get("ME", 0.0), 1),
               "customers": rep.customers}
        for m in gc.raw_materials:
            bid = [b for b in d.bids if b.good == m]
            row[f"{m}_reserve"] = round(lots[m].reserve_price, 2) if m in lots else ""
            row[f"{m}_lot"] = lots[m].quantity if m in lots else 0
            row[f"{m}_bid_p"] = round(bid[0].price_per_unit, 2) if bid else ""
            row[f"{m}_bid_q"] = sum(b.quantity for b in bid)
            row[f"{m}_won"] = rep.units_won.get("ME", {}).get(m, 0)
            row[f"{m}_clear_min"] = round(rep.clearing_price.get(m, 0.0), 2)
            row[f"{m}_stock_end"] = sim.teams_state["ME"].inventory.get(m, 0)
        for g in gc.finished_goods:
            ls = [l for l in d.listings if l.good == g]
            row[f"{g}_listed"] = rep.listed.get("ME", {}).get(g, 0)
            row[f"{g}_p_min"] = round(min(l.price for l in ls), 1) if ls else ""
            row[f"{g}_p_max"] = round(max(l.price for l in ls), 1) if ls else ""
            row[f"{g}_produced"] = rep.produced.get("ME", {}).get(g, 0)
            row[f"{g}_sold"] = rep.units_sold.get("ME", {}).get(g, 0)
            row[f"{g}_mkt_sold"] = rep.sold_by_good.get(g, 0)
            row[f"{g}_mkt_price"] = round(rep.revenue_by_good.get(g, 0.0) / rep.sold_by_good[g], 1) \
                if rep.sold_by_good.get(g) else ""
            row[f"{g}_stockouts"] = rep.stockouts.get(g, 0)
            row[f"{g}_stock_end"] = sim.teams_state["ME"].inventory.get(g, 0)
        rows.append(row)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(a.agent).name}_{Path(a.config).stem}_{a.field}_s{a.seed}.csv"
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------- summary
    caps = {t: s.capital for t, s in sim.teams_state.items()}
    rank = 1 + sum(1 for t in caps if caps[t] > caps["ME"])
    print(f"\n== {Path(a.agent).name} | {Path(a.config).stem} | field={a.field} ({len(caps)} teams) | seed {a.seed}")
    print(f"final capital {caps['ME']:,.0f} (start {cfg.initial_state.starting_capital:,.0f}) -> rank {rank}/{len(caps)}")
    top = sorted(caps.items(), key=lambda kv: -kv[1])[:5]
    print("top 5: " + ", ".join(f"{t} {c:,.0f}" for t, c in top))
    spend = sum(r["spend"] for r in rows)
    rev = sum(r["revenue"] for r in rows)
    print(f"spend {spend:,.0f}  revenue {rev:,.0f}  (revenue/spend {rev / max(spend, 1):.2f})")
    print("\nper good: produced, sold, sell-through, our share of market units, avg price we got vs market, end stock")
    for g in gc.finished_goods:
        prod = sum(r[f"{g}_produced"] for r in rows)
        sold = sum(r[f"{g}_sold"] for r in rows)
        listed = sum(r[f"{g}_listed"] for r in rows)
        mkt = sum(r[f"{g}_mkt_sold"] for r in rows)
        so = sum(r[f"{g}_stockouts"] for r in rows)
        print(f"  {g:11s} prod {prod:4d} sold {sold:4d} listed-turns {listed:5d} "
              f"sell-thru {sold / max(listed, 1):5.0%} share {sold / max(mkt, 1):5.0%} "
              f"unserved customers {so:4d} end stock {rows[-1][f'{g}_stock_end']}")
    print("\nper material: units won, avg bid / avg reserve, fill rate, end stock (wasted)")
    for m in gc.raw_materials:
        won = sum(r[f"{m}_won"] for r in rows)
        asked = sum(r[f"{m}_bid_q"] for r in rows)
        bids = [r[f"{m}_bid_p"] for r in rows if r[f"{m}_bid_p"] != ""]
        res = [r[f"{m}_reserve"] for r in rows if r[f"{m}_bid_p"] != ""]
        ratio = (sum(bids) / len(bids)) / (sum(res) / len(res)) if bids else 0
        print(f"  {m:8s} won {won:4d}/{asked:4d} asked (fill {won / max(asked, 1):4.0%})  "
              f"bid/reserve {ratio:4.2f}  end stock {rows[-1][f'{m}_stock_end']}")
    print("\nphases (capital): " + " ".join(f"t{r['turn']}:{r['capital_end'] / 1000:.1f}k"
                                            for r in rows if r['turn'] in (1, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40)))
    print(f"CSV: {out / name}")


if __name__ == "__main__":
    main()
