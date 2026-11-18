"""Write a 'zoo' of simple, diverse opponents (like other teams' agents) into sparring_zoo/.

Each zoo agent: bids reserve * bid_mult for enough materials to keep `stock` units of
every good, prices at material cost * markup (optionally random in [lo, hi]), lists
all stock in one listing. Folder names become team IDs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sparring_zoo"

TEMPLATE = '''import random

from dme_simulator.agents import AssemblyOrder, BaseAgent, Bid, Decision, Listing, Observation
from dme_simulator.config import GoodsConfig


class ZooAgent(BaseAgent):
    bid_mult = {bid_mult}
    markup_lo = {lo}
    markup_hi = {hi}
    stock = {stock}
    split = {split}

    def __init__(self):
        self.rng = random.Random({seed})

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        gc = goods_config
        lots = {{lot.good: lot for lot in state.auction_lots}}
        cost = {{m: lots[m].reserve_price * self.bid_mult if m in lots else 10.0 for m in gc.raw_materials}}
        inv = dict(state.inventory)
        want = {{m: 0 for m in gc.raw_materials}}
        assembly = []
        for name, rec in gc.recipes.items():
            need = max(0, self.stock - inv.get(rec.output_good, 0))
            assembly.append(AssemblyOrder(name, need))
            for m, q in rec.inputs.items():
                want[m] += q * need
        bids = []
        for m, lot in lots.items():
            q = min(lot.quantity, max(0, want.get(m, 0) - inv.get(m, 0)))
            if q > 0:
                bids.append(Bid(m, q, lot.reserve_price * self.bid_mult))
        listings = []
        for name, rec in gc.recipes.items():
            g = rec.output_good
            c = sum(cost[m] * q for m, q in rec.inputs.items())
            n = inv.get(g, 0) + self.stock
            mk = self.rng.uniform(self.markup_lo, self.markup_hi)
            if self.split:
                listings += [Listing(g, 1, c * mk) for _ in range(n)]
            else:
                listings.append(Listing(g, n, c * mk))
        return Decision(bids=bids, assembly=assembly, listings=listings)
'''

ZOO = {
    "zoo_cheap": dict(bid_mult=1.2, lo=2.0, hi=2.0, stock=3, split=False),
    "zoo_mid": dict(bid_mult=1.3, lo=4.0, hi=4.0, stock=3, split=False),
    "zoo_high": dict(bid_mult=1.2, lo=8.0, hi=8.0, stock=3, split=False),
    "zoo_random": dict(bid_mult=1.4, lo=2.0, hi=10.0, stock=4, split=False),
    "zoo_bidder": dict(bid_mult=3.0, lo=3.0, hi=5.0, stock=5, split=False),
    "zoo_splitter": dict(bid_mult=1.5, lo=3.0, hi=6.0, stock=4, split=True),
    "zoo_hoarder": dict(bid_mult=1.3, lo=3.0, hi=3.0, stock=10, split=False),
    "zoo_premium": dict(bid_mult=1.1, lo=12.0, hi=12.0, stock=2, split=False),
}


def random_zoo(n: int, seed: int = 7) -> dict:
    """Extra random opponents so the field is portal-sized (~30 teams)."""
    import random
    rng = random.Random(seed)
    out = {}
    for i in range(n):
        lo = rng.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
        out[f"zoo_r{i:02d}"] = dict(
            bid_mult=round(rng.choice([1.05, 1.2, 1.5, 2.0, 3.0]), 2),
            lo=lo, hi=round(lo * rng.choice([1.0, 1.0, 1.5, 2.5]), 2),
            stock=rng.choice([1, 2, 3, 5, 8]),
            split=rng.random() < 0.25,
        )
    return out


def main() -> None:
    import sys
    if OUT.exists():
        shutil.rmtree(OUT)
    zoo = dict(ZOO)
    zoo.update(random_zoo(int(sys.argv[1]) if len(sys.argv) > 1 else 14))
    for i, (name, p) in enumerate(zoo.items()):
        (OUT / name).mkdir(parents=True)
        (OUT / name / "agent.py").write_text(TEMPLATE.format(seed=i, **p))
    print(f"written {len(zoo)} zoo agents")


if __name__ == "__main__":
    main()
