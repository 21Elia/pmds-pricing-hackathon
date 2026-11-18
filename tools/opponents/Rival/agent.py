"""Library of opponent styles for local testing (port of yesterday's legacy/sim/opponents.py
to the official DME rules). Construct with Rival(style="..."); default style is "fixed".

Styles:
  fixed      fixed markup on material cost, fixed bid premium (the team that never adapts)
  undercut   lists below its last price until it hits cost (price war)
  highprice  very high markup, bids at reserve
  overbid    bids 2x reserve "to be sure", large quantities, low markup
  hoarder    tries to corner the scarcest material, sells at high markup
  dumper     sells near cost for the first turns (market share), then raises
  bandit     epsilon-greedy over markups, rewarded by realised profit
  valuebid   adaptive prices like ours, but bids up to half the product's value
  trembler   random but valid actions
"""

from __future__ import annotations

import math
import random
import zlib

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig

STYLES = ("fixed", "undercut", "highprice", "overbid", "hoarder", "dumper",
          "bandit", "valuebid", "trembler")


class Rival(BaseAgent):
    def __init__(self, style: str = "fixed", seed: int = 0, markup: float | None = None,
                 units: int | None = None, bid_mult: float | None = None):
        self.style = style if style in STYLES else "fixed"
        self.rng = random.Random(zlib.crc32(f"{style}:{seed}".encode()))
        self.markup = markup
        self.units = units
        self.bid_mult_override = bid_mult
        self.price = {}
        self.sales_idx = 0
        self.auct_idx = 0
        self.turn0 = None
        self.arms = [1.5, 2.0, 3.0, 4.0, 6.0]
        self.arm_value = [0.0] * len(self.arms)
        self.arm_n = [0] * len(self.arms)
        self.arm = 2
        self.last_capital = None

    # ------------------------------------------------------------------
    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            return self._decide(state, goods_config)
        except Exception:
            return Decision()

    def _decide(self, state, gc):
        lots = {l.good: l for l in state.auction_lots}
        ref = {m: (lots[m].reserve_price if m in lots else 1.0) for m in gc.raw_materials}
        sold = {}
        for tr in state.sales_transactions[self.sales_idx:]:
            sold[tr.good] = sold.get(tr.good, 0) + tr.quantity
        self.sales_idx = len(state.sales_transactions)
        self.auct_idx = len(state.auction_transactions)

        cost = {}
        recipe_for = {}
        for name, r in gc.recipes.items():
            c = sum(ref.get(m, 1.0) * q for m, q in r.inputs.items())
            if r.output_good not in cost or c < cost[r.output_good]:
                cost[r.output_good] = c
                recipe_for[r.output_good] = (name, r)

        s = self.style
        units = self.units if self.units is not None else 3
        bid_mult = 1.2
        budget_share = 0.6
        goods = [g for g in gc.finished_goods if g in recipe_for]

        if s == "fixed":
            mk = self.markup or 3.0
            price = {g: cost[g] * mk for g in goods}
        elif s == "highprice":
            mk = self.markup or 6.0
            price = {g: cost[g] * mk for g in goods}
            bid_mult, units = 1.0, (self.units or 2)
        elif s == "overbid":
            mk = self.markup or 2.5
            price = {g: cost[g] * mk for g in goods}
            bid_mult, units, budget_share = 2.0, (self.units or 6), 0.8
        elif s == "undercut":
            price = {}
            for g in goods:
                p = self.price.get(g, cost[g] * 4.0)
                p = p * 0.93
                price[g] = max(cost[g] * 1.05, p)
            units = self.units or 5
        elif s == "dumper":
            t = state.turn
            mk = 1.15 if t <= 12 else (self.markup or 4.0)
            price = {g: cost[g] * mk for g in goods}
            units = self.units or (6 if t <= 12 else 3)
        elif s == "bandit":
            if self.last_capital is not None:
                reward = state.capital - self.last_capital
                self.arm_n[self.arm] += 1
                self.arm_value[self.arm] += (reward - self.arm_value[self.arm]) / self.arm_n[self.arm]
            if self.rng.random() < 0.15:
                self.arm = self.rng.randrange(len(self.arms))
            else:
                self.arm = max(range(len(self.arms)), key=lambda i: self.arm_value[i])
            price = {g: cost[g] * self.arms[self.arm] for g in goods}
        elif s == "valuebid":
            price = {}
            for g in goods:
                p = self.price.get(g, cost[g] * 4.0)
                held = state.inventory.get(g, 0)
                if g in self.price:
                    p *= 1.08 if held == 0 else 0.93
                price[g] = max(cost[g] * 1.1, p)
            units = self.units or 4
        elif s == "hoarder":
            mk = self.markup or 5.0
            price = {g: cost[g] * mk for g in goods}
        else:  # trembler
            price = {g: cost[g] * self.rng.uniform(1.0, 6.0) for g in goods}
            units = self.rng.randint(0, 6)
            bid_mult = self.rng.uniform(1.0, 1.8)
        self.price = dict(price)

        if self.bid_mult_override is not None:
            bid_mult = self.bid_mult_override
        # demand per good: build to `units` held
        need = {g: max(0, units - state.inventory.get(g, 0)) for g in goods}
        raw = {}
        for g, q in need.items():
            for m, a in recipe_for[g][1].inputs.items():
                raw[m] = raw.get(m, 0) + a * q

        bids = []
        budget = state.capital * budget_share
        if s == "hoarder" and lots:
            scarce = min(lots.values(), key=lambda l: l.quantity / max(1, raw.get(l.good, 1)))
            q = int(min(scarce.quantity * 0.7, budget * 0.5 / (scarce.reserve_price * 1.5)))
            if q > 0:
                bids.append(Bid(scarce.good, q, scarce.reserve_price * 1.5))
                budget -= q * scarce.reserve_price * 1.5
        for m, q in raw.items():
            if m not in lots or q <= 0:
                continue
            q = max(0, q - state.inventory.get(m, 0))
            if s == "valuebid":
                values = [price[g] / max(1, sum(recipe_for[g][1].inputs.values()))
                          for g in goods if m in recipe_for[g][1].inputs]
                p = max(lots[m].reserve_price, 0.5 * (sum(values) / len(values)) if values else 0)
                p = min(p, lots[m].reserve_price * 3.0)
            else:
                p = lots[m].reserve_price * bid_mult
            q = min(q, int(max(0.0, budget) // max(p, 1e-9)), lots[m].quantity)
            if q > 0 and math.isfinite(p):
                bids.append(Bid(m, int(q), float(p)))
                budget -= q * p

        assembly = [AssemblyOrder(recipe_for[g][0], int(need[g])) for g in goods if need[g] > 0]
        listings = []
        for g in goods:
            q = state.inventory.get(g, 0) + need[g]
            if q > 0 and price[g] > 0 and math.isfinite(price[g]):
                listings.append(Listing(g, int(q), float(price[g])))
        self.last_capital = state.capital
        return Decision(bids=bids, assembly=assembly, listings=listings)
