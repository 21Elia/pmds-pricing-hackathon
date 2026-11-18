"""Gabri+ : Gabriele's PolimiAgent structure (one method per phase), made adaptive.

Changes vs PolimiAgent baseline:
- sell markup per good adapts to sell-through (sold out -> up, unsold -> down)
- every unit is its own 1-unit listing on a small price ladder (customers pick among listings)
- bid markup per material rises when we are not filled, decays when we are
- units per good follow recent sales instead of a fixed 1
- only reads the NEW tail of transaction lists (the history no longer grows)
All parameters can be overridden from the constructor: GabriPlus(sell_markup=3.0, ...).
"""

from __future__ import annotations

import math

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig


class GabriPlus(BaseAgent):
    # Tunable parameters
    bid_markup = 0.5          # initial bid = reserve * (1 + bid_markup)
    bid_markup_max = 5.0      # never above reserve * (1 + this)
    sell_markup = 4.0         # initial sale price = materials cost * sell_markup
    min_markup = 1.3          # never sell below cost * this
    ladder = (0.7, 1.1)       # per-unit listing prices span [low, high] * reference
    min_units = 2             # stock target per good = min_units + cover * recent sales
    cover = 2.0
    max_units = 25
    budget_share = 0.7

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if not hasattr(type(self), k):
                raise TypeError(f"unknown parameter {k}")
            setattr(self, k, v)
        self.me = None
        self.markup = {}       # good -> current sell markup
        self.rate = {}         # good -> recent units sold per turn
        self.bidm = {}         # material -> current bid markup
        self.asked = {}        # material -> units bid last turn
        self.listed = {}       # good -> units listed last turn
        self.seen_a = 0
        self.seen_s = 0

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            self._learn(state, goods_config)
            costs = self._costs(state, goods_config)
            plan = self._plan(state, goods_config, costs)
            return Decision(
                bids=self._bids(state, goods_config, plan),
                assembly=self._assembly(goods_config, plan, costs),
                listings=self._listings(state, goods_config, costs, plan),
            )
        except Exception:
            return Decision()

    # --- memory -----------------------------------------------------------
    def _learn(self, state, gc):
        if self.me is None:
            m = [t for t, c in state.public_capital.items()
                 if abs(c - state.capital) < 1e-6 and state.public_inventories.get(t) == state.inventory]
            if len(m) == 1:
                self.me = m[0]
        mine = lambda who: self.me is None or who == self.me
        won = {}
        for tx in state.auction_transactions[self.seen_a:]:
            if mine(tx.buyer):
                won[tx.good] = won.get(tx.good, 0) + tx.quantity
        self.seen_a = len(state.auction_transactions)
        sold = {}
        for tx in state.sales_transactions[self.seen_s:]:
            if mine(tx.seller):
                sold[tx.good] = sold.get(tx.good, 0) + tx.quantity
        self.seen_s = len(state.sales_transactions)

        for m in gc.raw_materials:
            self.bidm.setdefault(m, self.bid_markup)
            q = self.asked.get(m, 0)
            if q > 0:
                fill = won.get(m, 0) / q
                if fill < 0.5:
                    self.bidm[m] = min(self.bid_markup_max, self.bidm[m] * 1.3 + 0.05)
                elif fill > 0.95:
                    self.bidm[m] = max(0.0, self.bidm[m] * 0.92)
        for g in gc.finished_goods:
            self.markup.setdefault(g, self.sell_markup)
            self.rate.setdefault(g, 1.0)
            s = sold.get(g, 0)
            self.rate[g] = 0.7 * self.rate[g] + 0.3 * s
            on_sale = min(self.listed.get(g, 0), s + state.inventory.get(g, 0))
            if on_sale > 0:
                f = s / on_sale
                if f >= 0.999:
                    self.markup[g] *= 1.12
                elif f >= 0.5:
                    self.markup[g] *= 1.03
                elif f > 0:
                    self.markup[g] *= 0.95
                else:
                    self.markup[g] *= 0.85
                self.markup[g] = max(self.markup[g], self.min_markup)

    def _costs(self, state, gc):
        lots = {l.good: l for l in state.auction_lots}
        unit = {m: lots[m].reserve_price * (1 + self.bidm.get(m, self.bid_markup)) if m in lots else 1.0
                for m in gc.raw_materials}
        best = {}
        for name, r in gc.recipes.items():
            c = sum(unit.get(m, 1.0) * a for m, a in r.inputs.items())
            if r.output_good not in best or c < best[r.output_good][0]:
                best[r.output_good] = (c, name, r)
        return best

    def _plan(self, state, gc, costs):
        # units to build per good, best margin per material unit first
        order = sorted(costs, key=lambda g: -(costs[g][0] * (self.markup[g] - 1))
                       / max(1, sum(costs[g][2].inputs.values())))
        plan = {}
        for g in order:
            target = min(self.max_units, int(round(self.min_units + self.cover * self.rate[g])))
            plan[g] = max(0, target - state.inventory.get(g, 0))
        return plan

    # Phase 1: auction
    def _bids(self, state, gc, plan) -> list[Bid]:
        wanted = {}
        costs = self._costs(state, gc)  # same costs as decide(): bid markups unchanged in between
        for g, q in plan.items():
            for m, a in costs[g][2].inputs.items():
                wanted[m] = wanted.get(m, 0) + a * q
        bids, budget, self.asked = [], state.capital * self.budget_share, {}
        for lot in state.auction_lots:
            q = min(max(0, wanted.get(lot.good, 0) - state.inventory.get(lot.good, 0)), lot.quantity)
            p = lot.reserve_price * (1 + self.bidm.get(lot.good, self.bid_markup))
            q = min(q, int(budget // p)) if p > 0 else 0
            if q > 0 and math.isfinite(p):
                bids.append(Bid(lot.good, int(q), float(p)))
                budget -= q * p
                self.asked[lot.good] = int(q)
        return bids

    # Phase 2: production (the simulator caps it at the materials you actually hold)
    def _assembly(self, gc, plan, costs) -> list[AssemblyOrder]:
        return [AssemblyOrder(costs[g][1], int(q)) for g, q in plan.items() if q > 0]

    # Phase 3: sales: one unit per listing on a ladder around cost * markup
    def _listings(self, state, gc, costs, plan) -> list[Listing]:
        listings, self.listed = [], {}
        lo, hi = self.ladder
        for g in gc.finished_goods:
            if g not in costs:
                continue
            n = state.inventory.get(g, 0) + plan.get(g, 0)
            if n <= 0:
                continue
            ref = costs[g][0] * self.markup[g]
            floor = costs[g][0] * self.min_markup
            for i in range(n):
                frac = 0.5 if n == 1 else i / (n - 1)
                p = max(floor, ref * (lo + (hi - lo) * frac))
                if math.isfinite(p) and p > 0:
                    listings.append(Listing(g, 1, float(round(p, 2))))
            self.listed[g] = n
        return listings
