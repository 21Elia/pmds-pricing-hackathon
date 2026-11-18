"""PolimiDataScientists - adaptive pricing / bidding agent for the DME simulator.

Per turn:
  1. read only the NEW tail of our auction/sales transactions
  2. adapt list prices on sell-through (sold + leftover = what we really listed)
  3. target a small stock per good (end of game is unknown and stock is worth 0)
  4. bid for the missing kits; the bid moves between the reserve price and the
     material's value (product price minus the other inputs), more aggressive
     when we were not filled; a full kit's bids are capped at 80% of the
     product price (without the cap, copies of us overbid each other into losses)
  5. assemble and list in the same turn, optionally split into several
     listings (customers pick among listings) and a premium tier
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


class PolimiAgent(BaseAgent):
    def __init__(
        self,
        bid_mode: str = "value",     # "value": reserve..value ; "markup": reserve * markup (v1)
        share0: float = 0.10,        # initial share of (value - reserve) we add to reserve
        share_max: float = 0.60,
        value_cap: float = 0.85,     # never bid above this fraction of material value
        value_alloc: str = "full",   # "full": each input may take the whole margin (aggressive)
                                     # "prop": product price split over inputs by cost share
        kit_cap: float = 0.80,       # a full kit's bids never exceed kit_cap * product price (0 = off, v2)
        split: int = 1000,           # listings per good (1000 = one listing per unit: customers pick among LISTINGS)
        tier: float = 0.0,           # fraction of units listed at a premium price
        tier_mult: float = 1.3,
        init_mult: float = 4.0,      # first list price = mult * material cost
        min_margin: float = 1.15,    # never list below cost * this
        up: float = 1.08,            # price step after a sell-out
        down: float = 0.88,          # price step after selling nothing
        budget_share: float = 0.5,
        init_demand: float = 2.0,
        max_demand: float = 30.0,
    ):
        self.bid_mode = bid_mode
        self.share0, self.share_max, self.value_cap = share0, share_max, value_cap
        self.value_alloc, self.kit_cap = value_alloc, kit_cap
        self.split, self.tier, self.tier_mult = max(1, int(split)), tier, tier_mult
        self.init_mult, self.min_margin = init_mult, min_margin
        self.up, self.down = up, down
        self.budget_share = budget_share
        self.init_demand, self.max_demand = init_demand, max_demand

        self.me = None
        self.price = {}          # good -> current list price
        self.demand = {}         # good -> estimated sellable units per turn
        self.listed = {}         # good -> qty listed last turn
        self.unit_cost = {}      # material -> estimated unit cost
        self.share = {}          # material -> bid aggressiveness (value mode)
        self.markup = {}         # material -> bid markup (markup mode)
        self.bid_qty = {}        # material -> qty bid last turn
        self.sales_idx = 0
        self.auct_idx = 0

    # ------------------------------------------------------------------
    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            return self._decide(state, goods_config)
        except Exception:
            try:
                return self._fallback(state, goods_config)
            except Exception:
                return Decision()

    # ------------------------------------------------------------------
    def _identify(self, state):
        if self.me is not None:
            return
        cands = [
            t for t, c in state.public_capital.items()
            if abs(c - state.capital) < 1e-6
            and state.public_inventories.get(t, {}) == state.inventory
        ]
        if len(cands) == 1:
            self.me = cands[0]

    def _mine(self, who):
        return self.me is None or who == self.me

    # ------------------------------------------------------------------
    def _decide(self, state, gc):
        self._identify(state)
        lots = {lot.good: lot for lot in state.auction_lots}

        # --- new transactions only ---
        sold = {}
        for tr in state.sales_transactions[self.sales_idx:]:
            if self._mine(tr.seller):
                sold[tr.good] = sold.get(tr.good, 0) + tr.quantity
        self.sales_idx = len(state.sales_transactions)

        won, paid = {}, {}
        for tr in state.auction_transactions[self.auct_idx:]:
            if self._mine(tr.buyer):
                won[tr.good] = won.get(tr.good, 0) + tr.quantity
                paid[tr.good] = paid.get(tr.good, 0.0) + tr.price * tr.quantity
        self.auct_idx = len(state.auction_transactions)

        # --- material cost estimate (what a unit costs us now) ---
        for m in gc.raw_materials:
            self.share.setdefault(m, self.share0)
            self.markup.setdefault(m, 1.05)
            r = lots[m].reserve_price if m in lots else None
            est = self.unit_cost.get(m)
            if m in won and won[m] > 0:
                p = paid[m] / won[m]
                est = p if est is None else 0.5 * est + 0.5 * p
            elif r is not None:
                guess = r * (1.0 + self.share[m]) if self.bid_mode == "value" else r * self.markup[m]
                est = guess if est is None else 0.7 * est + 0.3 * guess
            if est is not None:
                self.unit_cost[m] = est

        # --- adapt bid aggressiveness on last turn's fill rate ---
        for m, q in self.bid_qty.items():
            if q <= 0:
                continue
            fill = won.get(m, 0) / q
            if fill >= 0.95:
                self.share[m] = max(0.0, self.share[m] - 0.03)
                self.markup[m] = max(1.0, self.markup[m] - 0.02)
            elif fill < 0.5:
                self.share[m] = min(self.share_max, self.share[m] + 0.08)
                self.markup[m] = min(1.6, self.markup[m] + 0.06)
            else:
                self.share[m] = min(self.share_max, self.share[m] + 0.03)
                self.markup[m] = min(1.6, self.markup[m] + 0.02)

        # --- cheapest recipe per good and its cost ---
        good_cost, best_recipe = {}, {}
        for name, rec in gc.recipes.items():
            c = sum(self.unit_cost.get(m, 0.0) * q for m, q in rec.inputs.items())
            g = rec.output_good
            if g not in good_cost or c < good_cost[g]:
                good_cost[g] = c
                best_recipe[g] = (name, rec)

        # --- prices and demand from last turn's sell-through ---
        for g in gc.finished_goods:
            cost = good_cost.get(g, 0.0)
            if g not in self.price:
                self.price[g] = max(cost * self.init_mult, 1.0)
                self.demand[g] = self.init_demand
            s = sold.get(g, 0)
            lq = (s + state.inventory.get(g, 0)) if self.listed.get(g, 0) > 0 else 0
            if lq > 0:
                f = s / lq
                if f >= 0.999:
                    self.price[g] *= self.up
                    self.demand[g] = min(self.max_demand, max(self.demand[g], s) + 1.0)
                elif f >= 0.5:
                    self.price[g] *= 1.02
                    self.demand[g] = 0.6 * self.demand[g] + 0.4 * (s + 0.5)
                elif f > 0:
                    self.price[g] *= 0.95
                    self.demand[g] = max(1.0, 0.6 * self.demand[g] + 0.4 * s)
                else:
                    self.price[g] *= self.down
                    self.demand[g] = max(1.0, 0.7 * self.demand[g])
            self.price[g] = max(self.price[g], cost * self.min_margin, 0.01)

        # --- production plan: profitable goods, best margin ratio first ---
        inv = state.inventory
        margin = {g: self.price[g] - good_cost.get(g, 0.0) for g in gc.finished_goods}
        order = sorted(
            [g for g in gc.finished_goods if g in best_recipe and margin[g] > 0],
            key=lambda g: -margin[g] / max(good_cost.get(g, 1.0), 1.0),
        )
        need_build = {g: max(0, int(math.ceil(self.demand[g])) - inv.get(g, 0)) for g in order}

        raw_need = {}
        for g in order:
            q = need_build[g]
            if q > 0:
                for m, a in best_recipe[g][1].inputs.items():
                    raw_need[m] = raw_need.get(m, 0) + a * q

        # --- material values: what one unit is worth to us (best use) ---
        ref = {m: (lots[m].reserve_price if m in lots else self.unit_cost.get(m, 0.0))
               for m in gc.raw_materials}
        value = {}
        for g in order:
            rec = best_recipe[g][1]
            kit_ref = sum(ref.get(m, 0.0) * a for m, a in rec.inputs.items())
            for m, a in rec.inputs.items():
                if self.value_alloc == "prop" and kit_ref > 0:
                    # the kit is worth the product price, split by each input's cost share
                    v = ref.get(m, 0.0) * self.price[g] / kit_ref
                else:
                    others = sum(self.unit_cost.get(o, 0.0) * qa
                                 for o, qa in rec.inputs.items() if o != m)
                    v = (self.price[g] - others) / a
                value[m] = max(value.get(m, 0.0), v)

        # --- bid price per material ---
        bid_price = {}
        for m in raw_need:
            if m not in lots:
                continue
            r = lots[m].reserve_price
            if self.bid_mode == "value":
                v = value.get(m, r)
                if v <= r:
                    continue
                p = min(r + self.share[m] * (v - r), self.value_cap * v)
                bid_price[m] = max(p, r)
            else:
                bid_price[m] = r * self.markup[m]

        # --- kit cap: the bids for one full kit stay below kit_cap * product price ---
        if self.kit_cap > 0:
            factor = {}
            for g in order:
                if need_build[g] <= 0:
                    continue
                rec = best_recipe[g][1]
                if any(m not in bid_price for m in rec.inputs if m in lots):
                    continue
                kit_bid = sum(bid_price.get(m, self.unit_cost.get(m, 0.0)) * a
                              for m, a in rec.inputs.items())
                kit_res = sum((lots[m].reserve_price if m in lots else self.unit_cost.get(m, 0.0)) * a
                              for m, a in rec.inputs.items())
                cap = self.kit_cap * self.price[g]
                if kit_bid > cap:
                    f = 0.0 if kit_bid <= kit_res else max(0.0, min(1.0, (cap - kit_res) / (kit_bid - kit_res)))
                    for m in rec.inputs:
                        factor[m] = min(factor.get(m, 1.0), f)
            for m, f in factor.items():
                if m in bid_price:
                    r = lots[m].reserve_price
                    bid_price[m] = r + (bid_price[m] - r) * f

        plan = []
        for m, q in raw_need.items():
            q -= inv.get(m, 0)
            if q <= 0 or m not in bid_price:
                continue
            plan.append([m, min(q, lots[m].quantity), bid_price[m]])
        budget = max(0.0, state.capital * self.budget_share)
        total = sum(q * p for _, q, p in plan)
        scale = min(1.0, budget / total) if total > 0 else 1.0
        bids, self.bid_qty = [], {}
        for m, q, p in plan:
            qq = int(q * scale)
            if qq > 0 and math.isfinite(p) and p > 0:
                bids.append(Bid(m, qq, float(round(p, 4))))
                self.bid_qty[m] = qq

        # --- assembly (simulator caps by materials actually held) ---
        assembly = [AssemblyOrder(best_recipe[g][0], int(need_build[g]))
                    for g in order if need_build[g] > 0]

        # --- listings ---
        listings, self.listed = [], {}
        for g in gc.finished_goods:
            q = inv.get(g, 0) + need_build.get(g, 0)
            p = self.price[g]
            if q <= 0 or not math.isfinite(p) or p <= 0:
                continue
            self.listed[g] = int(q)
            n_hi = int(q * self.tier) if self.tier > 0 else 0
            if n_hi > 0:
                listings.append(Listing(g, n_hi, float(round(p * self.tier_mult, 4))))
            rest = q - n_hi
            k = min(self.split, rest)
            for i in range(k):
                part = rest // k + (1 if i < rest % k else 0)
                if part > 0:
                    listings.append(Listing(g, int(part), float(round(p, 4))))
        return Decision(bids=bids, assembly=assembly, listings=listings)

    # ------------------------------------------------------------------
    def _fallback(self, state, gc):
        listings = []
        for g, q in state.inventory.items():
            if g in gc.finished_goods and q > 0:
                p = self.price.get(g, 100.0)
                listings.append(Listing(g, int(q), float(p if math.isfinite(p) and p > 0 else 100.0)))
        return Decision(listings=listings)
