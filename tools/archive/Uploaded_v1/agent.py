"""PolimiDataScientists - adaptive pricing / bidding agent for the DME simulator."""

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
    # --- tunables ---
    INIT_PRICE_MULT = 4.0      # first list price = mult * material cost
    MIN_MARGIN = 1.15          # never list below cost * this
    BID_MARKUP0 = 1.05         # start bidding at reserve * this
    BID_MARKUP_MAX = 1.6
    BUDGET_SHARE = 0.5         # max share of capital committed to bids per turn
    INIT_DEMAND = 2.0          # expected units sold per good per turn at start
    MAX_DEMAND = 30.0

    def __init__(self):
        self.me = None
        self.price = {}          # good -> current list price
        self.demand = {}         # good -> estimated sellable units per turn
        self.listed = {}         # good -> qty listed last turn
        self.unit_cost = {}      # material -> last known unit cost
        self.markup = {}         # material -> bid markup over reserve
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

    def _recipe_cost(self, recipe):
        return sum(self.unit_cost.get(m, 0.0) * q for m, q in recipe.inputs.items())

    # ------------------------------------------------------------------
    def _decide(self, state, gc):
        self._identify(state)
        lots = {lot.good: lot for lot in state.auction_lots}

        # material cost estimate: blend of reserve and what we actually paid
        for m in gc.raw_materials:
            if m in lots:
                r = lots[m].reserve_price
                prev = self.unit_cost.get(m)
                self.unit_cost[m] = r * self.markup.get(m, self.BID_MARKUP0) if prev is None \
                    else 0.5 * prev + 0.5 * r * self.markup.get(m, self.BID_MARKUP0)
            self.markup.setdefault(m, self.BID_MARKUP0)

        # --- read new transactions (only the tail) ---
        sold = {}
        for tr in state.sales_transactions[self.sales_idx:]:
            if self._mine(tr.seller):
                sold[tr.good] = sold.get(tr.good, 0) + tr.quantity
        self.sales_idx = len(state.sales_transactions)

        won = {}
        for tr in state.auction_transactions[self.auct_idx:]:
            if self._mine(tr.buyer):
                won[tr.good] = won.get(tr.good, 0) + tr.quantity
                prev = self.unit_cost.get(tr.good, tr.price)
                self.unit_cost[tr.good] = 0.5 * prev + 0.5 * tr.price
        self.auct_idx = len(state.auction_transactions)

        # --- adapt bid markups on fill rate ---
        for m, q in self.bid_qty.items():
            if q <= 0:
                continue
            fill = won.get(m, 0) / q
            if fill >= 0.95:
                self.markup[m] = max(1.0, self.markup[m] - 0.02)
            elif fill < 0.5:
                self.markup[m] = min(self.BID_MARKUP_MAX, self.markup[m] + 0.06)
            else:
                self.markup[m] = min(self.BID_MARKUP_MAX, self.markup[m] + 0.02)

        # cost per finished good (cheapest recipe)
        good_cost = {}
        good_recipes = {}
        for name, rec in gc.recipes.items():
            c = self._recipe_cost(rec)
            g = rec.output_good
            good_recipes.setdefault(g, []).append((c, name, rec))
            if g not in good_cost or c < good_cost[g]:
                good_cost[g] = c

        # --- adapt prices and demand on last turn's sell-through ---
        for g in gc.finished_goods:
            cost = good_cost.get(g, 0.0)
            if g not in self.price:
                self.price[g] = max(cost * self.INIT_PRICE_MULT, 1.0)
                self.demand[g] = self.INIT_DEMAND
            s = sold.get(g, 0)
            # actually listed = sold + leftover (we list all stock we hold)
            lq = (s + state.inventory.get(g, 0)) if self.listed.get(g, 0) > 0 else 0
            if lq > 0:
                f = s / lq
                if f >= 0.999:
                    self.price[g] *= 1.08
                    self.demand[g] = min(self.MAX_DEMAND, max(self.demand[g], s) + 1.0)
                elif f >= 0.5:
                    self.price[g] *= 1.02
                    self.demand[g] = 0.6 * self.demand[g] + 0.4 * (s + 0.5)
                elif f > 0:
                    self.price[g] *= 0.95
                    self.demand[g] = max(1.0, 0.6 * self.demand[g] + 0.4 * s)
                else:
                    self.price[g] *= 0.88
                    self.demand[g] = max(1.0, 0.7 * self.demand[g])
            self.price[g] = max(self.price[g], cost * self.MIN_MARGIN, 0.01)

        # --- production plan: target stock = demand, profitable goods only ---
        inv = dict(state.inventory)
        margin = {g: self.price[g] - good_cost.get(g, 0.0) for g in gc.finished_goods}
        order = sorted(
            [g for g in gc.finished_goods if g in good_recipes and margin[g] > 0],
            key=lambda g: -margin[g] / max(good_cost.get(g, 1.0), 1.0),
        )
        need_build = {}
        for g in order:
            target = int(math.ceil(self.demand[g]))
            need_build[g] = max(0, target - inv.get(g, 0))

        # materials required
        raw_need = {}
        for g in order:
            q = need_build[g]
            if q <= 0:
                continue
            _, _, rec = min(good_recipes[g], key=lambda x: x[0])
            for m, a in rec.inputs.items():
                raw_need[m] = raw_need.get(m, 0) + a * q

        bids = []
        budget = max(0.0, state.capital * self.BUDGET_SHARE)
        plan = []
        for m, q in raw_need.items():
            q = q - inv.get(m, 0)
            if q <= 0 or m not in lots:
                continue
            lot = lots[m]
            q = min(q, lot.quantity)
            p = lot.reserve_price * self.markup[m]
            plan.append([m, q, p])
        total = sum(q * p for _, q, p in plan)
        scale = min(1.0, budget / total) if total > 0 else 1.0
        self.bid_qty = {}
        for m, q, p in plan:
            qq = int(q * scale)
            if qq > 0 and math.isfinite(p) and p > 0:
                bids.append(Bid(m, qq, round(p, 4)))
                self.bid_qty[m] = qq

        # assembly: request needed units; simulator caps by materials held
        assembly = []
        for g in order:
            q = need_build[g]
            if q > 0:
                _, name, _ = min(good_recipes[g], key=lambda x: x[0])
                assembly.append(AssemblyOrder(name, int(q)))

        # listings: everything we hold or plan to build
        listings = []
        self.listed = {}
        for g in gc.finished_goods:
            q = inv.get(g, 0) + need_build.get(g, 0)
            if q <= 0:
                continue
            p = self.price[g]
            if not math.isfinite(p) or p <= 0:
                continue
            listings.append(Listing(g, int(q), round(p, 4)))
            # what can actually be listed is capped by real stock; estimate it
            self.listed[g] = int(q)
        return Decision(bids=bids, assembly=assembly, listings=listings)

    # ------------------------------------------------------------------
    def _fallback(self, state, gc):
        listings = [
            Listing(g, int(q), 100.0)
            for g, q in state.inventory.items()
            if g in gc.finished_goods and q > 0
        ]
        return Decision(listings=listings)
