"""Tight v3 (= v1 + aggressive opening): adaptive bidding, margin-ranked production, per-unit price ladder.

Ideas:
- Pay-your-bid auction: bid just above what wins. Per material we keep a
  price level that rises when we were not filled and decays when we were.
- Production: keep a small stock of every finished good, giving scarce
  materials to the goods with the highest margin first.
- Sales: every unit is listed as its own 1-unit listing on a price ladder
  around a reference price. Each listing counts as a separate offer for the
  customers, and the ladder tells us whether the reference is too high or low.
"""

from __future__ import annotations

import math

from dme_simulator.agents import (
    AssemblyOrder, BaseAgent, Bid, Decision, Listing, Observation,
)
from dme_simulator.config import GoodsConfig


class TightAgent(BaseAgent):
    # --- pricing ---
    prior_markup = 4.0          # first reference price = material cost * this
    ladder_low = 0.6           # ladder spans [low, high] * reference
    ladder_high = 1.0
    explore_turns = 6           # wider ladder and faster moves early on
    step_up = 0.07
    step_down = 0.15
    min_markup = 1.3            # never list below material cost * this
    # --- stock and production ---
    min_stock = 2
    max_stock = 30
    cover_turns = 3.0           # target stock = sales rate * cover + min_stock
    # --- bidding ---
    bid_start = 1.5            # first bid = reserve * this
    bid_up = 1.5
    bid_down = 0.94
    value_share = 0.8          # max bid per material unit = share of product value
    budget_share = 0.8          # max share of capital committed per turn
    # --- opening: in the first turns few teams have goods, so be first to market ---
    max_bid_mult = 4.0       # never bid more than reserve * this
    open_turns = 5
    open_stock = 6              # minimum stock target per good during the opening
    open_bid = 3.0              # bid at least reserve * this during the opening
    early_low = 0.75            # price ladder during the exploration turns
    early_high = 1.6

    def __init__(self) -> None:
        self.me: str | None = None
        self.ref_price: dict[str, float] = {}
        self.sales_rate: dict[str, float] = {}
        self.bid_level: dict[str, float] = {}
        self.unit_cost: dict[str, float] = {}
        self.last_bids: dict[str, tuple[int, float]] = {}
        self.last_ladder: dict[str, list[float]] = {}
        self.seen_auction = 0
        self.seen_sales = 0

    # ------------------------------------------------------------------
    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            return self._decide(state, goods_config)
        except Exception:
            return self._fallback(state, goods_config)

    def _fallback(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        listings = []
        for good in goods_config.finished_goods:
            qty = int(state.inventory.get(good, 0))
            price = self.ref_price.get(good)
            if qty > 0 and price and math.isfinite(price) and price > 0:
                listings.append(Listing(good, qty, float(price)))
        return Decision(listings=listings)

    # ------------------------------------------------------------------
    def _identify(self, state: Observation) -> None:
        matches = [
            team
            for team, inv in state.public_inventories.items()
            if abs(state.public_capital.get(team, -1e18) - state.capital) < 1e-6
            and {k: v for k, v in inv.items() if v} == {k: v for k, v in state.inventory.items() if v}
        ]
        if len(matches) == 1:
            self.me = matches[0]

    def _mine(self, who: str) -> bool:
        return self.me is None or who == self.me

    def _decide(self, state: Observation, gc: GoodsConfig) -> Decision:
        self._identify(state)
        turn = state.turn
        inv = {k: int(v) for k, v in state.inventory.items()}
        lots = {lot.good: lot for lot in state.auction_lots}

        # --- learn from last turn's auction --------------------------------
        new_auction = state.auction_transactions[self.seen_auction:]
        self.seen_auction = len(state.auction_transactions)
        won: dict[str, int] = {}
        for tx in new_auction:
            if not self._mine(tx.buyer):
                continue
            won[tx.good] = won.get(tx.good, 0) + tx.quantity
            old = self.unit_cost.get(tx.good, tx.price)
            self.unit_cost[tx.good] = 0.7 * old + 0.3 * tx.price

        # --- learn from last turn's sales ----------------------------------
        new_sales = state.sales_transactions[self.seen_sales:]
        self.seen_sales = len(state.sales_transactions)
        sold_prices: dict[str, list[float]] = {}
        for tx in new_sales:
            if not self._mine(tx.seller):
                continue
            sold_prices.setdefault(tx.good, []).extend([tx.price] * tx.quantity)

        # material cost estimates (what we pay, else the reserve)
        mat_cost = {
            m: self.unit_cost.get(m, lots[m].reserve_price if m in lots else 1.0)
            for m in gc.raw_materials
        }
        # cheapest recipe per finished good
        recipe_for: dict[str, str] = {}
        good_cost: dict[str, float] = {}
        for name, rec in gc.recipes.items():
            c = sum(mat_cost.get(m, 1.0) * q for m, q in rec.inputs.items())
            g = rec.output_good
            if g not in good_cost or c < good_cost[g]:
                good_cost[g], recipe_for[g] = c, name

        early = turn <= self.explore_turns
        for g in gc.finished_goods:
            cost = max(good_cost.get(g, 1.0), 1e-6)
            if g not in self.ref_price:
                self.ref_price[g] = cost * self.prior_markup
                self.sales_rate[g] = 1.0
            ladder = self.last_ladder.get(g, [])
            sold = sorted(sold_prices.get(g, []))
            n_sold = len(sold)
            self.sales_rate[g] = 0.7 * self.sales_rate[g] + 0.3 * n_sold
            # units really on sale last turn = sold + still in stock (we list all)
            listed = min(len(ladder), n_sold + inv.get(g, 0))
            if listed > 0:
                up = self.step_up * (2.0 if early else 1.0)
                down = self.step_down * (1.5 if early else 1.0)
                ref = self.ref_price[g]
                if n_sold >= listed:
                    ref *= 1 + 2 * up                       # sold out: clearly too cheap
                elif n_sold and sold[-1] >= ref * 1.1:
                    ref *= 1 + up                           # upper rungs sell
                elif n_sold:
                    ref *= 1 - 0.3 * down                   # only cheap rungs sell
                else:
                    ref *= 1 - down                         # nothing sold
                self.ref_price[g] = max(ref, cost * self.min_markup)

        # --- production plan -------------------------------------------------
        target = {
            g: int(min(self.max_stock,
                       round(self.sales_rate[g] * self.cover_turns)
                       + (max(self.min_stock, self.open_stock) if turn <= self.open_turns else self.min_stock)))
            for g in gc.finished_goods
        }
        margin = {g: self.ref_price[g] - good_cost.get(g, 0.0) for g in gc.finished_goods}
        ranked = sorted(
            (g for g in gc.finished_goods if g in recipe_for and margin[g] > 0),
            key=lambda g: margin[g] / max(1, sum(gc.recipes[recipe_for[g]].inputs.values())),
            reverse=True,
        )
        need = {g: max(0, target[g] - inv.get(g, 0)) for g in ranked}

        # --- bid levels: raise only when a material is really starving us ----
        for mat, (asked, price) in self.last_bids.items():
            if asked <= 0:
                continue
            fill = won.get(mat, 0) / asked
            starving = any(
                mat in gc.recipes[recipe_for[g]].inputs
                and inv.get(g, 0) < max(1, target[g] // 2)
                for g in ranked
            )
            if fill < 0.8 and starving:
                self.bid_level[mat] = price * self.bid_up
            elif fill < 0.8:
                self.bid_level[mat] = price
            else:
                self.bid_level[mat] = price * self.bid_down
        assembly = [AssemblyOrder(recipe_for[g], need[g]) for g in ranked if need[g] > 0]

        # materials required for the plan, minus what we hold
        want = {m: 0 for m in gc.raw_materials}
        for g in ranked:
            for m, q in gc.recipes[recipe_for[g]].inputs.items():
                want[m] = want.get(m, 0) + q * need[g]

        # --- bids --------------------------------------------------------------
        bids = []
        self.last_bids = {}
        budget = state.capital * self.budget_share
        for m in gc.raw_materials:
            lot = lots.get(m)
            if lot is None or lot.quantity <= 0:
                continue
            qty = min(max(0, want.get(m, 0) - inv.get(m, 0)), lot.quantity)
            if qty <= 0:
                continue
            # value cap: share of the best product's price per input unit
            cap = 0.0
            for g in ranked:
                rec = gc.recipes[recipe_for[g]]
                if m in rec.inputs:
                    cap = max(cap, self.value_share * self.ref_price[g] / sum(rec.inputs.values()))
            level = self.bid_level.get(m, lot.reserve_price * self.bid_start)
            if turn <= self.open_turns:
                level = max(level, lot.reserve_price * self.open_bid)
            price = max(lot.reserve_price * 1.001, level)
            price = min(price, lot.reserve_price * self.max_bid_mult)
            if cap > 0:
                price = min(price, max(cap, lot.reserve_price * 1.001))
            qty = min(qty, int(budget // price))
            if qty <= 0:
                continue
            budget -= qty * price
            bids.append(Bid(m, int(qty), float(price)))
            self.last_bids[m] = (int(qty), float(price))

        # --- listings: one unit per listing on a ladder ------------------------
        listings = []
        self.last_ladder = {}
        low, high = (self.early_low, self.early_high) if early else (self.ladder_low, self.ladder_high)
        for g in gc.finished_goods:
            planned = inv.get(g, 0) + need.get(g, 0)
            if planned <= 0:
                continue
            ref = self.ref_price[g]
            floor = good_cost.get(g, 0.0) * self.min_markup
            n = planned
            prices = []
            for i in range(n):
                frac = 0.5 if n == 1 else i / (n - 1)
                p = ref * (low + (high - low) * frac)
                prices.append(round(max(p, floor), 2))
            # cheapest first: if production falls short, the expensive rungs drop
            ladder = sorted(prices)
            for p in ladder:
                listings.append(Listing(g, 1, float(p)))
            self.last_ladder[g] = ladder
        return Decision(bids=bids, assembly=assembly, listings=listings)
