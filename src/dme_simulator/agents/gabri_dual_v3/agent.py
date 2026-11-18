"""gabri_dual_v3 = gabri_dual_agent with a LEAN trader in `default` instead of doing nothing.

In the detected `default` scenario (12,000 start, the four default materials) most
of ~50 teams lose money and the top 5 end at only +200..+700. v2 (full trading) ended
24th on the portal; idle 12,000 ranks ~8th. The teams just above 12,000 ("123",
"456") make a few small, quick trades. v3 does the same:
- sniper window (turns 1-2): on turn 1 nobody holds stock yet, so customers can only
  buy from teams that buy, build and list in that same turn. v3 bids 5x the reserve
  for ONE kit of each of the 2 most expensive goods (laptop, console), builds and
  lists them at once at 1.6 x what the kit cost (max spend ~1,200);
- if only part of a kit is won, it buys just the missing pieces to finish it;
- no other buying (mid-game lean buying lost money in every local test: in a crowded
  default even units priced near cost do not sell, there are too many offers);
- known horizon: prices cut from l_liquidate, all stock dumped on the last turn.
If the sniper bids win nothing, v3 behaves exactly like idle (12,000).
Every other scenario is unchanged: same PolimiAgent logic as gabri_dual_agent.

Original PolimiAgent notes:

Mechanics this exploits (see simulation._clear_sales / _clear_auction):
- every listing is a separate offer in the customer's logit choice, so many
  1-unit listings = more market share -> keep a healthy stock listed;
- unsold stock carries over for free and materials are cheap next to what
  customers pay -> under-pricing and stock-outs are the real losses;
- auction winnings are capped by capital, so bidding can never bankrupt us.

Pricing (dynamic pricing with built-in exploration): each good has a reference
price; its units are listed one per listing on a ladder [low, high] * ref.
The ladder is the experiment: which rungs sold tells whether ref is too low
(top rungs / everything sold -> raise) or too high (nothing / only cheap rungs
sold -> lower). `ladder_gamma` < 1 puts more rungs near the top of the ladder
(fewer cheap units cannibalising high-valuation buyers).

A bucketed bandit over prices was tried first and failed: with 1-3 buyers per
good per turn the per-price sale probability is too noisy to estimate.

All tunables are plain class attributes (`    name = value`) for tools/tune.py.
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
    # --- pricing ---
    prior_markup = 8.5    # first reference price = material cost * this
    ladder_low = 0.93        # ladder spans [low, high] * reference
    ladder_high = 1.0
    ladder_gamma = 1.0      # rung spacing exponent (<1: rungs crowd near the top)
    explore_turns = 6       # wider ladder and faster moves early on
    explore_low = 0.75
    explore_high = 1.6
    step_up = 0.07
    step_down = 0.15
    top_frac = 1.1          # a sale at >= ref * top_frac counts as "top rung sold"
    cheap_down = 0.3        # fraction of step_down applied when only cheap rungs sold
    min_markup = 1.3        # never list below material cost * this
    # --- stock and production ---
    min_stock = 6
    max_stock = 30
    cover_turns = 4.0       # target stock = sales rate * cover + min_stock
    rate_alpha = 0.3        # EMA weight of last turn's sales
    # --- bidding ---
    bid_start = 1.85         # first bid = reserve * this
    bid_up = 1.5            # multiply bid when fill < fill_target
    bid_down = 0.9         # otherwise
    fill_target = 0.8
    value_share = 0.64       # max bid per material unit = share of product price per input unit
    budget_share = 0.8      # max share of capital committed per turn
    buy_extra = 0.0         # extra material bought on top of the plan (fraction)
    starve_only = False     # raise bids only if a good using the material is low on stock
    # --- opening: few teams have goods early, be first to market (off when open_turns = 0) ---
    open_turns = 0          # 5 won locally but lost badly on the real leaderboard
    open_stock = 6         # minimum stock target during the opening
    open_bid = 3.0          # bid at least reserve * this during the opening
    # --- scenario switch: in "default" (12000 start, 4 materials) most teams lose money and
    #     doing nothing ranks ~4th on the real leaderboard, so stay out; trade everywhere else ---
    quiet_capital = 12000.0
    quiet_materials = ("BATTERY", "CASING", "CHIP", "DISPLAY")
    # --- lean trader, only in the detected default scenario ---
    lean = 1                    # 0 = original behaviour: do nothing in default
    l_horizon = 40              # the default scenario lasts 40 turns
    l_stop_buy = 35             # last turn we buy materials
    l_liquidate = 37            # from this turn prices fall to clear stock
    l_bid = 1.05                # new kits: bid = reserve * this
    l_complete_bid = 1.5        # missing pieces of a started kit: bid = reserve * this
    l_kits = 2                  # new kits bid for per turn
    l_max_exposure = 400.0      # max money in materials + unsold goods (at cost)
    l_markup = 2.0              # first price = cost * this
    l_floor = 1.25              # never below cost * this before liquidation
    l_up = 0.10                 # price *= 1 + l_up after a sell-out
    l_down = 0.10               # price *= 1 - l_down when nothing sold
    # early "sniper" window: on turn 1 nobody has stock yet, so customers can only buy
    # from teams that buy, build and list in that same turn
    l_early = 2                 # sniper turns (0 = off)
    l_early_goods = 2           # how many goods (the most expensive recipes)
    l_early_kits = 1            # kits per good per sniper turn
    l_early_bid = 5.0           # sniper bid = reserve * this
    l_early_markup = 1.6        # sniper price = what the kit costs at the sniper bid * this
    l_mid = 0                   # 1 = also buy a few cheap kits after the sniper window

    def __init__(self) -> None:
        self.quiet: bool | None = None
        self.me: str | None = None
        self.ref_price: dict[str, float] = {}
        self.sales_rate: dict[str, float] = {}
        self.bid_level: dict[str, float] = {}
        self.unit_cost: dict[str, float] = {}
        self.last_bids: dict[str, int] = {}
        self.last_ladder: dict[str, list[float]] = {}
        self.seen_auction = 0
        self.seen_sales = 0
        self.l_paid: dict[str, float] = {}       # average price paid per material unit
        self.l_bought: dict[str, int] = {}
        self.l_ref: dict[str, float] = {}
        self.l_listed: dict[str, int] = {}
        self.l_seen_a = 0
        self.l_seen_s = 0

    # ------------------------------------------------------------------
    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        if self.quiet is None:  # first call: capital is still the starting capital
            self.quiet = (abs(state.capital - self.quiet_capital) < 1e-6
                          and sorted(goods_config.raw_materials) == list(self.quiet_materials))
        if self.quiet:
            if not self.lean:
                return Decision()
            try:
                return self._lean(state, goods_config)
            except Exception:  # noqa: BLE001 -- never crash; doing nothing is the safe fallback
                return Decision()
        try:
            return self._decide(state, goods_config)
        except Exception:  # noqa: BLE001 -- a crash would forfeit the turn; list stock instead
            return self._fallback(state)

    def _lean(self, state: Observation, gc: GoodsConfig) -> Decision:
        """default scenario: a few cheap complete kits, sold a little above cost."""
        self._identify(state)
        turn = state.turn
        inv = {k: int(v) for k, v in state.inventory.items()}
        lots = {lot.good: lot for lot in state.auction_lots}
        for tx in state.auction_transactions[self.l_seen_a:]:
            if self._mine(tx.buyer):
                n0 = self.l_bought.get(tx.good, 0)
                p0 = self.l_paid.get(tx.good, tx.price)
                self.l_paid[tx.good] = (p0 * n0 + tx.price * tx.quantity) / (n0 + tx.quantity)
                self.l_bought[tx.good] = n0 + tx.quantity
        self.l_seen_a = len(state.auction_transactions)
        sold: dict[str, int] = {}
        for tx in state.sales_transactions[self.l_seen_s:]:
            if self._mine(tx.seller):
                sold[tx.good] = sold.get(tx.good, 0) + tx.quantity
        self.l_seen_s = len(state.sales_transactions)

        def unit_cost(m: str) -> float:
            if m in self.l_paid:
                return self.l_paid[m]
            return lots[m].reserve_price * self.l_bid if m in lots else 0.0

        recipe_for: dict[str, str] = {}
        cost: dict[str, float] = {}
        for name, rec in gc.recipes.items():
            c = sum(unit_cost(m) * q for m, q in rec.inputs.items())
            g = rec.output_good
            if g not in cost or c < cost[g]:
                cost[g], recipe_for[g] = max(c, 1e-6), name

        # price per good: start a little above cost, up after a sell-out, down when nothing sells
        for g in recipe_for:
            if g not in self.l_ref:
                self.l_ref[g] = cost[g] * self.l_markup
            listed = self.l_listed.get(g, 0)
            if listed > 0:
                s = sold.get(g, 0)
                if s >= listed:
                    self.l_ref[g] *= 1 + self.l_up
                elif s == 0:
                    self.l_ref[g] *= 1 - self.l_down
            self.l_ref[g] = max(self.l_ref[g], cost[g] * self.l_floor)

        # build every complete kit we hold, best margin first
        ranked = sorted(recipe_for, key=lambda g: self.l_ref[g] - cost[g], reverse=True)
        raw = {m: inv.get(m, 0) for m in gc.raw_materials}
        assembly, built = [], {}
        for g in ranked:
            rec = gc.recipes[recipe_for[g]]
            k = min((raw[m] // q for m, q in rec.inputs.items() if q > 0), default=0)
            if k > 0:
                assembly.append(AssemblyOrder(recipe_for[g], int(k)))
                built[g] = k
                for m, q in rec.inputs.items():
                    raw[m] -= q * k

        # exposure: materials + finished goods we hold, at cost
        exposure = (sum(inv.get(m, 0) * unit_cost(m) for m in gc.raw_materials)
                    + sum(inv.get(g, 0) * cost.get(g, 0.0) for g in gc.finished_goods))
        bids: list[Bid] = []
        if turn <= self.l_early:
            # sniper: a few complete kits of the most expensive goods, built and listed now
            reserve_cost = {g: sum(lots[m].reserve_price * q for m, q in gc.recipes[recipe_for[g]].inputs.items()
                                   if m in lots) for g in recipe_for}
            picks = sorted(recipe_for, key=lambda g: reserve_cost[g], reverse=True)[:self.l_early_goods]
            want_e: dict[str, int] = {}
            for g in picks:
                rec = gc.recipes[recipe_for[g]]
                if not all(m in lots for m in rec.inputs):
                    continue
                for m, q in rec.inputs.items():
                    want_e[m] = want_e.get(m, 0) + q * self.l_early_kits
                assembly.append(AssemblyOrder(recipe_for[g], self.l_early_kits))   # uses what we win now
                built[g] = built.get(g, 0) + self.l_early_kits
                if turn == 1 or g not in self.l_ref:
                    self.l_ref[g] = max(self.l_ref.get(g, 0.0),
                                        reserve_cost[g] * self.l_early_bid * self.l_early_markup)
            for m, q in want_e.items():
                lot = lots[m]
                q = min(q, lot.quantity)
                if q > 0:
                    bids.append(Bid(m, int(q), float(lot.reserve_price * self.l_early_bid)))
        elif turn <= self.l_stop_buy and (self.l_mid or any(q > 0 for q in raw.values())):
            want: dict[str, tuple[int, float]] = {}
            leftover = {m: q for m, q in raw.items() if q > 0}
            if leftover:
                # finish a started kit: the recipe that needs the fewest extra pieces
                def missing(g: str) -> dict[str, int]:
                    rec = gc.recipes[recipe_for[g]]
                    return {m: q - min(q, raw.get(m, 0)) for m, q in rec.inputs.items() if q > raw.get(m, 0)}
                usable = [g for g in ranked if any(m in leftover for m in gc.recipes[recipe_for[g]].inputs)]
                if usable:
                    g = min(usable, key=lambda x: sum(missing(x).values()))
                    for m, q in missing(g).items():
                        if m in lots:
                            want[m] = (q, lots[m].reserve_price * self.l_complete_bid)
            elif self.l_mid and exposure < self.l_max_exposure:
                # start new kits of the best good, if they fit in the exposure budget
                g = ranked[0] if ranked else None
                if g is not None:
                    rec = gc.recipes[recipe_for[g]]
                    kit = sum(lots[m].reserve_price * self.l_bid * q for m, q in rec.inputs.items() if m in lots)
                    k = int(min(self.l_kits, (self.l_max_exposure - exposure) // max(kit, 1e-6)))
                    if k > 0 and all(m in lots for m in rec.inputs):
                        for m, q in rec.inputs.items():
                            want[m] = (q * k, lots[m].reserve_price * self.l_bid)
            for m, (q, p) in want.items():
                lot = lots[m]
                q = min(q, lot.quantity)
                if q > 0:
                    bids.append(Bid(m, int(q), float(max(p, lot.reserve_price * 1.0001))))

        # list every finished unit as its own 1-unit offer
        listings = []
        self.l_listed = {}
        left = self.l_horizon - turn                     # turns left after this one
        for g in recipe_for:
            n = inv.get(g, 0) + built.get(g, 0)
            if n <= 0:
                continue
            ref = self.l_ref[g]
            if left <= 0:
                ref = cost[g] * 0.3                       # last turn: anything beats keeping it
            elif turn >= self.l_liquidate:
                ref = max(cost[g] * 0.9, ref * (0.6 + 0.1 * left))
            for i in range(n):
                listings.append(Listing(g, 1, float(round(max(ref * (1 + 0.05 * i), 0.01), 2))))
            self.l_listed[g] = n
        return Decision(bids=bids, assembly=assembly, listings=listings)

    def _fallback(self, state: Observation) -> Decision:
        return Decision(listings=[
            Listing(g, int(q), float(self.ref_price[g]))
            for g, q in state.inventory.items()
            if q > 0 and math.isfinite(self.ref_price.get(g, math.nan)) and self.ref_price[g] > 0
        ])

    def _identify(self, state: Observation) -> None:
        mine = {k: v for k, v in state.inventory.items() if v}
        matches = [
            team for team, inv in state.public_inventories.items()
            if abs(state.public_capital.get(team, -1e18) - state.capital) < 1e-6
            and {k: v for k, v in inv.items() if v} == mine
        ]
        if len(matches) == 1:
            self.me = matches[0]

    def _mine(self, who: str) -> bool:
        return self.me is None or who == self.me

    # ------------------------------------------------------------------
    def _decide(self, state: Observation, gc: GoodsConfig) -> Decision:
        self._identify(state)
        inv = {k: int(v) for k, v in state.inventory.items()}
        lots = {lot.good: lot for lot in state.auction_lots}
        early = state.turn <= self.explore_turns

        # --- learn from last auction ------------------------------------
        won: dict[str, int] = {}
        for tx in state.auction_transactions[self.seen_auction:]:
            if self._mine(tx.buyer):
                won[tx.good] = won.get(tx.good, 0) + tx.quantity
                old = self.unit_cost.get(tx.good, tx.price)
                self.unit_cost[tx.good] = 0.7 * old + 0.3 * tx.price
        self.seen_auction = len(state.auction_transactions)

        mat_cost = {
            m: self.unit_cost.get(m, lots[m].reserve_price if m in lots else 1.0)
            for m in gc.raw_materials
        }
        recipe_for: dict[str, str] = {}
        cost: dict[str, float] = {}
        for name, rec in gc.recipes.items():
            c = sum(mat_cost.get(m, 1.0) * q for m, q in rec.inputs.items())
            g = rec.output_good
            if g not in cost or c < cost[g]:
                cost[g], recipe_for[g] = max(c, 1e-6), name

        # --- learn from last sales: move the reference price ----------------
        sold_prices: dict[str, list[float]] = {}
        for tx in state.sales_transactions[self.seen_sales:]:
            if self._mine(tx.seller):
                sold_prices.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
        self.seen_sales = len(state.sales_transactions)

        for g in recipe_for:
            if g not in self.ref_price:
                self.ref_price[g] = cost[g] * self.prior_markup
                self.sales_rate[g] = 1.0
            sold = sorted(sold_prices.get(g, []))
            self.sales_rate[g] = (1 - self.rate_alpha) * self.sales_rate[g] + self.rate_alpha * len(sold)
            # units really on sale last turn = sold + still in stock (we list everything)
            listed = min(len(self.last_ladder.get(g, [])), len(sold) + inv.get(g, 0))
            if listed > 0:
                up = self.step_up * (2.0 if early else 1.0)
                down = self.step_down * (1.5 if early else 1.0)
                ref = self.ref_price[g]
                if len(sold) >= listed:
                    ref *= 1 + 2 * up                  # sold out: clearly too cheap
                elif sold and sold[-1] >= ref * self.top_frac:
                    ref *= 1 + up                      # upper rungs sell
                elif sold:
                    ref *= 1 - self.cheap_down * down  # only cheap rungs sell
                else:
                    ref *= 1 - down                    # nothing sold
                self.ref_price[g] = max(ref, cost[g] * self.min_markup)
        price = self.ref_price

        # --- production plan ----------------------------------------------
        opening = state.turn <= self.open_turns
        base_stock = max(self.min_stock, self.open_stock) if opening else self.min_stock
        target = {
            g: int(min(self.max_stock, round(self.sales_rate[g] * self.cover_turns) + base_stock))
            for g in recipe_for
        }
        inputs = {g: sum(gc.recipes[recipe_for[g]].inputs.values()) for g in recipe_for}
        ranked = sorted(
            (g for g in recipe_for if price[g] > cost[g]),
            key=lambda g: (price[g] - cost[g]) / max(1, inputs[g]),
            reverse=True,
        )
        need = {g: max(0, target[g] - inv.get(g, 0)) for g in ranked}
        assembly = [AssemblyOrder(recipe_for[g], need[g]) for g in ranked if need[g] > 0]

        # --- bids: escalate on unfilled orders, decay when filled ----------
        for m, asked in self.last_bids.items():
            if won.get(m, 0) / asked >= self.fill_target:
                self.bid_level[m] *= self.bid_down
            elif not self.starve_only or any(
                m in gc.recipes[recipe_for[g]].inputs and inv.get(g, 0) < max(1, target[g] // 2)
                for g in ranked
            ):
                self.bid_level[m] *= self.bid_up
        want = {m: 0 for m in gc.raw_materials}
        for g in ranked:
            for m, q in gc.recipes[recipe_for[g]].inputs.items():
                want[m] += q * need[g]
        bids = []
        self.last_bids = {}
        budget = state.capital * self.budget_share
        for m in gc.raw_materials:
            lot = lots.get(m)
            if lot is None or lot.quantity <= 0:
                continue
            qty = min(math.ceil(want[m] * (1 + self.buy_extra)) - inv.get(m, 0), lot.quantity)
            if qty <= 0:
                continue
            cap = max(
                (self.value_share * price[g] / max(1, inputs[g])
                 for g in ranked if m in gc.recipes[recipe_for[g]].inputs),
                default=0.0,
            )
            floor = lot.reserve_price * 1.001
            level = self.bid_level.get(m, lot.reserve_price * self.bid_start)
            if opening:
                level = max(level, lot.reserve_price * self.open_bid)
            level = max(floor,min(level, cap) if cap > 0 else level)
            self.bid_level[m] = level
            qty = min(qty, int(budget // level))
            if qty <= 0:
                continue
            budget -= qty * level
            bids.append(Bid(m, int(qty), float(level)))
            self.last_bids[m] = int(qty)

        # --- listings: every unit its own 1-unit offer on the ladder -------
        listings = []
        self.last_ladder = {}
        low, high = (self.explore_low, self.explore_high) if early else (self.ladder_low, self.ladder_high)
        for g in recipe_for:
            n = inv.get(g, 0) + need.get(g, 0)
            if n <= 0:
                continue
            floor = cost[g] * self.min_markup
            ladder = sorted(
                round(max(floor, price[g] * (low + (high - low) * self._rung(i, n))), 2)
                for i in range(n)
            )
            # cheapest first: if production falls short, the expensive rungs are the ones dropped
            listings.extend(Listing(g, 1, float(p)) for p in ladder)
            self.last_ladder[g] = ladder
        return Decision(bids=bids, assembly=assembly, listings=listings)

    def _rung(self, i: int, n: int) -> float:
        """Position of rung i of n on the ladder, in [0, 1]."""
        return 0.5 if n == 1 else (i / (n - 1)) ** self.ladder_gamma


assert PolimiAgent()._rung(0, 5) == 0.0 and PolimiAgent()._rung(4, 5) == 1.0
