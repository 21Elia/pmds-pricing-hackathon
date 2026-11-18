"""tight_v8: adaptive agent - reads the rivals from public information, no hardcoded scenario.

Base trading logic = gabri_dual_agent's PolimiAgent (top 3 on the portal's extended
scenario). gabri_dual_agent recognised the portal `default` by its numbers and stayed
idle there; new final scenarios make that useless, so v8 decides from what it SEES:
the market monitor (market.py) values every rival at cash + stock at replacement cost
and tracks whether the market makes or destroys money, who is active, who is broke,
how much stock rivals hold per good and how much cash they could bid with.

What was tested (Davide's tools/evaluate.py, isolated, random + hard + portal suites):
- full trading everywhere (adaptive=0, DEFAULT): mean rank 2.65, 1st in 62% of matches;
  gabri_dual_agent (idle in default) 4.06. In the "hard" default suite (41 aggressive
  teams, calibrated on the portal) full trading won 69% of matches.
- adaptive=1, earned budget (start at 5% of capital, grow while our value rises):
  mean rank 10.4, loses money - small trades still overpay and leave half-built kits.
- adaptive=2, stop-loss on our value at replacement cost: mean rank 7.7-11.2 - our value
  dips below the start in every healthy game (we pay above reserve, goods count at cost),
  so it stops us right before we cash in.
- use_rival_prices (match rivals' median sale price when visible): neutral to slightly
  worse on random markets.
The monitor stays (it is cheap and logs the signals) for the next ideas.

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

from .market import MarketMonitor


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
    keep_log = False            # keep every turn's market signals in self.log (analysis only)
    # --- rivals' sale prices (only when the scenario shows everyone's transactions) ---
    use_rival_prices = 0
    rp_above = 1.3              # nothing sold and our ref > median rival price * this -> cut
    rp_match = 1.05             # ... down to median rival price * this
    # --- earned budget: how much money may be tied in materials + unsold goods ---
    adaptive = 0                # 0 = always trade at full size (plain PolimiAgent logic)
                                # 1 = earned budget (tested: loses money, small trades overpay)
                                # 2 = full trading + stop-loss on our own value
    sl_after = 8                # stop-loss active from this turn
    sl_loss = 0.03              # stop buying for good if our value < start * (1 - sl_loss)
    e0_frac = 0.05              # starting budget = this share of the starting capital
    e_min_frac = 0.02           # the budget never falls below this share
    e_max_frac = 1.0            # ... nor grows above this share of current capital
    e_grow = 1.5                # budget *= this after a turn where our value went up
    e_shrink = 0.7              # budget *= this after a turn where it went down
    e_grace = 3                 # no shrinking during the first turns (nothing sold yet)
    e_ema = 0.5                 # smoothing of our per-turn profit

    def __init__(self, **overrides) -> None:
        # any class parameter can be overridden: PolimiAgent(prior_markup=6.0, ...)
        for k, v in overrides.items():
            if not hasattr(type(self), k):
                raise TypeError(f"unknown parameter {k}")
            setattr(self, k, v)
        self.monitor = MarketMonitor()
        self.log: list[dict] = []
        self.e_budget: float | None = None
        self.my_prev_value: float | None = None
        self.my_profit_ema: float = 0.0
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

    # ------------------------------------------------------------------
    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            self._identify(state)
            sig = self.monitor.update(state, goods_config, self.me)
        except Exception:  # noqa: BLE001 -- the monitor must never break the turn
            sig = {}
        try:
            d = self._decide(state, goods_config)
        except Exception:  # noqa: BLE001 -- a crash would forfeit the turn; list stock instead
            return self._fallback(state)
        try:
            if self.adaptive == 1:
                d = self._apply_budget(state, goods_config, d, sig)
            elif self.adaptive == 2:
                d = self._apply_stop_loss(state, goods_config, d, sig)
        except Exception:  # noqa: BLE001 -- budget logic failing must not cost the turn
            pass
        return d

    def _apply_stop_loss(self, state: Observation, gc: GoodsConfig, d: Decision, sig: dict) -> Decision:
        """Full trading, but stop buying for good once our value (cash + stock at
        replacement cost) has fallen sl_loss below the start after sl_after turns."""
        mat, good = MarketMonitor.replacement_costs(state, gc)
        inv = state.inventory
        value = state.capital + sum(int(inv.get(m, 0)) * mat[m] for m in gc.raw_materials) \
            + sum(int(inv.get(g, 0)) * good.get(g, 0.0) for g in gc.finished_goods)
        if getattr(self, "sl_start", None) is None:
            self.sl_start = state.capital
            self.sl_stopped = False
        if state.turn >= self.sl_after and value < self.sl_start * (1 - self.sl_loss):
            self.sl_stopped = True
        if self.keep_log:
            self.log.append({**sig, "my_value": value, "stopped": self.sl_stopped})
        if not self.sl_stopped:
            return d
        self.last_bids = {}
        return Decision(bids=[], assembly=d.assembly, listings=d.listings)

    def _apply_budget(self, state: Observation, gc: GoodsConfig, d: Decision, sig: dict) -> Decision:
        """Earned budget: scale our bids so that money tied in stock stays within a budget
        that grows while our own value (cash + stock at replacement cost) rises."""
        mat, good = MarketMonitor.replacement_costs(state, gc)
        inv = state.inventory
        stock_value = (sum(int(inv.get(m, 0)) * mat[m] for m in gc.raw_materials)
                       + sum(int(inv.get(g, 0)) * good.get(g, 0.0) for g in gc.finished_goods))
        value = state.capital + stock_value
        if self.e_budget is None:
            self.e_budget = self.e0_frac * state.capital
            self.e_start = state.capital
        if self.my_prev_value is not None:
            profit = value - self.my_prev_value
            self.my_profit_ema = (1 - self.e_ema) * self.my_profit_ema + self.e_ema * profit
            if self.my_profit_ema > 0:
                self.e_budget *= self.e_grow
            elif state.turn > self.e_grace:
                self.e_budget *= self.e_shrink
        self.my_prev_value = value
        lo = self.e_min_frac * self.e_start
        hi = max(lo, self.e_max_frac * state.capital)
        self.e_budget = min(hi, max(lo, self.e_budget))
        if self.keep_log:
            self.log.append({**sig, "budget": self.e_budget, "my_value": value, "my_profit_ema": self.my_profit_ema})
        room = self.e_budget - stock_value
        cost = sum(b.quantity * b.price_per_unit for b in d.bids)
        if cost <= room or cost <= 0:
            return d
        scale = max(0.0, room) / cost
        bids = []
        self.last_bids = {}
        for b in d.bids:
            q = int(b.quantity * scale)
            if q > 0:
                bids.append(Bid(b.good, q, b.price_per_unit))
                self.last_bids[b.good] = q
        return Decision(bids=bids, assembly=d.assembly, listings=d.listings)

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
        rival_prices: dict[str, list[float]] = {}
        for tx in state.sales_transactions[self.seen_sales:]:
            if self._mine(tx.seller):
                sold_prices.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
            elif self.me is not None:
                # only visible when the scenario shows everyone's transactions
                rival_prices.setdefault(tx.good, []).extend([tx.price] * tx.quantity)
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
                rp = sorted(rival_prices.get(g, []))
                if self.use_rival_prices and len(rp) >= 2:
                    market = rp[len(rp) // 2]               # median price rivals actually got
                    if not sold and ref > market * self.rp_above:
                        ref = market * self.rp_match         # we priced ourselves out: match the market
                    elif len(sold) >= listed and ref < market:
                        ref = (ref + market) / 2             # selling out below the market: move up
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
