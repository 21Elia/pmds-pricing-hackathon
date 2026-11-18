"""Market monitor for tight_v8: what the public information says about the rivals.

Everything here comes from the Observation every team receives:
- public_capital / public_inventories: every team's cash and stock, every turn;
- auction_lots: what suppliers offer this turn and at which reserve price.

Key idea, "market health": value every rival at cash + stock at REPLACEMENT cost
(materials at the current reserve, finished goods at their cheapest recipe cost
at the reserve). The turn-to-turn change of that value is the rival's real profit:
- buying materials at the reserve and building goods changes nothing;
- overpaying at auction or selling below cost lowers it;
- selling to customers above cost raises it.
Summed over the active rivals, it tells whether this market makes or destroys money,
without knowing which scenario we are in.
"""

from __future__ import annotations


class MarketMonitor:
    def __init__(self, ema: float = 0.35) -> None:
        self.ema = ema
        self.turns = 0
        self.start_cap: dict[str, float] = {}
        self.prev_value: dict[str, float] = {}
        self.prev_cap: dict[str, float] = {}
        self.prev_inv: dict[str, dict[str, int]] = {}
        self.active: set[str] = set()
        self.health: float | None = None   # EMA of rivals' profit per turn / start capital
        self.signals: dict = {}

    @staticmethod
    def replacement_costs(state, gc) -> tuple[dict[str, float], dict[str, float]]:
        lots = {lot.good: lot for lot in state.auction_lots}
        mat = {m: (lots[m].reserve_price if m in lots else 0.0) for m in gc.raw_materials}
        good: dict[str, float] = {}
        for rec in gc.recipes.values():
            c = sum(mat.get(m, 0.0) * q for m, q in rec.inputs.items())
            g = rec.output_good
            good[g] = min(good.get(g, c), c)
        return mat, good

    def update(self, state, gc, me: str | None) -> dict:
        self.turns += 1
        mat, good = self.replacement_costs(state, gc)
        rivals = [t for t in state.public_capital if t != me]
        n_rivals = len(rivals)
        start_avg = 0.0
        d_value = d_cash = 0.0
        losing = active_now = broke = 0
        rival_cash = 0.0
        fin_stock = {g: 0 for g in gc.finished_goods}
        raw_stock = {m: 0 for m in gc.raw_materials}
        sellers = {g: 0 for g in gc.finished_goods}
        for t in rivals:
            cap = float(state.public_capital.get(t, 0.0))
            inv = state.public_inventories.get(t, {})
            self.start_cap.setdefault(t, cap)
            start_avg += self.start_cap[t]
            value = cap + sum(int(inv.get(m, 0)) * mat[m] for m in gc.raw_materials) \
                + sum(int(inv.get(g, 0)) * good.get(g, 0.0) for g in gc.finished_goods)
            if t in self.prev_value:
                changed = abs(cap - self.prev_cap[t]) > 1e-6 or inv != self.prev_inv[t]
                if changed:
                    self.active.add(t)
                    active_now += 1
                if t in self.active:
                    d_value += value - self.prev_value[t]
                    d_cash += cap - self.prev_cap[t]
            if t in self.active:
                rival_cash += cap
                losing += value < self.start_cap[t] - 1e-6
                broke += cap < 0.1 * self.start_cap[t]
                for g in gc.finished_goods:
                    q = int(inv.get(g, 0))
                    fin_stock[g] += q
                    sellers[g] += q > 0
                for m in gc.raw_materials:
                    raw_stock[m] += int(inv.get(m, 0))
            self.prev_value[t] = value
            self.prev_cap[t] = cap
            self.prev_inv[t] = dict(inv)
        start_avg /= max(1, n_rivals)
        n_active = len(self.active)
        per_rival = d_value / max(1, n_active) / max(1.0, start_avg)
        if self.turns > 1 and n_active > 0:
            self.health = per_rival if self.health is None else (1 - self.ema) * self.health + self.ema * per_rival
        lot_value = sum(lot.quantity * lot.reserve_price for lot in state.auction_lots)
        self.signals = {
            "turn": state.turn,
            "n_rivals": n_rivals,
            "n_active": n_active,
            "active_now": active_now,
            "losing_frac": losing / max(1, n_active),
            "broke": broke,
            "market_profit": d_value,           # rivals' profit this turn at replacement value
            "market_cash": d_cash,
            "health": self.health,             # EMA of profit per active rival / start capital
            "rival_cash": rival_cash,
            "lot_value": lot_value,
            "cash_cover": rival_cash / max(1.0, lot_value),   # how many times rivals could buy all lots
            "fin_stock": fin_stock,
            "sellers": sellers,
            "raw_stock": raw_stock,
        }
        return self.signals
