"""Our team's agent. Each phase has its own method, so it can be improved independently."""

from __future__ import annotations

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
    # Tunable parameters
    bid_markup = 0.1      # bid = reserve_price * (1 + bid_markup)
    units_per_recipe = 1  # how many units of each recipe to buy materials for
    sell_markup = 2.0     # sale price = materials cost * sell_markup

    def __init__(self):
        # Memory kept across turns (the same instance is used for the whole match)
        self.history: list[Observation] = []

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self.history.append(state)
        prices = {lot.good: lot.reserve_price * (1 + self.bid_markup) for lot in state.auction_lots}
        return Decision(
            bids=self._bids(state, goods_config, prices),
            assembly=self._assembly(state, goods_config),
            listings=self._listings(state, goods_config, prices),
        )

    # Phase 1: auction
    def _bids(self, state, goods_config, prices) -> list[Bid]:
        wanted = {m: 0 for m in goods_config.raw_materials}
        for recipe in goods_config.recipes.values():
            for material, amount in recipe.inputs.items():
                wanted[material] += amount * self.units_per_recipe
        return [
            Bid(lot.good, max(0, wanted.get(lot.good, 0) - state.inventory.get(lot.good, 0)), prices[lot.good])
            for lot in state.auction_lots
        ]

    # Phase 2: production (the simulator caps it at the materials you actually hold)
    def _assembly(self, state, goods_config) -> list[AssemblyOrder]:
        return [AssemblyOrder(name, self.units_per_recipe) for name in goods_config.recipes]

    # Phase 3: sales (listings are capped at the inventory you actually hold)
    def _listings(self, state, goods_config, prices) -> list[Listing]:
        listings = []
        for recipe in goods_config.recipes.values():
            cost = sum(prices.get(m, 0.0) * a for m, a in recipe.inputs.items())
            # shortcut: this ignores units built this turn; fine because the listing quantity is capped anyway
            qty = state.inventory.get(recipe.output_good, 0) + self.units_per_recipe
            if cost > 0:
                listings.append(Listing(recipe.output_good, qty, cost * self.sell_markup))
        return listings
