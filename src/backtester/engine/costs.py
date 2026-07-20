from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    slippage_bps: float = 0.0
    fee_bps: float = 0.0

    def execution_price(self, price: float, is_buy: bool) -> float:
        adj = price * (self.slippage_bps / 10_000)
        return price + adj if is_buy else price - adj

    def fee(self, notional: float) -> float:
        return abs(notional) * (self.fee_bps / 10_000)
