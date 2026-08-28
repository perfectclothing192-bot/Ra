"""
Micro Mean-Reversion agent for XAUUSD.

Subclasses AdvancedTradingAgent to reuse its risk management, position
sizing, execution, and persistence machinery, adding the mean-reversion
strategy validated in backtest_micro_scalp.py: 88-day M5 XAUUSD backtest
returned -3.59% (profit factor 0.92) - net negative, not a proven edge.
This is deployed in PAPER mode purely to keep collecting live data on it
alongside the backtest, not because it has earned live capital.
"""

from datetime import datetime
from typing import List

import numpy as np

from trading_agent import AdvancedTradingAgent, PriceBar, Signal, SignalStrength, TradingMode

MEAN_WINDOW = 20   # bars - 20 M5 bars = 100 minutes, matching backtest_micro_scalp.py
BAND_K = 2.0
STOP_K = 3.5


class MeanReversionAgent(AdvancedTradingAgent):
    """
    Gold (XAUUSD) micro mean-reversion agent.

    Strategy: Micro Mean-Reversion (see backtest_micro_scalp.py for the
    validated backtest of this exact logic)
    - Rolling mean/stdev of close price over MEAN_WINDOW M5 bars.
    - LONG when price < mean - BAND_K*stdev (oversold), betting on
      reversion back toward the mean.
    - SHORT when price > mean + BAND_K*stdev, mirrored.
    - Target: the rolling mean itself. Stop: entry -/+ STOP_K*stdev.
    """

    def __init__(self,
                 mode: TradingMode = TradingMode.PAPER,
                 account_equity: float = 150000,
                 risk_per_trade: float = 0.0025,  # 0.25% - matches the scalper agent, given trade frequency
                 daily_loss_limit: float = 0.03,
                 ig_api_key: str = None,
                 ig_username: str = None):
        super().__init__(
            mode=mode,
            account_equity=account_equity,
            risk_per_trade=risk_per_trade,
            daily_loss_limit=daily_loss_limit,
            ig_api_key=ig_api_key,
            ig_username=ig_username,
        )
        self.price_history = {"XAUUSD": []}
        self.logger.info("Strategy: Micro Mean-Reversion | Asset: XAUUSD | M5")

    def micro_mean_reversion_signal(self, bars: List[PriceBar]) -> Signal:
        if len(bars) < MEAN_WINDOW + 1:
            return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "micro_mean_reversion", 0, datetime.now())

        closes = np.array([b.close for b in bars])
        window = closes[-MEAN_WINDOW:]
        mean = np.mean(window)
        std = np.std(window)
        price = closes[-1]

        if std == 0:
            return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "micro_mean_reversion", 0, datetime.now())

        oversold = price < mean - std * BAND_K
        overbought = price > mean + std * BAND_K

        if oversold:
            entry_price = price
            stop_loss = entry_price - std * STOP_K
            take_profit = mean
            if take_profit <= entry_price:
                return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "micro_mean_reversion", 0, datetime.now())
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
            return Signal(
                asset="XAUUSD", direction="BUY", strength=SignalStrength.MEDIUM,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                risk_reward_ratio=risk_reward, strategy="micro_mean_reversion",
                confidence=0.5, timestamp=datetime.now(),
            )

        if overbought:
            entry_price = price
            stop_loss = entry_price + std * STOP_K
            take_profit = mean
            if take_profit >= entry_price:
                return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "micro_mean_reversion", 0, datetime.now())
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
            return Signal(
                asset="XAUUSD", direction="SELL", strength=SignalStrength.MEDIUM,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                risk_reward_ratio=risk_reward, strategy="micro_mean_reversion",
                confidence=0.5, timestamp=datetime.now(),
            )

        return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "micro_mean_reversion", 0, datetime.now())
