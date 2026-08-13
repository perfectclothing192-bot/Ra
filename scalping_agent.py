"""
Scalping trading agent for XAUUSD (Gold).

Subclasses AdvancedTradingAgent to reuse its risk management, position
sizing, execution, and persistence machinery, adding a fast, short-timeframe
scalping strategy tuned for M1 candles instead of the swing-style strategies
in trading_agent.py.
"""

from datetime import datetime
from typing import List

import numpy as np

from trading_agent import AdvancedTradingAgent, PriceBar, Signal, SignalStrength, TradingMode


class ScalpingTradingAgent(AdvancedTradingAgent):
    """
    Gold (XAUUSD) scalping agent.

    Strategy: EMA Ribbon Momentum Scalp
    - EMA 5 / EMA 13 / EMA 50 ribbon on M1 candles
    - Enter on a fresh EMA5/EMA13 crossover in the direction of the EMA50 slope
    - RSI filter to avoid chasing an already-exhausted move
    - Tight ATR-based stop/target (0.6x / 1.0x ATR) for a fast in-and-out
      trade, instead of the multi-ATR swing targets used elsewhere
    """

    def __init__(self,
                 mode: TradingMode = TradingMode.PAPER,
                 account_equity: float = 150000,
                 risk_per_trade: float = 0.0025,  # 0.25% - smaller due to high trade frequency
                 daily_loss_limit: float = 0.03,  # 3% - tighter halt for scalping
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
        self.logger.info("Scalping strategy: EMA Ribbon Momentum Scalp | Asset: XAUUSD")

    def ema_ribbon_scalp_signal(self, bars: List[PriceBar]) -> Signal:
        if len(bars) < 55:
            return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "ema_ribbon_scalp", 0, datetime.now())

        closes = np.array([bar.close for bar in bars])

        ema_5 = self._ema(closes, 5)
        ema_13 = self._ema(closes, 13)
        ema_50 = self._ema(closes, 50)
        rsi = self._rsi(closes, 14)
        atr = self._atr(bars, 14)[-1]

        if atr == 0:
            return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "ema_ribbon_scalp", 0, datetime.now())

        current_price = closes[-1]

        # Fresh crossover: EMA5 crosses EMA13 on this bar
        bullish_cross = ema_5[-2] <= ema_13[-2] and ema_5[-1] > ema_13[-1]
        bearish_cross = ema_5[-2] >= ema_13[-2] and ema_5[-1] < ema_13[-1]

        # Trend filter: EMA50 slope agrees with crossover direction
        ema_50_rising = ema_50[-1] > ema_50[-5]
        ema_50_falling = ema_50[-1] < ema_50[-5]

        # Momentum filter: RSI not already exhausted in the trade direction
        rsi_ok_long = 45 < rsi[-1] < 70
        rsi_ok_short = 30 < rsi[-1] < 55

        if bullish_cross and ema_50_rising and rsi_ok_long:
            entry_price = current_price
            stop_loss = entry_price - atr * 0.6
            take_profit = entry_price + atr * 1.0
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
            return Signal(
                asset="XAUUSD", direction="BUY",
                strength=SignalStrength.STRONG if rsi[-1] < 60 else SignalStrength.MEDIUM,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                risk_reward_ratio=risk_reward, strategy="ema_ribbon_scalp",
                confidence=0.7, timestamp=datetime.now(),
            )

        if bearish_cross and ema_50_falling and rsi_ok_short:
            entry_price = current_price
            stop_loss = entry_price + atr * 0.6
            take_profit = entry_price - atr * 1.0
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
            return Signal(
                asset="XAUUSD", direction="SELL",
                strength=SignalStrength.STRONG if rsi[-1] > 40 else SignalStrength.MEDIUM,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                risk_reward_ratio=risk_reward, strategy="ema_ribbon_scalp",
                confidence=0.7, timestamp=datetime.now(),
            )

        return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "ema_ribbon_scalp", 0, datetime.now())

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Wilder's RSI"""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.zeros(len(closes))
        avg_loss = np.zeros(len(closes))
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50  # neutral until warmed up
        return rsi
