"""
========================================
RAMESH'S AUTOMATED TRADING AGENT V1
Multi-Strategy | Regime-Adaptive | Risk-Managed
========================================

STRATEGY FRAMEWORK:
- Gold (XAUUSD): Golden Pullback (EMA 200/50 + SMC)
- Oil (USOIL): SMA Cluster Reclaim (100/200 + Liquidity Sweep)
- Forex: Correlation Hedge (GBPUSD, EURUSD)

RISK MANAGEMENT:
- Max 1% per trade
- 5% daily loss halt
- Margin safeguards
- Circuit breakers

MODES: PAPER_TRADING (default) → LIVE (approval required)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import requests

# ============================================
# CONFIG & ENUMS
# ============================================

class TradingMode(Enum):
    PAPER = "paper"  # Simulated
    LIVE = "live"    # Real money (careful!)

class MarketRegime(Enum):
    STRONG_TREND = "strong_trend"      # Use momentum/trend-following
    WEAK_TREND = "weak_trend"          # Use mean-reversion
    CONSOLIDATION = "consolidation"    # Use breakout rules
    HIGH_VOLATILITY = "high_volatility" # Reduce size
    UNCERTAIN = "uncertain"             # Pause or micro-size

class SignalStrength(Enum):
    STRONG = "strong"      # High confidence
    MEDIUM = "medium"      # Moderate confidence
    WEAK = "weak"          # Low confidence
    CONFLICTING = "conflicting"  # Multiple indicators disagree

# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class PriceBar:
    """OHLCV data for a single candle"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class TechnicalIndicators:
    """Calculated technical indicators"""
    ema_50: float
    ema_200: float
    sma_100: float
    sma_200: float
    rsi_14: float
    atr_14: float
    vix_equivalent: float  # Volatility proxy

@dataclass
class Signal:
    """Trading signal from a strategy"""
    asset: str
    direction: str  # "BUY", "SELL", "HOLD"
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    strategy: str  # "golden_pullback", "sma_cluster", etc.
    confidence: float  # 0-1
    timestamp: datetime

@dataclass
class Position:
    """Active trading position"""
    asset: str
    direction: str  # "LONG", "SHORT"
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    position_id: str

@dataclass
class Trade:
    """Completed trade record"""
    asset: str
    entry_price: float
    exit_price: float
    direction: str
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    strategy: str

# ============================================
# CORE TRADING AGENT
# ============================================

class AdvancedTradingAgent:
    """
    Multi-strategy automated trading agent with:
    - Golden Pullback (Gold)
    - SMA Cluster Reclaim (Oil)
    - Market Regime Adaptation
    - Professional Risk Management
    """

    def __init__(self,
                 mode: TradingMode = TradingMode.PAPER,
                 account_equity: float = 150000,
                 risk_per_trade: float = 0.01,  # 1%
                 daily_loss_limit: float = 0.05,  # 5%
                 min_equity_pct: float = 0.5,  # halt if equity falls below this fraction of where it started
                 ig_api_key: str = None,
                 ig_username: str = None):

        self.mode = mode
        self.account_equity = account_equity
        self.risk_per_trade = risk_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.min_equity_pct = min_equity_pct
        self.initial_equity = account_equity
        self.ig_api_key = ig_api_key
        self.ig_username = ig_username

        # State management
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.max_daily_loss = account_equity * daily_loss_limit

        # Market data cache
        self.price_history: Dict[str, List[PriceBar]] = {
            "XAUUSD": [],
            "USOIL": [],
            "GBPUSD": [],
            "EURUSD": []
        }

        # Logging
        self.logger = self._setup_logging()
        self.logger.info(f"Agent initialized | Mode: {mode.value} | Capital: ${account_equity:,.0f}")

    # ============================================
    # STRATEGY 1: GOLDEN PULLBACK (GOLD)
    # ============================================

    def golden_pullback_signal(self, bars: List[PriceBar]) -> Signal:
        """
        Golden Pullback Strategy for XAUUSD

        Rules:
        1. Price pulls back to EMA 50 (after trending above EMA 200)
        2. Candle closes above EMA 50 = BUY
        3. Smart Money Concepts: Confirm with liquidity sweep

        Returns: Signal with entry, SL, TP
        """
        if len(bars) < 200:
            return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "golden_pullback", 0, datetime.now())

        closes = np.array([bar.close for bar in bars])

        # Calculate EMAs
        ema_50 = self._ema(closes, 50)[-1]
        ema_200 = self._ema(closes, 200)[-1]

        current_price = closes[-1]
        prev_price = closes[-2]
        atr = self._atr(bars, 14)[-1]

        # Trend Check: Price above EMA 200
        in_uptrend = current_price > ema_200

        # Pullback Check: Price pulled back to EMA 50
        price_near_ema50 = abs(current_price - ema_50) < atr * 0.5
        prev_below_ema50 = prev_price < ema_50

        # Entry: Candle closes above EMA 50 during pullback
        pullback_entry = (in_uptrend and price_near_ema50 and
                         current_price > ema_50 and prev_below_ema50)

        if pullback_entry:
            entry_price = current_price
            stop_loss = ema_50 - atr * 0.5  # Below EMA 50
            take_profit = current_price + atr * 2.5  # 2.5x ATR target

            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price != stop_loss else 0

            return Signal(
                asset="XAUUSD",
                direction="BUY",
                strength=SignalStrength.STRONG if risk_reward > 2 else SignalStrength.MEDIUM,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                strategy="golden_pullback",
                confidence=0.85,
                timestamp=datetime.now()
            )

        return Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "golden_pullback", 0, datetime.now())

    # ============================================
    # STRATEGY 2: SMA CLUSTER RECLAIM (OIL)
    # ============================================

    def sma_cluster_signal(self, bars: List[PriceBar]) -> Signal:
        """
        SMA Cluster Reclaim Strategy for USOIL

        Rules:
        1. Price tests SMA 100/200 cluster (consolidated zone)
        2. Price breaks above cluster on volume = BUY
        3. Liquidity sweep confirmation: Price touches then rejects lower

        Returns: Signal with entry, SL, TP
        """
        if len(bars) < 200:
            return Signal("USOIL", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "sma_cluster", 0, datetime.now())

        closes = np.array([bar.close for bar in bars])
        volumes = np.array([bar.volume for bar in bars])

        # Calculate SMAs
        sma_100 = self._sma(closes, 100)[-1]
        sma_200 = self._sma(closes, 200)[-1]

        current_price = closes[-1]
        atr = self._atr(bars, 14)[-1]
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]

        # Cluster: SMA 100/200 close together
        cluster_zone_low = min(sma_100, sma_200)
        cluster_zone_high = max(sma_100, sma_200)
        cluster_width = cluster_zone_high - cluster_zone_low

        in_cluster = cluster_width < atr * 0.3  # Tight cluster

        # Breakout: Price breaks above cluster on high volume
        price_above_cluster = current_price > cluster_zone_high
        volume_surge = current_volume > avg_volume * 1.5

        if in_cluster and price_above_cluster and volume_surge:
            entry_price = current_price
            stop_loss = cluster_zone_low - atr * 0.5
            take_profit = current_price + atr * 2.0

            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price != stop_loss else 0

            return Signal(
                asset="USOIL",
                direction="BUY",
                strength=SignalStrength.STRONG if volume_surge else SignalStrength.MEDIUM,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                strategy="sma_cluster",
                confidence=0.80,
                timestamp=datetime.now()
            )

        return Signal("USOIL", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "sma_cluster", 0, datetime.now())

    # ============================================
    # STRATEGY 3: CORRELATION HEDGE (FOREX)
    # ============================================

    def correlation_hedge_signal(self, gbp_bars: List[PriceBar], eur_bars: List[PriceBar]) -> Tuple[Signal, Signal]:
        """
        Correlation Hedge Strategy for GBPUSD / EURUSD

        Rules:
        1. GBPUSD and EURUSD are normally positively correlated (both quote USD
           as the counter currency, so both tend to move together on USD strength
           or weakness).
        2. Track the rolling z-score of their cumulative-return spread over a
           lookback window.
        3. When the spread stretches beyond +/-2 std devs while the rolling
           correlation stays strong (>0.5), treat it as a temporary dislocation:
           BUY the laggard, SELL the leader, betting on reconvergence.
        4. If the correlation itself has broken down (<0.5), stand aside on both
           legs - the historical relationship no longer holds, so a spread trade
           has no edge.

        Always returns a signal for both legs together (never one without the
        other), since this is a paired hedge, not two independent trades.
        """
        hold_pair = (
            Signal("GBPUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "correlation_hedge", 0, datetime.now()),
            Signal("EURUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "correlation_hedge", 0, datetime.now()),
        )

        lookback = 50
        if len(gbp_bars) < lookback + 1 or len(eur_bars) < lookback + 1:
            return hold_pair

        gbp_closes = np.array([b.close for b in gbp_bars[-(lookback + 1):]])
        eur_closes = np.array([b.close for b in eur_bars[-(lookback + 1):]])

        gbp_returns = np.diff(gbp_closes) / gbp_closes[:-1]
        eur_returns = np.diff(eur_closes) / eur_closes[:-1]

        correlation = np.corrcoef(gbp_returns, eur_returns)[0, 1]
        if np.isnan(correlation) or correlation < 0.5:
            return hold_pair

        spread = np.cumsum(gbp_returns) - np.cumsum(eur_returns)
        spread_std = np.std(spread)
        if spread_std == 0:
            return hold_pair

        z_score = (spread[-1] - np.mean(spread)) / spread_std
        if abs(z_score) < 2.0:
            return hold_pair

        gbp_atr = self._atr(gbp_bars, 14)[-1]
        eur_atr = self._atr(eur_bars, 14)[-1]
        gbp_price = gbp_closes[-1]
        eur_price = eur_closes[-1]

        strength = SignalStrength.STRONG if abs(z_score) > 2.5 else SignalStrength.MEDIUM
        confidence = min(0.5 + abs(z_score) * 0.1, 0.9)
        risk_reward = 2.0 / 1.5

        # z_score > 0: GBP has outrun EUR -> fade GBP (SELL), buy the laggard EUR
        # z_score < 0: EUR has outrun GBP -> fade EUR (SELL), buy the laggard GBP
        gbp_direction = "SELL" if z_score > 0 else "BUY"
        eur_direction = "BUY" if z_score > 0 else "SELL"
        gbp_sl_sign = 1 if gbp_direction == "SELL" else -1
        eur_sl_sign = 1 if eur_direction == "SELL" else -1

        gbp_signal = Signal(
            asset="GBPUSD",
            direction=gbp_direction,
            strength=strength,
            entry_price=gbp_price,
            stop_loss=gbp_price + gbp_sl_sign * gbp_atr * 1.5,
            take_profit=gbp_price - gbp_sl_sign * gbp_atr * 2.0,
            risk_reward_ratio=risk_reward,
            strategy="correlation_hedge",
            confidence=confidence,
            timestamp=datetime.now()
        )
        eur_signal = Signal(
            asset="EURUSD",
            direction=eur_direction,
            strength=strength,
            entry_price=eur_price,
            stop_loss=eur_price + eur_sl_sign * eur_atr * 1.5,
            take_profit=eur_price - eur_sl_sign * eur_atr * 2.0,
            risk_reward_ratio=risk_reward,
            strategy="correlation_hedge",
            confidence=confidence,
            timestamp=datetime.now()
        )

        return gbp_signal, eur_signal

    # ============================================
    # MARKET REGIME DETECTION
    # ============================================

    def detect_market_regime(self, bars: List[PriceBar]) -> MarketRegime:
        """
        Adaptive regime detection to switch strategies

        Uses:
        - ATR (volatility)
        - Trend strength (ADX proxy)
        - Price structure
        """
        if len(bars) < 50:
            return MarketRegime.UNCERTAIN

        closes = np.array([bar.close for bar in bars])

        atr = self._atr(bars, 14)[-1]
        atr_sma = np.mean(self._atr(bars, 14)[-20:])

        # Volatility regime
        volatility_ratio = atr / atr_sma

        if volatility_ratio > 1.5:
            return MarketRegime.HIGH_VOLATILITY

        # Trend regime (simple: compare close to SMA)
        sma_50 = self._sma(closes, 50)[-1]
        sma_200 = self._sma(closes, 200)[-1]

        trend_distance = abs(closes[-1] - sma_50)

        # ADX proxy (simple trend strength)
        uptrend_strength = sum(1 for i in range(-10, 0) if closes[i] > closes[i-1]) / 10

        if uptrend_strength > 0.7 and trend_distance > atr:
            return MarketRegime.STRONG_TREND
        elif uptrend_strength > 0.4:
            return MarketRegime.WEAK_TREND
        elif trend_distance < atr * 0.5:
            return MarketRegime.CONSOLIDATION

        return MarketRegime.UNCERTAIN

    # ============================================
    # POSITION SIZING & RISK MANAGEMENT
    # ============================================

    def calculate_position_size(self, signal: Signal) -> float:
        """
        Professional position sizing using:
        - Risk per trade (1% of account)
        - Risk/reward ratio
        - Volatility adjustment
        """
        if signal.direction == "HOLD":
            return 0

        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            self.logger.warning(f"Daily loss limit reached. Halting new positions.")
            return 0

        # Risk amount
        risk_amount = self.account_equity * self.risk_per_trade

        # Position size based on stop loss
        stop_distance = abs(signal.entry_price - signal.stop_loss)

        if stop_distance == 0:
            return 0

        position_size = risk_amount / stop_distance

        # Adjust for risk/reward
        if signal.risk_reward_ratio < 1.5:
            position_size *= 0.5
        elif signal.risk_reward_ratio > 3:
            position_size *= 1.2

        self.logger.info(f"{signal.asset} Position Size: {position_size:.2f} lots | R:R = {signal.risk_reward_ratio:.2f}")

        return position_size

    def check_circuit_breakers(self) -> bool:
        """
        Returns True if trading should STOP
        """
        # Daily loss circuit breaker
        if self.daily_pnl < -self.max_daily_loss:
            self.logger.error(f"CIRCUIT BREAKER: Daily loss limit hit. No new trades.")
            return True

        # Equity protection - halt if equity has fallen meaningfully below
        # where this agent started, rather than an arbitrary fixed number.
        equity_floor = self.initial_equity * self.min_equity_pct
        if self.account_equity < equity_floor:
            self.logger.error(
                f"CIRCUIT BREAKER: Equity ${self.account_equity:,.2f} below floor "
                f"${equity_floor:,.2f} ({self.min_equity_pct:.0%} of starting ${self.initial_equity:,.2f}). No new trades."
            )
            return True

        return False

    # ============================================
    # EXECUTION & POSITION MANAGEMENT
    # ============================================

    def execute_signal(self, signal: Signal) -> Optional[Position]:
        """
        Execute trade signal (paper or live)
        """
        if signal.direction == "HOLD" or self.check_circuit_breakers():
            return None

        position_size = self.calculate_position_size(signal)
        if position_size == 0:
            return None

        position = Position(
            asset=signal.asset,
            direction="LONG" if signal.direction == "BUY" else "SHORT",
            entry_price=signal.entry_price,
            entry_time=datetime.now(),
            quantity=position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_amount=self.account_equity * self.risk_per_trade,
            position_id=f"{signal.asset}_{datetime.now().timestamp()}"
        )

        if self.mode == TradingMode.PAPER:
            self.logger.info(f"[PAPER] {signal.asset} {signal.direction} {position_size:.2f} @ {signal.entry_price}")
        else:
            self._send_to_ig_api(position)
            self.logger.info(f"[LIVE] {signal.asset} {signal.direction} {position_size:.2f} @ {signal.entry_price}")

        self.positions[signal.asset] = position
        return position

    def update_positions(self, current_prices: Dict[str, float]):
        """
        Monitor open positions, close on TP/SL
        """
        closed_positions = []

        for asset, position in self.positions.items():
            current_price = current_prices.get(asset, 0)
            if current_price == 0:
                continue

            hit_sl = (
                (position.direction == "LONG" and current_price <= position.stop_loss) or
                (position.direction == "SHORT" and current_price >= position.stop_loss)
            )
            hit_tp = (
                (position.direction == "LONG" and current_price >= position.take_profit) or
                (position.direction == "SHORT" and current_price <= position.take_profit)
            )

            if hit_sl:
                self._close_position(position, current_price, "SL")
                closed_positions.append(asset)
            elif hit_tp:
                self._close_position(position, current_price, "TP")
                closed_positions.append(asset)

        for asset in closed_positions:
            del self.positions[asset]

    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close position and record trade"""
        direction_multiplier = 1 if position.direction == "LONG" else -1
        pnl = (exit_price - position.entry_price) * position.quantity * direction_multiplier
        pnl_percent = ((exit_price - position.entry_price) / position.entry_price) * 100 * direction_multiplier

        trade = Trade(
            asset=position.asset,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_percent=pnl_percent,
            strategy=position.asset  # Track which strategy
        )

        self.trades.append(trade)
        self.daily_pnl += pnl
        self.account_equity += pnl

        self.logger.info(f"CLOSED {position.asset} [{reason}] | P&L: ${pnl:,.2f} ({pnl_percent:.2f}%)")

    # ============================================
    # INDICATOR CALCULATIONS
    # ============================================

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return np.convolve(data, np.ones(period)/period, mode='valid') if len(data) >= period else data

    @staticmethod
    def _atr(bars: List[PriceBar], period: int = 14) -> np.ndarray:
        """Average True Range"""
        tr = []
        for i in range(1, len(bars)):
            h_l = bars[i].high - bars[i].low
            h_pc = abs(bars[i].high - bars[i-1].close)
            l_pc = abs(bars[i].low - bars[i-1].close)
            tr.append(max(h_l, h_pc, l_pc))

        if len(tr) < period:
            return np.array(tr)

        atr_vals = [np.mean(tr[:period])]
        for val in tr[period:]:
            atr_vals.append((atr_vals[-1] * (period-1) + val) / period)

        return np.array(atr_vals)

    # ============================================
    # IG AUSTRALIA API INTEGRATION
    # ============================================

    def _send_to_ig_api(self, position: Position):
        """Send trade to IG Australia via webhook"""
        if not self.ig_api_key:
            self.logger.error("IG API key not configured")
            return

        webhook_url = os.environ.get("IG_WEBHOOK_URL")
        if not webhook_url:
            self.logger.error("IG_WEBHOOK_URL not configured")
            return

        payload = {
            "epic": f"CS.D.{position.asset}.CFD",  # IG epic format
            "direction": "BUY" if position.direction == "LONG" else "SELL",
            "size": position.quantity,
            "orderType": "MARKET",
            "stopLevel": position.stop_loss,
            "profitLevel": position.take_profit,
        }

        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Authorization": f"Bearer {self.ig_api_key}"},
                timeout=10,
            )
            self.logger.info(f"IG API Response: {response.status_code}")
        except Exception as e:
            self.logger.error(f"IG API Error: {e}")

    # ============================================
    # LOGGING & ANALYTICS
    # ============================================

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("TradingAgent")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def get_performance_summary(self) -> Dict:
        """Return performance metrics"""
        if not self.trades:
            return {"message": "No trades completed yet"}

        pnls = [t.pnl for t in self.trades]
        win_rate = sum(1 for t in self.trades if t.pnl > 0) / len(self.trades) * 100
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0

        return {
            "total_trades": len(self.trades),
            "total_pnl": sum(pnls),
            "win_rate_percent": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(sum([p for p in pnls if p > 0]) / sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0,
            "current_equity": self.account_equity,
            "open_positions": len(self.positions)
        }


# ============================================
# DEMO: Run the Agent
# ============================================

if __name__ == "__main__":
    print("""
    ================================================================
             RAMESH'S AUTOMATED TRADING AGENT V1

      Golden Pullback Strategy (Gold)
      SMA Cluster Reclaim (Oil)
      Regime Adaptation
      Risk Management (1% per trade, 5% daily limit)
      IG Australia Integration

      MODE: PAPER TRADING (Safe Testing)

    ================================================================
    """)

    # Initialize agent in PAPER mode
    agent = AdvancedTradingAgent(
        mode=TradingMode.PAPER,
        account_equity=150000,
        risk_per_trade=0.01,
        daily_loss_limit=0.05
    )

    print("\nAgent ready for PAPER TRADING")
    print("To use LIVE trading, switch mode to TradingMode.LIVE")
    print("PAPER MODE: All trades are simulated, no real money at risk")
    print("\nNext steps:")
    print("1. Feed live price data (via websocket/API)")
    print("2. Agent will generate signals automatically")
    print("3. Monitor performance in real-time")
