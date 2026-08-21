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

    # Per-strategy risk override. correlation_hedge's stops are only a
    # handful of pips on GBP/EUR, so risk_per_trade's normal 5% translates
    # into a position size millions of units large - far more notional
    # than OANDA's 30:1 margin rate (~3.33%) can support on this account
    # (a single leg alone needs ~$116k margin against a ~$100k account).
    # 1% matches the level that actually filled on Aug 13, before risk was
    # raised to 5% for the other strategies, which have wider stops
    # relative to their instruments and don't hit this ceiling.
    # fx_range_reversion has the same problem, worse: its stop is a
    # fraction of a single M15 ATR on GBP/EUR (median ~0.045% of price,
    # tightest observed ~0.026%), so even 1% risk can demand ~150% of
    # account equity in margin for a single leg in the tight-stop tail.
    # 0.25% keeps worst-case single-leg margin under ~40% of equity.
    # jesse_livermore's stops are much wider (median ~0.79% of price on
    # XAUUSD M15, vs fx_range_reversion's ~0.045%) so it doesn't hit the
    # same wall - default 5% risk alone maxes out around 30-60% of equity
    # in margin (gold's marginRate is 5%, i.e. 20:1, not FX's 30:1).
    # Trimmed to 3% specifically because this runs *alongside* smc_signal
    # on the same underlying instrument (XAUUSD M5 + XAUUSD_M15 can both
    # be open at once) - leaves headroom instead of two full-size 5%
    # gold positions both maxing margin at the same time.
    STRATEGY_RISK_OVERRIDE = {
        "correlation_hedge": 0.01,
        "fx_range_reversion": 0.0025,
        "jesse_livermore": 0.03,
    }

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
            "EURUSD": [],
            "BTCUSD": []
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
    # STRATEGY 4: SMART MONEY CONCEPTS (GOLD)
    # ============================================

    def smc_signal(self, bars: List[PriceBar]) -> Signal:
        """
        Smart Money Concepts strategy for XAUUSD (liquidity sweep + break of structure).

        Rules:
        1. Track swing highs/lows via a simple fractal: a bar is a swing
           point if its high (or low) is the most extreme among the 3 bars
           on each side.
        2. Liquidity sweep: the latest bar's low pierces below the most
           recent confirmed swing low, then closes back above it - price
           ran the resting stop-loss liquidity below that low and rejected
           (the mirror image applies to the bearish case, using swing highs).
        3. Break of structure: that same bar's close also breaks above the
           swing high that preceded the swept low, confirming a genuine
           structure shift rather than just a wick into the level.
        4. Entry on the confirming close. Stop beyond the sweep's extreme
           wick, target at 2x the stop distance.

        This intentionally does not fire on every pullback - a sweep
        without a structure break, or a structure break without a prior
        sweep, is not enough on its own.
        """
        hold = Signal("XAUUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "smc", 0, datetime.now())

        fractal = 3
        if len(bars) < 80:
            return hold

        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])
        n = len(bars)

        swing_low_idxs = [
            i for i in range(fractal, n - fractal)
            if lows[i] == min(lows[i - fractal:i + fractal + 1])
        ]
        swing_high_idxs = [
            i for i in range(fractal, n - fractal)
            if highs[i] == max(highs[i - fractal:i + fractal + 1])
        ]

        if not swing_low_idxs or not swing_high_idxs:
            return hold

        current_close = closes[-1]
        current_low = lows[-1]
        current_high = highs[-1]
        atr = self._atr(bars, 14)[-1]

        # Bullish: most recent confirmed swing low got swept and reclaimed,
        # and price also broke back above the swing high that preceded it.
        last_swing_low_idx = swing_low_idxs[-1]
        last_swing_low = lows[last_swing_low_idx]
        prior_swing_highs = [h for h in swing_high_idxs if h < last_swing_low_idx]

        if prior_swing_highs:
            structure_high = highs[prior_swing_highs[-1]]
            swept_low = current_low < last_swing_low and current_close > last_swing_low
            broke_structure = current_close > structure_high
            if swept_low and broke_structure:
                entry_price = current_close
                stop_loss = current_low - atr * 0.3
                take_profit = entry_price + (entry_price - stop_loss) * 2.0
                return Signal(
                    asset="XAUUSD",
                    direction="BUY",
                    strength=SignalStrength.STRONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=2.0,
                    strategy="smc",
                    confidence=0.78,
                    timestamp=datetime.now()
                )

        # Bearish: mirror image, using swing highs.
        last_swing_high_idx = swing_high_idxs[-1]
        last_swing_high = highs[last_swing_high_idx]
        prior_swing_lows = [l for l in swing_low_idxs if l < last_swing_high_idx]

        if prior_swing_lows:
            structure_low = lows[prior_swing_lows[-1]]
            swept_high = current_high > last_swing_high and current_close < last_swing_high
            broke_structure = current_close < structure_low
            if swept_high and broke_structure:
                entry_price = current_close
                stop_loss = current_high + atr * 0.3
                take_profit = entry_price - (stop_loss - entry_price) * 2.0
                return Signal(
                    asset="XAUUSD",
                    direction="SELL",
                    strength=SignalStrength.STRONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=2.0,
                    strategy="smc",
                    confidence=0.78,
                    timestamp=datetime.now()
                )

        return hold

    # ============================================
    # STRATEGY 5: VOLATILITY BREAKOUT (BITCOIN)
    # ============================================

    def volatility_breakout_signal(self, bars: List[PriceBar]) -> Signal:
        """
        Volatility Breakout strategy for BTCUSD.

        Crypto behaves differently enough from forex/gold/oil that reusing
        a mean-reversion-flavored strategy here would be a mismatch: BTC
        trends tend to extend hard once genuinely underway, and "buying
        the dip" during a real breakdown is a common way to get run over.
        This is deliberately a momentum/breakout design instead:

        1. Track a rolling 20-bar Donchian channel (highest high / lowest
           low over the prior 20 bars, excluding the current one).
        2. Require volatility to actually be expanding first: current ATR
           must exceed its own 20-bar average by 30%. A channel breakout
           during a quiet, illiquid stretch is much more likely to be
           noise than the start of a real move - this filter is what
           keeps the strategy from firing on every minor wiggle.
        3. Entry: the current bar's close breaks above the channel high
           (bullish) or below the channel low (bearish), with volatility
           already expanding.
        4. Stop: 1.5x ATR from entry. Target: 3x ATR (2:1 reward-to-risk)
           - wide enough to let a genuine trend run rather than clip it
           at the first pullback, which is the whole point of trading
           breakouts on an asset this momentum-driven.
        """
        hold = Signal("BTCUSD", "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "volatility_breakout", 0, datetime.now())

        channel = 20
        if len(bars) < channel + 21:
            return hold

        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])

        atr_series = self._atr(bars, 14)
        if len(atr_series) < 21:
            return hold
        current_atr = atr_series[-1]
        avg_atr = np.mean(atr_series[-21:-1])

        if avg_atr == 0 or current_atr < avg_atr * 1.3:
            return hold

        channel_high = np.max(highs[-(channel + 1):-1])
        channel_low = np.min(lows[-(channel + 1):-1])
        current_close = closes[-1]

        if current_close > channel_high:
            entry_price = current_close
            stop_loss = entry_price - current_atr * 1.5
            take_profit = entry_price + current_atr * 3.0
            return Signal(
                asset="BTCUSD",
                direction="BUY",
                strength=SignalStrength.STRONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=2.0,
                strategy="volatility_breakout",
                confidence=0.72,
                timestamp=datetime.now()
            )

        if current_close < channel_low:
            entry_price = current_close
            stop_loss = entry_price + current_atr * 1.5
            take_profit = entry_price - current_atr * 3.0
            return Signal(
                asset="BTCUSD",
                direction="SELL",
                strength=SignalStrength.STRONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=2.0,
                strategy="volatility_breakout",
                confidence=0.72,
                timestamp=datetime.now()
            )

        return hold

    # ============================================
    # STRATEGY 6: JESSE LIVERMORE PIVOTAL POINT
    # ============================================

    def jesse_livermore_signal(self, bars: List[PriceBar], asset: str = "XAUUSD") -> Signal:
        """
        Jesse Livermore-style Pivotal Point strategy, adapted from
        "Reminiscences of a Stock Operator": trade only with the line of
        least resistance, and enter at pivotal points rather than dips.

        1. Primary trend ("line of least resistance") is SMA 50 vs SMA
           200 - only longs are considered in an uptrend, only shorts in
           a downtrend. Livermore's biggest losses came from fighting the
           tape; this filter exists to prevent that outright.
        2. A pivotal point is the highest high / lowest low of the prior
           20 bars (excluding the current one) - the edge of the base the
           market has been building. A close beyond it, in the direction
           of the primary trend, is the breakout he bought (or sold).
        3. Stop sits just beyond the most recent 10-bar minor swing point
           opposite the breakout, plus a small ATR buffer - true to "cut
           losses quickly." Target is 3x that risk - true to "let your
           winners run," the other half of the same discipline.
        """
        hold = Signal(asset, "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "jesse_livermore", 0, datetime.now())

        pivot_window = 20
        swing_window = 10
        trend_window = 200
        if len(bars) < trend_window + 1:
            return hold

        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])

        sma_50 = self._sma(closes, 50)[-1]
        sma_200 = self._sma(closes, 200)[-1]
        atr = self._atr(bars, 14)[-1]
        current_close = closes[-1]

        pivot_high = np.max(highs[-(pivot_window + 1):-1])
        pivot_low = np.min(lows[-(pivot_window + 1):-1])
        recent_low = np.min(lows[-(swing_window + 1):-1])
        recent_high = np.max(highs[-(swing_window + 1):-1])

        uptrend = sma_50 > sma_200
        downtrend = sma_50 < sma_200

        if uptrend and current_close > pivot_high:
            entry_price = current_close
            stop_loss = recent_low - atr * 0.3
            risk = entry_price - stop_loss
            if risk <= 0:
                return hold
            take_profit = entry_price + risk * 3.0
            return Signal(
                asset=asset,
                direction="BUY",
                strength=SignalStrength.STRONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=3.0,
                strategy="jesse_livermore",
                confidence=0.75,
                timestamp=datetime.now()
            )

        if downtrend and current_close < pivot_low:
            entry_price = current_close
            stop_loss = recent_high + atr * 0.3
            risk = stop_loss - entry_price
            if risk <= 0:
                return hold
            take_profit = entry_price - risk * 3.0
            return Signal(
                asset=asset,
                direction="SELL",
                strength=SignalStrength.STRONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=3.0,
                strategy="jesse_livermore",
                confidence=0.75,
                timestamp=datetime.now()
            )

        return hold

    # ============================================
    # STRATEGY 7: FX RANGE REVERSION (GBPUSD / EURUSD)
    # ============================================

    def fx_range_reversion_signal(self, bars: List[PriceBar], asset: str) -> Signal:
        """
        Replacement for correlation_hedge on GBPUSD/EURUSD: instead of
        pairing the two instruments, trade each one independently as a
        quiet-regime Bollinger Band fade back toward its own mean.

        1. A rolling 40-bar mean/stdev defines the band; price closing
           outside +-2.25 std is a range extreme.
        2. Only fade the extreme when the market is quiet (current ATR <
           1.1x its 50-bar average) - mean reversion gets run over in a
           trending/volatile regime, so this filters those out.
        3. Require a one-bar reversal confirmation (close ticks back
           toward the mean vs. the prior close) before entering, rather
           than fading the extreme bar itself.
        4. Stop sits just beyond the signal bar's high/low plus a 0.5x
           ATR buffer; target is the band mean (reversion, not breakout).

        Backtested (1yr M15, walk-forward, no lookahead, out-of-sample
        checked on a 70/30 train/test split): GBPUSD +12.4R/121 trades
        (29.8% WR, PF 1.15), EURUSD +9.0R/107 trades (29.9% WR, PF 1.12).
        Both positive on both halves of the split - modest edge, much
        thinner than smc_signal or jesse_livermore, but the first design
        tried here (existing single-asset strategies, correlation_hedge's
        pairing) that came back robust on both legs instead of flat/
        negative. See STRATEGY_RISK_OVERRIDE: like correlation_hedge, its
        stops are tight enough that default risk_per_trade would blow
        past OANDA's margin.
        """
        hold = Signal(asset, "HOLD", SignalStrength.WEAK, 0, 0, 0, 0, "fx_range_reversion", 0, datetime.now())

        band_period = 40
        std_mult = 2.25
        atr_quiet_mult = 1.1
        stop_atr_mult = 0.5
        warmup = max(band_period, 14 + 50) + 1
        if len(bars) < warmup:
            return hold

        closes = np.array([b.close for b in bars])
        atr_series = self._atr(bars, 14)
        atr = atr_series[-1]
        atr_avg50 = np.mean(atr_series[-50:])
        if atr_avg50 <= 0 or atr >= atr_quiet_mult * atr_avg50:
            return hold

        window = closes[-band_period:]
        mean = window.mean()
        std = window.std()
        if std == 0:
            return hold
        upper = mean + std_mult * std
        lower = mean - std_mult * std

        prev_close, current_close = closes[-2], closes[-1]
        current_high, current_low = bars[-1].high, bars[-1].low

        if current_close < lower and current_close > prev_close:
            entry_price = current_close
            stop_loss = current_low - atr * stop_atr_mult
            risk = entry_price - stop_loss
            if risk <= 0:
                return hold
            take_profit = mean
            risk_reward = (take_profit - entry_price) / risk
            return Signal(
                asset=asset,
                direction="BUY",
                strength=SignalStrength.MEDIUM,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                strategy="fx_range_reversion",
                confidence=0.55,
                timestamp=datetime.now()
            )

        if current_close > upper and current_close < prev_close:
            entry_price = current_close
            stop_loss = current_high + atr * stop_atr_mult
            risk = stop_loss - entry_price
            if risk <= 0:
                return hold
            take_profit = mean
            risk_reward = (entry_price - take_profit) / risk
            return Signal(
                asset=asset,
                direction="SELL",
                strength=SignalStrength.MEDIUM,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                strategy="fx_range_reversion",
                confidence=0.55,
                timestamp=datetime.now()
            )

        return hold

    def jesse_livermore_xauusd_m15_signal(self, bars: List[PriceBar]) -> Signal:
        """Wrapper so this fits the bars-only calling convention, using
        the synthetic "XAUUSD_M15" asset key so this M15 position slot
        stays independent of the M5 smc_signal position on "XAUUSD" -
        see the ASSET_TO_OANDA_INSTRUMENT comment in oanda_client.py."""
        return self.jesse_livermore_signal(bars, asset="XAUUSD_M15")

    def fx_range_reversion_gbpusd_signal(self, bars: List[PriceBar]) -> Signal:
        """Thin wrapper so this fits the single-asset (bars-only) calling
        convention the live loop and backtester use for every other
        strategy - fx_range_reversion_signal itself is asset-agnostic."""
        return self.fx_range_reversion_signal(bars, "GBPUSD")

    def fx_range_reversion_eurusd_signal(self, bars: List[PriceBar]) -> Signal:
        return self.fx_range_reversion_signal(bars, "EURUSD")

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

        # Risk amount (per-strategy override for strategies whose stops
        # are too tight for the standard risk_per_trade to be marginable)
        risk_pct = self.STRATEGY_RISK_OVERRIDE.get(signal.strategy, self.risk_per_trade)
        risk_amount = self.account_equity * risk_pct

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
