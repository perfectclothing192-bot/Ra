"""
Standalone backtest of a Smart Money Concepts (SMC) liquidity-sweep strategy
against 1 year of real OANDA historical M15 candles. Same research-script
pattern as backtest_vwap.py / backtest_sma_cross.py - not part of the live
trading loops, run manually:

    python backtest_smc_liquidity.py [ASSET] [--days N]

Strategy: Liquidity Sweep + Reclaim, filtered by higher-timeframe structure
- Swing points: a bar is a confirmed swing low/high if its low/high is the
  extreme within a +/-5 bar fractal window. It can only be confirmed once
  the 5 bars *after* it exist, so live/backtest signals never use a swing
  point before it's actually knowable (no lookahead).
- Structure bias: price above EMA200 = bullish bias (only take bullish
  sweeps), price below EMA200 = bearish bias (only take bearish sweeps) -
  the standard SMC idea of only hunting liquidity in the direction of the
  higher-timeframe trend, not against it.
- Liquidity sweep + reclaim (the entry): price wicks below (bullish case) a
  recent confirmed swing low - sweeping the stop-loss liquidity resting
  there - then the same candle closes back above that swing low. Mirrors
  for a bearish sweep of a recent swing high.
- Stop loss: the sweep candle's wick extreme (the actual liquidity-grab
  low/high), plus a small ATR buffer. Take profit: 2.5x ATR (same multiple
  as golden_pullback_signal, for comparability).
- One position at a time, 1% risk per trade (halved past 10% drawdown,
  same as the live agents), $150,000 starting equity.

No lookahead: every input to a signal at bar i only uses bars[0:i+1], and
swing points are only usable once the confirming bars have actually passed.
"""

import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from oanda_client import fetch_candles_range

FRACTAL_WINDOW = 5      # bars on each side to confirm a swing point
SWING_LOOKBACK = 20     # how many recent confirmed swings to consider sweepable


@dataclass
class BacktestTrade:
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: float
    pnl: float
    reason: str


def compute_ema(closes, period):
    return np.array(
        __import__("pandas").Series(closes).ewm(span=period, adjust=False).mean()
    )


def compute_atr(bars, period=14):
    atr = np.zeros(len(bars))
    trs = [0.0]
    for i in range(1, len(bars)):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(bars) > period:
        atr[period] = np.mean(trs[1:period + 1])
        for i in range(period + 1, len(bars)):
            atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period
    return atr


def find_confirmed_swing(bars, idx):
    """
    Check whether bars[idx] is a fractal swing point, confirmable only once
    bars[idx + FRACTAL_WINDOW] exists. Returns ("high"|"low"|None, price).
    """
    lo = idx - FRACTAL_WINDOW
    hi = idx + FRACTAL_WINDOW
    if lo < 0 or hi >= len(bars):
        return None, None
    window_highs = [bars[j].high for j in range(lo, hi + 1)]
    window_lows = [bars[j].low for j in range(lo, hi + 1)]
    if bars[idx].high == max(window_highs):
        return "high", bars[idx].high
    if bars[idx].low == min(window_lows):
        return "low", bars[idx].low
    return None, None


def run_backtest(bars, account_equity: float = 150000, risk_per_trade: float = 0.01):
    closes = np.array([b.close for b in bars])
    ema200 = compute_ema(closes, 200)
    atr = compute_atr(bars, 14)

    swing_lows = deque(maxlen=SWING_LOOKBACK)   # list of prices
    swing_highs = deque(maxlen=SWING_LOOKBACK)
    confirmed_up_to = -1  # last bar index whose swing status we've checked

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None

    warmup = 200
    for i in range(warmup, len(bars)):
        bar = bars[i]
        current_atr = atr[i]

        # Confirm any newly-knowable swing points (as of bar i, bar i - FRACTAL_WINDOW
        # is the newest one that could just now be confirmed).
        newly_confirmable = i - FRACTAL_WINDOW
        while confirmed_up_to < newly_confirmable:
            confirmed_up_to += 1
            kind, price = find_confirmed_swing(bars, confirmed_up_to)
            if kind == "low":
                swing_lows.append(price)
            elif kind == "high":
                swing_highs.append(price)

        if current_atr == 0:
            equity_curve.append(equity)
            continue

        if position:
            hit_sl = (
                (position["direction"] == "LONG" and bar.low <= position["stop_loss"]) or
                (position["direction"] == "SHORT" and bar.high >= position["stop_loss"])
            )
            hit_tp = (
                (position["direction"] == "LONG" and bar.high >= position["take_profit"]) or
                (position["direction"] == "SHORT" and bar.low <= position["take_profit"])
            )
            if hit_sl or hit_tp:
                exit_price = position["stop_loss"] if hit_sl else position["take_profit"]
                direction_mult = 1 if position["direction"] == "LONG" else -1
                pnl = (exit_price - position["entry_price"]) * position["quantity"] * direction_mult
                trades.append(BacktestTrade(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], exit_time=bar.timestamp,
                    exit_price=exit_price, quantity=position["quantity"], pnl=pnl,
                    reason="TP" if hit_tp else "SL",
                ))
                equity += pnl
                peak_equity = max(peak_equity, equity)
                position = None
            equity_curve.append(equity)
            continue

        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0
        risk_scale = 0.5 if drawdown > 0.10 else 1.0

        bullish_bias = bar.close > ema200[i]
        bearish_bias = bar.close < ema200[i]

        bullish_sweep = None
        if bullish_bias and swing_lows:
            # nearest swing low below current price that this bar's wick swept and reclaimed
            candidates = [lvl for lvl in swing_lows if bar.low < lvl <= bar.close]
            if candidates:
                bullish_sweep = max(candidates)  # the closest one above the wick

        bearish_sweep = None
        if bearish_bias and swing_highs:
            candidates = [lvl for lvl in swing_highs if bar.high > lvl >= bar.close]
            if candidates:
                bearish_sweep = min(candidates)

        if bullish_sweep is not None:
            entry_price = bar.close
            stop_loss = bar.low - current_atr * 0.1
            take_profit = entry_price + current_atr * 2.5
            stop_distance = entry_price - stop_loss
            if stop_distance > 0:
                risk_amount = equity * risk_per_trade * risk_scale
                quantity = risk_amount / stop_distance
                position = {"direction": "LONG", "entry_price": entry_price, "stop_loss": stop_loss,
                            "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}
        elif bearish_sweep is not None:
            entry_price = bar.close
            stop_loss = bar.high + current_atr * 0.1
            take_profit = entry_price - current_atr * 2.5
            stop_distance = stop_loss - entry_price
            if stop_distance > 0:
                risk_amount = equity * risk_per_trade * risk_scale
                quantity = risk_amount / stop_distance
                position = {"direction": "SHORT", "entry_price": entry_price, "stop_loss": stop_loss,
                            "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}

        equity_curve.append(equity)

    return trades, equity, equity_curve


def summarize(trades, starting_equity, ending_equity, equity_curve, bars):
    if not trades:
        print("No trades were generated.")
        return

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(trades) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")

    peak = starting_equity
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)

    total_return = (ending_equity - starting_equity) / starting_equity * 100
    days = (bars[-1].timestamp - bars[0].timestamp).days

    print(f"Period: {bars[0].timestamp.date()} -> {bars[-1].timestamp.date()} ({days} days, {len(bars)} M15 bars)")
    print(f"Total trades: {len(trades)}")
    print(f"Win rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Avg win: ${avg_win:,.2f}  |  Avg loss: ${avg_loss:,.2f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Starting equity: ${starting_equity:,.2f}")
    print(f"Ending equity:   ${ending_equity:,.2f}")
    print(f"Total return: {total_return:+.2f}%")
    print(f"Max drawdown: {max_dd * 100:.2f}%")


if __name__ == "__main__":
    asset = "XAUUSD"
    days = 365
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        asset = args[0]
        args = args[1:]
    if "--days" in args:
        days = int(args[args.index("--days") + 1])

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print(f"Fetching {days}d of M15 {asset} candles from OANDA ({start.date()} -> {end.date()})...")
    bars = fetch_candles_range(asset, "M15", start, end)
    print(f"Fetched {len(bars)} bars.\n")

    trades, ending_equity, equity_curve = run_backtest(bars)
    summarize(trades, 150000, ending_equity, equity_curve, bars)
