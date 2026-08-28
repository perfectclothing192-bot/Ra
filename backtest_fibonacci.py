"""
Standalone backtest of a Fibonacci retracement continuation strategy against
1 year of real OANDA historical M15 candles. Same research-script pattern as
the other backtest_*.py scripts - not part of the live trading loops, run
manually:

    python backtest_fibonacci.py [ASSET] [--days N]

Strategy: Fibonacci Retracement Continuation
- Reuses the same fractal swing-point detection as backtest_smc_liquidity.py
  (a swing is confirmed once the FRACTAL_WINDOW bars after it exist - no
  lookahead). Tracks the most recent confirmed swing high and swing low.
- Leg direction: whichever of the two swings is more recent defines the
  leg's end (B); the other is the leg's start (A).
    - low(A) -> high(B), B more recent: up-leg. Retracement zone is the
      50%-61.8% pullback down from B toward A ("golden pocket").
      A bullish reaction inside that zone (close > prior close) = LONG,
      betting the up-leg continues.
    - high(A) -> low(B), B more recent: down-leg, mirrored for SHORT.
- Stop loss: the 78.6% retracement level of the same leg (a deeper pullback
  invalidates the setup). Take profit: 2.5x ATR (same multiple as
  golden_pullback_signal, for comparability across the other backtests).
- Only one trade is taken per unique leg (a fresh entry requires a newly
  confirmed swing point), to avoid re-entering the same zone repeatedly as
  price oscillates inside it.
- One position at a time, 1% risk per trade (halved past 10% drawdown,
  same as the live agents), $150,000 starting equity.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from oanda_client import fetch_candles_range

FRACTAL_WINDOW = 5
ZONE_50 = 0.5
ZONE_618 = 0.618
STOP_786 = 0.786


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
    atr = compute_atr(bars, 14)

    last_high = None  # (index, price)
    last_low = None
    confirmed_up_to = -1
    used_legs = set()

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None

    warmup = 210
    for i in range(warmup, len(bars)):
        bar = bars[i]
        prev_close = bars[i - 1].close
        current_atr = atr[i]

        newly_confirmable = i - FRACTAL_WINDOW
        while confirmed_up_to < newly_confirmable:
            confirmed_up_to += 1
            kind, price = find_confirmed_swing(bars, confirmed_up_to)
            if kind == "high":
                last_high = (confirmed_up_to, price)
            elif kind == "low":
                last_low = (confirmed_up_to, price)

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

        if not last_high or not last_low:
            equity_curve.append(equity)
            continue

        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0
        risk_scale = 0.5 if drawdown > 0.10 else 1.0

        idx_high, price_high = last_high
        idx_low, price_low = last_low
        leg_range = price_high - price_low
        if leg_range <= 0:
            equity_curve.append(equity)
            continue

        if idx_high > idx_low:
            # up-leg: low -> high, retrace down into the golden pocket for a LONG
            leg_id = ("up", idx_low, idx_high)
            zone_top = price_high - leg_range * ZONE_50
            zone_bottom = price_high - leg_range * ZONE_618
            stop_level = price_high - leg_range * STOP_786
            in_zone = bar.low <= zone_top and bar.close >= zone_bottom
            reversal = bar.close > prev_close
            if leg_id not in used_legs and in_zone and reversal:
                entry_price = bar.close
                stop_loss = stop_level
                take_profit = entry_price + current_atr * 2.5
                stop_distance = entry_price - stop_loss
                if stop_distance > 0:
                    risk_amount = equity * risk_per_trade * risk_scale
                    quantity = risk_amount / stop_distance
                    position = {"direction": "LONG", "entry_price": entry_price, "stop_loss": stop_loss,
                                "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}
                    used_legs.add(leg_id)
        else:
            # down-leg: high -> low, retrace up into the golden pocket for a SHORT
            leg_id = ("down", idx_high, idx_low)
            zone_bottom = price_low + leg_range * ZONE_50
            zone_top = price_low + leg_range * ZONE_618
            stop_level = price_low + leg_range * STOP_786
            in_zone = bar.high >= zone_bottom and bar.close <= zone_top
            reversal = bar.close < prev_close
            if leg_id not in used_legs and in_zone and reversal:
                entry_price = bar.close
                stop_loss = stop_level
                take_profit = entry_price - current_atr * 2.5
                stop_distance = stop_loss - entry_price
                if stop_distance > 0:
                    risk_amount = equity * risk_per_trade * risk_scale
                    quantity = risk_amount / stop_distance
                    position = {"direction": "SHORT", "entry_price": entry_price, "stop_loss": stop_loss,
                                "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}
                    used_legs.add(leg_id)

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
