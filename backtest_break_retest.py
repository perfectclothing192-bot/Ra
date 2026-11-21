"""
Standalone backtest of a "break and retest" strategy against 1 year of real
OANDA historical M15 candles. Same research-script pattern as the other
backtest_*.py scripts - not part of the live trading loops, run manually:

    python backtest_break_retest.py [ASSET] [--days N]

Strategy: Swing-Structure Break + Retest
- Swing points: a bar is a confirmed swing low/high if its low/high is the
  extreme within a +/-FRACTAL_WINDOW bar fractal window. It can only be
  confirmed once the FRACTAL_WINDOW bars *after* it exist, so signals never
  use a swing point before it's actually knowable (no lookahead) - same
  fractal convention as backtest_smc_liquidity.py.
- The break: the most recently confirmed swing high/low is tracked as the
  active level. A bar that *closes* beyond it (above a swing high, or below
  a swing low) is a structure break. That level is then "consumed" (the
  script waits for the next swing point to track a new one) and becomes a
  pending retest, valid for RETEST_WINDOW bars.
- The retest: within the window, if a bar closes back through the broken
  level, the break is invalidated (false breakout, no trade). Otherwise, the
  first bar whose wick touches back to the level but still *closes* beyond
  it (the level holding as new support/resistance) triggers entry in the
  breakout direction.
- Stop loss: the retest bar's wick extreme (the level getting touched, i.e.
  the actual low/high of the retest), plus a small ATR buffer. Take profit:
  2.5x ATR (same multiple as golden_pullback_signal / backtest_smc_liquidity,
  for comparability).
- Only one pending break (either direction) is tracked at a time, and only
  one position at a time, 1% risk per trade (halved past 10% drawdown, same
  as the live agents), $150,000 starting equity.

No lookahead: every input to a signal at bar i only uses bars[0:i+1], and
swing points are only usable once the confirming bars have actually passed.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from oanda_client import fetch_candles_range

FRACTAL_WINDOW = 5     # bars on each side to confirm a swing point
RETEST_WINDOW = 20     # bars a break has to be retested before it expires


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
    atr = compute_atr(bars, 14)

    last_swing_high = None
    last_swing_low = None
    confirmed_up_to = -1

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None
    pending_break = None  # dict: direction ("up"|"down"), level, break_idx

    warmup = max(FRACTAL_WINDOW, 14) + 1
    for i in range(warmup, len(bars)):
        bar = bars[i]
        current_atr = atr[i]

        newly_confirmable = i - FRACTAL_WINDOW
        while confirmed_up_to < newly_confirmable:
            confirmed_up_to += 1
            kind, price = find_confirmed_swing(bars, confirmed_up_to)
            if kind == "low":
                last_swing_low = price
            elif kind == "high":
                last_swing_high = price

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

        if pending_break:
            level = pending_break["level"]
            if i - pending_break["break_idx"] > RETEST_WINDOW:
                pending_break = None
            elif pending_break["direction"] == "up":
                if bar.close < level:
                    pending_break = None  # false breakout
                elif bar.low <= level:
                    entry_price = bar.close
                    stop_loss = bar.low - current_atr * 0.1
                    take_profit = entry_price + current_atr * 2.5
                    stop_distance = entry_price - stop_loss
                    if stop_distance > 0:
                        risk_amount = equity * risk_per_trade * risk_scale
                        quantity = risk_amount / stop_distance
                        position = {"direction": "LONG", "entry_price": entry_price,
                                    "stop_loss": stop_loss, "take_profit": take_profit,
                                    "quantity": quantity, "entry_time": bar.timestamp}
                    pending_break = None
            else:  # "down"
                if bar.close > level:
                    pending_break = None  # false breakdown
                elif bar.high >= level:
                    entry_price = bar.close
                    stop_loss = bar.high + current_atr * 0.1
                    take_profit = entry_price - current_atr * 2.5
                    stop_distance = stop_loss - entry_price
                    if stop_distance > 0:
                        risk_amount = equity * risk_per_trade * risk_scale
                        quantity = risk_amount / stop_distance
                        position = {"direction": "SHORT", "entry_price": entry_price,
                                    "stop_loss": stop_loss, "take_profit": take_profit,
                                    "quantity": quantity, "entry_time": bar.timestamp}
                    pending_break = None
        else:
            if last_swing_high is not None and bar.close > last_swing_high:
                pending_break = {"direction": "up", "level": last_swing_high, "break_idx": i}
                last_swing_high = None
            elif last_swing_low is not None and bar.close < last_swing_low:
                pending_break = {"direction": "down", "level": last_swing_low, "break_idx": i}
                last_swing_low = None

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
