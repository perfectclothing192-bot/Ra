"""
Standalone backtest of a Donchian Channel breakout strategy (classic
trend-following, Turtle Trading-style) against 1 year of real OANDA
historical M15 candles. Same research-script pattern as the other
backtest_*.py scripts - not part of the live trading loops, run manually:

    python backtest_donchian_breakout.py [ASSET] [--days N]

Strategy: Donchian Breakout + ATR Trailing Stop
- Donchian channel: the highest high and lowest low over the prior
  CHANNEL_PERIOD bars (not including the current bar - no lookahead).
- LONG entry: current bar closes above the prior-N-bar highest high.
- SHORT entry: current bar closes below the prior-N-bar lowest low.
- Exit: NOT a fixed take-profit - a trailing ATR stop that only ever
  tightens in the trade's favor (2x ATR behind the best close seen since
  entry), which is the actual mechanism behind trend-following's asymmetric
  win/loss profile (many small losses when a breakout fails immediately,
  occasional large wins when a trend actually runs).
- One position at a time, 1% risk per trade (halved past 10% drawdown,
  same as the live agents), $150,000 starting equity.

No lookahead: the channel at bar i is computed from bars[i-CHANNEL_PERIOD:i],
never including bar i itself.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from oanda_client import fetch_candles_range

CHANNEL_PERIOD = 20   # classic Turtle "system 1" entry lookback
TRAIL_ATR_MULT = 2.0
INITIAL_STOP_ATR_MULT = 2.0


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


def run_backtest(bars, account_equity: float = 150000, risk_per_trade: float = 0.01):
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    atr = compute_atr(bars, 14)

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None  # dict: direction, entry_price, stop, quantity, entry_time, best_close

    warmup = max(CHANNEL_PERIOD, 14) + 1
    for i in range(warmup, len(bars)):
        bar = bars[i]
        current_atr = atr[i]
        if current_atr == 0:
            equity_curve.append(equity)
            continue

        if position:
            # Trail the stop before checking for a hit, using the prior bar's close
            # (the trail only tightens, never loosens).
            if position["direction"] == "LONG":
                position["best_close"] = max(position["best_close"], closes[i - 1])
                new_stop = position["best_close"] - current_atr * TRAIL_ATR_MULT
                position["stop"] = max(position["stop"], new_stop)
                hit_stop = bar.low <= position["stop"]
            else:
                position["best_close"] = min(position["best_close"], closes[i - 1])
                new_stop = position["best_close"] + current_atr * TRAIL_ATR_MULT
                position["stop"] = min(position["stop"], new_stop)
                hit_stop = bar.high >= position["stop"]

            if hit_stop:
                exit_price = position["stop"]
                direction_mult = 1 if position["direction"] == "LONG" else -1
                pnl = (exit_price - position["entry_price"]) * position["quantity"] * direction_mult
                trades.append(BacktestTrade(
                    direction=position["direction"], entry_time=position["entry_time"],
                    entry_price=position["entry_price"], exit_time=bar.timestamp,
                    exit_price=exit_price, quantity=position["quantity"], pnl=pnl,
                    reason="TRAIL_STOP",
                ))
                equity += pnl
                peak_equity = max(peak_equity, equity)
                position = None
            equity_curve.append(equity)
            continue

        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0
        risk_scale = 0.5 if drawdown > 0.10 else 1.0

        channel_high = np.max(highs[i - CHANNEL_PERIOD:i])
        channel_low = np.min(lows[i - CHANNEL_PERIOD:i])

        if bar.close > channel_high:
            entry_price = bar.close
            stop = entry_price - current_atr * INITIAL_STOP_ATR_MULT
            stop_distance = entry_price - stop
            risk_amount = equity * risk_per_trade * risk_scale
            quantity = risk_amount / stop_distance
            position = {"direction": "LONG", "entry_price": entry_price, "stop": stop,
                        "quantity": quantity, "entry_time": bar.timestamp, "best_close": entry_price}
        elif bar.close < channel_low:
            entry_price = bar.close
            stop = entry_price + current_atr * INITIAL_STOP_ATR_MULT
            stop_distance = stop - entry_price
            risk_amount = equity * risk_per_trade * risk_scale
            quantity = risk_amount / stop_distance
            position = {"direction": "SHORT", "entry_price": entry_price, "stop": stop,
                        "quantity": quantity, "entry_time": bar.timestamp, "best_close": entry_price}

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
