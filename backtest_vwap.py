"""
Standalone backtest of a VWAP Pullback Reclaim strategy against 1 year of
real OANDA historical M15 candles. Not part of the live trading loops - this
is a research/validation script, run manually:

    python backtest_vwap.py [ASSET] [--days N]

Strategy: VWAP Pullback Reclaim (mirrors golden_pullback_signal's shape in
trading_agent.py, swapping EMA50 for a session VWAP as the pullback level,
so it's a fair apples-to-apples comparison against the strategy already
running live):
- VWAP resets each UTC calendar day, computed from typical price
  ((H+L+C)/3) weighted by each candle's tick volume.
- Uptrend bias: price above VWAP. Entry: price pulls back to within 0.5x
  ATR of VWAP, then closes back above it (a fresh reclaim, prior bar closed
  at/below VWAP).
- Downtrend bias mirrors this for SHORT entries.
- Stop loss: VWAP -/+ 0.5x ATR. Take profit: entry +/- 2.5x ATR (same
  multiples as golden_pullback_signal).
- One position at a time, 1% risk per trade, $150,000 starting equity -
  matching the live swing agent's defaults.

No lookahead: VWAP/ATR/signals at bar i only use bars[0:i+1].
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from oanda_client import fetch_candles_range
from trading_agent import AdvancedTradingAgent


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


def compute_vwap(bars):
    """Session VWAP, reset each UTC calendar day. Returns a numpy array aligned with bars."""
    vwap = np.zeros(len(bars))
    day_cum_pv = 0.0
    day_cum_vol = 0.0
    current_day = None
    for i, bar in enumerate(bars):
        day = bar.timestamp.date()
        if day != current_day:
            current_day = day
            day_cum_pv = 0.0
            day_cum_vol = 0.0
        typical_price = (bar.high + bar.low + bar.close) / 3
        vol = max(bar.volume, 1)  # OANDA's tick volume is never 0 for a real candle, but guard anyway
        day_cum_pv += typical_price * vol
        day_cum_vol += vol
        vwap[i] = day_cum_pv / day_cum_vol
    return vwap


def compute_atr(bars, period=14):
    """Wilder's ATR, aligned with bars (first `period` entries are 0 - not enough history yet)."""
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


def run_backtest(asset: str, bars, account_equity: float = 150000, risk_per_trade: float = 0.01):
    vwap = compute_vwap(bars)
    atr = compute_atr(bars, 14)

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None  # dict: direction, entry_price, stop_loss, take_profit, quantity, entry_time

    for i in range(15, len(bars)):
        bar = bars[i]
        prev_bar = bars[i - 1]
        current_atr = atr[i]
        if current_atr == 0:
            equity_curve.append(equity)
            continue

        # Manage open position first
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

        # Graduated drawdown risk scaling, matching the live agent's logic
        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0
        risk_scale = 0.5 if drawdown > 0.10 else 1.0

        price = bar.close
        prev_price = prev_bar.close
        v = vwap[i]
        prev_v = vwap[i - 1]

        near_vwap = abs(price - v) < current_atr * 0.5
        bullish_reclaim = price > v and near_vwap and prev_price <= prev_v
        bearish_reclaim = price < v and near_vwap and prev_price >= prev_v

        if bullish_reclaim:
            entry_price = price
            stop_loss = v - current_atr * 0.5
            take_profit = entry_price + current_atr * 2.5
            stop_distance = entry_price - stop_loss
            if stop_distance > 0:
                risk_amount = equity * risk_per_trade * risk_scale
                quantity = risk_amount / stop_distance
                position = {"direction": "LONG", "entry_price": entry_price, "stop_loss": stop_loss,
                            "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}
        elif bearish_reclaim:
            entry_price = price
            stop_loss = v + current_atr * 0.5
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

    trades, ending_equity, equity_curve = run_backtest(asset, bars)
    summarize(trades, 150000, ending_equity, equity_curve, bars)
