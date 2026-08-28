"""
Standalone backtest of an SMA 9/20 crossover strategy against 1 year of real
OANDA historical M15 candles. Same research-script pattern as
backtest_vwap.py - not part of the live trading loops, run manually:

    python backtest_sma_cross.py [ASSET] [--days N]

Strategy: SMA 9/20 Crossover (classic fast/slow moving-average cross)
- LONG on a fresh bullish cross: SMA9 was <= SMA20 on the prior bar, now > it.
- SHORT on a fresh bearish cross: SMA9 was >= SMA20 on the prior bar, now < it.
- Stop loss: entry -/+ 1.5x ATR. Take profit: entry +/- 3.0x ATR (2:1 R:R).
- One position at a time, 1% risk per trade (halved past 10% drawdown, same
  as the live agents), $150,000 starting equity.

No lookahead: SMA/ATR/signals at bar i only use bars[0:i+1].
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from oanda_client import fetch_candles_range


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


def compute_sma(closes, period):
    sma = np.zeros(len(closes))
    cumsum = np.cumsum(closes)
    for i in range(len(closes)):
        if i + 1 < period:
            continue
        if i + 1 == period:
            sma[i] = cumsum[i] / period
        else:
            sma[i] = sma[i - 1] + (closes[i] - closes[i - period]) / period
    return sma


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


def run_backtest(bars, account_equity: float = 150000, risk_per_trade: float = 0.01,
                  fast: int = 9, slow: int = 20):
    closes = np.array([b.close for b in bars])
    sma_fast = compute_sma(closes, fast)
    sma_slow = compute_sma(closes, slow)
    atr = compute_atr(bars, 14)

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None

    warmup = max(slow, 14) + 1
    for i in range(warmup, len(bars)):
        bar = bars[i]
        current_atr = atr[i]
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

        bullish_cross = sma_fast[i - 1] <= sma_slow[i - 1] and sma_fast[i] > sma_slow[i]
        bearish_cross = sma_fast[i - 1] >= sma_slow[i - 1] and sma_fast[i] < sma_slow[i]

        if bullish_cross:
            entry_price = bar.close
            stop_loss = entry_price - current_atr * 1.5
            take_profit = entry_price + current_atr * 3.0
            stop_distance = entry_price - stop_loss
            risk_amount = equity * risk_per_trade * risk_scale
            quantity = risk_amount / stop_distance
            position = {"direction": "LONG", "entry_price": entry_price, "stop_loss": stop_loss,
                        "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}
        elif bearish_cross:
            entry_price = bar.close
            stop_loss = entry_price + current_atr * 1.5
            take_profit = entry_price - current_atr * 3.0
            stop_distance = stop_loss - entry_price
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
