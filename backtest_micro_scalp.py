"""
Standalone backtest of a sub-minute mean-reversion micro-scalp strategy on
XAUUSD, using OANDA's finest candle resolution (S5 = 5-second bars) - the
closest legitimate proxy to "every second" scalping that OANDA's REST API
can actually supply (no raw tick/order-book data is available at any
resolution). Same research-script pattern as the other backtest_*.py
scripts - not part of the live trading loops, run manually:

    python backtest_micro_scalp.py [--days N]

CRITICAL DIFFERENCE from the other backtest_*.py scripts: this one
deducts a realistic round-trip SPREAD_COST from every trade. None of the
M15+ backtests bothered, because spread is a rounding error at that scale;
at 5-second resolution it is often the majority of the price move, so
leaving it out here would make the result meaningless.

Strategy: Micro Mean-Reversion
- Rolling mean and stdev of close price over MEAN_WINDOW S5 bars (100s).
- LONG when price drops below mean - BAND_K*stdev (oversold micro-move),
  betting on reversion back toward the mean.
- SHORT when price rises above mean + BAND_K*stdev, mirrored.
- Target: the rolling mean itself (a full reversion). Stop: entry -/+
  STOP_K*stdev (a further move away invalidates the reversion bet).
- One position at a time, 0.25% risk per trade (matching the live scalper
  agent's default), $150,000 starting equity, drawdown risk-scaling at
  10% same as the other agents.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from oanda_client import fetch_candles_range

MEAN_WINDOW = 20     # 20 S5 bars = 100 seconds
BAND_K = 2.0
STOP_K = 3.5
SPREAD_COST = 0.30   # $ round-trip cost per unit, approximating OANDA's typical XAU_USD spread


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


def run_backtest(bars, account_equity: float = 150000, risk_per_trade: float = 0.0025):
    closes = np.array([b.close for b in bars])

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None

    for i in range(MEAN_WINDOW, len(bars)):
        bar = bars[i]
        window = closes[i - MEAN_WINDOW:i]
        mean = np.mean(window)
        std = np.std(window)

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
                gross_pnl = (exit_price - position["entry_price"]) * position["quantity"] * direction_mult
                spread_cost = SPREAD_COST * position["quantity"]
                pnl = gross_pnl - spread_cost
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

        if std == 0:
            equity_curve.append(equity)
            continue

        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0
        risk_scale = 0.5 if drawdown > 0.10 else 1.0

        price = bar.close
        oversold = price < mean - std * BAND_K
        overbought = price > mean + std * BAND_K

        if oversold:
            entry_price = price
            stop_loss = entry_price - std * STOP_K
            take_profit = mean
            stop_distance = entry_price - stop_loss
            if stop_distance > 0 and take_profit > entry_price:
                risk_amount = equity * risk_per_trade * risk_scale
                quantity = risk_amount / stop_distance
                position = {"direction": "LONG", "entry_price": entry_price, "stop_loss": stop_loss,
                            "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}
        elif overbought:
            entry_price = price
            stop_loss = entry_price + std * STOP_K
            take_profit = mean
            stop_distance = stop_loss - entry_price
            if stop_distance > 0 and take_profit < entry_price:
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
    total_spread_cost = SPREAD_COST * sum(t.quantity for t in trades)

    peak = starting_equity
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)

    total_return = (ending_equity - starting_equity) / starting_equity * 100
    span = bars[-1].timestamp - bars[0].timestamp

    print(f"Period: {bars[0].timestamp} -> {bars[-1].timestamp} ({span}, {len(bars)} S5 bars)")
    print(f"Total trades: {len(trades)}  ({len(trades) / max(span.total_seconds() / 3600, 1):.1f} trades/hour of data)")
    print(f"Win rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Avg win: ${avg_win:,.2f}  |  Avg loss: ${avg_loss:,.2f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Total spread cost paid: ${total_spread_cost:,.2f}")
    print(f"Starting equity: ${starting_equity:,.2f}")
    print(f"Ending equity:   ${ending_equity:,.2f}")
    print(f"Total return: {total_return:+.2f}%")
    print(f"Max drawdown: {max_dd * 100:.2f}%")


if __name__ == "__main__":
    days = 14
    args = sys.argv[1:]
    if "--days" in args:
        days = int(args[args.index("--days") + 1])

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print(f"Fetching {days}d of S5 XAUUSD candles from OANDA ({start.date()} -> {end.date()})...")
    bars = fetch_candles_range("XAUUSD", "S5", start, end)
    print(f"Fetched {len(bars)} bars.\n")

    trades, ending_equity, equity_curve = run_backtest(bars)
    summarize(trades, 150000, ending_equity, equity_curve, bars)
