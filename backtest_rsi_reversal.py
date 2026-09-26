"""
Standalone backtest of an RSI overbought/oversold trend-reversal strategy on
XAUUSD M1 candles. Same research-script pattern as the other backtest_*.py
scripts - not part of the live trading loops, run manually:

    python backtest_rsi_reversal.py [--days N] [--granularity M1]

Strategy: RSI Turn Reversal
- RSI(RSI_PERIOD) on M1 closes.
- LONG when RSI crosses back above OVERSOLD (was <= 30, now > 30) - price
  got oversold and is turning back up.
- SHORT when RSI crosses back below OVERBOUGHT (was >= 70, now < 70) -
  mirrored, price got overbought and is turning back down.
- Stop: entry -/+ ATR_STOP_MULT * ATR(14). Target: entry +/- ATR_TP_MULT *
  ATR(14). One position at a time, 0.25% risk per trade (matching the live
  scalper agent's default), $150,000 starting equity, drawdown risk-scaling
  at 10% same as the other agents.
- Deducts a realistic round-trip SPREAD_COST per trade, same as
  backtest_micro_scalp.py - at M1 resolution spread is a meaningful share
  of a typical move, so it isn't safe to ignore the way the M15+ backtests do.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from oanda_client import fetch_candles_range

RSI_PERIOD = 14
OVERSOLD = 30.0
OVERBOUGHT = 70.0
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
ATR_TP_MULT = 2.5
SPREAD_COST = 0.30  # $ round-trip cost per unit, approximating OANDA's typical XAU_USD spread


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


def compute_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
    """Wilder's RSI. rsi[i] only ever depends on closes[0..i] - safe to
    precompute over the whole series without introducing lookahead."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi[period] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rsi[i] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

    return rsi


def compute_atr(bars, period: int = ATR_PERIOD) -> np.ndarray:
    """Wilder's ATR, index-aligned with `bars` (atr[i] undefined/0 for i < period)."""
    n = len(bars)
    tr = np.zeros(n)
    for i in range(1, n):
        h_l = bars[i].high - bars[i].low
        h_pc = abs(bars[i].high - bars[i - 1].close)
        l_pc = abs(bars[i].low - bars[i - 1].close)
        tr[i] = max(h_l, h_pc, l_pc)

    atr = np.zeros(n)
    if n <= period:
        return atr
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def run_backtest(bars, account_equity: float = 150000, risk_per_trade: float = 0.0025):
    closes = np.array([b.close for b in bars])
    rsi = compute_rsi(closes)
    atr = compute_atr(bars)

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None

    start_i = max(RSI_PERIOD, ATR_PERIOD) + 1

    for i in range(start_i, len(bars)):
        bar = bars[i]

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

        bar_atr = atr[i]
        if bar_atr <= 0:
            equity_curve.append(equity)
            continue

        prev_rsi, cur_rsi = rsi[i - 1], rsi[i]
        turned_up_from_oversold = prev_rsi <= OVERSOLD and cur_rsi > OVERSOLD
        turned_down_from_overbought = prev_rsi >= OVERBOUGHT and cur_rsi < OVERBOUGHT

        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0
        risk_scale = 0.5 if drawdown > 0.10 else 1.0

        price = bar.close
        if turned_up_from_oversold:
            entry_price = price
            stop_loss = entry_price - bar_atr * ATR_STOP_MULT
            take_profit = entry_price + bar_atr * ATR_TP_MULT
            stop_distance = entry_price - stop_loss
            if stop_distance > 0:
                risk_amount = equity * risk_per_trade * risk_scale
                quantity = risk_amount / stop_distance
                position = {"direction": "LONG", "entry_price": entry_price, "stop_loss": stop_loss,
                            "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}
        elif turned_down_from_overbought:
            entry_price = price
            stop_loss = entry_price + bar_atr * ATR_STOP_MULT
            take_profit = entry_price - bar_atr * ATR_TP_MULT
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
    total_spread_cost = SPREAD_COST * sum(t.quantity for t in trades)

    peak = starting_equity
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)

    total_return = (ending_equity - starting_equity) / starting_equity * 100
    span = bars[-1].timestamp - bars[0].timestamp

    print(f"Period: {bars[0].timestamp} -> {bars[-1].timestamp} ({span}, {len(bars)} bars)")
    print(f"Total trades: {len(trades)}  ({len(trades) / max(span.total_seconds() / 86400, 1):.1f} trades/day of data)")
    print(f"Win rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Avg win: ${avg_win:,.2f}  |  Avg loss: ${avg_loss:,.2f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Total spread cost paid: ${total_spread_cost:,.2f}")
    print(f"Starting equity: ${starting_equity:,.2f}")
    print(f"Ending equity:   ${ending_equity:,.2f}")
    print(f"Total return: {total_return:+.2f}%")
    print(f"Max drawdown: {max_dd * 100:.2f}%")


if __name__ == "__main__":
    days = 365
    granularity = "M1"
    args = sys.argv[1:]
    if "--days" in args:
        days = int(args[args.index("--days") + 1])
    if "--granularity" in args:
        granularity = args[args.index("--granularity") + 1]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print(f"Fetching {days}d of {granularity} XAUUSD candles from OANDA ({start.date()} -> {end.date()})...")
    bars = fetch_candles_range("XAUUSD", granularity, start, end)
    print(f"Fetched {len(bars)} bars.\n")

    trades, ending_equity, equity_curve = run_backtest(bars)
    summarize(trades, 150000, ending_equity, equity_curve, bars)
