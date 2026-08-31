"""
Backtest of the ACTUAL live sma_cluster_signal strategy (trading_agent.py)
against 1 year of real OANDA historical M15 USOIL candles - a faithful
replication of the exact logic already running in the swing agent, not a
new strategy idea. Used as a "control" comparison since the live agent
hasn't had a single USOIL trade trigger yet (it's a rare, long-only
consolidation-breakout setup by design).

    python backtest_sma_cluster_live.py [--days N]

No lookahead: SMA/ATR/volume-avg at bar i only use bars[0:i+1] (mirrors
trading_agent.py's sma_cluster_signal, which always looks at the trailing
window ending at the current bar).
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


def run_backtest(bars, account_equity: float = 150000, risk_per_trade: float = 0.01):
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    sma_100 = compute_sma(closes, 100)
    sma_200 = compute_sma(closes, 200)
    atr = compute_atr(bars, 14)

    equity = account_equity
    peak_equity = account_equity
    trades = []
    equity_curve = [equity]
    position = None

    for i in range(200, len(bars)):
        bar = bars[i]
        current_atr = atr[i]

        if position:
            hit_sl = bar.low <= position["stop_loss"]
            hit_tp = bar.high >= position["take_profit"]
            if hit_sl or hit_tp:
                exit_price = position["stop_loss"] if hit_sl else position["take_profit"]
                pnl = (exit_price - position["entry_price"]) * position["quantity"]
                trades.append(BacktestTrade(
                    direction="LONG", entry_time=position["entry_time"],
                    entry_price=position["entry_price"], exit_time=bar.timestamp,
                    exit_price=exit_price, quantity=position["quantity"], pnl=pnl,
                    reason="TP" if hit_tp else "SL",
                ))
                equity += pnl
                peak_equity = max(peak_equity, equity)
                position = None
            equity_curve.append(equity)
            continue

        if current_atr == 0 or i < 20:
            equity_curve.append(equity)
            continue

        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0
        risk_scale = 0.5 if drawdown > 0.10 else 1.0

        cluster_low = min(sma_100[i], sma_200[i])
        cluster_high = max(sma_100[i], sma_200[i])
        cluster_width = cluster_high - cluster_low
        in_cluster = cluster_width < current_atr * 0.3

        avg_volume = np.mean(volumes[i - 20:i])
        volume_surge = volumes[i] > avg_volume * 1.5
        price_above_cluster = bar.close > cluster_high

        if in_cluster and price_above_cluster and volume_surge:
            entry_price = bar.close
            stop_loss = cluster_low - current_atr * 0.5
            take_profit = entry_price + current_atr * 2.0
            stop_distance = entry_price - stop_loss
            if stop_distance > 0:
                risk_amount = equity * risk_per_trade * risk_scale
                quantity = risk_amount / stop_distance
                position = {"entry_price": entry_price, "stop_loss": stop_loss,
                            "take_profit": take_profit, "quantity": quantity, "entry_time": bar.timestamp}

        equity_curve.append(equity)

    return trades, equity, equity_curve


def summarize(trades, starting_equity, ending_equity, equity_curve, bars):
    if not trades:
        print("No trades were generated - this matches the live agent's own experience so far.")
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
    days = 365
    args = sys.argv[1:]
    if "--days" in args:
        days = int(args[args.index("--days") + 1])

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print(f"Fetching {days}d of M15 USOIL candles from OANDA ({start.date()} -> {end.date()})...")
    bars = fetch_candles_range("USOIL", "M15", start, end)
    print(f"Fetched {len(bars)} bars.\n")

    trades, ending_equity, equity_curve = run_backtest(bars)
    summarize(trades, 150000, ending_equity, equity_curve, bars)
