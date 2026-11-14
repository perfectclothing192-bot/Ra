"""
Order-flow (volume-delta) backtest for XAUUSD on OANDA historical candles.

OANDA's REST API only exposes OHLCV candles (tick-count volume), not real
bid/ask depth or trade-by-trade order flow. This script builds the closest
usable proxy from that data:

  buy_volume  = volume * (close - low) / (high - low)
  sell_volume = volume * (high - close) / (high - low)
  delta       = buy_volume - sell_volume
  CVD         = cumulative sum of delta (cumulative volume delta)

Strategy: CVD / price divergence.
  - Price falling over the lookback window while CVD (smoothed) is rising
    -> accumulation despite falling price -> BUY.
  - Price rising over the lookback window while CVD (smoothed) is falling
    -> distribution despite rising price -> SELL.
  A minimum-magnitude filter on both legs keeps out noise.

This is explicitly a proxy, not real order flow. Treat results accordingly.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

OANDA_ENV = os.environ.get("OANDA_ENV", "practice")
BASE_URL = "https://api-fxpractice.oanda.com" if OANDA_ENV == "practice" else "https://api-fxtrade.oanda.com"
INSTRUMENT = "XAU_USD"


def _headers() -> dict:
    api_key = os.environ.get("OANDA_API_KEY")
    if not api_key:
        raise RuntimeError("OANDA_API_KEY not configured")
    return {"Authorization": f"Bearer {api_key}"}


def fetch_historical_candles(granularity: str, lookback_days: int) -> pd.DataFrame:
    """Paginate backward from now to collect `lookback_days` of candles."""
    url = f"{BASE_URL}/v3/instruments/{INSTRUMENT}/candles"
    end_cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    all_rows = []
    to_time = datetime.now(timezone.utc)
    seen_times = set()

    while True:
        params = {
            "granularity": granularity,
            "count": 5000,
            "price": "M",
            "to": to_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        }
        resp = None
        for attempt in range(5):
            resp = requests.get(url, params=params, headers=_headers(), timeout=30)
            if resp.status_code == 200:
                break
            time.sleep(5)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        if not candles:
            break

        batch_rows = []
        for c in candles:
            if not c.get("complete"):
                continue
            t = c["time"]
            if t in seen_times:
                continue
            seen_times.add(t)
            mid = c["mid"]
            batch_rows.append({
                "time": datetime.fromisoformat(t.replace("Z", "+00:00")),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })

        if not batch_rows:
            break

        all_rows.extend(batch_rows)
        earliest = min(r["time"] for r in batch_rows)
        if earliest.tzinfo is None:
            earliest = earliest.replace(tzinfo=timezone.utc)

        if earliest <= end_cutoff or earliest >= to_time:
            break

        to_time = earliest

    df = pd.DataFrame(all_rows).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[df["time"] >= pd.Timestamp(end_cutoff)]
    return df.reset_index(drop=True)


def compute_order_flow_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    buy_frac = ((df["close"] - df["low"]) / rng).fillna(0.5).clip(0, 1)
    df["buy_volume"] = df["volume"] * buy_frac
    df["sell_volume"] = df["volume"] * (1 - buy_frac)
    df["delta"] = df["buy_volume"] - df["sell_volume"]
    df["cvd"] = df["delta"].cumsum()
    return df


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def generate_signals(df: pd.DataFrame, lookback: int, cvd_ema_span: int,
                      min_price_atr_mult: float, min_cvd_z: float) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = atr(df, 14)
    df["cvd_ema"] = df["cvd"].ewm(span=cvd_ema_span, adjust=False).mean()

    price_change = df["close"] - df["close"].shift(lookback)
    cvd_change = df["cvd_ema"] - df["cvd_ema"].shift(lookback)
    cvd_change_std = cvd_change.rolling(100, min_periods=20).std()
    cvd_z = cvd_change / cvd_change_std.replace(0, np.nan)

    price_threshold = df["atr"] * min_price_atr_mult

    bullish_div = (price_change < -price_threshold) & (cvd_z > min_cvd_z)
    bearish_div = (price_change > price_threshold) & (cvd_z < -min_cvd_z)

    df["signal"] = "HOLD"
    df.loc[bullish_div, "signal"] = "BUY"
    df.loc[bearish_div, "signal"] = "SELL"
    df["price_change"] = price_change
    df["cvd_z"] = cvd_z
    return df


@dataclass
class SimTrade:
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl_price: float
    pnl_r: float
    exit_reason: str


def run_backtest(df: pd.DataFrame, sl_atr_mult: float, tp_atr_mult: float,
                  starting_equity: float, risk_per_trade: float) -> dict:
    trades: List[SimTrade] = []
    position: Optional[dict] = None
    equity = starting_equity
    equity_curve = []

    rows = df.to_dict("records")

    for i in range(1, len(rows)):
        row = rows[i]
        prev = rows[i - 1]

        if position is not None:
            direction = position["direction"]
            sl = position["stop_loss"]
            tp = position["take_profit"]
            exit_price = None
            exit_reason = None

            if direction == "LONG":
                if row["low"] <= sl:
                    exit_price, exit_reason = sl, "stop_loss"
                elif row["high"] >= tp:
                    exit_price, exit_reason = tp, "take_profit"
            else:
                if row["high"] >= sl:
                    exit_price, exit_reason = sl, "stop_loss"
                elif row["low"] <= tp:
                    exit_price, exit_reason = tp, "take_profit"

            if exit_price is not None:
                entry_price = position["entry_price"]
                if direction == "LONG":
                    pnl_price = exit_price - entry_price
                else:
                    pnl_price = entry_price - exit_price
                risk_price = abs(entry_price - sl)
                pnl_r = pnl_price / risk_price if risk_price else 0.0
                pnl_dollars = pnl_r * (starting_equity * risk_per_trade)
                equity += pnl_dollars

                trades.append(SimTrade(
                    direction=direction,
                    entry_time=position["entry_time"],
                    entry_price=entry_price,
                    exit_time=row["time"],
                    exit_price=exit_price,
                    stop_loss=sl,
                    take_profit=tp,
                    pnl_price=pnl_price,
                    pnl_r=pnl_r,
                    exit_reason=exit_reason,
                ))
                position = None

        if position is None and prev["signal"] in ("BUY", "SELL"):
            entry_price = row["open"]
            cur_atr = prev["atr"]
            if pd.isna(cur_atr) or cur_atr <= 0:
                continue
            if prev["signal"] == "BUY":
                sl = entry_price - sl_atr_mult * cur_atr
                tp = entry_price + tp_atr_mult * cur_atr
                direction = "LONG"
            else:
                sl = entry_price + sl_atr_mult * cur_atr
                tp = entry_price - tp_atr_mult * cur_atr
                direction = "SHORT"

            position = {
                "direction": direction,
                "entry_time": row["time"],
                "entry_price": entry_price,
                "stop_loss": sl,
                "take_profit": tp,
            }

        equity_curve.append({"time": row["time"], "equity": equity})

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_equity": equity,
    }


def summarize(trades: List[SimTrade], starting_equity: float, final_equity: float,
              equity_curve: List[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"total_trades": 0, "message": "No trades generated with current thresholds."}

    wins = [t for t in trades if t.pnl_r > 0]
    losses = [t for t in trades if t.pnl_r <= 0]
    gross_win_r = sum(t.pnl_r for t in wins)
    gross_loss_r = abs(sum(t.pnl_r for t in losses))
    profit_factor = (gross_win_r / gross_loss_r) if gross_loss_r > 0 else float("inf")

    equities = [starting_equity] + [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak else 0.0
        max_dd = max(max_dd, dd)

    return {
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100 * len(wins) / n, 2),
        "avg_r": round(sum(t.pnl_r for t in trades) / n, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf (no losses)",
        "total_return_pct": round(100 * (final_equity - starting_equity) / starting_equity, 2),
        "starting_equity": starting_equity,
        "final_equity": round(final_equity, 2),
        "max_drawdown_pct": round(100 * max_dd, 2),
        "stop_loss_exits": sum(1 for t in trades if t.exit_reason == "stop_loss"),
        "take_profit_exits": sum(1 for t in trades if t.exit_reason == "take_profit"),
    }


def main():
    parser = argparse.ArgumentParser(description="Order-flow proxy backtest for XAUUSD on OANDA data")
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--divergence-lookback", type=int, default=20)
    parser.add_argument("--cvd-ema-span", type=int, default=20)
    parser.add_argument("--min-price-atr-mult", type=float, default=1.0)
    parser.add_argument("--min-cvd-z", type=float, default=1.0)
    parser.add_argument("--sl-atr-mult", type=float, default=1.5)
    parser.add_argument("--tp-atr-mult", type=float, default=2.25)
    parser.add_argument("--starting-equity", type=float, default=150000)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--out", default=None, help="Optional path to dump full trade log as JSON")
    args = parser.parse_args()

    print(f"Fetching {args.lookback_days}d of {args.granularity} XAU_USD candles from OANDA...")
    df = fetch_historical_candles(args.granularity, args.lookback_days)
    print(f"Fetched {len(df)} candles: {df['time'].min()} to {df['time'].max()}")

    df = compute_order_flow_proxy(df)
    df = generate_signals(df, args.divergence_lookback, args.cvd_ema_span,
                           args.min_price_atr_mult, args.min_cvd_z)

    signal_counts = df["signal"].value_counts().to_dict()
    print(f"Signal counts: {signal_counts}")

    result = run_backtest(df, args.sl_atr_mult, args.tp_atr_mult,
                           args.starting_equity, args.risk_per_trade)
    summary = summarize(result["trades"], args.starting_equity, result["final_equity"], result["equity_curve"])

    print(json.dumps(summary, indent=2, default=str))

    if args.out:
        trade_log = [
            {
                "direction": t.direction,
                "entry_time": t.entry_time.isoformat(),
                "entry_price": t.entry_price,
                "exit_time": t.exit_time.isoformat(),
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "pnl_price": t.pnl_price,
                "pnl_r": t.pnl_r,
                "exit_reason": t.exit_reason,
            }
            for t in result["trades"]
        ]
        with open(args.out, "w") as f:
            json.dump({
                "params": vars(args),
                "summary": summary,
                "trades": trade_log,
                "equity_curve": [{"time": e["time"].isoformat(), "equity": e["equity"]} for e in result["equity_curve"]],
            }, f, indent=2, default=str)
        print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
