"""
Backtest harness for fib_retracement_signal, the Fibonacci-retracement
continuation strategy already live on the "*_FIB" synthetic slots
(XAUUSD_FIB, USOIL_FIB, GBPUSD_FIB, EURUSD_FIB, BTCUSD_FIB) in
run_loop.py's STRATEGIES dict. Same 1yr M15 chronological 70/30
train/test methodology as backtest_candidates.py, re-run here as a
standalone, requested re-check of this specific strategy.

Usage:
    OANDA_API_KEY=... OANDA_ACCOUNT_ID=... OANDA_ENV=practice \
        python3 backtest_fib_retracement.py
"""

from datetime import datetime, timedelta, timezone

from oanda_client import ASSET_TO_OANDA_INSTRUMENT, BASE_URL, _headers, _get_with_retry
from trading_agent import AdvancedTradingAgent, PriceBar

GRANULARITY = "M15"
LOOKBACK_DAYS = 365
WINDOW = 250  # matches run_loop.py's fetch_candles(count=250)

# (backtest asset label, strategy_label) - mirrors the live "*_FIB" wrappers
CANDIDATES = [
    ("XAUUSD", "fib_retracement_xauusd"),
    ("USOIL", "fib_retracement_usoil"),
    ("GBPUSD", "fib_retracement_gbpusd"),
    ("EURUSD", "fib_retracement_eurusd"),
    ("BTCUSD", "fib_retracement_btcusd"),
]


def fetch_historical_candles(instrument: str, granularity: str, from_dt: datetime, to_dt: datetime):
    bars = []
    cursor = from_dt
    while cursor < to_dt:
        url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": 5000,
            "price": "M",
        }
        response = _get_with_retry(url, params=params, headers=_headers())
        candles = response.json().get("candles", [])
        if not candles:
            break

        new_bars = []
        for candle in candles:
            if not candle.get("complete"):
                continue
            ts = datetime.fromisoformat(candle["time"].replace("Z", "+00:00"))
            if ts > to_dt:
                continue
            mid = candle["mid"]
            new_bars.append(PriceBar(
                timestamp=ts,
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=int(candle.get("volume", 0)),
            ))

        if not new_bars:
            break
        bars.extend(new_bars)

        last_ts = candles[-1].get("time")
        last_ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        if last_ts <= cursor:
            break
        cursor = last_ts + timedelta(seconds=1)

        if len(candles) < 5000:
            break

    return bars


def simulate(agent, asset: str, strategy_label: str, bars, window: int = WINDOW):
    """Same walk-forward, one-position-at-a-time simulation as
    backtest_candidates.py's simulate(), calling fib_retracement_signal
    directly with this asset's own strategy_label (so the Signal's
    .strategy field matches what the live wrapper would produce)."""
    trades = []
    position = None

    for i in range(window, len(bars)):
        bar = bars[i]
        had_position = position is not None

        if position is not None:
            hit_sl = (
                (position["direction"] == "LONG" and bar.low <= position["stop_loss"]) or
                (position["direction"] == "SHORT" and bar.high >= position["stop_loss"])
            )
            hit_tp = (
                (position["direction"] == "LONG" and bar.high >= position["take_profit"]) or
                (position["direction"] == "SHORT" and bar.low <= position["take_profit"])
            )
            if hit_sl:
                trades.append({"entry_time": position["entry_time"], "exit_time": bar.timestamp, "r": -1.0})
                position = None
            elif hit_tp:
                risk = abs(position["entry_price"] - position["stop_loss"])
                reward = abs(position["take_profit"] - position["entry_price"])
                r = (reward / risk) if risk else 0.0
                trades.append({"entry_time": position["entry_time"], "exit_time": bar.timestamp, "r": r})
                position = None

        if not had_position and position is None:
            window_bars = bars[i - window + 1: i + 1]
            signal = agent.fib_retracement_signal(window_bars, asset=asset, strategy_label=strategy_label)
            if signal.direction in ("BUY", "SELL"):
                position = {
                    "direction": "LONG" if signal.direction == "BUY" else "SHORT",
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "entry_time": bar.timestamp,
                }

    return trades


def summarize(trades):
    if not trades:
        return {"n": 0, "total_r": 0.0, "win_rate": 0.0, "profit_factor": None}
    total_r = sum(t["r"] for t in trades)
    wins = [t["r"] for t in trades if t["r"] > 0]
    losses = [t["r"] for t in trades if t["r"] <= 0]
    win_rate = len(wins) / len(trades) * 100
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None
    return {"n": len(trades), "total_r": total_r, "win_rate": win_rate, "profit_factor": profit_factor}


def format_summary(label, s):
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "n/a"
    return f"{label}: {s['n']} trades, {s['total_r']:+.1f}R, {s['win_rate']:.1f}% WR, PF {pf}"


def main():
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=LOOKBACK_DAYS)

    agent = AdvancedTradingAgent()

    assets = sorted({asset for asset, _ in CANDIDATES})
    print(f"Fetching {LOOKBACK_DAYS}d of {GRANULARITY} candles for {', '.join(assets)}...")
    candle_cache = {}
    for asset in assets:
        instrument = ASSET_TO_OANDA_INSTRUMENT[asset]
        bars = fetch_historical_candles(instrument, GRANULARITY, from_dt, to_dt)
        candle_cache[asset] = bars
        print(f"  {asset} ({instrument}): {len(bars)} bars, {bars[0].timestamp.date()} -> {bars[-1].timestamp.date()}")

    print()
    for asset, strategy_label in CANDIDATES:
        bars = candle_cache[asset]
        split_idx = int(len(bars) * 0.7)

        full_trades = simulate(agent, asset, strategy_label, bars)

        train_bars = bars[:split_idx]
        test_bars = bars[split_idx:]
        train_trades = simulate(agent, asset, strategy_label, train_bars)
        test_trades = simulate(agent, asset, strategy_label, test_bars)

        full_s = summarize(full_trades)
        train_s = summarize(train_trades)
        test_s = summarize(test_trades)

        print(f"=== {strategy_label} on {asset} ===")
        print("  " + format_summary("Full year ", full_s))
        print("  " + format_summary("Train 70% ", train_s))
        print("  " + format_summary("Test  30% ", test_s))
        print()


if __name__ == "__main__":
    main()
