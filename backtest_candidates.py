"""
Backtest harness for the two "data points" quant-style candidate strategies
already implemented in trading_agent.py but never wired into run_loop.py's
STRATEGIES dict: rsi2_pullback_signal and donchian_breakout_signal (both
labeled "Candidate replacement for fx_range_reversion on GBPUSD/EURUSD" in
their docstrings - both are generic (bars, asset) methods, so also tried
here on the three commodity assets already live under other strategies:
XAUUSD (gold), USOIL, XAGUSD (silver)).

Every strategy actually live in run_loop.py has a documented 1-year M15
backtest, checked out-of-sample on a chronological 70/30 train/test split,
before being wired in - see the comments in run_loop.py's STRATEGIES dict.
This script applies the same methodology to the two unvalidated candidates
so they can be judged by the same bar before ever being considered for
live trading.

Usage:
    OANDA_API_KEY=... OANDA_ACCOUNT_ID=... OANDA_ENV=practice \
        python3 backtest_candidates.py
"""

import os
from datetime import datetime, timedelta, timezone

from oanda_client import ASSET_TO_OANDA_INSTRUMENT, BASE_URL, _headers, _get_with_retry
from trading_agent import AdvancedTradingAgent, PriceBar

GRANULARITY = "M15"
LOOKBACK_DAYS = 365
WINDOW = 250  # bars fed to each signal call, matching run_loop.py's fetch_candles(count=250)

# (backtest asset label, strategy function name, kwargs)
CANDIDATES = [
    ("GBPUSD", "rsi2_pullback_signal", {}),
    ("EURUSD", "rsi2_pullback_signal", {}),
    ("GBPUSD", "donchian_breakout_signal", {}),
    ("EURUSD", "donchian_breakout_signal", {}),
    ("XAUUSD", "rsi2_pullback_signal", {}),
    ("USOIL", "rsi2_pullback_signal", {}),
    ("XAGUSD", "rsi2_pullback_signal", {}),
    ("XAUUSD", "donchian_breakout_signal", {}),
    ("USOIL", "donchian_breakout_signal", {}),
    ("XAGUSD", "donchian_breakout_signal", {}),
    # mark_douglas_signal - mechanical EMA trend-continuation (trend +
    # slope filter, shallow pullback, fixed 2R target) - never backtested
    # anywhere before this, unlike every other strategy in this file.
    ("GBPUSD", "mark_douglas_signal", {}),
    ("EURUSD", "mark_douglas_signal", {}),
    ("XAUUSD", "mark_douglas_signal", {}),
    ("USOIL", "mark_douglas_signal", {}),
    ("XAGUSD", "mark_douglas_signal", {}),
]


def fetch_historical_candles(instrument: str, granularity: str, from_dt: datetime, to_dt: datetime):
    """Paginate OANDA's candles endpoint (max 5000/request) forward from
    from_dt to to_dt using from+count, advancing the cursor past the last
    candle returned each round."""
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
            break  # guard against an unadvancing cursor
        cursor = last_ts + timedelta(seconds=1)

        if len(candles) < 5000:
            break

    return bars


def simulate(agent, method_name, asset: str, bars, window: int = WINDOW, **kwargs):
    """
    Walk bars one at a time, feeding a trailing window of `window` bars into
    the named signal method exactly as run_loop.py's live poll loop does
    each cycle (fetch_candles(count=250)). One position open at a time;
    once opened at the signal bar's close, a position is held until a later
    bar's high/low touches its stop-loss or take-profit - no time-based or
    re-signal exit, matching update_positions()'s live behavior. A bar is
    never used to both close the previous trade and open a new one, to
    avoid any lookahead ambiguity.
    """
    method = getattr(agent, method_name)
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
            # Stop checked first when a single bar's range spans both -
            # the conservative assumption (avoids overstating results).
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
            signal = method(window_bars, asset, **kwargs)
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

    assets = sorted({asset for asset, _, _ in CANDIDATES})
    print(f"Fetching {LOOKBACK_DAYS}d of {GRANULARITY} candles for {', '.join(assets)}...")
    candle_cache = {}
    for asset in assets:
        instrument = ASSET_TO_OANDA_INSTRUMENT[asset]
        bars = fetch_historical_candles(instrument, GRANULARITY, from_dt, to_dt)
        candle_cache[asset] = bars
        print(f"  {asset} ({instrument}): {len(bars)} bars, {bars[0].timestamp.date()} -> {bars[-1].timestamp.date()}")

    print()
    for asset, method_name, kwargs in CANDIDATES:
        bars = candle_cache[asset]
        split_idx = int(len(bars) * 0.7)

        full_trades = simulate(agent, method_name, asset, bars, **kwargs)

        # Chronological 70/30 split: re-simulate each half independently
        # (not just slicing full_trades) so each half only ever sees bars
        # within its own window, matching how this repo's other strategies
        # were validated.
        train_bars = bars[:split_idx]
        test_bars = bars[split_idx:]
        train_trades = simulate(agent, method_name, asset, train_bars, **kwargs)
        test_trades = simulate(agent, method_name, asset, test_bars, **kwargs)

        full_s = summarize(full_trades)
        train_s = summarize(train_trades)
        test_s = summarize(test_trades)

        print(f"=== {method_name} on {asset} ===")
        print("  " + format_summary("Full year ", full_s))
        print("  " + format_summary("Train 70% ", train_s))
        print("  " + format_summary("Test  30% ", test_s))
        print()


if __name__ == "__main__":
    main()
