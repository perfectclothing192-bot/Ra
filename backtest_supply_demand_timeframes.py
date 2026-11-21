"""
Timeframe sweep for the new supply_demand_signal candidate (see its
docstring in trading_agent.py) - same train/test methodology as
backtest_candidates.py (chronological 70/30 split, R-multiples), just
looped across granularities on one instrument to answer "which timeframe
works best" directly, instead of the single hardcoded M15 that
backtest_candidates.py always uses.

WINDOW stays 250 bars (not scaled by granularity) on purpose: that's
exactly what run_loop.py's live fetch_candles(count=250) would hand the
signal at any granularity, so this matches what actually deploying the
strategy at a different granularity would look like.

Usage:
    OANDA_API_KEY=... OANDA_ACCOUNT_ID=... OANDA_ENV=practice \
        python3 backtest_supply_demand_timeframes.py [ASSET] [--days N]
"""

import sys
from datetime import datetime, timedelta, timezone

from backtest_candidates import fetch_historical_candles, simulate, summarize, format_summary
from oanda_client import ASSET_TO_OANDA_INSTRUMENT
from trading_agent import AdvancedTradingAgent

GRANULARITIES = ["M15", "M30", "H1", "H4"]
WINDOW = 250
METHOD_NAME = "supply_demand_signal"


def main():
    asset = "XAUUSD"
    days = 365
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        asset = args[0]
        args = args[1:]
    if "--days" in args:
        days = int(args[args.index("--days") + 1])

    instrument = ASSET_TO_OANDA_INSTRUMENT[asset]
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days)
    agent = AdvancedTradingAgent()

    print(f"supply_demand_signal on {asset} ({instrument}), {days}d, "
          f"{', '.join(GRANULARITIES)}\n")

    for granularity in GRANULARITIES:
        bars = fetch_historical_candles(instrument, granularity, from_dt, to_dt)
        if len(bars) < WINDOW + 10:
            print(f"=== {granularity} ===\n  Not enough bars ({len(bars)}) for a {WINDOW}-bar window, skipping.\n")
            continue

        split_idx = int(len(bars) * 0.7)
        train_bars = bars[:split_idx]
        test_bars = bars[split_idx:]

        full_trades = simulate(agent, METHOD_NAME, asset, bars, window=WINDOW)
        train_trades = simulate(agent, METHOD_NAME, asset, train_bars, window=WINDOW)
        test_trades = simulate(agent, METHOD_NAME, asset, test_bars, window=WINDOW)

        full_s = summarize(full_trades)
        train_s = summarize(train_trades)
        test_s = summarize(test_trades)

        print(f"=== {granularity} ({len(bars)} bars, {bars[0].timestamp.date()} -> {bars[-1].timestamp.date()}) ===")
        print("  " + format_summary("Full year ", full_s))
        print("  " + format_summary("Train 70% ", train_s))
        print("  " + format_summary("Test  30% ", test_s))
        print()


if __name__ == "__main__":
    main()
