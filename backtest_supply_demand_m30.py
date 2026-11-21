"""
Proper multi-year validation of supply_demand_signal on M30, following up
on the quick timeframe sweep in backtest_supply_demand_timeframes.py which
only had 1yr/29 trades on M30 - too thin to trust. OANDA practice has 5yr
of M30 XAUUSD history available, so this:

1. Runs the same chronological 70/30 split as backtest_candidates.py, but
   over the full 5yr of XAUUSD M30 history instead of 1yr, for a much
   larger sample.
2. Breaks the 5yr into yearly folds (re-simulated independently, same
   no-lookahead rule as backtest_candidates.py) to check whether any edge
   is consistent across years or concentrated in one regime.
3. Repeats the same 5yr 70/30 check on GBPUSD, EURUSD, USOIL, XAGUSD -
   the same cross-asset generalization check backtest_candidates.py runs
   for every new candidate - to see whether this is a genuine supply/demand
   edge or XAUUSD-specific noise.

WINDOW=250 bars throughout, matching run_loop.py's live fetch_candles(count=250).

Usage:
    OANDA_API_KEY=... OANDA_ACCOUNT_ID=... OANDA_ENV=practice \
        python3 backtest_supply_demand_m30.py
"""

from datetime import datetime, timedelta, timezone

from backtest_candidates import fetch_historical_candles, simulate, summarize, format_summary
from oanda_client import ASSET_TO_OANDA_INSTRUMENT
from trading_agent import AdvancedTradingAgent

GRANULARITY = "M30"
LOOKBACK_DAYS = 365 * 5
WINDOW = 250
METHOD_NAME = "supply_demand_signal"
OTHER_ASSETS = ["GBPUSD", "EURUSD", "USOIL", "XAGUSD"]


def main():
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=LOOKBACK_DAYS)
    agent = AdvancedTradingAgent()

    # --- 1 & 2: XAUUSD, full 5yr + yearly folds ---
    instrument = ASSET_TO_OANDA_INSTRUMENT["XAUUSD"]
    bars = fetch_historical_candles(instrument, GRANULARITY, from_dt, to_dt)
    print(f"=== XAUUSD {GRANULARITY}: {len(bars)} bars, "
          f"{bars[0].timestamp.date()} -> {bars[-1].timestamp.date()} ===\n")

    split_idx = int(len(bars) * 0.7)
    train_bars, test_bars = bars[:split_idx], bars[split_idx:]
    full_s = summarize(simulate(agent, METHOD_NAME, "XAUUSD", bars, window=WINDOW))
    train_s = summarize(simulate(agent, METHOD_NAME, "XAUUSD", train_bars, window=WINDOW))
    test_s = summarize(simulate(agent, METHOD_NAME, "XAUUSD", test_bars, window=WINDOW))
    print("5yr 70/30 split:")
    print("  " + format_summary("Full 5yr  ", full_s))
    print("  " + format_summary("Train 70% ", train_s))
    print("  " + format_summary("Test  30% ", test_s))
    print()

    print("Yearly folds (each re-simulated independently, no lookahead across fold boundaries):")
    start = bars[0].timestamp
    for yr in range(5):
        lo = start + timedelta(days=365 * yr)
        hi = start + timedelta(days=365 * (yr + 1))
        fold_bars = [b for b in bars if lo <= b.timestamp < hi]
        if len(fold_bars) < WINDOW + 10:
            print(f"  Year {yr + 1} ({lo.date()} -> {hi.date()}): not enough bars ({len(fold_bars)}), skipped")
            continue
        fold_s = summarize(simulate(agent, METHOD_NAME, "XAUUSD", fold_bars, window=WINDOW))
        print(f"  Year {yr + 1} ({fold_bars[0].timestamp.date()} -> {fold_bars[-1].timestamp.date()}): "
              + format_summary("", fold_s).lstrip(": "))
    print()

    # --- 3: cross-asset generalization check, same 5yr/70-30 methodology ---
    print("Cross-asset check (5yr, 70/30 split, same method/window):")
    for asset in OTHER_ASSETS:
        inst = ASSET_TO_OANDA_INSTRUMENT[asset]
        abars = fetch_historical_candles(inst, GRANULARITY, from_dt, to_dt)
        if len(abars) < WINDOW + 10:
            print(f"  {asset}: not enough bars ({len(abars)}), skipped")
            continue
        asplit = int(len(abars) * 0.7)
        atrain, atest = abars[:asplit], abars[asplit:]
        afull_s = summarize(simulate(agent, METHOD_NAME, asset, abars, window=WINDOW))
        atrain_s = summarize(simulate(agent, METHOD_NAME, asset, atrain, window=WINDOW))
        atest_s = summarize(simulate(agent, METHOD_NAME, asset, atest, window=WINDOW))
        print(f"  === {asset} ({len(abars)} bars, {abars[0].timestamp.date()} -> {abars[-1].timestamp.date()}) ===")
        print("    " + format_summary("Full ", afull_s))
        print("    " + format_summary("Train", atrain_s))
        print("    " + format_summary("Test ", atest_s))


if __name__ == "__main__":
    main()
