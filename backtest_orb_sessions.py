"""
Backtest: does anchoring XAUUSD's Opening Range Breakout to a real
trading session (London or New York open) beat the fixed-UTC-midnight
range orb_15m_signal actually trades live?

Compares, over 2 years of M15 bars (vs. the 1-year window every other
strategy in this repo was validated on):
    orb_15m_signal           - the deployed baseline (UTC-midnight range)
    london_breakout_signal   - already in trading_agent.py, previously
                                backtested only on 1yr, no edge found
    ny_session_breakout_signal - new, added 2026-10-22 for this comparison

Reuses backtest_candidates.py's exact methodology (same simulate() bar
walk, same chronological 70/30 train/test split, same R-multiple
accounting) so results are directly comparable to every other strategy's
documented backtest in this repo.

Usage:
    OANDA_API_KEY=... OANDA_ACCOUNT_ID=... OANDA_ENV=practice \
        python3 backtest_orb_sessions.py
"""

from datetime import datetime, timedelta, timezone

from oanda_client import ASSET_TO_OANDA_INSTRUMENT
from trading_agent import AdvancedTradingAgent
from backtest_candidates import (
    fetch_historical_candles,
    simulate,
    summarize,
    format_summary,
    GRANULARITY,
)

LOOKBACK_DAYS = 730  # 2 years, vs. this repo's usual 1yr validation window
ASSET = "XAUUSD"
STRATEGIES = ["orb_15m_signal", "london_breakout_signal", "ny_session_breakout_signal"]


def main():
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=LOOKBACK_DAYS)

    agent = AdvancedTradingAgent()

    instrument = ASSET_TO_OANDA_INSTRUMENT[ASSET]
    print(f"Fetching {LOOKBACK_DAYS}d of {GRANULARITY} candles for {ASSET} ({instrument})...")
    bars = fetch_historical_candles(instrument, GRANULARITY, from_dt, to_dt)
    print(f"  {len(bars)} bars, {bars[0].timestamp.date()} -> {bars[-1].timestamp.date()}")
    print()

    split_idx = int(len(bars) * 0.7)
    train_bars = bars[:split_idx]
    test_bars = bars[split_idx:]
    print(f"Train window: {train_bars[0].timestamp.date()} -> {train_bars[-1].timestamp.date()} ({len(train_bars)} bars)")
    print(f"Test  window: {test_bars[0].timestamp.date()} -> {test_bars[-1].timestamp.date()} ({len(test_bars)} bars)")
    print()

    for method_name in STRATEGIES:
        full_trades = simulate(agent, method_name, ASSET, bars)
        train_trades = simulate(agent, method_name, ASSET, train_bars)
        test_trades = simulate(agent, method_name, ASSET, test_bars)

        full_s = summarize(full_trades)
        train_s = summarize(train_trades)
        test_s = summarize(test_trades)

        print(f"=== {method_name} on {ASSET} (2yr) ===")
        print("  " + format_summary("Full 2yr  ", full_s))
        print("  " + format_summary("Train 70% ", train_s))
        print("  " + format_summary("Test  30% ", test_s))
        print()


if __name__ == "__main__":
    main()
