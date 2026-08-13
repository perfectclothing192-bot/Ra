"""
PAPER-trading loop for the agent, backed by OANDA price data.

Polls OANDA for recent candles on each covered asset, feeds them to the
agent's strategies, and executes/manages PAPER positions.

Two run modes:
- Continuous (default): a long-lived worker process (e.g. on Railway) that
  polls every POLL_INTERVAL_SECONDS forever.
- `--once`: runs a single poll/evaluate/execute cycle and exits. Intended
  for scheduled/cron-style execution where nothing stays running between
  invocations; state (positions, trades, equity) is loaded from and saved
  back to STATE_FILE so consecutive runs behave like one continuous agent.

Environment variables:
- OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV: see oanda_client.py
- POLL_INTERVAL_SECONDS: seconds between polls in continuous mode (default 300)
- OANDA_GRANULARITY: OANDA candle granularity, e.g. M15, H1 (default M15)
- ACCOUNT_EQUITY, RISK_PER_TRADE, DAILY_LOSS_LIMIT: agent risk settings
- STATE_FILE: path to persist agent state across invocations (default trading_state.json)

This loop always runs the agent in PAPER mode. No real orders are placed.
"""

import argparse
import json
import os
import time

from trading_agent import AdvancedTradingAgent, TradingMode
from oanda_client import fetch_candles, ASSET_TO_OANDA_INSTRUMENT
from state_io import save_state, load_state

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
GRANULARITY = os.environ.get("OANDA_GRANULARITY", "M15")
STATE_FILE = os.environ.get("STATE_FILE", "trading_state.json")

# Only assets with an implemented strategy are traded automatically.
STRATEGIES = {
    "XAUUSD": "golden_pullback_signal",
    "USOIL": "sma_cluster_signal",
}

# Paired strategy: GBPUSD/EURUSD are traded together via correlation_hedge_signal.
CORRELATION_PAIR = ("GBPUSD", "EURUSD")
CORRELATION_LOOKBACK = 51


def poll_once(agent):
    """Run a single poll/evaluate/execute cycle. Returns a list of event dicts."""
    events = []
    current_prices = {}
    for asset in ASSET_TO_OANDA_INSTRUMENT:
        bars = fetch_candles(asset, granularity=GRANULARITY, count=250)
        agent.price_history[asset] = bars
        if bars:
            current_prices[asset] = bars[-1].close

    agent.update_positions(current_prices)

    for asset, strategy_method in STRATEGIES.items():
        bars = agent.price_history.get(asset, [])
        if len(bars) < 200:
            agent.logger.info(f"{asset}: waiting for more history ({len(bars)}/200 bars)")
            events.append({"asset": asset, "status": "waiting_history", "bars": len(bars)})
            continue
        price = current_prices.get(asset)
        if asset in agent.positions:
            agent.logger.info(f"{asset}: price={price} | position open, skipping signal check")
            events.append({"asset": asset, "price": price, "status": "position_open"})
            continue
        signal = getattr(agent, strategy_method)(bars)
        agent.logger.info(f"{asset}: price={price} | signal={signal.direction} ({signal.strategy})")
        events.append({"asset": asset, "price": price, "signal": signal.direction, "strategy": signal.strategy})
        if signal.direction != "HOLD":
            agent.execute_signal(signal)

    gbp_asset, eur_asset = CORRELATION_PAIR
    gbp_bars = agent.price_history.get(gbp_asset, [])
    eur_bars = agent.price_history.get(eur_asset, [])
    if len(gbp_bars) < CORRELATION_LOOKBACK or len(eur_bars) < CORRELATION_LOOKBACK:
        agent.logger.info(f"{gbp_asset}/{eur_asset}: waiting for more history")
        events.append({"asset": f"{gbp_asset}/{eur_asset}", "status": "waiting_history"})
    elif gbp_asset in agent.positions or eur_asset in agent.positions:
        agent.logger.info(f"{gbp_asset}/{eur_asset}: position open on one leg, skipping signal check")
        events.append({"asset": f"{gbp_asset}/{eur_asset}", "status": "position_open"})
    else:
        gbp_signal, eur_signal = agent.correlation_hedge_signal(gbp_bars, eur_bars)
        agent.logger.info(f"{gbp_asset}: price={current_prices.get(gbp_asset)} | signal={gbp_signal.direction} (correlation_hedge)")
        agent.logger.info(f"{eur_asset}: price={current_prices.get(eur_asset)} | signal={eur_signal.direction} (correlation_hedge)")
        events.append({"asset": gbp_asset, "price": current_prices.get(gbp_asset), "signal": gbp_signal.direction, "strategy": "correlation_hedge"})
        events.append({"asset": eur_asset, "price": current_prices.get(eur_asset), "signal": eur_signal.direction, "strategy": "correlation_hedge"})
        if gbp_signal.direction != "HOLD" and eur_signal.direction != "HOLD":
            agent.execute_signal(gbp_signal)
            agent.execute_signal(eur_signal)

    return events


def build_agent():
    agent = AdvancedTradingAgent(
        mode=TradingMode.PAPER,
        account_equity=float(os.environ.get("ACCOUNT_EQUITY", "150000")),
        risk_per_trade=float(os.environ.get("RISK_PER_TRADE", "0.01")),
        daily_loss_limit=float(os.environ.get("DAILY_LOSS_LIMIT", "0.05")),
    )
    load_state(agent, STATE_FILE)
    return agent


def run_once():
    agent = build_agent()
    try:
        events = poll_once(agent)
    except Exception as e:
        agent.logger.error(f"Loop iteration error: {e}")
        events = [{"error": str(e)}]
    save_state(agent, STATE_FILE)
    return events


def run_continuous():
    agent = build_agent()
    agent.logger.info(f"Starting PAPER trading loop | poll every {POLL_INTERVAL_SECONDS}s | granularity {GRANULARITY}")
    while True:
        try:
            poll_once(agent)
        except Exception as e:
            agent.logger.error(f"Loop iteration error: {e}")
        save_state(agent, STATE_FILE)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single poll cycle and exit, instead of looping forever")
    args = parser.parse_args()

    if args.once:
        result = run_once()
        print(json.dumps(result, default=str))
    else:
        run_continuous()
