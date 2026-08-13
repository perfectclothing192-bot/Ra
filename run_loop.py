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
- OANDA_EXECUTE: "true" to also place real market orders on the configured
  OANDA account for every signal (default "false" - the agent's own PAPER
  simulation below always runs regardless)
- OANDA_MIRROR_SCALE: fraction of the internally-computed position size
  actually sent to OANDA when OANDA_EXECUTE is on (default 0.01)

This loop always runs the agent's own simulation in PAPER mode - no real
orders from that simulation itself. Separately, when OANDA_EXECUTE=true, it
also places real orders on the OANDA account configured via OANDA_ACCOUNT_ID
(safe on a practice account, real money on a live one).
"""

import argparse
import json
import os
import time

from trading_agent import AdvancedTradingAgent, TradingMode
from oanda_client import fetch_candles, place_market_order, ASSET_TO_OANDA_INSTRUMENT
from state_io import save_state, load_state
from status_server import start_status_server, update_status, serialize_positions, serialize_trades

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
GRANULARITY = os.environ.get("OANDA_GRANULARITY", "M15")
STATE_FILE = os.environ.get("STATE_FILE", "trading_state.json")

# When enabled, every signal this loop acts on is also placed as a real market
# order on the configured OANDA account (practice or live, per OANDA_ENV), so
# trades are visible in the OANDA platform itself - independent of, and not
# necessarily identical to, this app's own internal PAPER simulation below
# (which drives the dashboard/status endpoint from its own simulated fills).
OANDA_EXECUTE = os.environ.get("OANDA_EXECUTE", "false").lower() == "true"
# Fraction of the internally-computed position size actually sent to OANDA,
# to keep order size sane on an account sized independently of ACCOUNT_EQUITY.
OANDA_MIRROR_SCALE = float(os.environ.get("OANDA_MIRROR_SCALE", "0.01"))


def mirror_to_oanda(agent, position):
    if not OANDA_EXECUTE or position is None:
        return
    units = max(1, round(position.quantity * OANDA_MIRROR_SCALE))
    if position.direction == "SHORT":
        units = -units
    try:
        result = place_market_order(position.asset, units, stop_loss=position.stop_loss, take_profit=position.take_profit)
        fill = result.get("orderFillTransaction", {})
        agent.logger.info(f"[OANDA] {position.asset} order filled: {units} units | tradeID={fill.get('id')}")
    except Exception as e:
        agent.logger.error(f"[OANDA] {position.asset} order failed: {e}")


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
    agent.last_prices = current_prices

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
            position = agent.execute_signal(signal)
            mirror_to_oanda(agent, position)

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
            gbp_position = agent.execute_signal(gbp_signal)
            eur_position = agent.execute_signal(eur_signal)
            mirror_to_oanda(agent, gbp_position)
            mirror_to_oanda(agent, eur_position)

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


def publish_status(agent, events):
    positions = serialize_positions(agent.positions, getattr(agent, "last_prices", None))
    unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions.values())
    update_status(
        agent_name="swing",
        assets=list(STRATEGIES.keys()) + list(CORRELATION_PAIR),
        granularity=GRANULARITY,
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        mode=agent.mode.value,
        account_equity=agent.account_equity,
        daily_pnl=agent.daily_pnl,
        unrealized_pnl=unrealized_pnl,
        positions=positions,
        recent_trades=serialize_trades(agent.trades),
        last_events=events,
    )


def run_continuous():
    agent = build_agent()
    agent.logger.info(f"Starting PAPER trading loop | poll every {POLL_INTERVAL_SECONDS}s | granularity {GRANULARITY}")

    port = os.environ.get("PORT")
    if port:
        start_status_server(int(port))
        agent.logger.info(f"Status endpoint listening on :{port}/status")

    while True:
        try:
            events = poll_once(agent)
        except Exception as e:
            agent.logger.error(f"Loop iteration error: {e}")
            events = [{"error": str(e)}]
        publish_status(agent, events)
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
