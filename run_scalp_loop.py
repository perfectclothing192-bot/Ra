"""
PAPER-trading loop for the Gold (XAUUSD) scalping agent, backed by OANDA
price data.

Runs as its own process, separate from run_loop.py (which covers the swing
strategies), because scalping needs a much faster candle granularity and
poll interval than the M15/300s swing setup.

Built to stay up unattended: OANDA API calls retry with backoff on transient
network/rate-limit/5xx errors (see oanda_client._get_with_retry), a failed
poll cycle is caught and logged rather than crashing the process (see
run_once/run_continuous below), and polling is skipped cleanly over the
weekend FX/Gold market closure instead of erroring every cycle.

Two run modes:
- Continuous (default): a long-lived worker process that polls every
  SCALP_POLL_INTERVAL_SECONDS forever.
- `--once`: runs a single poll/evaluate/execute cycle and exits. State
  (positions, trades, equity) is loaded from and saved back to
  SCALP_STATE_FILE so consecutive runs behave like one continuous agent.

Environment variables:
- OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV: see oanda_client.py
- SCALP_POLL_INTERVAL_SECONDS: seconds between polls in continuous mode (default 60)
- SCALP_GRANULARITY: OANDA candle granularity (default M1)
- SCALP_ACCOUNT_EQUITY, SCALP_RISK_PER_TRADE, SCALP_DAILY_LOSS_LIMIT: agent risk settings
- SCALP_STATE_FILE: path to persist agent state across invocations (default scalp_state.json)
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

from scalping_agent import ScalpingTradingAgent
from trading_agent import TradingMode
from oanda_client import fetch_candles, place_market_order, is_market_closed
from state_io import save_state, load_state
from status_server import start_status_server, update_status, serialize_positions, serialize_trades

POLL_INTERVAL_SECONDS = int(os.environ.get("SCALP_POLL_INTERVAL_SECONDS", "60"))
GRANULARITY = os.environ.get("SCALP_GRANULARITY", "M1")
STATE_FILE = os.environ.get("SCALP_STATE_FILE", "scalp_state.json")

ASSET = "XAUUSD"
MIN_BARS = 55

# When enabled, every signal this loop acts on is also placed as a real market
# order on the configured OANDA account (practice or live, per OANDA_ENV), so
# trades are visible in the OANDA platform itself - independent of, and not
# necessarily identical to, this app's own internal PAPER simulation below
# (which drives the dashboard/status endpoint from its own simulated fills).
OANDA_EXECUTE = os.environ.get("OANDA_EXECUTE", "false").lower() == "true"
# Fraction of the internally-computed position size actually sent to OANDA,
# to keep order size sane on an account sized independently of SCALP_ACCOUNT_EQUITY.
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


def poll_once(agent):
    """Run a single poll/evaluate/execute cycle. Returns a list of event dicts."""
    events = []

    if is_market_closed():
        agent.logger.info(f"{ASSET}: market closed (weekend) - skipping poll")
        events.append({"asset": ASSET, "status": "market_closed"})
        return events

    bars = fetch_candles(ASSET, granularity=GRANULARITY, count=250)
    agent.price_history[ASSET] = bars
    current_price = bars[-1].close if bars else None

    if current_price is not None:
        agent.update_positions({ASSET: current_price})
    agent.last_prices = {ASSET: current_price} if current_price is not None else {}

    if len(bars) < MIN_BARS:
        agent.logger.info(f"{ASSET}: waiting for more history ({len(bars)}/{MIN_BARS} bars)")
        events.append({"asset": ASSET, "status": "waiting_history", "bars": len(bars)})
        return events

    if ASSET in agent.positions:
        agent.logger.info(f"{ASSET}: price={current_price} | position open, skipping signal check")
        events.append({"asset": ASSET, "price": current_price, "status": "position_open"})
        return events

    signal = agent.ema_ribbon_scalp_signal(bars)
    agent.logger.info(f"{ASSET}: price={current_price} | signal={signal.direction} ({signal.strategy})")
    events.append({"asset": ASSET, "price": current_price, "signal": signal.direction, "strategy": signal.strategy})
    if signal.direction != "HOLD":
        position = agent.execute_signal(signal)
        mirror_to_oanda(agent, position)

    return events


def build_agent():
    agent = ScalpingTradingAgent(
        mode=TradingMode.PAPER,
        account_equity=float(os.environ.get("SCALP_ACCOUNT_EQUITY", "150000")),
        risk_per_trade=float(os.environ.get("SCALP_RISK_PER_TRADE", "0.0025")),
        daily_loss_limit=float(os.environ.get("SCALP_DAILY_LOSS_LIMIT", "0.03")),
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
        agent_name="scalper",
        asset=ASSET,
        strategy="ema_ribbon_scalp",
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
    agent.logger.info(f"Starting scalping PAPER trading loop | poll every {POLL_INTERVAL_SECONDS}s | granularity {GRANULARITY}")

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
