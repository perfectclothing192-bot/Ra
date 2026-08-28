"""
PAPER-trading loop for the Gold (XAUUSD) micro mean-reversion agent, backed
by OANDA M5 price data.

Runs as its own process, separate from run_loop.py (swing, M15) and
run_scalp_loop.py (scalp, M1), matching this strategy's own validated
granularity - see backtest_micro_scalp.py, which found this exact
strategy (MEAN_WINDOW=20, BAND_K=2.0, STOP_K=3.5) net negative on an
88-day M5 backtest (-3.59% return, profit factor 0.92). It's deployed here
in PAPER mode purely to keep collecting live data on it, not because it
has earned live capital - see meanrev_agent.py's module docstring.

Two run modes:
- Continuous (default): a long-lived worker process that polls every
  MEANREV_POLL_INTERVAL_SECONDS forever.
- `--once`: runs a single poll/evaluate/execute cycle and exits. State
  (positions, trades, equity) is loaded from and saved back to
  MEANREV_STATE_FILE so consecutive runs behave like one continuous agent.

Environment variables:
- OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV: see oanda_client.py
- MEANREV_POLL_INTERVAL_SECONDS: seconds between polls in continuous mode (default 300)
- MEANREV_GRANULARITY: OANDA candle granularity (default M5)
- MEANREV_ACCOUNT_EQUITY, MEANREV_RISK_PER_TRADE, MEANREV_DAILY_LOSS_LIMIT: agent risk settings
- MEANREV_STATE_FILE: path to persist agent state across invocations (default meanrev_state.json)
- OANDA_EXECUTE, OANDA_MIRROR_SCALE: same meaning as run_scalp_loop.py - off by default
- ETORO_EXECUTE, ETORO_ORDER_AMOUNT: same meaning as run_scalp_loop.py - off by default, unverified
"""

import argparse
import json
import os
import time

from meanrev_agent import MeanReversionAgent
from trading_agent import TradingMode
from oanda_client import fetch_candles, place_market_order, is_market_closed
from state_io import save_state, load_state
from status_server import start_status_server, update_status, serialize_positions, serialize_trades
import etoro_client

POLL_INTERVAL_SECONDS = int(os.environ.get("MEANREV_POLL_INTERVAL_SECONDS", "300"))
GRANULARITY = os.environ.get("MEANREV_GRANULARITY", "M5")
STATE_FILE = os.environ.get("MEANREV_STATE_FILE", "meanrev_state.json")

ASSET = "XAUUSD"
MIN_BARS = 25  # MEAN_WINDOW (20) plus a small buffer

OANDA_EXECUTE = os.environ.get("OANDA_EXECUTE", "false").lower() == "true"
OANDA_MIRROR_SCALE = float(os.environ.get("OANDA_MIRROR_SCALE", "0.01"))

ETORO_EXECUTE = os.environ.get("ETORO_EXECUTE", "false").lower() == "true"
ETORO_ORDER_AMOUNT = float(os.environ.get("ETORO_ORDER_AMOUNT", "20"))


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


def mirror_to_etoro(agent, position):
    if not ETORO_EXECUTE or position is None:
        return
    direction = "BUY" if position.direction == "LONG" else "SELL"
    try:
        result = etoro_client.place_order(
            position.asset, direction, ETORO_ORDER_AMOUNT,
            stop_loss_rate=position.stop_loss, take_profit_rate=position.take_profit,
        )
        agent.logger.info(f"[ETORO] {position.asset} order response: {result}")
    except Exception as e:
        agent.logger.error(f"[ETORO] {position.asset} order failed: {e}")


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

    signal = agent.micro_mean_reversion_signal(bars)
    agent.logger.info(f"{ASSET}: price={current_price} | signal={signal.direction} ({signal.strategy})")
    events.append({"asset": ASSET, "price": current_price, "signal": signal.direction, "strategy": signal.strategy})
    if signal.direction != "HOLD":
        position = agent.execute_signal(signal)
        mirror_to_oanda(agent, position)
        mirror_to_etoro(agent, position)

    return events


def build_agent():
    agent = MeanReversionAgent(
        mode=TradingMode.PAPER,
        account_equity=float(os.environ.get("MEANREV_ACCOUNT_EQUITY", "150000")),
        risk_per_trade=float(os.environ.get("MEANREV_RISK_PER_TRADE", "0.0025")),
        daily_loss_limit=float(os.environ.get("MEANREV_DAILY_LOSS_LIMIT", "0.03")),
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
        agent_name="meanrev",
        asset=ASSET,
        strategy="micro_mean_reversion",
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
    agent.logger.info(f"Starting mean-reversion PAPER trading loop | poll every {POLL_INTERVAL_SECONDS}s | granularity {GRANULARITY}")

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
