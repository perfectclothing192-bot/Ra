"""
PAPER-trading loop for the agent, mirrored onto a real OANDA practice account.

Polls OANDA for recent candles, feeds them to the agent's strategies, and
when a signal fires, places a REAL market order on the configured OANDA
account (practice by default) with the strategy's stop-loss/take-profit
attached. This is still zero real-money risk as long as OANDA_ENV=practice
(the default) — it's a demo account — but it is no longer a purely
internal simulation: trades actually appear in the OANDA account/app, and
OANDA's own server enforces the stop/target, not this script.

Position sizing is based on the account's real OANDA balance (fetched each
poll), not a fictional number, so 1%-per-trade risk is risk of real
(demo) capital.

Two run modes:
- Continuous (default): a long-lived worker process (e.g. on Railway) that
  polls every POLL_INTERVAL_SECONDS forever.
- `--once`: runs a single poll/evaluate/execute cycle and exits. Intended
  for scheduled/cron-style execution where nothing stays running between
  invocations; state (positions, trades, OANDA trade IDs) is loaded from
  and saved back to STATE_FILE so consecutive runs behave like one
  continuous agent.

Environment variables:
- OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV: see oanda_client.py
  (OANDA_ENV=practice is the default and the only mode this has been
  tested against — do not point this at a live/funded account casually)
- POLL_INTERVAL_SECONDS: seconds between polls in continuous mode (default 300)
- OANDA_GRANULARITY: OANDA candle granularity, e.g. M15, H1 (default M15)
- RISK_PER_TRADE, DAILY_LOSS_LIMIT: agent risk settings
- STATE_FILE: path to persist agent state across invocations (default trading_state.json)
"""

import argparse
import json
import os
import time
from datetime import datetime

from trading_agent import AdvancedTradingAgent, TradingMode, Trade
from oanda_client import (
    fetch_candles,
    ASSET_TO_OANDA_INSTRUMENT,
    get_account_summary,
    place_market_order,
    get_trade,
)
from state_io import save_state, load_state

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
GRANULARITY = os.environ.get("OANDA_GRANULARITY", "M15")
STATE_FILE = os.environ.get("STATE_FILE", "trading_state.json")

# Only assets with an implemented strategy are traded automatically.
# XAUUSD checks each strategy in order and takes the first non-HOLD signal,
# so Golden Pullback and SMC never fight over the same position.
STRATEGIES = {
    "XAUUSD": ["golden_pullback_signal", "smc_signal"],
    "USOIL": ["sma_cluster_signal"],
}

# Paired strategy: GBPUSD/EURUSD are traded together via correlation_hedge_signal.
CORRELATION_PAIR = ("GBPUSD", "EURUSD")
CORRELATION_LOOKBACK = 51


def sync_oanda_state(agent):
    """
    Pull the real account balance (source of truth for equity/position
    sizing) and reconcile any OANDA-mirrored positions that have closed on
    OANDA's side (via its own stop-loss/take-profit) since the last poll.
    """
    summary = get_account_summary()
    balance = float(summary["balance"])
    if getattr(agent, "oanda_initial_balance", None) is None:
        agent.oanda_initial_balance = balance
    agent.account_equity = balance
    agent.daily_pnl = balance - agent.oanda_initial_balance
    # Circuit breaker's equity floor should be relative to the real OANDA
    # starting balance, not the agent's constructor default.
    agent.initial_equity = agent.oanda_initial_balance

    for asset, trade_id in list(agent.oanda_trade_ids.items()):
        trade = get_trade(trade_id)
        if trade["state"] == "CLOSED":
            position = agent.positions.pop(asset, None)
            del agent.oanda_trade_ids[asset]
            if position is not None:
                exit_price = float(trade["averageClosePrice"])
                realized_pl = float(trade["realizedPL"])
                agent.trades.append(Trade(
                    asset=asset,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    direction=position.direction,
                    quantity=position.quantity,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(),
                    pnl=realized_pl,
                    pnl_percent=(realized_pl / agent.account_equity) * 100,
                    strategy=asset,
                ))
                agent.logger.info(f"OANDA CLOSED {asset} | realized P&L: ${realized_pl:,.2f}")


def mirror_to_oanda(agent, asset, position):
    """Place a real OANDA order matching a just-opened local position.
    On failure, reverts the local position so state doesn't diverge from
    what's actually on the account."""
    units = int(round(position.quantity))
    if position.direction == "SHORT":
        units = -units
    try:
        fill = place_market_order(asset, units, position.stop_loss, position.take_profit)
        trade_id = fill["tradeOpened"]["tradeID"]
        agent.oanda_trade_ids[asset] = trade_id
        agent.logger.info(f"OANDA order filled: {asset} {units} units @ {fill['price']} (tradeID {trade_id})")
        return True
    except Exception as e:
        agent.logger.error(f"OANDA order FAILED for {asset}: {e} — reverting local position")
        agent.positions.pop(asset, None)
        return False


def close_oanda_trade(agent, asset):
    """Force-close a mirrored OANDA trade (used to unwind a hedge leg whose
    partner order failed to fill)."""
    trade_id = agent.oanda_trade_ids.pop(asset, None)
    agent.positions.pop(asset, None)
    if trade_id is None:
        return
    try:
        import requests
        from oanda_client import BASE_URL, _headers
        account_id = os.environ.get("OANDA_ACCOUNT_ID")
        requests.put(
            f"{BASE_URL}/v3/accounts/{account_id}/trades/{trade_id}/close",
            headers=_headers(), json={"units": "ALL"}, timeout=15,
        ).raise_for_status()
        agent.logger.info(f"Unwound {asset} OANDA trade {trade_id} (hedge partner leg failed to fill)")
    except Exception as e:
        agent.logger.error(f"Failed to unwind {asset} OANDA trade {trade_id}: {e}")


def poll_once(agent):
    """Run a single poll/evaluate/execute cycle. Returns a list of event dicts."""
    events = []

    sync_oanda_state(agent)

    current_prices = {}
    for asset in ASSET_TO_OANDA_INSTRUMENT:
        bars = fetch_candles(asset, granularity=GRANULARITY, count=250)
        agent.price_history[asset] = bars
        if bars:
            current_prices[asset] = bars[-1].close

    for asset, strategy_methods in STRATEGIES.items():
        bars = agent.price_history.get(asset, [])
        if len(bars) < 200:
            agent.logger.info(f"{asset}: waiting for more history ({len(bars)}/200 bars)")
            events.append({"asset": asset, "status": "waiting_history", "bars": len(bars)})
            continue
        price = current_prices.get(asset)
        if asset in agent.positions:
            agent.logger.info(f"{asset}: price={price} | position open on OANDA, skipping signal check")
            events.append({"asset": asset, "price": price, "status": "position_open"})
            continue
        signal = None
        for method_name in strategy_methods:
            signal = getattr(agent, method_name)(bars)
            if signal.direction != "HOLD":
                break
        agent.logger.info(f"{asset}: price={price} | signal={signal.direction} ({signal.strategy})")
        events.append({"asset": asset, "price": price, "signal": signal.direction, "strategy": signal.strategy})
        if signal.direction != "HOLD":
            position = agent.execute_signal(signal)
            if position is not None:
                mirror_to_oanda(agent, asset, position)

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
            gbp_ok = mirror_to_oanda(agent, gbp_asset, gbp_position) if gbp_position else False
            eur_ok = mirror_to_oanda(agent, eur_asset, eur_position) if eur_position else False
            if gbp_ok and not eur_ok:
                close_oanda_trade(agent, gbp_asset)
            elif eur_ok and not gbp_ok:
                close_oanda_trade(agent, eur_asset)

    return events


def build_agent():
    agent = AdvancedTradingAgent(
        mode=TradingMode.PAPER,
        account_equity=float(os.environ.get("ACCOUNT_EQUITY", "150000")),
        risk_per_trade=float(os.environ.get("RISK_PER_TRADE", "0.05")),
        daily_loss_limit=float(os.environ.get("DAILY_LOSS_LIMIT", "0.05")),
        min_equity_pct=float(os.environ.get("MIN_EQUITY_PCT", "0.5")),
    )
    agent.oanda_trade_ids = {}
    agent.oanda_initial_balance = None
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
    agent.logger.info(f"Starting OANDA-mirrored trading loop | poll every {POLL_INTERVAL_SECONDS}s | granularity {GRANULARITY}")
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
