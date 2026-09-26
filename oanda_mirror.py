"""
Shared OANDA order-mirroring helpers used by run_loop.py, run_scalp_loop.py,
and run_meanrev_loop.py.

All three loops mirror trades from their own independent internal PAPER
simulation onto the SAME configured OANDA account (OANDA_ACCOUNT_ID). That
account is a NETTING account, not a hedging one: OANDA only ever holds one
net position per instrument, so when two of our bots are both live on the
same instrument (all three currently trade XAUUSD), one bot's entry in the
opposite direction doesn't open a second position - it reduces or closes
the other bot's position via a MARKET_ORDER_TRADE_CLOSE. Analysis of the
live account's trade history found this had happened on 56% of gold
closes, contaminating every bot's P&L with exits it never intended and
never triggered by that position's own stop-loss/take-profit.

mirror_open() fixes this by checking for any already-open live trade on
the instrument before mirroring a new entry, and skipping the mirror (the
internal PAPER simulation still runs normally either way) if one exists -
so only one bot at a time holds real exposure on a given instrument.

Separately, the old code never told OANDA when the internal simulation
closed a position - the real trade only ever closed via its own native
SL/TP, or by getting netted by another bot's order as above. A position
closed by any other path (e.g. the mean-reversion agent's MAX_HOLD_HOURS
force-close) left the real OANDA trade open and orphaned indefinitely.
mirror_close() closes the tracked live trade explicitly whenever the
internal position closes, for any reason.
"""

import os

from oanda_client import ASSET_TO_OANDA_INSTRUMENT, place_market_order, get_open_trades, close_trade

OANDA_EXECUTE = os.environ.get("OANDA_EXECUTE", "false").lower() == "true"
OANDA_MIRROR_SCALE = float(os.environ.get("OANDA_MIRROR_SCALE", "0.01"))


def mirror_open(agent, position, agent_tag: str):
    """Mirror a newly opened internal position to OANDA, unless another
    bot already has live exposure on the same instrument."""
    if not OANDA_EXECUTE or position is None:
        return
    if not hasattr(agent, "live_trade_ids"):
        agent.live_trade_ids = {}

    instrument = ASSET_TO_OANDA_INSTRUMENT.get(position.asset)
    try:
        existing = get_open_trades(instrument)
    except Exception as e:
        agent.logger.error(f"[OANDA] {position.asset} could not check open trades, skipping mirror to be safe: {e}")
        return
    if existing:
        agent.logger.warning(
            f"[OANDA] {position.asset} skipping live mirror - {instrument} already has an open live "
            f"trade (id={existing[0].get('id')}), avoiding a netting collision with another agent"
        )
        return

    units = max(1, round(position.quantity * OANDA_MIRROR_SCALE))
    if position.direction == "SHORT":
        units = -units
    try:
        result = place_market_order(
            position.asset, units, stop_loss=position.stop_loss, take_profit=position.take_profit,
            client_id=position.position_id, client_tag=agent_tag, client_comment=position.asset,
        )
        fill = result.get("orderFillTransaction", {})
        trade_id = (fill.get("tradeOpened") or {}).get("tradeID") or fill.get("id")
        if trade_id:
            agent.live_trade_ids[position.asset] = trade_id
        agent.logger.info(f"[OANDA] {position.asset} order filled: {units} units | tradeID={trade_id}")
    except Exception as e:
        agent.logger.error(f"[OANDA] {position.asset} order failed: {e}")


def mirror_close(agent, asset: str):
    """Explicitly close the tracked live trade for `asset` when the
    internal position closes, so a real trade is never left orphaned by an
    internal-only exit (e.g. a max-hold-time force-close) that OANDA's own
    SL/TP would never have triggered."""
    if not OANDA_EXECUTE:
        return
    live_trade_ids = getattr(agent, "live_trade_ids", {})
    trade_id = live_trade_ids.pop(asset, None)
    if not trade_id:
        return
    try:
        close_trade(trade_id)
        agent.logger.info(f"[OANDA] {asset} closed live trade id={trade_id}")
    except Exception as e:
        agent.logger.error(f"[OANDA] {asset} failed to close live trade id={trade_id}: {e}")
