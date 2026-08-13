"""
Continuous PAPER-trading loop for the agent, backed by OANDA price data.

Polls OANDA for recent candles on each covered asset, feeds them to the
agent's strategies, and executes/manages PAPER positions. Intended to run
as a long-lived worker process (e.g. on Railway).

Environment variables:
- OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV: see oanda_client.py
- POLL_INTERVAL_SECONDS: seconds between polls (default 300)
- OANDA_GRANULARITY: OANDA candle granularity, e.g. M15, H1 (default M15)
- ACCOUNT_EQUITY, RISK_PER_TRADE, DAILY_LOSS_LIMIT: agent risk settings

This loop always runs the agent in PAPER mode. No real orders are placed.
"""

import os
import time

from trading_agent import AdvancedTradingAgent, TradingMode
from oanda_client import fetch_candles, ASSET_TO_OANDA_INSTRUMENT

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
GRANULARITY = os.environ.get("OANDA_GRANULARITY", "M15")

# Only assets with an implemented strategy are traded automatically.
STRATEGIES = {
    "XAUUSD": "golden_pullback_signal",
    "USOIL": "sma_cluster_signal",
}

# Paired strategy: GBPUSD/EURUSD are traded together via correlation_hedge_signal.
CORRELATION_PAIR = ("GBPUSD", "EURUSD")
CORRELATION_LOOKBACK = 51


def run():
    agent = AdvancedTradingAgent(
        mode=TradingMode.PAPER,
        account_equity=float(os.environ.get("ACCOUNT_EQUITY", "150000")),
        risk_per_trade=float(os.environ.get("RISK_PER_TRADE", "0.01")),
        daily_loss_limit=float(os.environ.get("DAILY_LOSS_LIMIT", "0.05")),
    )
    agent.logger.info(f"Starting PAPER trading loop | poll every {POLL_INTERVAL_SECONDS}s | granularity {GRANULARITY}")

    while True:
        try:
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
                    continue
                price = current_prices.get(asset)
                if asset in agent.positions:
                    agent.logger.info(f"{asset}: price={price} | position open, skipping signal check")
                    continue
                signal = getattr(agent, strategy_method)(bars)
                agent.logger.info(f"{asset}: price={price} | signal={signal.direction} ({signal.strategy})")
                if signal.direction != "HOLD":
                    agent.execute_signal(signal)

            gbp_asset, eur_asset = CORRELATION_PAIR
            gbp_bars = agent.price_history.get(gbp_asset, [])
            eur_bars = agent.price_history.get(eur_asset, [])
            if len(gbp_bars) < CORRELATION_LOOKBACK or len(eur_bars) < CORRELATION_LOOKBACK:
                agent.logger.info(
                    f"{gbp_asset}/{eur_asset}: waiting for more history "
                    f"({len(gbp_bars)}/{CORRELATION_LOOKBACK}, {len(eur_bars)}/{CORRELATION_LOOKBACK} bars)"
                )
            elif gbp_asset in agent.positions or eur_asset in agent.positions:
                agent.logger.info(f"{gbp_asset}/{eur_asset}: position open on one leg, skipping signal check")
            else:
                gbp_signal, eur_signal = agent.correlation_hedge_signal(gbp_bars, eur_bars)
                agent.logger.info(
                    f"{gbp_asset}: price={current_prices.get(gbp_asset)} | signal={gbp_signal.direction} (correlation_hedge)"
                )
                agent.logger.info(
                    f"{eur_asset}: price={current_prices.get(eur_asset)} | signal={eur_signal.direction} (correlation_hedge)"
                )
                if gbp_signal.direction != "HOLD" and eur_signal.direction != "HOLD":
                    agent.execute_signal(gbp_signal)
                    agent.execute_signal(eur_signal)

        except Exception as e:
            agent.logger.error(f"Loop iteration error: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
