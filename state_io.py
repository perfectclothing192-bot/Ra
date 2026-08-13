"""
Persistence helpers for AdvancedTradingAgent state (open positions, closed
trades, equity), so a PAPER run can survive across separate process
invocations instead of only living in one long-running process's memory.
"""

import json
from dataclasses import asdict
from datetime import datetime

from trading_agent import AdvancedTradingAgent, Position, Trade


def save_state(agent: AdvancedTradingAgent, path: str):
    state = {
        "account_equity": agent.account_equity,
        "daily_pnl": agent.daily_pnl,
        "positions": {
            asset: {**asdict(pos), "entry_time": pos.entry_time.isoformat()}
            for asset, pos in agent.positions.items()
        },
        "trades": [
            {
                **asdict(t),
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
            }
            for t in agent.trades
        ],
        # Maps asset -> OANDA tradeID, for positions actually placed on OANDA.
        # Not part of the core agent model; only meaningful to the OANDA-mirroring loop.
        "oanda_trade_ids": getattr(agent, "oanda_trade_ids", {}),
        "oanda_initial_balance": getattr(agent, "oanda_initial_balance", None),
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(agent: AdvancedTradingAgent, path: str):
    try:
        with open(path) as f:
            state = json.load(f)
    except FileNotFoundError:
        return

    agent.account_equity = state.get("account_equity", agent.account_equity)
    agent.daily_pnl = state.get("daily_pnl", agent.daily_pnl)

    agent.positions = {}
    for asset, pos in state.get("positions", {}).items():
        pos = dict(pos)
        pos["entry_time"] = datetime.fromisoformat(pos["entry_time"])
        agent.positions[asset] = Position(**pos)

    agent.trades = []
    for t in state.get("trades", []):
        t = dict(t)
        t["entry_time"] = datetime.fromisoformat(t["entry_time"])
        t["exit_time"] = datetime.fromisoformat(t["exit_time"])
        agent.trades.append(Trade(**t))

    agent.oanda_trade_ids = state.get("oanda_trade_ids", {})
    agent.oanda_initial_balance = state.get("oanda_initial_balance")
