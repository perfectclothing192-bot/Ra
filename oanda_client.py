"""
OANDA v20 REST API client for price data and order execution.

Requires environment variables:
- OANDA_API_KEY: personal access token (generate in OANDA account settings)
- OANDA_ACCOUNT_ID: practice or live account ID (e.g. 101-011-XXXXXXX-001)
- OANDA_ENV: "practice" (default) or "live"
"""

import os
from datetime import datetime
from typing import List

import requests

from trading_agent import PriceBar

OANDA_ENV = os.environ.get("OANDA_ENV", "practice")
BASE_URL = "https://api-fxpractice.oanda.com" if OANDA_ENV == "practice" else "https://api-fxtrade.oanda.com"

# Map internal asset names to OANDA instrument codes.
# "XAUUSD_M15" is a synthetic second key for the same XAU_USD instrument -
# it lets the M15 Jesse Livermore strategy hold its own independent
# position/trade-ID slot (agent.positions / agent.oanda_trade_ids are
# keyed by asset string) alongside the M5 SMC strategy on plain "XAUUSD",
# without either one clobbering the other's tracked state.
ASSET_TO_OANDA_INSTRUMENT = {
    "XAUUSD": "XAU_USD",
    "XAUUSD_M15": "XAU_USD",
    "USOIL": "WTICO_USD",
    "GBPUSD": "GBP_USD",
    "EURUSD": "EUR_USD",
    "BTCUSD": "BTC_USD",
}


def _headers() -> dict:
    api_key = os.environ.get("OANDA_API_KEY")
    if not api_key:
        raise RuntimeError("OANDA_API_KEY not configured")
    return {"Authorization": f"Bearer {api_key}"}


def fetch_candles(asset: str, granularity: str = "M15", count: int = 250) -> List[PriceBar]:
    """Fetch the most recent completed candles for an asset."""
    instrument = ASSET_TO_OANDA_INSTRUMENT.get(asset)
    if not instrument:
        raise ValueError(f"No OANDA instrument mapping for asset '{asset}'")

    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    params = {"granularity": granularity, "count": count, "price": "M"}
    response = requests.get(url, params=params, headers=_headers(), timeout=15)
    response.raise_for_status()
    data = response.json()

    bars = []
    for candle in data.get("candles", []):
        if not candle.get("complete"):
            continue
        mid = candle["mid"]
        bars.append(PriceBar(
            timestamp=datetime.fromisoformat(candle["time"].replace("Z", "+00:00")),
            open=float(mid["o"]),
            high=float(mid["h"]),
            low=float(mid["l"]),
            close=float(mid["c"]),
            volume=int(candle.get("volume", 0)),
        ))
    return bars


def get_account_summary() -> dict:
    """Fetch account balance, currency, and margin info."""
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID not configured")
    url = f"{BASE_URL}/v3/accounts/{account_id}/summary"
    response = requests.get(url, headers=_headers(), timeout=15)
    response.raise_for_status()
    return response.json()["account"]


def place_market_order(asset: str, units: int, stop_loss_price: float = None, take_profit_price: float = None) -> dict:
    """
    Place a real market order on the OANDA account (practice or live, per OANDA_ENV).
    units: positive = buy/long, negative = sell/short.
    Returns the orderFillTransaction dict, which includes tradeOpened.tradeID.
    """
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID not configured")
    instrument = ASSET_TO_OANDA_INSTRUMENT.get(asset)
    if not instrument:
        raise ValueError(f"No OANDA instrument mapping for asset '{asset}'")

    order = {
        "type": "MARKET",
        "instrument": instrument,
        "units": str(units),
        "timeInForce": "FOK",
        "positionFill": "DEFAULT",
    }
    if stop_loss_price is not None:
        order["stopLossOnFill"] = {"price": f"{stop_loss_price:.5f}"}
    if take_profit_price is not None:
        order["takeProfitOnFill"] = {"price": f"{take_profit_price:.5f}"}

    url = f"{BASE_URL}/v3/accounts/{account_id}/orders"
    response = requests.post(url, headers=_headers(), json={"order": order}, timeout=15)
    response.raise_for_status()
    data = response.json()
    if "orderFillTransaction" not in data:
        raise RuntimeError(f"Order for {asset} did not fill: {data}")
    return data["orderFillTransaction"]


def get_open_trades() -> list:
    """List currently open trades on the account."""
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID not configured")
    url = f"{BASE_URL}/v3/accounts/{account_id}/openTrades"
    response = requests.get(url, headers=_headers(), timeout=15)
    response.raise_for_status()
    return response.json()["trades"]


def get_trade_close_info(trade_id: str, lookback: int = 300) -> dict:
    """
    Find the exit price and realized P&L for a trade that's no longer
    open, by scanning recent transaction history for the fill that
    closed it.

    Needed because this OANDA practice environment doesn't keep closed
    trades queryable the documented way - GET /trades/{id} 404s and
    GET /trades?state=CLOSED returns empty immediately after a trade
    closes, rather than returning the trade with state=CLOSED. The
    transaction log is the one place the close details reliably persist.

    Returns None if no closing fill for this trade is found within the
    lookback window (still open, or genuinely not found).
    """
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID not configured")
    summary = get_account_summary()
    last_id = int(summary["lastTransactionID"])
    from_id = max(1, last_id - lookback)
    url = f"{BASE_URL}/v3/accounts/{account_id}/transactions/idrange"
    response = requests.get(url, headers=_headers(), params={"from": from_id, "to": last_id}, timeout=15)
    response.raise_for_status()
    for txn in response.json().get("transactions", []):
        if txn.get("type") != "ORDER_FILL":
            continue
        for closed in txn.get("tradesClosed", []):
            if closed.get("tradeID") == str(trade_id):
                return {"exit_price": float(txn["price"]), "realized_pl": float(closed["realizedPL"])}
    return None


def get_trade(trade_id: str) -> dict:
    """Fetch a single trade's current details (works for open or closed trades)."""
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID not configured")
    url = f"{BASE_URL}/v3/accounts/{account_id}/trades/{trade_id}"
    response = requests.get(url, headers=_headers(), timeout=15)
    response.raise_for_status()
    return response.json()["trade"]


def fetch_current_price(asset: str) -> float:
    """Fetch the current mid price for an asset."""
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID not configured")
    instrument = ASSET_TO_OANDA_INSTRUMENT.get(asset)
    if not instrument:
        raise ValueError(f"No OANDA instrument mapping for asset '{asset}'")

    url = f"{BASE_URL}/v3/accounts/{account_id}/pricing"
    params = {"instruments": instrument}
    response = requests.get(url, params=params, headers=_headers(), timeout=15)
    response.raise_for_status()
    prices = response.json().get("prices", [])
    if not prices:
        raise RuntimeError(f"No price returned for instrument '{instrument}'")

    bid = float(prices[0]["bids"][0]["price"])
    ask = float(prices[0]["asks"][0]["price"])
    return (bid + ask) / 2
