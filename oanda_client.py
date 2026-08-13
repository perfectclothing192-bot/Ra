"""
OANDA v20 REST API client for price data.

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

# Map internal asset names to OANDA instrument codes
ASSET_TO_OANDA_INSTRUMENT = {
    "XAUUSD": "XAU_USD",
    "USOIL": "WTICO_USD",
    "GBPUSD": "GBP_USD",
    "EURUSD": "EUR_USD",
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
