"""
OANDA v20 REST API client for price data.

Requires environment variables:
- OANDA_API_KEY: personal access token (generate in OANDA account settings)
- OANDA_ACCOUNT_ID: practice or live account ID (e.g. 101-011-XXXXXXX-001)
- OANDA_ENV: "practice" (default) or "live"
"""

import os
import time
from datetime import datetime, timedelta, timezone
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

# OANDA rejects orders whose price fields exceed the instrument's allowed
# decimal precision (displayPrecision). Metals/commodities allow far fewer
# decimals than FX majors.
ASSET_PRICE_PRECISION = {
    "XAUUSD": 3,
    "USOIL": 3,
    "GBPUSD": 5,
    "EURUSD": 5,
}


def _headers() -> dict:
    api_key = os.environ.get("OANDA_API_KEY")
    if not api_key:
        raise RuntimeError("OANDA_API_KEY not configured")
    return {"Authorization": f"Bearer {api_key}"}


def _get_with_retry(url: str, params: dict, headers: dict, timeout: int = 15,
                     max_retries: int = 3, backoff_base: float = 2.0) -> requests.Response:
    """GET with retry/backoff for transient failures (network errors, rate limits, 5xx),
    so a single blip doesn't fail a whole poll cycle when polling frequently."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            time.sleep(backoff_base * (2 ** attempt))
    raise last_exc


def is_market_closed(now: datetime = None) -> bool:
    """
    Approximate check for the FX/Gold weekend closure: OANDA halts trading
    from Friday ~21:00 UTC to Sunday ~21:00 UTC. This is an approximation of
    the real boundary (5pm US Eastern, which shifts with US DST) but is
    close enough to avoid polling into the weekend gap.
    """
    now = now or datetime.now(timezone.utc)
    weekday = now.weekday()  # Monday=0 ... Sunday=6
    if weekday == 5:  # Saturday: closed all day
        return True
    if weekday == 4 and now.hour >= 21:  # Friday after close
        return True
    if weekday == 6 and now.hour < 21:  # Sunday before reopen
        return True
    return False


def _post_with_retry(url: str, json_body: dict, headers: dict, timeout: int = 15,
                      max_retries: int = 3, backoff_base: float = 2.0) -> requests.Response:
    """POST with retry/backoff for transient failures (network errors, rate limits, 5xx)."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=json_body, headers=headers, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            time.sleep(backoff_base * (2 ** attempt))
    raise last_exc


def place_market_order(asset: str, units: float, stop_loss: float = None, take_profit: float = None) -> dict:
    """
    Place a real market order against the configured OANDA account (practice
    or live, per OANDA_ENV). This actually executes on OANDA's platform - on
    a practice account that's still risk-free, but it's a genuinely different
    thing from this app's own internal PAPER simulation (which drives the
    dashboard/status endpoint): the two track independently and won't match
    exactly, since fill price/timing come from OANDA's own execution.
    """
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("OANDA_ACCOUNT_ID not configured")
    instrument = ASSET_TO_OANDA_INSTRUMENT.get(asset)
    if not instrument:
        raise ValueError(f"No OANDA instrument mapping for asset '{asset}'")

    precision = ASSET_PRICE_PRECISION.get(asset, 5)

    order = {
        "type": "MARKET",
        "instrument": instrument,
        "units": str(int(round(units))),
        "timeInForce": "FOK",
        "positionFill": "DEFAULT",
    }
    if stop_loss:
        order["stopLossOnFill"] = {"price": f"{stop_loss:.{precision}f}"}
    if take_profit:
        order["takeProfitOnFill"] = {"price": f"{take_profit:.{precision}f}"}

    url = f"{BASE_URL}/v3/accounts/{account_id}/orders"
    response = _post_with_retry(url, {"order": order}, _headers())
    body = response.json()
    if response.status_code >= 400:
        raise RuntimeError(f"OANDA order rejected: {body}")
    return body


def fetch_candles(asset: str, granularity: str = "M15", count: int = 250) -> List[PriceBar]:
    """Fetch the most recent completed candles for an asset."""
    instrument = ASSET_TO_OANDA_INSTRUMENT.get(asset)
    if not instrument:
        raise ValueError(f"No OANDA instrument mapping for asset '{asset}'")

    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    params = {"granularity": granularity, "count": count, "price": "M"}
    response = _get_with_retry(url, params, _headers())
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


def fetch_candles_range(asset: str, granularity: str, start: datetime, end: datetime) -> List[PriceBar]:
    """
    Fetch all completed candles between start and end (inclusive), paginating
    since OANDA caps a single request at 5000 candles. Used for backtesting
    over long windows that a single fetch_candles(count=...) can't cover.
    """
    instrument = ASSET_TO_OANDA_INSTRUMENT.get(asset)
    if not instrument:
        raise ValueError(f"No OANDA instrument mapping for asset '{asset}'")

    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    bars = []
    cursor = start
    while cursor < end:
        # OANDA rejects from+to+count together - page forward with from+count.
        params = {
            "granularity": granularity,
            "price": "M",
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": 5000,
        }
        response = _get_with_retry(url, params, _headers())
        response.raise_for_status()
        candles = response.json().get("candles", [])
        if not candles:
            break

        last_time = cursor
        for candle in candles:
            if not candle.get("complete"):
                continue
            mid = candle["mid"]
            ts = datetime.fromisoformat(candle["time"].replace("Z", "+00:00"))
            bars.append(PriceBar(
                timestamp=ts,
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=int(candle.get("volume", 0)),
            ))
            last_time = ts

        if len(candles) < 5000 or last_time <= cursor:
            break
        cursor = last_time + timedelta(seconds=1)

    return [b for b in bars if b.timestamp <= end]


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
    response = _get_with_retry(url, params, _headers())
    response.raise_for_status()
    prices = response.json().get("prices", [])
    if not prices:
        raise RuntimeError(f"No price returned for instrument '{instrument}'")

    bid = float(prices[0]["bids"][0]["price"])
    ask = float(prices[0]["asks"][0]["price"])
    return (bid + ask) / 2
