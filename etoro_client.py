"""
eToro Public API client for order placement via Agent Portfolios.

Requires environment variables:
- ETORO_API_KEY: Public API key (eToro: Settings -> Trading -> API Key Management)
- ETORO_USER_KEY: User key paired with the API key (shown once at creation -
  treat it like a password)
- ETORO_ENV: "demo" (default) or "real"

*** UNVERIFIED AGAINST A LIVE ACCOUNT ***
eToro launched this API in February 2026. Its own published docs are
inconsistent about the exact endpoint path - one reference page cites
"/api/v1/trading/real/orders", another cites
"/api/v2/trading/execution/orders" for the same "create an order" action.
This client was written against eToro's published documentation only; no
eToro API credentials were available to actually call it and confirm the
request/response shape, the way oanda_client.py was verified against a real
OANDA practice account. Before setting ETORO_EXECUTE=true anywhere:
  1. Create an Agent Portfolio and API key at https://api-portal.etoro.com
  2. Confirm the current endpoint path and request schema against that
     portal's live reference for your account (docs may have changed)
  3. Test manually against ETORO_ENV=demo and inspect the real response
     shape before trusting this against ETORO_ENV=real
"""

import os
import time
import uuid

import requests

BASE_URL = "https://public-api.etoro.com/api/v2"
ETORO_ENV = os.environ.get("ETORO_ENV", "demo")

# Map internal asset names to eToro instrument symbols.
# UNVERIFIED: eToro's exact symbol strings for gold/oil/FX pairs should be
# confirmed against GET /api/v1/instruments (or the portal's instrument
# list) before relying on this mapping.
ASSET_TO_ETORO_SYMBOL = {
    "XAUUSD": "XAUUSD",
    "USOIL": "OILCrude",
    "GBPUSD": "GBPUSD",
    "EURUSD": "EURUSD",
}


def _headers() -> dict:
    api_key = os.environ.get("ETORO_API_KEY")
    user_key = os.environ.get("ETORO_USER_KEY")
    if not api_key or not user_key:
        raise RuntimeError("ETORO_API_KEY / ETORO_USER_KEY not configured")
    return {
        "x-api-key": api_key,
        "x-user-key": user_key,
        "x-request-id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }


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


def place_order(asset: str, direction: str, amount: float, leverage: float = 1,
                 stop_loss_rate: float = None, take_profit_rate: float = None) -> dict:
    """
    Place a market order on eToro (demo or real, per ETORO_ENV).

    direction: "BUY" or "SELL". eToro only supports "buy" and "sellShort" as
    transaction types per its docs; SELL maps to "sellShort", which eToro
    requires a stopLossRate for (as does leverage > 1).
    amount: position size in account currency (a dollar amount to invest),
    not asset units - this is a different sizing model than OANDA's
    unit-based orders, so callers should NOT reuse OANDA-style quantities
    here directly.
    stop_loss_rate / take_profit_rate: price levels, per eToro's naming
    ("Rate") - UNVERIFIED whether these are absolute instrument prices (as
    assumed here) or expressed some other way; confirm against a real
    response before trusting this on ETORO_ENV=real.
    """
    symbol = ASSET_TO_ETORO_SYMBOL.get(asset)
    if not symbol:
        raise ValueError(f"No eToro symbol mapping for asset '{asset}'")

    path = "trading/execution/orders" if ETORO_ENV == "real" else "trading/execution/demo/orders"
    transaction_type = "buy" if direction == "BUY" else "sellShort"

    if transaction_type == "sellShort" and stop_loss_rate is None:
        raise ValueError("stop_loss_rate is required for sellShort orders per eToro's API")
    if leverage > 1 and stop_loss_rate is None:
        raise ValueError("stop_loss_rate is required when leverage > 1 per eToro's API")

    order = {
        "symbol": symbol,
        "transactionType": transaction_type,
        "amount": amount,
        "leverage": leverage,
    }
    if stop_loss_rate is not None:
        order["stopLossRate"] = stop_loss_rate
    if take_profit_rate is not None:
        order["takeProfitRate"] = take_profit_rate

    url = f"{BASE_URL}/{path}"
    response = _post_with_retry(url, order, _headers())
    body = response.json()
    if response.status_code >= 400:
        raise RuntimeError(f"eToro order rejected: {body}")
    return body
