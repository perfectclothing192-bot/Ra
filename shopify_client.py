"""
Thin REST client for the Shopify Admin API. Used by ShopifyMarketingAgent
(shopify_agent.py) to read new orders/customers/abandoned checkouts and to
create discount codes. Mirrors the style of oanda_client.py.

Requires environment variables:
- SHOPIFY_STORE_DOMAIN: e.g. perfectstore12345.myshopify.com
- SHOPIFY_ACCESS_TOKEN: Admin API access token (custom app, Admin API scopes:
  read_orders, read_customers, read_checkouts, write_price_rules,
  write_discounts)
- SHOPIFY_API_VERSION: optional, defaults to 2024-10
"""

import os
from datetime import datetime

import requests

STORE_DOMAIN = os.environ.get("SHOPIFY_STORE_DOMAIN", "")
ACCESS_TOKEN = os.environ.get("SHOPIFY_ACCESS_TOKEN", "")
API_VERSION = os.environ.get("SHOPIFY_API_VERSION", "2024-10")

_session = requests.Session()


def _base_url() -> str:
    if not STORE_DOMAIN:
        raise RuntimeError("SHOPIFY_STORE_DOMAIN is not configured")
    return f"https://{STORE_DOMAIN}/admin/api/{API_VERSION}"


def _headers() -> dict:
    if not ACCESS_TOKEN:
        raise RuntimeError("SHOPIFY_ACCESS_TOKEN is not configured")
    return {"X-Shopify-Access-Token": ACCESS_TOKEN, "Content-Type": "application/json"}


def _get(path: str, params: dict = None) -> dict:
    response = _session.get(f"{_base_url()}{path}", headers=_headers(), params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def _post(path: str, payload: dict) -> dict:
    response = _session.post(f"{_base_url()}{path}", headers=_headers(), json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def get_orders_since(since_id: int = 0, limit: int = 50) -> list:
    """Orders created after since_id (ascending by id, Shopify's since_id default)."""
    data = _get("/orders.json", {"since_id": since_id, "limit": limit, "status": "any"})
    return data.get("orders", [])


def get_customers_since(since_id: int = 0, limit: int = 50) -> list:
    data = _get("/customers.json", {"since_id": since_id, "limit": limit})
    return data.get("customers", [])


def get_customer_email(customer_id: int) -> str:
    data = _get(f"/customers/{customer_id}.json")
    return data.get("customer", {}).get("email")


def get_abandoned_checkouts_since(created_at_min: datetime, limit: int = 50) -> list:
    """Open (not-yet-completed) checkouts created since `created_at_min`."""
    data = _get("/checkouts.json", {
        "created_at_min": created_at_min.isoformat(),
        "status": "open",
        "limit": limit,
    })
    return data.get("checkouts", [])


def _find_price_rule(title: str) -> dict:
    data = _get("/price_rules.json", {"limit": 250})
    for rule in data.get("price_rules", []):
        if rule.get("title") == title:
            return rule
    return None


def create_percentage_discount(code: str, percentage: float, starts_at: datetime,
                                ends_at: datetime = None, once_per_customer: bool = True,
                                entitled_collection_ids: list = None) -> dict:
    """Creates a percentage-off discount code (price rule + code), applied
    store-wide unless `entitled_collection_ids` restricts it to specific
    collections. No-ops and returns the existing price rule if one titled
    `code` already exists, so bootstrap is safe to re-run."""
    existing = _find_price_rule(code)
    if existing:
        return existing

    attributes = {
        "title": code,
        "target_type": "line_item",
        "allocation_method": "across",
        "value_type": "percentage",
        "value": f"-{percentage}",
        "customer_selection": "all",
        "once_per_customer": once_per_customer,
        "starts_at": starts_at.isoformat(),
    }
    if entitled_collection_ids:
        attributes["target_selection"] = "entitled"
        attributes["entitled_collection_ids"] = entitled_collection_ids
    else:
        attributes["target_selection"] = "all"
    if ends_at:
        attributes["ends_at"] = ends_at.isoformat()

    price_rule = _post("/price_rules.json", {"price_rule": attributes})["price_rule"]
    _post(f"/price_rules/{price_rule['id']}/discount_codes.json", {"discount_code": {"code": code}})
    return price_rule
