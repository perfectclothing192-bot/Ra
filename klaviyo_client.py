"""
Thin client for the Klaviyo API (JSON:API). Used by ShopifyMarketingAgent
(shopify_agent.py) to mirror Shopify customer/order/abandoned-checkout
activity into Klaviyo lists and events, and to create the email templates
from content_library.py. Mirrors the style of oanda_client.py /
shopify_client.py.

Requires environment variables:
- KLAVIYO_API_KEY: a private API key (Klaviyo account -> Settings -> API Keys)
- KLAVIYO_API_REVISION: optional, defaults to 2024-10-15

Note: Klaviyo flows themselves (trigger -> wait -> send steps) have no
practical REST equivalent, so this client only creates the building blocks
(lists, profiles, events, templates) - wiring a flow to fire off "added to
list X" using a given template is a one-time manual step in the Klaviyo UI.
"""

import os

import requests

API_KEY = os.environ.get("KLAVIYO_API_KEY", "")
API_REVISION = os.environ.get("KLAVIYO_API_REVISION", "2024-10-15")
BASE_URL = "https://a.klaviyo.com/api"

_session = requests.Session()


def _headers() -> dict:
    if not API_KEY:
        raise RuntimeError("KLAVIYO_API_KEY is not configured")
    return {
        "Authorization": f"Klaviyo-API-Key {API_KEY}",
        "revision": API_REVISION,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _get(path: str, params: dict = None) -> dict:
    response = _session.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def _post(path: str, payload: dict, expect_json: bool = True):
    response = _session.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=15)
    response.raise_for_status()
    return response.json() if expect_json and response.content else None


def get_or_create_list(name: str) -> str:
    """Returns the Klaviyo list id for `name`, creating it if needed."""
    existing = _get("/lists/", {"filter": f'equals(name,"{name}")'})
    if existing.get("data"):
        return existing["data"][0]["id"]
    created = _post("/lists/", {"data": {"type": "list", "attributes": {"name": name}}})
    return created["data"]["id"]


def upsert_profile(email: str, first_name: str = None) -> str:
    """Creates a profile and returns its Klaviyo id, or returns the existing
    profile's id if one with this email already exists."""
    attributes = {"email": email}
    if first_name:
        attributes["first_name"] = first_name
    try:
        created = _post("/profiles/", {"data": {"type": "profile", "attributes": attributes}})
        return created["data"]["id"]
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 409:
            raise
        return e.response.json()["errors"][0]["meta"]["duplicate_profile_id"]


def add_profile_to_list(list_id: str, profile_id: str):
    _post(f"/lists/{list_id}/relationships/profiles/", {
        "data": [{"type": "profile", "id": profile_id}]
    }, expect_json=False)


def track_event(email: str, metric_name: str, properties: dict = None, value: float = None):
    attributes = {
        "properties": properties or {},
        "metric": {"data": {"type": "metric", "attributes": {"name": metric_name}}},
        "profile": {"data": {"type": "profile", "attributes": {"email": email}}},
    }
    if value is not None:
        attributes["value"] = value
    _post("/events/", {"data": {"type": "event", "attributes": attributes}}, expect_json=False)


def create_template(name: str, subject: str, html: str) -> str:
    """Creates a Klaviyo email template ready to drop into a flow's email
    step. Returns the existing template's id if one named `name` already
    exists (idempotent, safe to re-run). The subject is embedded as an HTML
    comment since Klaviyo templates don't carry a subject line themselves -
    that's set on the flow message when the template is attached to it."""
    existing = _get("/templates/", {"filter": f'equals(name,"{name}")'})
    if existing.get("data"):
        return existing["data"][0]["id"]
    created = _post("/templates/", {"data": {"type": "template", "attributes": {
        "name": name,
        "editor_type": "CODE",
        "html": f"<!-- SUBJECT: {subject} -->\n{html}",
    }}})
    return created["data"]["id"]
