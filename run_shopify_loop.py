"""
Polling loop for ShopifyMarketingAgent, mirroring run_loop.py's role for the
trading agent. Automates the "Zapier + Klaviyo popup" wiring described in
COMPLETE_AUTOMATION_SETUP_GUIDE.md: new Shopify customers/orders/abandoned
checkouts get mirrored into the right Klaviyo list/event, which is what
fires the corresponding Klaviyo flow.

Three run modes:
- `--bootstrap`: one-time setup - creates the 5 discount codes and the
  Klaviyo lists/email templates from content_library.py, then exits. Safe
  to re-run (every step no-ops if it already exists). Run this once before
  the first sync.
- `--once`: single sync cycle then exit, for cron-style scheduling. State
  (last order/customer id, re-engagement tracking) persists in STATE_FILE
  so consecutive runs behave like one continuous agent.
- Continuous (default): long-lived worker (e.g. on Railway) that polls
  every POLL_INTERVAL_SECONDS forever.

Environment variables:
- SHOPIFY_STORE_DOMAIN, SHOPIFY_ACCESS_TOKEN, SHOPIFY_API_VERSION: see shopify_client.py
- KLAVIYO_API_KEY, KLAVIYO_API_REVISION: see klaviyo_client.py
- POLL_INTERVAL_SECONDS: seconds between polls in continuous mode (default 900)
- REENGAGEMENT_DAYS: days without an order before a customer is flagged (default 30)
- STATE_FILE: path to persist agent state across invocations (default shopify_state.json)
"""

import argparse
import json
import os
import time
from datetime import datetime

from shopify_agent import ShopifyMarketingAgent
from status_server import start_status_server, update_status

POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "900"))
REENGAGEMENT_DAYS = int(os.environ.get("REENGAGEMENT_DAYS", "30"))
STATE_FILE = os.environ.get("STATE_FILE", "shopify_state.json")


def save_state(agent: ShopifyMarketingAgent, path: str):
    state = {
        "last_order_id": agent.last_order_id,
        "last_customer_id": agent.last_customer_id,
        "last_checkout_poll_at": agent.last_checkout_poll_at.isoformat(),
        "customer_last_order_at": agent.customer_last_order_at,
        "reengaged_customer_ids": list(agent.reengaged_customer_ids),
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(agent: ShopifyMarketingAgent, path: str):
    try:
        with open(path) as f:
            state = json.load(f)
    except FileNotFoundError:
        return

    agent.last_order_id = state.get("last_order_id", 0)
    agent.last_customer_id = state.get("last_customer_id", 0)
    if state.get("last_checkout_poll_at"):
        agent.last_checkout_poll_at = datetime.fromisoformat(state["last_checkout_poll_at"])
    agent.customer_last_order_at = state.get("customer_last_order_at", {})
    agent.reengaged_customer_ids = set(state.get("reengaged_customer_ids", []))


def build_agent() -> ShopifyMarketingAgent:
    agent = ShopifyMarketingAgent(reengagement_days=REENGAGEMENT_DAYS)
    load_state(agent, STATE_FILE)
    return agent


def run_once():
    agent = build_agent()
    try:
        events = agent.run_cycle()
    except Exception as e:
        agent.logger.error(f"Sync cycle error: {e}")
        events = [{"error": str(e)}]
    save_state(agent, STATE_FILE)
    return events


def run_continuous():
    agent = build_agent()
    agent.logger.info(f"Starting Shopify marketing sync loop | poll every {POLL_INTERVAL_SECONDS}s")

    port = os.environ.get("PORT")
    if port:
        start_status_server(int(port))
        agent.logger.info(f"Status endpoint listening on :{port}/status")

    while True:
        try:
            events = agent.run_cycle()
        except Exception as e:
            agent.logger.error(f"Sync cycle error: {e}")
            events = [{"error": str(e)}]
        update_status(
            agent_name="shopify_marketing",
            poll_interval_seconds=POLL_INTERVAL_SECONDS,
            reengagement_days=REENGAGEMENT_DAYS,
            last_order_id=agent.last_order_id,
            last_customer_id=agent.last_customer_id,
            last_events=events,
        )
        save_state(agent, STATE_FILE)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single sync cycle and exit")
    parser.add_argument("--bootstrap", action="store_true",
                         help="Create discount codes + Klaviyo lists/templates, then exit")
    args = parser.parse_args()

    if args.bootstrap:
        build_agent().bootstrap()
    elif args.once:
        print(json.dumps(run_once(), default=str))
    else:
        run_continuous()
