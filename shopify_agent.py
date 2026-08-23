"""
Marketing automation agent for the Perfect Store Shopify shop.

Plays the same role for Shopify marketing that trading_agent.py plays for
the OANDA trading bot: this module holds the agent's core logic, while
run_shopify_loop.py drives it on a schedule (see that file's docstring).

What it automates - replacing the Zapier + Klaviyo-popup setup steps in
COMPLETE_AUTOMATION_SETUP_GUIDE.md:
- bootstrap(): creates the 5 discount codes (Shopify) and the Klaviyo lists
  + email templates (from content_library.py), so they're ready to attach
  to flows. Safe to re-run - every step no-ops if it already exists.
- run_cycle(): on every poll, mirrors new Shopify customers/orders/
  abandoned checkouts into the matching Klaviyo list + event. That list
  membership / event is what actually fires each Klaviyo flow.
- Flags customers with no order in `reengagement_days` for the
  re-engagement list.

What it does NOT do: build the Klaviyo flow itself (trigger -> wait ->
send). Klaviyo's flow canvas has no practical REST equivalent, so wiring
"when profile added to list X, send template Y after Z hours" is a one-time
manual step in the Klaviyo UI, using the lists/templates this agent creates
and the delay_hours in content_library.py.
"""

import logging
from datetime import datetime, timedelta, timezone

import shopify_client
import klaviyo_client
from content_library import (
    DISCOUNT_CODES, WELCOME_EMAILS, ABANDONED_CART_EMAILS,
    POST_PURCHASE_EMAILS, REENGAGEMENT_EMAILS,
)

LIST_NEW_SUBSCRIBERS = "Perfect Store Customers"
LIST_CUSTOMERS = "Customers"
LIST_ABANDONED_CART = "Abandoned Cart"
LIST_REENGAGEMENT = "Re-engagement"


class ShopifyMarketingAgent:
    def __init__(self, reengagement_days: int = 30):
        self.reengagement_days = reengagement_days
        self.logger = self._setup_logging()

        # Runtime state - persisted/restored by run_shopify_loop.py so a
        # sync cycle picks up where the last one left off.
        self.last_order_id = 0
        self.last_customer_id = 0
        self.last_checkout_poll_at = datetime.now(timezone.utc) - timedelta(hours=1)
        self.customer_last_order_at = {}     # customer_id (str) -> isoformat datetime
        self.reengaged_customer_ids = set()  # customer ids already sent to the re-engagement list

    # ------------------------------------------------------------------
    # BOOTSTRAP (run once via `python run_shopify_loop.py --bootstrap`)
    # ------------------------------------------------------------------

    def bootstrap(self) -> dict:
        self.logger.info("Creating discount codes...")
        now = datetime.now(timezone.utc)
        for d in DISCOUNT_CODES:
            rule = shopify_client.create_percentage_discount(
                code=d["code"],
                percentage=d["percentage"],
                starts_at=now,
                ends_at=now + timedelta(days=d["expires_after_days"]),
            )
            self.logger.info(f"  {d['code']} ({d['usage']}): price_rule_id={rule.get('id')}")

        self.logger.info("Creating Klaviyo lists...")
        list_ids = {}
        for name in (LIST_NEW_SUBSCRIBERS, LIST_CUSTOMERS, LIST_ABANDONED_CART, LIST_REENGAGEMENT):
            list_ids[name] = klaviyo_client.get_or_create_list(name)
            self.logger.info(f"  {name}: list_id={list_ids[name]}")

        self.logger.info("Creating Klaviyo email templates...")
        sequences = (
            ("Welcome", WELCOME_EMAILS),
            ("Abandoned Cart", ABANDONED_CART_EMAILS),
            ("Post-Purchase", POST_PURCHASE_EMAILS),
            ("Re-engagement", REENGAGEMENT_EMAILS),
        )
        for sequence_name, emails in sequences:
            for email in emails:
                template_name = f"{sequence_name} - {email['name']}"
                template_id = klaviyo_client.create_template(
                    name=template_name,
                    subject=email["subject"],
                    html=_body_to_html(email["body"]),
                )
                self.logger.info(f"  {template_name} (+{email['delay_hours']}h): template_id={template_id}")

        self.logger.info(
            "Bootstrap complete. In Klaviyo, build 4 flows - Welcome, Abandoned "
            "Cart, Post-Purchase, Re-engagement - each triggered by 'added to "
            "list' on the matching list above, using the templates just created "
            "and the delay_hours in content_library.py for the wait steps."
        )
        return list_ids

    # ------------------------------------------------------------------
    # SYNC (called every poll cycle by run_shopify_loop.py)
    # ------------------------------------------------------------------

    def sync_new_customers(self) -> list:
        events = []
        for customer in shopify_client.get_customers_since(self.last_customer_id):
            self.last_customer_id = max(self.last_customer_id, customer["id"])
            email = customer.get("email")
            if not email:
                continue
            profile_id = klaviyo_client.upsert_profile(email, customer.get("first_name"))
            list_id = klaviyo_client.get_or_create_list(LIST_NEW_SUBSCRIBERS)
            klaviyo_client.add_profile_to_list(list_id, profile_id)
            self.logger.info(f"New subscriber synced: {email}")
            events.append({"type": "new_subscriber", "email": email})
        return events

    def sync_new_orders(self) -> list:
        events = []
        for order in shopify_client.get_orders_since(self.last_order_id):
            self.last_order_id = max(self.last_order_id, order["id"])
            customer = order.get("customer") or {}
            email = customer.get("email") or order.get("email")
            if not email:
                continue

            if customer.get("id"):
                self.customer_last_order_at[str(customer["id"])] = datetime.now(timezone.utc).isoformat()
                self.reengaged_customer_ids.discard(customer["id"])

            profile_id = klaviyo_client.upsert_profile(email, customer.get("first_name"))
            list_id = klaviyo_client.get_or_create_list(LIST_CUSTOMERS)
            klaviyo_client.add_profile_to_list(list_id, profile_id)
            klaviyo_client.track_event(email, "Placed Order", {
                "order_id": order["id"],
                "order_number": order.get("order_number"),
                "total_price": order.get("total_price"),
            }, value=float(order.get("total_price") or 0))
            self.logger.info(f"Order synced: {email} | #{order.get('order_number')}")
            events.append({"type": "order", "email": email, "order_number": order.get("order_number")})
        return events

    def sync_abandoned_checkouts(self) -> list:
        events = []
        poll_from = self.last_checkout_poll_at
        self.last_checkout_poll_at = datetime.now(timezone.utc)
        for checkout in shopify_client.get_abandoned_checkouts_since(poll_from):
            email = checkout.get("email")
            if not email:
                continue
            first_name = (checkout.get("customer") or {}).get("first_name")
            profile_id = klaviyo_client.upsert_profile(email, first_name)
            list_id = klaviyo_client.get_or_create_list(LIST_ABANDONED_CART)
            klaviyo_client.add_profile_to_list(list_id, profile_id)
            klaviyo_client.track_event(email, "Abandoned Checkout", {
                "checkout_id": checkout["id"],
                "abandoned_checkout_url": checkout.get("abandoned_checkout_url"),
                "total_price": checkout.get("total_price"),
            })
            self.logger.info(f"Abandoned checkout synced: {email}")
            events.append({"type": "abandoned_checkout", "email": email})
        return events

    def sync_reengagement_candidates(self) -> list:
        events = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.reengagement_days)
        for customer_id, last_order_at in list(self.customer_last_order_at.items()):
            if int(customer_id) in self.reengaged_customer_ids:
                continue
            if datetime.fromisoformat(last_order_at) > cutoff:
                continue
            email = shopify_client.get_customer_email(int(customer_id))
            if not email:
                continue
            profile_id = klaviyo_client.upsert_profile(email)
            list_id = klaviyo_client.get_or_create_list(LIST_REENGAGEMENT)
            klaviyo_client.add_profile_to_list(list_id, profile_id)
            self.reengaged_customer_ids.add(int(customer_id))
            self.logger.info(f"Re-engagement candidate synced: {email}")
            events.append({"type": "reengagement", "email": email})
        return events

    def run_cycle(self) -> list:
        events = []
        events += self.sync_new_customers()
        events += self.sync_new_orders()
        events += self.sync_abandoned_checkouts()
        events += self.sync_reengagement_candidates()
        return events

    @staticmethod
    def _setup_logging() -> logging.Logger:
        logger = logging.getLogger("ShopifyMarketingAgent")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger


def _body_to_html(body: str) -> str:
    return "<div>" + body.strip().replace("\n", "<br>\n") + "</div>"
