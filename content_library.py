"""
Email content and discount codes for the Perfect Store marketing automation,
transcribed from READY_TO_COPY_CONTENT_LIBRARY.md / COMPLETE_AUTOMATION_SETUP_GUIDE.md.

Pure data - no side effects. shopify_agent.py's bootstrap() turns this into
Klaviyo email templates and Shopify discount codes.

`delay_hours` on each email is the offset from that sequence's trigger event
(new subscriber / abandoned checkout / order placed) at which the source
guide says to send it - not from the previous email - to match how a
Klaviyo flow's "wait" steps are configured off the flow's entry trigger.
The one exception is the welcome sequence, where the guide's original
timing branches on "previous email opened"; here it's collapsed into a
straight cumulative delay from signup (0h, 3h, 21h, 45h) as a reasonable
default - adjust the wait steps in the Klaviyo flow if you want the
"opened" branching instead.
"""

WELCOME_EMAILS = [
    {
        "name": "Welcome + 20% Off",
        "delay_hours": 0,
        "subject": "Welcome! Here's 20% Off Your First Order \U0001F381",
        "body": """Hey [First Name],

Welcome to Perfect Store! \U0001F389

We're so glad you're here. To say thanks for joining our community,
here's something special just for you:

Use code: WELCOME20 for 20% off your first order

What you can expect from us:
- Exclusive discounts (like this one!)
- Early access to new collections
- Fashion & home decor tips
- Easter prep guides
- Member-only sales

SHOP NOW -> [INSERT STORE LINK]

Can't wait to have you!
Perfect Store Team

P.S. This discount expires in 48 hours. Don't miss out!""",
    },
    {
        "name": "Best Sellers",
        "delay_hours": 3,
        "subject": "Your VIP Preview: Our Best-Selling Items",
        "body": """Hi [First Name],

While you're thinking about your first order, wanted to share what
our customers are absolutely LOVING right now:

TOP 3 BESTSELLERS:

1. Oversized Streetwear Hoodies (AUD $67.50)
   - Available in 5 colors
   - Perfect for casual wear
   - "Best hoodie I've ever owned!" - Sarah M.

2. Canvas Wall Art Set (AUD $54-290)
   - Transform any room
   - Premium quality
   - Ships in 2-3 days

3. Easter Collection (AUD $5-36)
   - Everything you need for Easter
   - In stock NOW
   - Perfect gifts

Each comes with our 30-day happiness guarantee.

EXPLORE BESTSELLERS -> [INSERT BESTSELLERS COLLECTION LINK]

Questions? Reply to this email. We read every message!

Your VIP Code (WELCOME20) works on these too.

Perfect Store Team""",
    },
    {
        "name": "Complete Your Look",
        "delay_hours": 21,
        "subject": "Steal These 3 Outfit Combos \U0001F457",
        "body": """Hi [First Name],

Noticed you visited our fashion section? Let me help you build
the perfect wardrobe.

Here are 3 complete outfit ideas (all under AUD $150):

OUTFIT 1: Weekend Vibes
- Oversized Graphic Hoodie (AUD $67.50)
- Relaxed Fit Jeans (AUD $75)
- Total: AUD $142.50

OUTFIT 2: Casual Chic
- Steel Blue Graphic T-Shirt (AUD $25)
- Modal Loungewear Top (AUD $75)
- Total: AUD $100

OUTFIT 3: Date Night
- Coral Crush Urban Dress (AUD $75)
- Pair with any shoes
- Total: AUD $75

Each outfit is comfortable, stylish, and affordable.

SHOP OUTFITS -> [INSERT FASHION COLLECTION LINK]

20% off with WELCOME20

Perfect Store Team

P.S. Unsure what to order? Hit reply - we love styling requests!""",
    },
    {
        "name": "Final Reminder",
        "delay_hours": 45,
        "subject": "LAST CHANCE: 20% Off Expires Tonight ⏰",
        "body": """Hi [First Name],

Your VIP discount expires at midnight tonight!

WELCOME20 - 20% OFF

Whether you're after home decor, fashion, Easter gifts, or skincare
essentials, we've got you covered. And 20% off makes it even better.

USE CODE: WELCOME20 -> [INSERT STORE LINK]

Don't wait - this expires soon!

Perfect Store Team

P.S. Still not sure? Check out our reviews. 4.8 stars. 500+ happy customers.""",
    },
]

ABANDONED_CART_EMAILS = [
    {
        "name": "Gentle Reminder",
        "delay_hours": 1,
        "subject": "You Left Something Behind \U0001F45C",
        "body": """Hi [First Name],

Just noticed you left some great items in your cart:

[PRODUCT 1 + PRICE]
[PRODUCT 2 + PRICE]

Total: [TOTAL PRICE]

Want to complete your purchase? Here's a little incentive:

Use COMEBACK15 for 15% off

COMPLETE PURCHASE -> [INSERT ABANDONED CART LINK]

Takes just 60 seconds. We're here if you have questions!

Perfect Store Team

P.S. Stock on these items is limited. Grab yours before they're gone!""",
    },
    {
        "name": "Urgency Boost",
        "delay_hours": 24,
        "subject": "Last Chance: Your Items Are Selling Out Fast ⚠️",
        "body": """Hi [First Name],

Bad news: limited stock remaining on items in your cart.

The items you wanted:
- [PRODUCT 1] - [X] left
- [PRODUCT 2] - [X] left

Still have your 15% discount: COMEBACK15

SECURE YOURS NOW -> [INSERT ABANDONED CART LINK]

We can have these shipped to you in 2 days.

Perfect Store Team""",
    },
    {
        "name": "Final Option",
        "delay_hours": 48,
        "subject": "Last Chance to Save 15% (Expires Midnight)",
        "body": """Hi [First Name],

This is your final notice for your 15% discount.

Your items:
[PRODUCT 1: PRICE]
[PRODUCT 2: PRICE]

Subtotal: [TOTAL]
With COMEBACK15 (15% off): [DISCOUNTED TOTAL]

This discount expires at midnight tonight.

CLAIM YOUR DISCOUNT -> [INSERT ABANDONED CART LINK]

If you don't want these items, no worries! But this discount won't last forever.

Perfect Store Team""",
    },
]

POST_PURCHASE_EMAILS = [
    {
        "name": "Thank You + Tips",
        "delay_hours": 2,
        "subject": "Your Order is On Its Way! \U0001F69A",
        "body": """Hi [First Name],

Your order [#ORDER ID] is confirmed and shipping tomorrow!

Items ordered:
[PRODUCT NAME] x[QUANTITY]

Expected delivery: [DATE]

Track your order -> [TRACKING LINK]

WHILE YOU WAIT...

Check out these complementary items that pair perfectly with your purchase:

[RELATED PRODUCT 1: PRICE]
[RELATED PRODUCT 2: PRICE]

GET 10% OFF RELATED ITEMS
Code: PAIR10 (Valid for 7 days)

SHOP RELATED ITEMS -> [INSERT RELATED PRODUCTS LINK]

Questions? We're here! Reply to this email.

Perfect Store Team""",
    },
    {
        "name": "Product Tips",
        "delay_hours": 72,
        "subject": "How to Style Your New [Product] - 5 Ideas",
        "body": """Hi [First Name],

Your order should arrive soon! Here's how to style/use it:

If you got the Oversized Hoodie:
1. Pair with black jeans + white sneakers (casual)
2. Layer over a dress (elevated)
3. Oversized with cycling shorts (trendy)
4. Belt at waist + heels (dressy)
5. Solo with sweatpants (cozy)

SAVE THESE STYLING COMBOS
Check out our guide: [STYLING GUIDE LINK]

SHOP STYLING PIECES -> [INSERT FASHION LINK]

Perfect Store Team""",
    },
    {
        "name": "Review Request",
        "delay_hours": 168,
        "subject": "What Do You Think? ⭐",
        "body": """Hi [First Name],

Your order should be with you by now!

We'd LOVE to hear what you think. Your honest review helps other
customers find the perfect items.

LEAVE A REVIEW (Takes 60 seconds) -> [INSERT REVIEW LINK]

Quick review = you enter our monthly raffle for:
- AUD $100 gift card
- Free items
- Exclusive discounts

Plus, every review helps us improve!

Thanks for being an awesome customer,
Perfect Store Team

P.S. Not happy? We offer 30-day returns, no questions asked.""",
    },
]

REENGAGEMENT_EMAILS = [
    {
        "name": "We Miss You",
        "delay_hours": 0,
        "subject": "We Miss You! Here's 25% Off",
        "body": """Hi [First Name],

It's been a month since your last order. We miss you!

Things have changed in our store:
- NEW: Summer Collection (Fresh styles)
- NEW: 50% Off Flash Sale (This weekend)
- NEW: VIP Member Perks (Exclusive to you)

To welcome you back: Use COMEBACK25 for 25% off

WELCOME BACK -> [INSERT STORE LINK]

What's new: limited edition hoodies, updated home decor, Easter
(still relevant!), expanded skincare range.

We're excited to show you what's new.

Your VIP Code: COMEBACK25 (Valid 7 days)

Perfect Store Team

P.S. Miss a particular item? Hit reply - we can help you find it!""",
    },
]

# Mirrors the "DISCOUNT CODES (Create in Shopify)" section of
# READY_TO_COPY_CONTENT_LIBRARY.md.
DISCOUNT_CODES = [
    {"code": "WELCOME15", "percentage": 15, "expires_after_days": 30, "usage": "First-time customers only"},
    {"code": "WELCOME20", "percentage": 20, "expires_after_days": 7, "usage": "Email subscribers only"},
    {"code": "COMEBACK15", "percentage": 15, "expires_after_days": 1, "usage": "Abandoned cart"},
    {"code": "COMEBACK25", "percentage": 25, "expires_after_days": 7, "usage": "Past customers (no purchase 30+ days)"},
    {"code": "BUNDLE20", "percentage": 20, "expires_after_days": 30, "usage": "Bundle products (all customers)"},
]
