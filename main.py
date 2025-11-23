# %%
import os
from dotenv import load_dotenv
from square.client import Client
from datetime import datetime, timedelta, timezone

load_dotenv()

# Read credentials from environment (.env)
ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN", "")
LOCATION_ID = os.getenv("SQUARE_LOCATION_ID", "")

# Initialize the Square client.  Set environment to "sandbox" for testing or
# "production" for live data [oai_citation:4‡pypi.org](https://pypi.org/project/squareup/13.0.0.20210721/#:~:text=To%20use%20the%20Square%20API%2C,Here%E2%80%99s%20how).
client = Client(
    access_token=ACCESS_TOKEN,
    environment="production"
)

def fetch_orders():
    """Fetch only the first 100 orders from the location."""
    body = {
        "location_ids": [LOCATION_ID],
        "query": {
            "filter": {
                "fulfillment_filter": {
                    "fulfillment_types": ["PICKUP"],
                }
            },
            "sort": {
                "sort_field": "CREATED_AT",
                "sort_order": "DESC"
            }
        },
        "limit": 100
    }

    result = client.orders.search_orders(body)
    if result.is_success():
        return result.body.get("orders", [])
    else:
        raise RuntimeError(result.errors)


orders = fetch_orders()
print("Done:", len(orders))

# %%
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

PACIFIC = ZoneInfo("America/Los_Angeles")

def build_dataframe_from_orders(orders):
    rows = []

    for order in orders:
        fulfillments = order.get("fulfillments") or []
        line_items = order.get("line_items") or []

        for f in fulfillments:
            pickup_details = f.get("pickup_details") or {}
            pickup_at = pickup_details.get("pickup_at")  # e.g. "2025-11-23T18:00:00Z"
            if not pickup_at:
                continue

            # UTC → Pacific
            dt_utc = datetime.fromisoformat(pickup_at.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone(PACIFIC)

            recipient = pickup_details.get("recipient") or {}
            recipient_name = (
                recipient.get("display_name")
                or (" ".join(
                    x for x in [
                        recipient.get("given_name"),
                        recipient.get("family_name")
                    ] if x
                ) or None)
            )
            recipient_email = recipient.get("email_address")
            recipient_phone = recipient.get("phone_number")

            # One row per line item
            for li in line_items:
                item_name = li.get("name")
                qty_str = li.get("quantity") or "0"
                try:
                    qty = int(qty_str)
                except ValueError:
                    qty = float(qty_str)

                rows.append({
                    "Fulfillment Date": dt_local,
                    "Recipient Name": recipient_name,
                    "Recipient Email": recipient_email,
                    "Recipient Phone": recipient_phone,
                    "Item Name": item_name,
                    "Item Quantity": qty,
                })

    return pd.DataFrame(rows)

# %%
import pandas as pd

def make_tomorrow_reports():
    orders = fetch_orders()  # your existing function: first 100 orders
    df = build_dataframe_from_orders(orders)

    # Tomorrow in Los Angeles time
    now_local = datetime.now(PACIFIC)
    tomorrow_local = now_local.date() + timedelta(days=1)

    # Keep only tomorrow’s pickups (LOS ANGELES / Pacific)
    df = df[df["Fulfillment Date"].dt.date == tomorrow_local]

    # Same logic you had before
    df["Fulfillment Time"] = df["Fulfillment Date"].dt.strftime("%I:%M %p")

    # 取货时间表 – grouped by time + recipient + item
    pickup_df = (
        df.groupby(
            [
                "Fulfillment Date",
                "Fulfillment Time",
                "Recipient Name",
                "Recipient Email",
                "Recipient Phone",
                "Item Name",
            ]
        )["Item Quantity"]
        .sum()
        .reset_index()
    )
    pickup_df = pickup_df.drop(columns=["Fulfillment Date"])

    date_str = tomorrow_local.isoformat()
    pickup_df.to_csv(f"取货时间表-{date_str}.csv", index=False)

    # 后厨生产表 – total quantity by item
    kitchen_df = (
        pickup_df.groupby(["Item Name"])["Item Quantity"]
        .sum()
        .reset_index()
    )
    kitchen_df.to_csv(f"后厨生产表-{date_str}.csv", index=False)

# Call this to generate tomorrow’s CSVs
make_tomorrow_reports()

# %%
