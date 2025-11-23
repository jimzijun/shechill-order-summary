import os
import json
from datetime import datetime, timedelta, time, date, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st

try:
    from square.client import Client
except ImportError:
    raise ImportError(
        "Could not import Client from square.client. "
        "Make sure you installed the official SDK, e.g. `pip install squareup`, "
        "and that no conflicting `square` package is installed."
    )

# ----- Config -----
LOCAL_TZ = ZoneInfo("America/Los_Angeles")  # adjust if needed

DEFAULT_LOCATION_ID = os.getenv("SQUARE_LOCATION_ID", "")
ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN", "")

DAYS_BACK = 14  # how many days back to search for updated orders

# ----- Helpers for time -----
def iso_utc(dt: datetime) -> str:
    """Return RFC3339 UTC string (e.g. 2025-11-22T15:00:00Z)."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def local_today() -> date:
    return datetime.now(LOCAL_TZ).date()


def local_date_from_rfc3339(ts: str | None) -> date | None:
    if not ts:
        return None
    dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt_utc.astimezone(LOCAL_TZ).date()


def local_dt_from_rfc3339(ts: str | None) -> datetime | None:
    if not ts:
        return None
    dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt_utc.astimezone(LOCAL_TZ)


# ----- Square fetch logic -----
def make_square_client(access_token: str, environment: str) -> Client:
    return Client(access_token=access_token, environment=environment)


def fetch_recent_pickup_orders(
    client: Client,
    location_id: str,
    days_back: int = DAYS_BACK,
) -> list[dict]:
    """
    Pull recent PICKUP orders from Square using updated_at filter.
    We cannot filter by pickup_at server-side, so we do that in code later.
    """
    orders_api = client.orders
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days_back)

    cursor = None
    all_orders: list[dict] = []

    while True:
        body = {
            "location_ids": [location_id],
            "query": {
                "filter": {
                    "date_time_filter": {
                        "updated_at": {
                            "start_at": iso_utc(start_utc),
                            "end_at": iso_utc(now_utc),
                        }
                    },
                    "fulfillment_filter": {
                        "fulfillment_types": ["PICKUP"],
                        # Valid states include: PROPOSED, RESERVED, PREPARED, COMPLETED, CANCELED, FAILED
                        "fulfillment_states": [
                            "PROPOSED",
                            "RESERVED",
                            "PREPARED",
                            "COMPLETED",
                        ],
                    },
                }
            },
            "limit": 1000,
        }
        if cursor:
            body["cursor"] = cursor

        result = orders_api.search_orders(body)
        if result.is_error():
            raise RuntimeError(result.errors)

        body = result.body
        orders = body.get("orders", [])
        # Drop draft orders client-side since Square search doesn't filter them out here
        non_draft_orders = [o for o in orders if o.get("state") != "DRAFT"]
        all_orders.extend(non_draft_orders)

        cursor = body.get("cursor")
        if not cursor:
            break

    return all_orders


def split_orders_by_pickup_date(orders: list[dict]):
    """Split recent pickup orders into (today_orders, tomorrow_orders) based on pickup_details.pickup_at."""
    today = local_today()
    tomorrow = today + timedelta(days=1)

    today_orders: list[dict] = []
    tomorrow_orders: list[dict] = []

    for o in orders:
        fulfillments = o.get("fulfillments") or []
        for f in fulfillments:
            if f.get("type") != "PICKUP":
                continue
            pickup_details = f.get("pickup_details") or {}
            pickup_at = pickup_details.get("pickup_at")
            pickup_date = local_date_from_rfc3339(pickup_at)

            if pickup_date == today:
                today_orders.append(o)
                break  # avoid duplicate add if multiple fulfillments
            elif pickup_date == tomorrow:
                tomorrow_orders.append(o)
                break

    return today_orders, tomorrow_orders


def orders_to_lineitem_df(orders: list[dict]) -> pd.DataFrame:
    """
    Flatten orders into a line-item DataFrame with the columns needed for exports.
    Each row is a pickup fulfillment line item.
    """
    rows = []

    for o in orders:
        fulfillments = o.get("fulfillments") or []
        line_items = o.get("line_items") or []

        for f in fulfillments:
            if f.get("type") != "PICKUP":
                continue

            pickup_details = f.get("pickup_details") or {}
            pickup_at = pickup_details.get("pickup_at")
            pickup_dt_local = local_dt_from_rfc3339(pickup_at)
            fulfillment_time = (
                pickup_dt_local.strftime("%Y-%m-%d %H:%M") if pickup_dt_local else None
            )

            recipient = pickup_details.get("recipient") or {}
            recipient_name = recipient.get("display_name")
            if not recipient_name:
                name_parts = [recipient.get("given_name"), recipient.get("family_name")]
                recipient_name = " ".join([p for p in name_parts if p]) or None

            recipient_email = recipient.get("email_address")
            recipient_phone = recipient.get("phone_number")

            for li in line_items:
                qty_str = li.get("quantity", "0")
                try:
                    qty = float(qty_str)
                except ValueError:
                    qty = None

                rows.append(
                    {
                        "Fulfillment Time": fulfillment_time,
                        "Recipient Name": recipient_name,
                        "Recipient Email": recipient_email,
                        "Recipient Phone": recipient_phone,
                        "Item Name": li.get("name"),
                        "Item Quantity": qty,
                    }
                )

    columns = [
        "Fulfillment Time",
        "Recipient Name",
        "Recipient Email",
        "Recipient Phone",
        "Item Name",
        "Item Quantity",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["Item Quantity"] = pd.to_numeric(df["Item Quantity"], errors="coerce")

    # Sort by fulfillment time for a clean pickup schedule
    df["_fulfillment_time_dt"] = pd.to_datetime(df["Fulfillment Time"], errors="coerce")
    df = df.sort_values(["_fulfillment_time_dt", "Recipient Name", "Item Name"]).drop(
        columns=["_fulfillment_time_dt"]
    )

    return df[columns]


def kitchen_production_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total quantity per item for the kitchen production sheet.
    """
    if df.empty:
        return pd.DataFrame(columns=["Item Name", "Item Quantity"])

    agg = (
        df.groupby(["Item Name"], dropna=False)["Item Quantity"]
        .sum()
        .reset_index()
        .sort_values(["Item Name"])
    )
    return agg


def dataframe_full_height(df: pd.DataFrame, row_px: int = 35, header_px: int = 38, padding_px: int = 16) -> int:
    """
    Compute a height that fits all rows so Streamlit shows the full table without scroll bars.
    """
    rows = max(len(df), 1)  # ensure at least one row height
    return header_px + rows * row_px + padding_px


# ----- Streamlit app -----
st.set_page_config(page_title="Square Pickup Orders – Today & Tomorrow", layout="wide")

st.title("Square Pickup Orders – Today & Tomorrow")

with st.sidebar:
    st.header("Square Settings")

    token_ok = bool(ACCESS_TOKEN)
    if not token_ok:
        st.error("SQUARE_ACCESS_TOKEN env var is not set.")
    st.write("Environment variables used (only production environment is supported):")

    access_token = st.text_input(
        "Access token",
        value=ACCESS_TOKEN,
        type="password",
        help="SQUARE_ACCESS_TOKEN",
    )

    location_id = st.text_input(
        "Location ID",
        value=DEFAULT_LOCATION_ID,
        help="SQUARE_LOCATION_ID (from your Square dashboard)",
    )

    st.caption("Data is cached briefly to avoid hitting Square on every refresh.")


if not access_token:
    st.stop()
if not location_id:
    st.error("Please provide a Square location ID.")
    st.stop()


@st.cache_data(ttl=120)  # cache for 2 minutes
def load_orders_cached(access_token: str, location_id: str):
    client = make_square_client(access_token, environment="production")
    recent_orders = fetch_recent_pickup_orders(client, location_id, days_back=DAYS_BACK)
    today_orders, tomorrow_orders = split_orders_by_pickup_date(recent_orders)
    return recent_orders, today_orders, tomorrow_orders


try:
    recent_orders, today_orders, tomorrow_orders = load_orders_cached(
        access_token, location_id
    )
except Exception as e:
    st.error(f"Error fetching orders from Square: {e}")
    st.stop()

today_df = orders_to_lineitem_df(today_orders)
tomorrow_df = orders_to_lineitem_df(tomorrow_orders)

today_prod = kitchen_production_table(today_df)
tomorrow_prod = kitchen_production_table(tomorrow_df)

orders_tab, = st.tabs(["Orders"])

with orders_tab:
    sub_tabs = st.tabs(["Today", "Tomorrow"])

    # --- Today ---
    with sub_tabs[0]:
        st.subheader(f"**Customer Order - {local_today().isoformat()}**")
        st.caption("One row per line item in each pickup order.")
        st.data_editor(
            today_df.reset_index(drop=True),
            height=dataframe_full_height(today_df),
            hide_index=True,
            disabled=True,
        )

        st.subheader(f"**Kitchen production - {local_today().isoformat()}**")
        st.caption("Total quantity per item for today.")
        st.data_editor(
            today_prod.reset_index(drop=True),
            height=dataframe_full_height(today_prod),
            hide_index=True,
            disabled=True,
        )

    # --- Tomorrow ---
    with sub_tabs[1]:
        st.subheader(f"**Customer Order - {(local_today() + timedelta(days=1)).isoformat()}**")
        st.caption("One row per line item in each pickup order.")
        st.data_editor(
            tomorrow_df.reset_index(drop=True),
            height=dataframe_full_height(tomorrow_df),
            hide_index=True,
            disabled=True,
        )

        st.subheader(f"**Kitchen production - {(local_today() + timedelta(days=1)).isoformat()}**")
        st.caption("Total quantity per item for tomorrow.")
        st.data_editor(
            tomorrow_prod.reset_index(drop=True),
            height=dataframe_full_height(tomorrow_prod),
            hide_index=True,
            disabled=True,
        )

    st.markdown("**Debug / Raw data (optional)**")
    with st.expander("Show raw Square JSON for recent pickup orders"):
        st.text(json.dumps(recent_orders, indent=2)[:20000])  # truncate for safety
