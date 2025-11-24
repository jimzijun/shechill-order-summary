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
    from square.http.auth.o_auth_2 import BearerAuthCredentials
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
    # access_token arg is deprecated; use bearer_auth_credentials instead
    return Client(
        bearer_auth_credentials=BearerAuthCredentials(access_token),
        environment=environment,
    )


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


def split_orders_by_pickup_date(orders: list[dict], days: int = 7):
    """
    Group recent pickup orders by pickup date for the next N days (starting today).
    Returns a dict keyed by date -> list[orders].
    """
    target_days = [local_today() + timedelta(days=i) for i in range(days)]
    grouped: dict[date, list[dict]] = {d: [] for d in target_days}

    for o in orders:
        fulfillments = o.get("fulfillments") or []
        added_dates: set[date] = set()

        for f in fulfillments:
            if f.get("type") != "PICKUP":
                continue
            pickup_details = f.get("pickup_details") or {}
            pickup_at = pickup_details.get("pickup_at")
            pickup_date = local_date_from_rfc3339(pickup_at)

            if pickup_date in grouped and pickup_date not in added_dates:
                grouped[pickup_date].append(o)
                added_dates.add(pickup_date)

    return grouped


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
                        "Variation Name": li.get("variation_name"),
                        "Item Quantity": qty,
                    }
                )

    columns = [
        "Fulfillment Time",
        "Recipient Name",
        "Recipient Email",
        "Recipient Phone",
        "Item Name",
        "Variation Name",
        "Item Quantity",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["Item Quantity"] = pd.to_numeric(df["Item Quantity"], errors="coerce")

    # Sort by fulfillment time for a clean pickup schedule
    df["_fulfillment_time_dt"] = pd.to_datetime(df["Fulfillment Time"], errors="coerce")
    df = df.sort_values(["_fulfillment_time_dt", "Recipient Name", "Item Name", "Variation Name"]).drop(
        columns=["_fulfillment_time_dt"]
    )

    return df[columns]


def kitchen_production_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total quantity per item for the kitchen production sheet.
    """
    if df.empty:
        return pd.DataFrame(columns=["Item Name", "Variation Name", "Item Quantity"])

    agg = (
        df.groupby(["Item Name", "Variation Name"], dropna=False)["Item Quantity"]
        .sum()
        .reset_index()
        .sort_values(["Item Name", "Variation Name"])
    )
    return agg


def dataframe_full_height(df: pd.DataFrame, row_px: int = 35, header_px: int = 38, padding_px: int = 16) -> int:
    """
    Compute a height that fits all rows so Streamlit shows the full table without scroll bars.
    """
    rows = max(len(df), 1)  # ensure at least one row height
    return header_px + rows * row_px


# ----- Streamlit app -----
st.set_page_config(page_title="Square Pickup Orders", layout="wide")

st.title("Square Pickup Orders Viewer")

def upcoming_days(days: int = 14) -> list[date]:
    """Return [today, tomorrow, ...] for the requested day count."""
    start = local_today()
    return [start + timedelta(days=i) for i in range(days)]


def day_label(idx: int, dt: date) -> str:
    """Human-friendly label for a date starting from today."""
    if idx == 0:
        return f"Today ({dt.isoformat()})"
    if idx == 1:
        return f"Tomorrow ({dt.isoformat()})"
    return f"{dt.strftime('%a')} ({dt.isoformat()})"


if "day_view" not in st.session_state:
    st.session_state["day_view"] = None

day_dates = upcoming_days(14)
base_day_options = [day_label(i, d) for i, d in enumerate(day_dates)]
base_label_to_date = dict(zip(base_day_options, day_dates))

access_token = ACCESS_TOKEN
location_id = DEFAULT_LOCATION_ID

if not access_token:
    st.error("SQUARE_ACCESS_TOKEN env var is not set.")
    st.stop()
if not location_id:
    st.error("Please provide a Square location ID (SQUARE_LOCATION_ID).")
    st.stop()


@st.cache_data(ttl=120)  # cache for 2 minutes
def load_orders_cached(access_token: str, location_id: str):
    client = make_square_client(access_token, environment="production")
    recent_orders = fetch_recent_pickup_orders(client, location_id, days_back=DAYS_BACK)
    orders_by_day = split_orders_by_pickup_date(recent_orders, days=7)
    return recent_orders, orders_by_day


try:
    recent_orders, orders_by_day = load_orders_cached(
        access_token, location_id
    )
except Exception as e:
    st.error(f"Error fetching orders from Square: {e}")
    st.stop()

orders_by_day = orders_by_day or {}
day_options_empty = {
    label: orders_to_lineitem_df(orders_by_day.get(day, [])).empty
    for label, day in base_label_to_date.items()
}

day_options = [
    f"{label} 🟢" if not day_options_empty[label] else f"{label} ⭕"
    for label in base_day_options
]
label_to_date = dict(zip(day_options, day_dates))

default_day_option = st.session_state["day_view"]
if default_day_option not in label_to_date:
    default_day_option = day_options[1] if len(day_options) > 1 else day_options[0]

with st.container(key="day_view_container"):
    selected_day_label = st.segmented_control(
        "Pickup Date",
        day_options,
        default=default_day_option,
        selection_mode="single",
        key="day_view_toggle",
    )


st.session_state["day_view"] = selected_day_label
selected_date = label_to_date.get(selected_day_label, local_today())

selected_df = orders_to_lineitem_df(orders_by_day.get(selected_date, []))
selected_prod = kitchen_production_table(selected_df)

st.subheader(f"**Customer Order - {selected_date.isoformat()}**")
st.caption(f"One row per line item in each pickup order for {selected_day_label}.")
st.data_editor(
    selected_df.reset_index(drop=True),
    height=dataframe_full_height(selected_df),
    hide_index=True,
    disabled=True,
)

st.subheader(f"**Kitchen production - {selected_date.isoformat()}**")
st.caption(f"Total quantity per item for {selected_day_label}.")
st.data_editor(
    selected_prod.reset_index(drop=True),
    height=dataframe_full_height(selected_prod),
    hide_index=True,
    disabled=True,
)

st.markdown("**Debug / Raw data (optional)**")
with st.expander("Show raw Square JSON for recent pickup orders"):
    st.text(json.dumps(recent_orders, indent=2)[:20000])  # truncate for safety
