# streamlit_app.py
# UAE Men Shoes — Global Lows (Supabase REST API + Streamlit)
# Requires: streamlit, pandas, requests, altair  (see requirements.txt)

import math
import requests
import pandas as pd
import streamlit as st
from urllib.parse import quote

# ---------------------------- App config ----------------------------
st.set_page_config(page_title="UAE Men Shoes — Global Lows", layout="wide")

# Read secrets safely (prevents crash if not set)
SUPABASE_URL = (st.secrets.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase secrets. Go to Settings → Secrets and set SUPABASE_URL + SUPABASE_ANON_KEY.")
    st.stop()

REST = f"{SUPABASE_URL}/rest/v1"
MV = "nam_uae_men_shoes_at_global_low"         # materialized view
PRICES = 'nam-uae-men-shoes-prices'            # raw time-series table

HDR = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Prefer": "count=exact",   # so Content-Range header includes total rows
}

# ---------------------------- Helpers ----------------------------
def pg_in(values: list[str]) -> str:
    """PostgREST IN() filter string with proper double-quote escaping."""
    esc = [v.replace('"', '""') for v in values]
    return 'in.(' + ",".join([f'"{e}"' for e in esc]) + ')'

def pct_fmt(x) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "—"

def http_get(url: str, params: dict) -> tuple[list, str | None]:
    """GET with simple error surface in Streamlit UI."""
    try:
        r = requests.get(url, params=params, headers=HDR, timeout=60)
        if not r.ok:
            st.error(f"Supabase REST error: {r.status_code}\n{r.text}")
            st.stop()
        return r.json(), r.headers.get("content-range")
    except requests.RequestException as e:
        st.error(f"Network error: {e}")
        st.stop()

@st.cache_data(ttl=300)
def load_options():
    """Fetch a big page and compute brands/categories/min-max locally (simple & robust)."""
    data, _ = http_get(f"{REST}/{MV}", {"select": "brand,category,latest_price,min_hits", "limit": "10000"})
    df = pd.DataFrame(data)
    brands = sorted([b for b in df.get("brand", pd.Series(dtype=str)).dropna().unique().tolist() if b])
    cats = sorted([c for c in df.get("category", pd.Series(dtype=str)).dropna().unique().tolist() if c]) or ["Men UAE shoes"]

    if df.empty:
        return brands, cats, 0.0, 0.0, 0, 0

    pmin, pmax = float(df["latest_price"].min()), float(df["latest_price"].max())
    hmin, hmax = int(df["min_hits"].min()), int(df["min_hits"].max())
    return brands, cats, pmin, pmax, hmin, hmax

def build_params(flt: dict, limit: int, offset: int) -> dict:
    """Translate UI filters → PostgREST query parameters."""
    p: dict[str, list | str] = {
        "select": "*",
        "has_higher": "eq.true",
        "order": f"{flt['order_by']}.{ 'desc' if flt['order_desc'] else 'asc'}",
        "limit": str(limit),
        "offset": str(offset),
        "count": "exact",
    }
    if flt["brands"]:
        p["brand"] = pg_in(flt["brands"])                # brand=in.("Nike","Adidas")

    if flt["category"]:
        p["category"] = f"eq.{flt['category']}"

    # numeric ranges
    lo, hi = flt["min_hits"]
    p["min_hits"] = [f"gte.{lo}", f"lte.{hi}"]

    plo, phi = flt["price_range"]
    p["latest_price"] = [f"gte.{plo}", f"lte.{phi}"]

    # percent sliders (UI is %, DB is fraction)
    for col, (lo_pct, hi_pct) in {
        "drop_pct_vs_prev": flt["drop_prev"],
        "delta_vs_30d_pct": flt["drop_30"],
        "delta_vs_90d_pct": flt["drop_90"],
    }.items():
        p[col] = [f"gte.{lo_pct/100.0}", f"lte.{hi_pct/100.0}"]

    return p

@st.cache_data(ttl=300)
def fetch_items(flt: dict, page: int, page_size: int) -> tuple[pd.DataFrame, int | None]:
    params = build_params(flt, page_size, page * page_size)
    data, cr = http_get(f"{REST}/{MV}", params)
    total = int(cr.split("/")[-1]) if cr and "/" in cr else None
    return pd.DataFrame(data), total

@st.cache_data(ttl=300)
def fetch_series(item_url: str, limit: int = 5000) -> pd.DataFrame:
    params = {
        "select": "timestamp,price",
        "url": f"eq.{quote(item_url, safe='')}",
        "order": "timestamp.asc",
        "limit": str(limit),
    }
    data, _ = http_get(f"{REST}/{PRICES}", params)
    df = pd.DataFrame(data)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Beirut")
    return df

# ---------------------------- UI: Filters first ----------------------------
st.title("UAE Men Shoes — Current Global Lows")
st.caption("Data live from Supabase REST API (materialized view + time series).")

brands, categories, pmin, pmax, hmin, hmax = load_options()

with st.form("filters_form"):
    st.subheader("Filters (choose, then Apply)")

    c1, c2, c3 = st.columns(3)
    with c1:
        chosen_brands = st.multiselect("Brands", options=brands)
        category = st.selectbox("Category", options=["(Any)"] + categories, index=0)
    with c2:
        hits_range = st.slider("Min hits", hmin, max(hmin, hmax), (hmin, max(hmin, hmax)))
        price_range = st.slider(
            "Price range (AED)",
            float(pmin),
            float(pmax or max(pmin, pmin + 1)),
            (float(pmin), float(pmax or max(pmin, pmin + 1))),
        )
    with c3:
        drop_prev = st.slider("Drop vs previous (%)", -100, 100, (-100, 100))
        drop_30 = st.slider("Drop vs 30‑day avg (%)", -100, 100, (-100, 100))
        drop_90 = st.slider("Drop vs 90‑day avg (%)", -100, 100, (-100, 100))

    st.markdown("---")
    cA, cB, cC = st.columns(3)
    with cA:
        order_by = st.selectbox(
            "Order by",
            [
                "delta_vs_30d_pct",
                "delta_vs_90d_pct",
                "drop_pct_vs_prev",
                "latest_price",
                "min_hits",
                "gap_to_second_lowest_pct",
                "days_since_first_low",
            ],
            index=0,
        )
    with cB:
        order_desc = st.toggle("Sort descending", True)
    with cC:
        page_size = st.select_slider("Page size", options=[12, 24, 48, 96], value=24)

    submitted = st.form_submit_button("Apply Filters")

if not submitted:
    st.info("↑ Set your filters, then click **Apply Filters**.")
    st.stop()

flt = {
    "brands": chosen_brands,
    "category": None if category == "(Any)" else category,
    "min_hits": hits_range,
    "price_range": price_range,
    "drop_prev": drop_prev,
    "drop_30": drop_30,
    "drop_90": drop_90,
    "order_by": order_by,
    "order_desc": order_desc,
}

# ---------------------------- Pagination ----------------------------
page = st.session_state.get("page", 0)
prev_col, _, next_col = st.columns([1, 6, 1])
with prev_col:
    if st.button("⟵ Prev", disabled=(page <= 0)):
        page = max(0, page - 1)
with next_col:
    if st.button("Next ⟶"):
        page = page + 1
st.session_state["page"] = page

# ---------------------------- Data & Grid ----------------------------
df, total = fetch_items(flt, page, page_size)
if df.empty:
    st.warning("No items match your filters.")
    st.stop()

st.caption(f"Matches: {total if total is not None else '—'}  •  Page {page + 1}")

ncols = 3
rows = math.ceil(len(df) / ncols)

for r in range(rows):
    cols = st.columns(ncols, gap="large")
    for c in range(ncols):
        i = r * ncols + c
        if i >= len(df):
            break
        row = df.iloc[i]
        with cols[c]:
            with st.container(border=True):
                # Header with image + metadata
                left, right = st.columns([1, 2])
                with left:
                    if row.get("image_link"):
                        st.image(row["image_link"], use_container_width=True)
                with right:
                    st.markdown(f"**{row.get('brand','')}**")
                    st.markdown(row.get("title", ""))
                    if row.get("url"):
                        st.link_button("Open product", row["url"], use_container_width=True)

                # Key metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Latest (AED)", f"{row.get('latest_price', 0):.2f}")
                    st.caption(f"Min hits: {row.get('min_hits', '—')}")
                with m2:
                    st.metric("Drop vs prev", pct_fmt(row.get("drop_pct_vs_prev")))
                    st.caption(f"30d Δ: {pct_fmt(row.get('delta_vs_30d_pct'))}")
                with m3:
                    st.metric("90d Δ", pct_fmt(row.get("delta_vs_90d_pct")))
                    st.caption(f"2nd-lowest gap: {pct_fmt(row.get('gap_to_second_lowest_pct'))}")

                # Chart (lazy-loaded in expander)
                with st.expander("Price history (AED)"):
                    ts = fetch_series(row["url"])
                    if ts.empty:
                        st.info("No time-series data.")
                    else:
                        import altair as alt
                        chart = (
                            alt.Chart(ts)
                            .mark_line()
                            .encode(
                                x=alt.X("timestamp:T", title="Time (Asia/Beirut)"),
                                y=alt.Y("price:Q", title="Price (AED)"),
                                tooltip=["timestamp:T", "price:Q"],
                            )
                            .properties(height=220)
                        )
                        st.altair_chart(chart, use_container_width=True)
