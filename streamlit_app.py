import math, requests, pandas as pd, streamlit as st
from urllib.parse import quote

st.set_page_config(page_title="UAE Men Shoes — Global Lows", layout="wide")

BASE = st.secrets["SUPABASE_URL"].rstrip("/")
REST = f"{BASE}/rest/v1"
API_KEY = st.secrets["SUPABASE_ANON_KEY"]
MV = "nam_uae_men_shoes_at_global_low"
PRICES = 'nam-uae-men-shoes-prices'
HDR = {"apikey": API_KEY, "Authorization": f"Bearer {API_KEY}", "Prefer": "count=exact"}

def _get(url, params):
    r = requests.get(url, params=params, headers=HDR, timeout=60)
    r.raise_for_status()
    return r.json(), r.headers.get("content-range")

@st.cache_data(ttl=300)
def options():
    data, _ = _get(f"{REST}/{MV}", {"select":"brand,category,latest_price,min_hits","limit":"10000"})
    df = pd.DataFrame(data)
    brands = sorted([b for b in df["brand"].dropna().unique().tolist() if b])
    cats = sorted([c for c in df["category"].dropna().unique().tolist() if c]) or ["Men UAE shoes"]
    pmin, pmax = (float(df["latest_price"].min()), float(df["latest_price"].max())) if not df.empty else (0.0, 0.0)
    hmin, hmax = (int(df["min_hits"].min()), int(df["min_hits"].max())) if not df.empty else (0, 0)
    return brands, cats, pmin, pmax, hmin, hmax

def pct_fmt(v): 
    try: return f"{float(v)*100:.1f}%"
    except: return "—"

def build_params(f, limit, offset):
    p = {"select":"*", "has_higher":"eq.true", "limit":str(limit), "offset":str(offset),
         "order": f"{f['order_by']}.{ 'desc' if f['order_desc'] else 'asc'}", "count":"exact"}
    if f["brands"]:
        p["brand"] = f'in.({",".join([f"\\"{b.replace(\\"\\",\\"\\\\\\")}\\"" for b in f["brands"]])})'
    if f["category"]: p["category"] = f"eq.{f['category']}"
    p["min_hits"] = [f"gte.{f['min_hits'][0]}", f"lte.{f['min_hits'][1]}"]
    p["latest_price"] = [f"gte.{f['price_range'][0]}", f"lte.{f['price_range'][1]}"]
    for col, (lo, hi) in {
        "drop_pct_vs_prev": f["drop_prev"],
        "delta_vs_30d_pct": f["drop_30"],
        "delta_vs_90d_pct": f["drop_90"],
    }.items():
        p[col] = [f"gte.{lo/100.0}", f"lte.{hi/100.0}"]  # percent → fraction
    return p

@st.cache_data(ttl=300)
def fetch_items(f, page, page_size):
    params = build_params(f, page_size, page*page_size)
    data, cr = _get(f"{REST}/{MV}", params)
    total = int(cr.split("/")[-1]) if cr and "/" in cr else None
    return pd.DataFrame(data), total

@st.cache_data(ttl=300)
def timeseries(url):
    data, _ = _get(f"{REST}/{PRICES}", {
        "select":"timestamp,price",
        "url": f"eq.{quote(url, safe='')}",
        "order":"timestamp.asc",
        "limit":"5000"
    })
    df = pd.DataFrame(data)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ---------- UI ----------
st.title("UAE Men Shoes — Current Global Lows (via Supabase API)")
brands, cats, pmin, pmax, hmin, hmax = options()

with st.form("filters"):
    c1, c2, c3 = st.columns(3)
    with c1:
        f_brands = st.multiselect("Brands", options=brands)
        f_cat = st.selectbox("Category", ["(Any)"] + cats)
    with c2:
        f_hits = st.slider("Min hits", hmin, max(hmin, hmax), (hmin, max(hmin, hmax)))
        f_price = st.slider("Price (AED)", float(pmin), float(pmax or max(pmin, pmin+1)),
                            (float(pmin), float(pmax or max(pmin, pmin+1))))
    with c3:
        f_prev = st.slider("Drop vs previous (%)", -100, 100, (-100, 100))
        f_30   = st.slider("Drop vs 30d avg (%)", -100, 100, (-100, 100))
        f_90   = st.slider("Drop vs 90d avg (%)", -100, 100, (-100, 100))
    st.markdown("---")
    cA, cB, cC = st.columns(3)
    with cA:
        order_by = st.selectbox("Order by", [
            "delta_vs_30d_pct","delta_vs_90d_pct","drop_pct_vs_prev",
            "latest_price","min_hits","gap_to_second_lowest_pct","days_since_first_low"
        ])
    with cB: order_desc = st.toggle("Sort descending", True)
    with cC: page_size = st.select_slider("Page size", [12,24,48,96], 24)
    submitted = st.form_submit_button("Apply Filters")

if not submitted:
    st.info("Choose filters then click **Apply Filters**.")
    st.stop()

f = {
    "brands": f_brands,
    "category": None if f_cat == "(Any)" else f_cat,
    "min_hits": f_hits,
    "price_range": f_price,
    "drop_prev": f_prev,
    "drop_30": f_30,
    "drop_90": f_90,
    "order_by": order_by,
    "order_desc": order_desc,
}

page = st.session_state.get("page", 0)
left, mid, right = st.columns([1,2,1])
with left:
    if st.button("⟵ Prev", disabled=page<=0): page = max(0, page-1)
with right:
    if st.button("Next ⟶"): page += 1
st.session_state["page"] = page

df, total = fetch_items(f, page, page_size)
if df.empty:
    st.warning("No items match your filters.")
    st.stop()

st.caption(f"Matches: {total if total is not None else '—'} • Page {page+1}")

ncols = 3
rows = math.ceil(len(df)/ncols)
for r in range(rows):
    cols = st.columns(ncols, gap="large")
    for c in range(ncols):
        i = r*ncols + c
        if i >= len(df): break
        row = df.iloc[i]
        with cols[c]:
            with st.container(border=True):
                a, b = st.columns([1,2])
                with a:
                    if row.get("image_link"): st.image(row["image_link"], use_container_width=True)
                with b:
                    st.markdown(f"**{row.get('brand','')}**")
                    st.markdown(row.get("title",""))
                    if row.get("url"): st.link_button("Open product", row["url"], use_container_width=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Latest (AED)", f"{row.get('latest_price',0):.2f}")
                    st.caption(f"Min hits: {row.get('min_hits','—')}")
                with m2:
                    st.metric("Drop vs prev", pct_fmt(row.get("drop_pct_vs_prev")))
                    st.caption(f"30d Δ: {pct_fmt(row.get('delta_vs_30d_pct'))}")
                with m3:
                    st.metric("90d Δ", pct_fmt(row.get('delta_vs_90d_pct')))
                    st.caption(f"2nd‑lowest gap: {pct_fmt(row.get('gap_to_second_lowest_pct'))}")
                with st.expander("Price history"):
                    ts = timeseries(row["url"])
                    if ts.empty:
                        st.info("No time‑series data.")
                    else:
                        import altair as alt
                        st.altair_chart(
                            alt.Chart(ts).mark_line().encode(
                                x=alt.X("timestamp:T", title="Time"),
                                y=alt.Y("price:Q", title="Price (AED)"),
                                tooltip=["timestamp:T","price:Q"]
                            ).properties(height=220),
                            use_container_width=True
                        )
