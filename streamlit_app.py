# streamlit_app.py
# UAE Men Shoes — Global Lows (Supabase REST API + Streamlit)
# Plotly charts + lazy-loaded price history + diagnostics logs

import math
import os
import sys
import time
import json
import platform
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objs as go


# ============================ Diagnostics logger ============================

class DiagLog:
    def __init__(self, name="logs"):
        self.name = name
        if "_diag_log" not in st.session_state:
            st.session_state["_diag_log"] = []
        self.buf = st.session_state["_diag_log"]

    def log(self, msg, data=None):
        ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f %Z")
        entry = {"ts": ts, "msg": str(msg)}
        if data is not None:
            try:
                json.dumps(data)
                entry["data"] = data
            except Exception:
                entry["data"] = str(data)
        self.buf.append(entry)

    def clear(self):
        st.session_state["_diag_log"] = []
        self.buf = st.session_state["_diag_log"]

    def render(self):
        with st.expander("🧰 Logs (diagnostics)", expanded=True):
            st.caption("Detailed boot/runtime logs (safe: no secrets)")
            if not self.buf:
                st.write("No logs yet.")
                return
            for e in self.buf[-200:]:
                st.write(f"[{e['ts']}] {e['msg']}")
                if "data" in e:
                    try:
                        st.code(json.dumps(e["data"], indent=2))
                    except Exception:
                        st.code(str(e["data"]))


LOG = DiagLog()

def safe_pkg_version(mod_name):
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "missing"

def boot_diag():
    LOG.clear()
    LOG.log("Boot: starting app")
    LOG.log(
        "Environment",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "timezone": time.tzname,
            "streamlit": safe_pkg_version("streamlit"),
            "pandas": safe_pkg_version("pandas"),
            "requests": safe_pkg_version("requests"),
            "plotly": safe_pkg_version("plotly"),
        },
    )


# ============================ App config / Secrets ============================

st.set_page_config(page_title="UAE Men Shoes — Global Lows", layout="wide")
boot_diag()

SUPABASE_URL = (st.secrets.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY")

LOG.log(
    "Secrets presence check",
    {
        "SUPABASE_URL_set": bool(SUPABASE_URL),
        "SUPABASE_ANON_KEY_set": bool(SUPABASE_ANON_KEY),
        "url_preview": SUPABASE_URL[:40] + ("…" if len(SUPABASE_URL) > 40 else ""),
    },
)

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase secrets. Go to Settings → Secrets and set SUPABASE_URL + SUPABASE_ANON_KEY.")
    LOG.log("Fatal: missing secrets -> stopping")
    LOG.render()
    st.stop()

REST = f"{SUPABASE_URL}/rest/v1"
MV = "nam_uae_men_shoes_at_global_low"          # materialized view
PRICES_PRIMARY = 'nam-uae-men-shoes-prices'     # likely on your cloud DB
PRICES_FALLBACK = 'prices'                      # used in your local app

HDR = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Prefer": "count=exact",
}

# Sidebar diagnostics toggles
with st.sidebar:
    st.markdown("### Diagnostics")
    VERBOSE_NET = st.toggle("Verbose network logs", value=True, help="Log each HTTP request/response meta")
    TEST_PING = st.button("Run connectivity test")


# ============================ HTTP helper ============================

def http_get(url: str, params: dict, label: str = "") -> tuple[list, str | None]:
    """GET with detailed timing + error surfacing."""
    t0 = time.perf_counter()
    if VERBOSE_NET:
        LOG.log("HTTP GET start", {"label": label, "url": url, "params": params})
    try:
        r = requests.get(url, params=params, headers=HDR, timeout=60)
        meta = {
            "label": label,
            "status_code": r.status_code,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "ok": r.ok,
            "url": r.url,
            "content_length": len(r.content or b""),
            "content_range": r.headers.get("content-range"),
            "ratelimit-remaining": r.headers.get("x-ratelimit-remaining"),
        }
        if VERBOSE_NET:
            LOG.log("HTTP GET end", meta)

        if not r.ok:
            body_preview = r.text[:1000]
            LOG.log("HTTP error body (preview)", {"body": body_preview})
            st.error(f"Supabase REST error: {r.status_code}\n{body_preview}")
            st.stop()

        try:
            js = r.json()
        except Exception as e:
            LOG.log("JSON decode error", {"error": str(e), "body_preview": r.text[:1000]})
            st.error("Failed to decode JSON from REST response.")
            st.stop()

        return js, r.headers.get("content-range")

    except requests.RequestException as e:
        LOG.log("Network exception", {"label": label, "error": str(e)})
        st.error(f"Network error: {e}")
        st.stop()


# ============================ Utilities ============================

def pg_in(values: list[str]) -> str:
    esc = [v.replace('"', '""') for v in values]
    return 'in.(' + ",".join([f'"{e}"' for e in esc]) + ')'

def pct_fmt(x) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "—"


# ============================ Data loaders (cached) ============================

@st.cache_data(ttl=300)
def load_options(limit_first_page: int = 2000):
    LOG.log("load_options: fetching", {"limit": limit_first_page})
    data, _ = http_get(
        f"{REST}/{MV}",
        {"select": "brand,category,latest_price,min_hits", "limit": str(limit_first_page)},
        label="load_options",
    )
    df = pd.DataFrame(data)
    LOG.log("load_options: fetched rows", {"rows": len(df)})

    brands = sorted([b for b in df.get("brand", pd.Series(dtype=str)).dropna().unique().tolist() if b])
    cats = sorted([c for c in df.get("category", pd.Series(dtype=str)).dropna().unique().tolist() if c]) or ["Men UAE shoes"]

    if df.empty:
        LOG.log("load_options: empty dataframe")
        return brands, cats, 0.0, 0.0, 0, 0

    pmin, pmax = float(df["latest_price"].min()), float(df["latest_price"].max())
    hmin, hmax = int(df["min_hits"].min()), int(df["min_hits"].max())
    LOG.log("load_options: computed ranges", {"pmin": pmin, "pmax": pmax, "hmin": hmin, "hmax": hmax})
    return brands, cats, pmin, pmax, hmin, hmax


def build_params(flt: dict, limit: int, offset: int) -> dict:
    p: dict[str, list | str] = {
        "select": "*",
        "has_higher": "eq.true",
        "order": f"{flt['order_by']}.{ 'desc' if flt['order_desc'] else 'asc'}",
        "limit": str(limit),
        "offset": str(offset),
        # Prefer: count=exact is already in headers
    }
    if flt["brands"]:
        p["brand"] = pg_in(flt["brands"])
    if flt["category"]:
        p["category"] = f"eq.{flt['category']}"

    lo, hi = flt["min_hits"]
    p["min_hits"] = [f"gte.{lo}", f"lte.{hi}"]

    plo, phi = flt["price_range"]
    p["latest_price"] = [f"gte.{plo}", f"lte.{phi}"]

    for col, (lo_pct, hi_pct) in {
        "drop_pct_vs_prev": flt["drop_prev"],
        "delta_vs_30d_pct": flt["drop_30"],
        "delta_vs_90d_pct": flt["drop_90"],
    }.items():
        p[col] = [f"gte.{lo_pct/100.0}", f"lte.{hi_pct/100.0}"]

    LOG.log("build_params", {"limit": limit, "offset": offset, "order": p["order"]})
    return p


@st.cache_data(ttl=300)
def fetch_items(flt: dict, page: int, page_size: int) -> tuple[pd.DataFrame, int | None]:
    params = build_params(flt, page_size, page * page_size)
    t0 = time.perf_counter()
    data, cr = http_get(f"{REST}/{MV}", params, label="fetch_items")
    total = int(cr.split("/")[-1]) if cr and "/" in cr else None
    df = pd.DataFrame(data)
    LOG.log(
        "fetch_items result",
        {"rows": len(df), "total": total, "page": page, "page_size": page_size, "elapsed_s": round(time.perf_counter() - t0, 3)},
    )
    return df, total


@st.cache_data(ttl=300)
def fetch_series_from(table_name: str, item_url: str, limit: int = 5000) -> pd.DataFrame:
    params = {
        "select": "timestamp,price",
        "url": f"eq.{item_url}",   # let requests encode
        "order": "timestamp.asc",
        "limit": str(limit),
    }
    t0 = time.perf_counter()
    data, _ = http_get(f"{REST}/{table_name}", params, label=f"fetch_series:{table_name}")
    df = pd.DataFrame(data)
    if not df.empty:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = ts.dt.tz_convert("Asia/Beirut")
    LOG.log("fetch_series result", {
        "table": table_name,
        "url_preview": item_url[:60] + "…",
        "rows": len(df),
        "elapsed_s": round(time.perf_counter() - t0, 3)
    })
    return df


@st.cache_data(ttl=300)
def fetch_series(item_url: str, limit: int = 5000) -> pd.DataFrame:
    df = fetch_series_from(PRICES_PRIMARY, item_url, limit)
    if df.empty:
        LOG.log("Primary prices table returned 0 rows; trying fallback", {"primary": PRICES_PRIMARY, "fallback": PRICES_FALLBACK})
        df = fetch_series_from(PRICES_FALLBACK, item_url, limit)
    return df


# ============================ Optional connectivity test ============================

if TEST_PING:
    try:
        LOG.log("Ping: MV limit=1")
        sample, cr = http_get(f"{REST}/{MV}", {"select": "brand,latest_price", "limit": "1"}, label="ping_mv")
        LOG.log("Ping MV ok", {"content_range": cr, "sample_rows": len(sample)})
        st.success("Connectivity OK (MV). See logs below.")
    except Exception as e:
        LOG.log("Ping MV failed", {"error": str(e)})
        st.error(f"Ping failed: {e}")


# ============================ UI ============================

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
    LOG.log("Form submitted", {"submitted": submitted})

if not submitted:
    st.info("↑ Set your filters, then click **Apply Filters**.")
    LOG.render()
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

# Pagination state
page = st.session_state.get("page", 0)
prev_col, _, next_col = st.columns([1, 6, 1])
with prev_col:
    if st.button("⟵ Prev", disabled=(page <= 0)):
        page = max(0, page - 1)
with next_col:
    if st.button("Next ⟶"):
        page = page + 1
st.session_state["page"] = page
LOG.log("Pagination", {"page": page})

# Data fetch
df, total = fetch_items(flt, page, page_size)
if df.empty:
    st.warning("No items match your filters.")
    LOG.log("No matches, stopping")
    LOG.render()
    st.stop()

st.caption(f"Matches: {total if total is not None else '—'}  •  Page {page + 1}")

# Cards grid
ncols = 3
rows = math.ceil(len(df) / ncols)
LOG.log("Rendering grid", {"rows": rows, "n_items": len(df)})

for r in range(rows):
    cols = st.columns(ncols, gap="large")
    for c in range(ncols):
        i = r * ncols + c
        if i >= len(df):
            break
        row = df.iloc[i]
        with cols[c]:
            with st.container(border=True):
                left, right = st.columns([1, 2])
                with left:
                    if row.get("image_link"):
                        st.image(row["image_link"], use_column_width=True)
                with right:
                    st.markdown(f"**{row.get('brand','')}**")
                    st.markdown(row.get("title", ""))
                    if row.get("url"):
                        st.link_button("Open product", row["url"], use_container_width=True)

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

                # ---- Lazy-loaded Plotly price history (no DB hit until opted in) ----
                with st.expander("📈 Price History (AED)", expanded=False):
                    chk_key = f"show_hist_{i}"
                    show = st.checkbox("Show price history", key=chk_key)
                    if show:
                        with st.spinner("Fetching history…"):
                            ts = fetch_series(row["url"])
                        if ts.empty:
                            st.info("No time-series data.")
                        else:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=ts["timestamp"],
                                y=ts["price"],
                                mode="lines+markers",
                                name="Price",
                            ))
                            fig.update_layout(
                                title="Price Over Time",
                                xaxis_title="Date (Asia/Beirut)",
                                yaxis_title="AED",
                                margin=dict(l=10, r=10, t=30, b=10),
                                height=300,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        LOG.log("Rendered series chart (lazy, plotly)", {
                            "url_preview": row["url"][:60] + "…",
                            "checked": True
                        })

# Final: render logs
LOG.render()
