# streamlit_app.py
# UAE Men Shoes — Global Lows (Supabase REST API + Streamlit)
# Stable filters in URL + Plotly (last 200 points) + diagnostics
# UI: clickable full-width image, 4 cards/row, equalized heights, responsive fonts
# Costs: Shipping/Margin/Cashback with USD conversion + landed USD (persist across Apply)

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # avoid inotify limit crash

import math
import sys
import time
import json
import platform
from datetime import datetime, timezone

import pandas as pd
import requests
import plotly.graph_objs as go
import streamlit as st


# ============================ Diagnostics logger ============================

class DiagLog:
    def __init__(self, name="logs"):
        self.name = name
        self._items: list[dict] = []

    def log(self, tag: str, data: dict | None = None):
        self._items.append({"ts": datetime.now(timezone.utc).isoformat(), "tag": tag, "data": data or {}})

    def render(self):
        if not self._items:
            st.caption("No logs.")
            return
        st.write(f"**Diagnostics ({len(self._items)})**")
        for e in self._items[-200:]:
            with st.expander(f"[{e['ts']}] {e['tag']}", expanded=False):
                try:
                    st.code(json.dumps(e["data"], indent=2))
                except Exception:
                    st.code(str(e["data"]))


LOG = DiagLog()

def safe_pkg_version(mod_name):
    try:
        mod = __import__(mod_name)
        v = getattr(mod, "__version__", "unknown")
    except Exception:
        v = "n/a"
    return v

def boot_diag():
    st.caption(
        f"Py {platform.python_version()} • "
        f"streamlit {safe_pkg_version('streamlit')} • "
        f"pandas {safe_pkg_version('pandas')} • "
        f"plotly {safe_pkg_version('plotly')} • "
        f"requests {safe_pkg_version('requests')}"
    )


# ============================ CSS ============================

st.markdown(
    """
    <style>
:root{
  --card-gap: .9rem;
  --title-size: clamp(13px, 1.2vw + .3rem, 16px);
  --brand-size: clamp(11px, 1vw + .2rem, 13px);
}
.card-grid{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--card-gap);
}
@media (max-width: 1400px){ .card-grid{ grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 980px){ .card-grid{ grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 620px){ .card-grid{ grid-template-columns: 1fr; } }

.card{
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: .8rem .9rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #fff;
}
.card a{ text-decoration: none; }
.card .brand{
  font-size: var(--brand-size);
  color: #374151;
  margin-bottom: .15rem;
}
.card .title{
  font-size: var(--title-size);
  color: #1f2937;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 2.6em;
  margin: 0 0 .5rem 0;
}
.card-thumb{
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: contain;
  background: #fff;
  border-radius: 12px;
  display: block;
}
.usd-red   { color:#dc2626; font-weight:700; margin:.15rem 0 0; }
.usd-landed { font-weight:700; margin:.15rem 0 .2rem; }
.small-cap { color:#6b7280; font-size: .85rem; }
.section-gap { margin-top:.35rem; }
button[aria-expanded="false"] p { margin-bottom: 0; }
div[data-testid="stExpander"] div[role="button"] p { font-size: .9rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================ Secrets & REST ============================

st.title("UAE Men Shoes — Global Lows")

SUPABASE_URL = (st.secrets.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase secrets. Go to Settings → Secrets and set SUPABASE_URL + SUPABASE_ANON_KEY.")
    LOG.log("Fatal: missing secrets -> stopping")
    LOG.render()
    st.stop()

REST = f"{SUPABASE_URL}/rest/v1"

# ⬇️  public anon key only; ensure RLS policies allow read for anon
HDR = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Prefer": "count=exact",
}

MV = "nam_global_shoes_at_global_low"
PRICES_TABLES = [
    "nam-uae-men-shoes-prices",
    "nam-uae-women-shoes-prices",
    "nam-ksa-men-shoes-prices",
    "prices",  # optional fallback if you still maintain it
]


# ============================ State & query params ======================

def qp_get() -> dict:
    return st.query_params.to_dict()

def qp_set(**kwargs):
    # merge into current
    qp = {**st.query_params, **{k: v for k, v in kwargs.items() if v is not None}}
    st.query_params.clear()
    st.query_params.update(qp)

def qp_clear(keys: list[str]):
    qp = dict(st.query_params)
    for k in keys:
        qp.pop(k, None)
    st.query_params.clear()
    st.query_params.update(qp)

def encode_filters_to_qp(flt: dict):
    qp_set(
        b=",".join(flt["brands"]) if flt["brands"] else None,
        c=flt["category"] or None,
        p=f"{int(flt['price_range'][0])}-{int(flt['price_range'][1])}",
        h=f"{int(flt['min_hits'][0])}-{int(flt['min_hits'][1])}",
        d=f"{flt['drop_prev'][0]}-{flt['drop_prev'][1]}",
        d30=f"{flt['drop_30'][0]}-{flt['drop_30'][1]}",
        d90=f"{flt['drop_90'][0]}-{flt['drop_90'][1]}",
        ob=flt["order_by"],
        od="1" if flt["order_desc"] else "0",
        ps=str(flt["page_size"]),
        ship=str(flt["ship_usd"]),
        m=str(flt["margin_pct"]),
        cb=str(flt["cashback_pct"]),
        page=str(flt["page"]),
    )

def parse_range(s: str | None, cast=int, default=(0, 0)):
    if not s: return default
    try:
        lo, hi = s.split("-", 1)
        return cast(lo), cast(hi)
    except Exception:
        return default

def parse_float(s, default):
    try:
        return float(s)
    except Exception:
        return default

def hydrate_from_qp():
    qp = qp_get()
    # basic filters
    if "w_order_by" not in st.session_state:
        st.session_state["w_order_by"] = qp.get("ob", "delta_vs_30d_pct")
    if "w_order_desc" not in st.session_state:
        st.session_state["w_order_desc"] = qp.get("od", "1") == "1"
    if "w_page" not in st.session_state:
        try: st.session_state["w_page"] = int(qp.get("page","0"))
        except Exception: pass
    if "w_page_size" not in st.session_state:
        try:
            st.session_state["w_page_size"] = int(qp.get("ps","24"))
        except Exception:
            pass
    # hydrate costs if not set
    if "w_ship_usd" not in st.session_state or st.session_state["w_ship_usd"] is None:
        st.session_state["w_ship_usd"] = parse_float(qp.get("ship", st.session_state["w_ship_usd"]), 7.0)
    if "w_margin_pct" not in st.session_state or st.session_state["w_margin_pct"] is None:
        st.session_state["w_margin_pct"] = parse_float(qp.get("m", st.session_state["w_margin_pct"]), 25.0)
    if "w_cashback_pct" not in st.session_state or st.session_state["w_cashback_pct"] is None:
        st.session_state["w_cashback_pct"] = parse_float(qp.get("cb", st.session_state["w_cashback_pct"]), 0.0)

    st.session_state["filters_applied"] = True


# ============================ HTTP ============================

def http_get(url: str, params: dict, label: str = "") -> tuple[list, str | None]:
    """HTTP GET with soft-fail: logs + warning instead of st.stop() on errors."""
    try:
        r = requests.get(url, params=params, headers=HDR, timeout=30)
        r.raise_for_status()
        return r.json(), r.headers.get("content-range")
    except requests.RequestException as e:
        try:
            LOG.log("HTTP error", {"label": label, "url": url, "params": params, "error": str(e)})
        except Exception:
            pass
        st.warning(f"{label}: temporary data error. Check logs.")
        return [], None


# ============================ Data loaders (cached) ============================

@st.cache_data(ttl=300)
def _scan_distinct_values_from_mv(col: str, page_size: int = 300, max_rows: int = 3000) -> list[str]:
    values: set[str] = set()
    offset = 0
    total = None
    seen = 0
    while True:
        params = {"select": col, col: "not.is.null", "order": f"{col}.asc", "limit": str(page_size), "offset": str(offset)}
        chunk, content_range = http_get(f"{REST}/{MV}", params, label=f"scan:{col}")
        if total is None and content_range and "/" in content_range:
            try: total = int(content_range.split("/")[-1])
            except Exception: total = None
        if not chunk: break
        for r in chunk:
            v = r.get(col)
            if v is not None: values.add(str(v))
        offset += len(chunk)
        seen += len(chunk)
        if (total is not None and offset >= total) or seen >= max_rows: break
    return sorted(values, key=lambda s: s.lower())

@st.cache_data(ttl=300)
def load_options():
    brands = _scan_distinct_values_from_mv("brand", page_size=300, max_rows=3000)
    categories = _scan_distinct_values_from_mv("category", page_size=300, max_rows=3000)
    def minmax(col, cast=float):
        lo_rows, _ = http_get(f"{REST}/{MV}", {"select": col, "order": f"{col}.asc", "limit": "1"}, label=f"min_{col}")
        hi_rows, _ = http_get(f"{REST}/{MV}", {"select": col, "order": f"{col}.desc", "limit": "1"}, label=f"max_{col}")
        lo = cast(lo_rows[0][col]) if lo_rows and lo_rows[0].get(col) is not None else (0 if cast is int else 0.0)
        hi = cast(hi_rows[0][col]) if hi_rows and hi_rows[0].get(col) is not None else (0 if cast is int else 0.0)
        return lo, hi
    pmin, pmax = minmax("latest_price", float)
    hmin, hmax = minmax("min_hits", int)
    return brands, categories, float(pmin), float(pmax), int(hmin), int(hmax)


# ============================ Utilities ============================

def pg_in(values: list[str]) -> str:
    # PostgREST "in.(a,b,c)" format; ensure URL-safe (quotes not needed for basic strings)
    esc = ",".join([str(v).replace(",", " ") for v in values if v])
    return f"in.({esc})"

def pct_fmt(x):
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "—"

def clamp01(x): return max(0.0, min(1.0, float(x)))

def build_params(flt: dict, limit: int, offset: int) -> dict:
    direction = "desc" if flt["order_desc"] else "asc"
    p: dict[str, list | str] = {
        "select": "*",
        "has_higher": "eq.true",
        "order": f"{flt['order_by']}.{direction}.nullslast,latest_price.asc,brand.asc",
        "limit": str(limit),
        "offset": str(offset),
    }
    if flt["brands"]:   p["brand"] = pg_in(flt["brands"])
    if flt["category"]: p["category"] = f"eq.{flt['category']}"
    lo, hi = flt["min_hits"];      p["min_hits"] = [f"gte.{lo}", f"lte.{hi}"]
    plo, phi = flt["price_range"]; p["latest_price"] = [f"gte.{plo}", f"lte.{phi}"]
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
    data, cr = http_get(f"{REST}/{MV}", params, label="fetch_items")
    total = int(cr.split("/")[-1]) if cr and "/" in cr else None
    df = pd.DataFrame(data)
    return df, total


# ============================ Series loader: latest N points per URL ============================

MAX_POINTS = 200

@st.cache_data(ttl=300, max_entries=512)
def fetch_series(item_url: str, n: int = MAX_POINTS) -> pd.DataFrame:
    """
    Fetch the last N (timestamp, price) points for a URL by searching the
    regional prices tables in order, falling back to 'prices' if present.
    """
    params_base = {
        "select": "timestamp,price",
        "url": f"eq.{item_url}",
        "order": "timestamp.desc",
        "limit": str(n),
    }

    df = pd.DataFrame()
    for table in PRICES_TABLES:
        data, _ = http_get(f"{REST}/{table}", params_base, label=f"fetch_series:{table}")
        df = pd.DataFrame(data)
        if not df.empty:
            break

    if df.empty:
        return df

    # normalize tz to Beirut for plotting
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Beirut")
    return (
        df[["timestamp", "price"]]
          .dropna(subset=["timestamp"])
          .drop_duplicates(subset=["timestamp"], keep="last")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )


# ============================ Sidebar ============================

with st.sidebar:
    st.markdown("### Diagnostics")
    st.toggle("Verbose network logs", key="t_verbose", help="Log each HTTP request/response meta")
    if st.button("Run connectivity test", key="btn_ping"):
        try:
            sample, cr = http_get(f"{REST}/{MV}", {"select": "brand,latest_price", "limit": "1"}, label="ping_mv")
            st.success("Connectivity OK (MV). See logs below.")
            LOG.log("Ping MV ok", {"content_range": cr, "sample_rows": len(sample)})
        except Exception as e:
            LOG.log("Ping MV failed", {"error": str(e)})
            st.error("Ping failed.")
    with st.expander("Logs", expanded=False):
        LOG.render()

boot_diag()


# ============================ Filters ============================

brands, categories, pmin, pmax, hmin, hmax = load_options()
hydrate_from_qp()

def number_stateful(key, label, value_default, **kwargs):
    if key in st.session_state:
        return st.number_input(label, key=key, **kwargs)
    else:
        return st.number_input(label, value=value_default, key=key, **kwargs)

def slider_stateful(key, label, min_value, max_value, value_default, **kwargs):
    if key in st.session_state:
        return st.slider(label, min_value=min_value, max_value=max_value, key=key, **kwargs)
    else:
        return st.slider(label, min_value=min_value, max_value=max_value, value=value_default, key=key, **kwargs)

def selectbox_stateful(key, label, options, index_default=0, **kwargs):
    if key in st.session_state:
        return st.selectbox(label, options=options, key=key, **kwargs)
    else:
        return st.selectbox(label, options=options, index=index_default, key=key, **kwargs)

def multiselect_stateful(key, label, options, default=None, **kwargs):
    if key in st.session_state:
        return st.multiselect(label, options=options, key=key, **kwargs)
    else:
        return st.multiselect(label, options=options, default=default or [], key=key, **kwargs)

def toggle_stateful(key, label, value_default=False, **kwargs):
    if key in st.session_state:
        return st.toggle(label, key=key, **kwargs)
    else:
        return st.toggle(label, value=value_default, key=key, **kwargs)

def select_slider_stateful(key, label, options, value_default, **kwargs):
    if key in st.session_state:
        return st.select_slider(label, options=options, key=key, **kwargs)
    else:
        return st.select_slider(label, options=options, value=value_default, key=key, **kwargs)


with st.form("filters"):
    st.subheader("Filters")

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_brands = multiselect_stateful("w_brands", "Brands", brands, default=brands[:12])
    with c2:
        sel_cat = selectbox_stateful("w_category", "Category", [""] + categories, index_default=0)
    with c3:
        price_range = slider_stateful("w_price_range", "Price (AED)", int(pmin), int(pmax), (int(pmin), int(pmax)))

    c4, c5, c6 = st.columns(3)
    with c4:
        min_hits = slider_stateful("w_min_hits", "Min Hits", int(hmin), max(100, int(hmax)), (int(hmin), int(hmax)))
    with c5:
        drop_prev = slider_stateful("w_drop_prev", "Drop vs prev (%)", -100, 100, (-100, 100))
    with c6:
        drop_30 = slider_stateful("w_drop_30", "Drop vs 30-day avg (%)", -100, 100, (-100, 100))

    st.markdown("---")
    cA, cB, cC = st.columns(3)
    with cA:
        order_by = selectbox_stateful("w_order_by", "Order by",
                                      options=["delta_vs_30d_pct","delta_vs_90d_pct","drop_pct_vs_prev",
                                               "latest_price","min_hits","gap_to_second_lowest_pct","days_since_first_low"],
                                      index_default=0)
    with cB:
        order_desc = toggle_stateful("w_order_desc", "Sort descending", True)
    with cC:
        page_size = select_slider_stateful("w_page_size", "Page size", options=[12, 24, 48, 96], value_default=24)

    cX, cY, cZ = st.columns(3)
    with cX:
        ship_usd = number_stateful("w_ship_usd", "Shipping USD", 7.0, step=0.5)
    with cY:
        margin_pct = number_stateful("w_margin_pct", "Margin %", 25.0, step=1.0)
    with cZ:
        cashback_pct = number_stateful("w_cashback_pct", "Cashback %", 0.0, step=0.5)

    apply = st.form_submit_button("Apply", use_container_width=True)
    if apply:
        st.session_state["w_page"] = 0
        encode_filters_to_qp({
            "brands": sel_brands,
            "category": sel_cat,
            "price_range": price_range,
            "min_hits": min_hits,
            "drop_prev": drop_prev,
            "drop_30": drop_30,
            "drop_90": (-100, 100),
            "order_by": order_by,
            "order_desc": order_desc,
            "page_size": page_size,
            "ship_usd": ship_usd,
            "margin_pct": margin_pct,
            "cashback_pct": cashback_pct,
            "page": 0,
        })
        st.rerun()


flt = {
    "brands": st.session_state.get("w_brands", []),
    "category": st.session_state.get("w_category", ""),
    "price_range": st.session_state.get("w_price_range", (0, 0)),
    "min_hits": st.session_state.get("w_min_hits", (0, 0)),
    "drop_prev": st.session_state.get("w_drop_prev", (-100, 100)),
    "drop_30": st.session_state.get("w_drop_30", (-100, 100)),
    "drop_90": (-100, 100),
    "order_by": st.session_state.get("w_order_by", "delta_vs_30d_pct"),
    "order_desc": st.session_state.get("w_order_desc", True),
    "page_size": st.session_state.get("w_page_size", 24),
    "page": st.session_state.get("w_page", 0),
    "ship_usd": st.session_state.get("w_ship_usd", 7.0),
    "margin_pct": st.session_state.get("w_margin_pct", 25.0),
    "cashback_pct": st.session_state.get("w_cashback_pct", 0.0),
}

st.caption(f"Page {flt['page']+1} • Page size {flt['page_size']}")


# ============================ Fetch & Render ============================

df, total = fetch_items(flt, page=flt["page"], page_size=flt["page_size"])
count = len(df)

if count == 0:
    st.info("No items match the current filters.")
else:
    st.write(f"**{total or count} items** (showing {count})")

    # Grid
    st.markdown('<div class="card-grid">', unsafe_allow_html=True)

    for _, row in df.iterrows():
        # card start
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # image + link
        link = row.get("link") or row.get("url") or "#"
        img = row.get("image_url") or row.get("image") or ""
        brand = str(row.get("brand", "") or "")
        title = str(row.get("title", "") or "")

        if img:
            st.markdown(f'<a href="{link}" target="_blank"><img class="card-thumb" src="{img}"/></a>',
                        unsafe_allow_html=True)

        # brand + title
        st.markdown(f'<div class="brand">{brand}</div>', unsafe_allow_html=True)
        st.markdown(f'<a href="{link}" target="_blank"><div class="title">{title}</div></a>', unsafe_allow_html=True)

        # prices
        latest = row.get("latest_price")
        min_price = row.get("min_price")
        second_low = row.get("second_lowest_price")
        latest_usd = row.get("latest_usd")
        landed_usd = row.get("landed_usd")

        if latest is not None:
            st.markdown(f"**AED {latest:,.2f}**")
        if latest_usd is not None:
            st.markdown(f'<div class="usd-red">~ ${latest_usd:,.2f}</div>', unsafe_allow_html=True)
        if landed_usd is not None:
            st.markdown(f'<div class="usd-landed">Landed ~ ${landed_usd:,.2f}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="small-cap">All-time low: AED {min_price:,.2f} • 2nd-lowest: AED {second_low:,.2f}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="small-cap">30d Δ: {pct_fmt(row.get("delta_vs_30d_pct"))}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="small-cap">90d Δ: {pct_fmt(row.get("delta_vs_90d_pct"))}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="small-cap">2nd-lowest gap: {pct_fmt(row.get("gap_to_second_lowest_pct"))}</div>',
            unsafe_allow_html=True
        )

        # time-series (guard for missing URL)
        with st.expander("📈 Price History (AED)", expanded=False):
            item_url = row.get("url")
            if not item_url:
                st.info("No URL available for time-series lookup.")
            else:
                ts = fetch_series(item_url, n=MAX_POINTS)
                if ts.empty:
                    st.info("No time-series data.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts["timestamp"], y=ts["price"], mode="lines+markers", name="Price"))
                    fig.update_layout(xaxis_title="Date (Asia/Beirut)", yaxis_title="AED",
                                      margin=dict(l=10, r=10, t=30, b=10), height=280)
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)  # card end

    st.markdown('</div>', unsafe_allow_html=True)  # grid end

    # Pager
    left, mid, right = st.columns(3)
    with left:
        if st.button("⟵ Prev", disabled=flt["page"] <= 0, use_container_width=True):
            st.session_state["w_page"] = max(0, flt["page"] - 1)
            qp_set(page=str(st.session_state["w_page"]))
            st.rerun()
    with right:
        more = (total is None) or ((flt["page"] + 1) * flt["page_size"] < (total or 0))
        if st.button("Next ⟶", disabled=not more, use_container_width=True):
            st.session_state["w_page"] = flt["page"] + 1
            qp_set(page=str(st.session_state["w_page"]))
            st.rerun()
