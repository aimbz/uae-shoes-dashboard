# streamlit_app.py
# UAE Men Shoes — Global Lows (Supabase REST API + Streamlit)
# Stable filters in URL + Plotly (last 200 points) + diagnostics
# UI tweaks: clickable full-width image, 4 cards/row, equalized heights, responsive fonts

# --- Disable Streamlit file watcher BEFORE importing streamlit (avoids inotify limit crash) ---
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

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

# Global CSS: responsive fonts, equalized card blocks, clean spacing
st.markdown("""
<style>
/* -------- Responsive font sizes (scale down on narrow viewports) -------- */
:root{
  --brand-size: clamp(0.9rem, 1.2vw, 1.05rem);
  --title-size: clamp(0.85rem, 1.1vw, 0.98rem);
  --metric-value: clamp(0.95rem, 1.4vw, 1.25rem);
  --metric-label: clamp(0.70rem, 1.0vw, 0.85rem);
}

/* Tighter page padding */
.block-container { padding-top: 0.75rem; }

/* Metric sizes */
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: var(--metric-value); line-height: 1.15; }
div[data-testid="stMetric"] div[data-testid="stMetricLabel"] { font-size: var(--metric-label); color: #6b7280; }
div[data-testid="stMetricDelta"] { font-size: 0.9rem; }

/* Our custom pieces inside each card */
.card-brand{ font-weight: 600; font-size: var(--brand-size); margin: .35rem 0 .15rem; }
.card-title{
  font-size: var(--title-size);
  color: #374151;
  display: -webkit-box;
  -webkit-line-clamp: 2;         /* clamp to 2 lines to stabilize height */
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 2.6em;              /* reserve space so all cards line up */
  margin: 0 0 .4rem 0;
}

/* Thumbnail: fixed aspect ratio so rows align, full width, rounded */
.card-thumb{
  width: 100%;
  aspect-ratio: 1 / 1;            /* square box */
  object-fit: contain;
  background: #fff;
  border-radius: 12px;
  display: block;
}

/* Make the collapsed expander header compact */
button[aria-expanded="false"] p { margin-bottom: 0; }

/* Ensure Streamlit containers inside columns stretch nicely */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlock"]) { height: 100%; }

/* Small screens: nudge sizes down a bit more */
@media (max-width: 1100px){
  :root{
    --brand-size: clamp(0.85rem, 1.3vw, 1rem);
    --title-size: clamp(0.80rem, 1.05vw, 0.92rem);
    --metric-value: clamp(0.90rem, 1.3vw, 1.15rem);
    --metric-label: clamp(0.65rem, 0.9vw, 0.80rem);
  }
}
</style>
""", unsafe_allow_html=True)

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
PRICES_PRIMARY = "nam-uae-men-shoes-prices"     # cloud table
PRICES_FALLBACK = "prices"                      # local/dev table

HDR = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Prefer": "count=exact",
}


# ============================ Persistent state & query param helpers ============================

if "filters_applied" not in st.session_state:
    st.session_state["filters_applied"] = False
if "page" not in st.session_state:
    st.session_state["page"] = 0

def qp_get() -> dict:
    try:
        return {k: v for k, v in st.query_params.items()}
    except Exception:
        raw = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else "") for k, v in raw.items()}

def qp_set(new_params: dict):
    qp = {k: (",".join(v) if isinstance(v, list) else str(v)) for k, v in new_params.items() if v is not None}
    try:
        st.query_params.update(qp)
    except Exception:
        st.experimental_set_query_params(**qp)

def qp_clear():
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()

def encode_filters_to_qp():
    brands = st.session_state.get("w_brands", [])
    category = st.session_state.get("w_category")
    category_param = "" if (not category or category == "(Any)") else category
    hits = st.session_state.get("w_hits")
    price = st.session_state.get("w_price")
    order = st.session_state.get("w_order_by") or ""
    desc = 1 if st.session_state.get("w_order_desc", True) else 0
    page = st.session_state.get("page", 0)
    ps = st.session_state.get("w_page_size", 24)

    qp_set({
        "applied": "1",
        "brands": ",".join(brands) if brands else "",
        "category": category_param,
        "hits": f"{hits[0]}-{hits[1]}" if hits else "",
        "price": f"{price[0]}-{price[1]}" if price else "",
        "order": order,
        "desc": str(desc),
        "page": str(page),
        "ps": str(ps),
    })

def parse_range(s: str, cast=float):
    try:
        lo, hi = s.split("-", 1)
        return (cast(lo), cast(hi))
    except Exception:
        return None

def hydrate_from_qp(brands_opts, categories_opts, pmin, pmax, hmin, hmax):
    qp = qp_get()
    if not qp or (qp.get("applied") != "1" and not any(qp.get(k) for k in ("brands","category","order","price","hits"))):
        return

    if "w_brands" not in st.session_state and qp.get("brands"):
        picked = [b for b in qp["brands"].split(",") if b]
        st.session_state["w_brands"] = [b for b in picked if b in brands_opts]

    if "w_category" not in st.session_state:
        cat = qp.get("category", "")
        st.session_state["w_category"] = cat if cat in categories_opts else "(Any)"

    if "w_hits" not in st.session_state:
        r = parse_range(qp.get("hits",""), int)
        if r:
            lo, hi = r
            st.session_state["w_hits"] = (max(hmin, lo), min(hmax, hi))

    if "w_price" not in st.session_state:
        r = parse_range(qp.get("price",""), float)
        if r:
            lo, hi = r
            st.session_state["w_price"] = (max(pmin, lo), min(pmax, hi))

    if "w_order_by" not in st.session_state and qp.get("order"):
        st.session_state["w_order_by"] = qp["order"]
    if "w_order_desc" not in st.session_state and "desc" in qp:
        st.session_state["w_order_desc"] = (str(qp["desc"]) == "1")

    if "page" not in st.session_state:
        try:
            st.session_state["page"] = max(0, int(qp.get("page","0")))
        except Exception:
            pass
    if "w_page_size" not in st.session_state:
        try:
            st.session_state["w_page_size"] = int(qp.get("ps","24"))
        except Exception:
            pass

    st.session_state["filters_applied"] = True
    LOG.log("Hydrated from query params", qp)


# ============================ HTTP helper ============================

def http_get(url: str, params: dict, label: str = "") -> tuple[list, str | None]:
    t0 = time.perf_counter()
    if st.session_state.get("t_verbose", True):
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
        if st.session_state.get("t_verbose", True):
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


# ============================ Utilities ============================

def pg_in(values: list[str]) -> str:
    esc = [v.replace('"', '""') for v in values]
    return 'in.(' + ",".join([f'"{e}"' for e in esc]) + ')'

def pct_fmt(x) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "—"


def build_params(flt: dict, limit: int, offset: int) -> dict:
    p: dict[str, list | str] = {
        "select": "*",
        "has_higher": "eq.true",
        "order": f"{flt['order_by']}.{ 'desc' if flt['order_desc'] else 'asc'}",
        "limit": str(limit),
        "offset": str(offset),
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


# ============================ Series loader: latest N points per URL ============================

MAX_POINTS = 200  # last N timestamps per product URL

@st.cache_data(ttl=300, max_entries=512)
def fetch_series(item_url: str, n: int = MAX_POINTS) -> pd.DataFrame:
    params = {
        "select": "timestamp,price",
        "url": f"eq.{item_url}",
        "order": "timestamp.desc",  # newest first
        "limit": str(n),
    }

    data, _ = http_get(f"{REST}/{PRICES_PRIMARY}", params, label="fetch_series:lastN:primary")
    df = pd.DataFrame(data)

    if df.empty:
        data, _ = http_get(f"{REST}/{PRICES_FALLBACK}", params, label="fetch_series:lastN:fallback")
        df = pd.DataFrame(data)

    if df.empty:
        return df

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert("Asia/Beirut")
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    LOG.log("fetch_series result (lastN)", {"rows": len(df), "limit": n})
    return df


# ============================ Stateful widget helpers ============================

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

def multiselect_stateful(key, label, options, default_default=None, **kwargs):
    if key in st.session_state:
        return st.multiselect(label, options=options, key=key, **kwargs)
    else:
        return st.multiselect(label, options=options, default=(default_default or []), key=key, **kwargs)

def toggle_stateful(key, label, value_default=False, **kwargs):
    if key in st.session_state:
        return st.toggle(label, key=key, **kwargs)
    else:
        return st.toggle(label, value=value_default, key=key, **kwargs)

def select_slider_stateful(key, label, options, value_default=None, **kwargs):
    if key in st.session_state:
        return st.select_slider(label, options=options, key=key, **kwargs)
    else:
        return st.select_slider(label, options=options, value=value_default, key=key, **kwargs)


# ============================ Sidebar (single block; unique keys) ============================

with st.sidebar:
    st.markdown("### Diagnostics")
    st.toggle(
        "Verbose network logs",
        value=True,
        help="Log each HTTP request/response meta",
        key="t_verbose",
    )

    if st.button("Run connectivity test", key="btn_ping"):
        try:
            LOG.log("Ping: MV limit=1")
            sample, cr = http_get(f"{REST}/{MV}", {"select": "brand,latest_price", "limit": "1"}, label="ping_mv")
            LOG.log("Ping MV ok", {"content_range": cr, "sample_rows": len(sample)})
            st.success("Connectivity OK (MV). See logs below.")
        except Exception as e:
            LOG.log("Ping MV failed", {"error": str(e)})
            st.error(f"Ping failed: {e}")

    st.markdown("---")
    if st.button("Reset filters", key="btn_reset"):
        st.session_state["filters_applied"] = False
        st.session_state["page"] = 0
        for k in list(st.session_state.keys()):
            if k.startswith("w_"):
                del st.session_state[k]
        qp_clear()
        st.rerun()


# ============================ UI ============================

st.title("UAE Men Shoes — Current Global Lows")
st.caption("Data live from Supabase REST API (materialized view + time series).")

brands, categories, pmin, pmax, hmin, hmax = load_options()
hydrate_from_qp(brands, categories, pmin, pmax, hmin, hmax)

with st.form("filters_form"):
    st.subheader("Filters (choose, then Apply)")

    c1, c2, c3 = st.columns(3)
    with c1:
        chosen_brands = multiselect_stateful("w_brands", "Brands", options=brands)
        category = selectbox_stateful("w_category", "Category", options=["(Any)"] + categories, index_default=0)
    with c2:
        hits_range = slider_stateful("w_hits", "Min hits", hmin, max(hmin, hmax), (hmin, max(hmin, hmax)))
        price_range = slider_stateful(
            "w_price", "Price range (AED)",
            float(pmin), float(pmax or max(pmin, pmin + 1)),
            (float(pmin), float(pmax or max(pmin, pmin + 1))),
        )
    with c3:
        drop_prev = slider_stateful("w_drop_prev", "Drop vs previous (%)", -100, 100, (-100, 100))
        drop_30 = slider_stateful("w_drop_30", "Drop vs 30-day avg (%)", -100, 100, (-100, 100))
        drop_90 = slider_stateful("w_drop_90", "Drop vs 90-day avg (%)", -100, 100, (-100, 100))

    st.markdown("---")
    cA, cB, cC = st.columns(3)
    with cA:
        order_by = selectbox_stateful(
            "w_order_by", "Order by",
            options=[
                "delta_vs_30d_pct",
                "delta_vs_90d_pct",
                "drop_pct_vs_prev",
                "latest_price",
                "min_hits",
                "gap_to_second_lowest_pct",
                "days_since_first_low",
            ],
            index_default=0,
        )
    with cB:
        order_desc = toggle_stateful("w_order_desc", "Sort descending", True)
    with cC:
        page_size = select_slider_stateful("w_page_size", "Page size", options=[12, 24, 48, 96], value_default=24)

    submitted = st.form_submit_button("Apply Filters", use_container_width=True)
    if submitted:
        st.session_state["filters_applied"] = True
        st.session_state["page"] = 0
        encode_filters_to_qp()
        LOG.log("Form submitted", {"submitted": True})

if not st.session_state.get("filters_applied", False):
    st.info("↑ Set your filters, then click **Apply Filters**.")
    LOG.render()
    st.stop()

flt = {
    "brands": st.session_state.get("w_brands", []),
    "category": None if st.session_state.get("w_category") in (None, "(Any)") else st.session_state.get("w_category"),
    "min_hits": st.session_state.get("w_hits"),
    "price_range": st.session_state.get("w_price"),
    "drop_prev": st.session_state.get("w_drop_prev"),
    "drop_30": st.session_state.get("w_drop_30"),
    "drop_90": st.session_state.get("w_drop_90"),
    "order_by": st.session_state.get("w_order_by"),
    "order_desc": st.session_state.get("w_order_desc", True),
}

# ============================ Results ============================

# Pagination controls
page = st.session_state.get("page", 0)
prev_col, _, next_col = st.columns([1, 6, 1])
with prev_col:
    if st.button("⟵ Prev", disabled=(page <= 0), key="btn_prev"):
        st.session_state["page"] = max(0, page - 1)
        encode_filters_to_qp()
        st.rerun()
with next_col:
    if st.button("Next ⟶", key="btn_next"):
        st.session_state["page"] = page + 1
        encode_filters_to_qp()
        st.rerun()

page = st.session_state.get("page", 0)
page_size = st.session_state.get("w_page_size", 24)
LOG.log("Pagination", {"page": page, "page_size": page_size})

# Fetch
df, total = fetch_items(flt, page, page_size)
if df.empty:
    st.warning("No items match your filters.")
    LOG.log("No matches, stopping")
    LOG.render()
    st.stop()

st.caption(f"Matches: {total if total is not None else '—'}  •  Page {page + 1}")

# ============================ Cards grid (4 per row, equalized heights) ============================

def html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

ncols = 4  # 4 items per row; Streamlit stacks on very small screens
rows = math.ceil(len(df) / ncols)
LOG.log("Rendering grid", {"rows": rows, "n_items": len(df), "ncols": ncols})

for r in range(rows):
    cols = st.columns(ncols, gap="large")
    for c in range(ncols):
        i = r * ncols + c
        if i >= len(df):
            break
        row = df.iloc[i]
        with cols[c]:
            with st.container(border=True):
                # ---------- Clickable full-width image (keeps rows even via aspect ratio) ----------
                if row.get("image_link"):
                    link = row.get("url") or "#"
                    img = html_escape(row.get("image_link", ""))
                    alt = html_escape(row.get("title", "") or row.get("brand","") or "product")
                    st.markdown(
                        f'<a href="{link}" target="_blank" rel="noopener">'
                        f'<img class="card-thumb" src="{img}" alt="{alt}"/></a>',
                        unsafe_allow_html=True,
                    )

                # Brand & title with clamped height -> equalizes cards
                st.markdown(f'<div class="card-brand">{html_escape(row.get("brand",""))}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">{html_escape(row.get("title",""))}</div>', unsafe_allow_html=True)

                # Metrics (smaller, responsive)
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

                # Price history (last 200 points, cached)
                with st.expander("📈 Price History (AED)", expanded=False):
                    ts = fetch_series(row["url"], n=MAX_POINTS)
                    if ts.empty:
                        st.info("No time-series data.")
                    else:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ts["timestamp"], y=ts["price"], mode="lines+markers", name="Price"))
                        fig.update_layout(
                            xaxis_title="Date (Asia/Beirut)",
                            yaxis_title="AED",
                            margin=dict(l=10, r=10, t=30, b=10),
                            height=280,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    LOG.log("Rendered series chart (lastN)", {"url_preview": str(row.get('url',''))[:60] + "…", "points": len(ts)})

# Final: logs
LOG.render()
