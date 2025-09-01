# streamlit_app.py
# UAE Men Shoes — Global Lows (Supabase REST API + Streamlit)
# Faster first paint: no global count=exact, narrower selects, capped scans, lazy plotly import

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import json
import platform
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="UAE Men Shoes — Global Lows", layout="wide")

# ───────────────────────── Diagnostics ─────────────────────────

class DiagLog:
    def __init__(self): self._items = []
    def log(self, tag, data=None):
        self._items.append({"ts": datetime.now(timezone.utc).isoformat(), "tag": tag, "data": data or {}})
    def render(self):
        if not self._items:
            st.caption("No logs.")
            return
        st.write(f"**Diagnostics ({len(self._items)})**")
        for e in self._items[-200:]:
            with st.expander(f"[{e['ts']}] {e['tag']}", expanded=False):
                try: st.code(json.dumps(e["data"], indent=2))
                except Exception: st.code(str(e["data"]))

LOG = DiagLog()

def _ver(mod):
    try:
        m = __import__(mod); return getattr(m, "__version__", "unknown")
    except Exception:
        return "n/a"

st.caption(f"Py • streamlit {_ver('streamlit')} • pandas {_ver('pandas')} • requests {_ver('requests')}")

# ───────────────────────── CSS ─────────────────────────

st.markdown("""
<style>
:root{ --card-gap:.9rem; --title-size:clamp(13px,1.2vw + .3rem,16px); --brand-size:clamp(11px,1vw + .2rem,13px);}
.card-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:var(--card-gap);}
@media (max-width:1400px){ .card-grid{ grid-template-columns:repeat(3,1fr);} }
@media (max-width:980px){ .card-grid{ grid-template-columns:repeat(2,1fr);} }
@media (max-width:620px){ .card-grid{ grid-template-columns:1fr;} }
.card{ border:1px solid #e5e7eb; border-radius:14px; padding:.8rem .9rem; height:100%; display:flex; flex-direction:column; background:#fff;}
.card a{ text-decoration:none;}
.card .brand{ font-size:var(--brand-size); color:#374151; margin-bottom:.15rem;}
.card .title{ font-size:var(--title-size); color:#1f2937; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; min-height:2.6em; margin:0 0 .5rem 0;}
.card-thumb{ width:100%; aspect-ratio:1/1; object-fit:contain; background:#fff; border-radius:12px; display:block;}
.usd-red{ color:#dc2626; font-weight:700; margin:.15rem 0 0;}
.usd-landed{ font-weight:700; margin:.15rem 0 .2rem;}
.small-cap{ color:#6b7280; font-size:.85rem;}
.section-gap{ margin-top:.35rem;}
button[aria-expanded="false"] p { margin-bottom: 0; }
div[data-testid="stExpander"] div[role="button"] p { font-size: .9rem !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Secrets & REST ─────────────────────────

st.title("UAE Men Shoes — Global Lows")

SUPABASE_URL = (st.secrets.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase secrets. Add SUPABASE_URL and SUPABASE_ANON_KEY in App → Settings → Secrets.")
    LOG.log("fatal_missing_secrets")
    LOG.render()
    st.stop()

REST = f"{SUPABASE_URL}/rest/v1"
AUTH_HDR = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    # NOTE: we DO NOT set Prefer: count=exact globally anymore (too slow).
}

MV = "nam_global_shoes_at_global_low"
PRICES_TABLES = [
    "nam-uae-men-shoes-prices",
    "nam-uae-women-shoes-prices",
    "nam-ksa-men-shoes-prices",
    "prices",
]

SELECT_COLS = [
    "brand","title","link","url","image_url","image",
    "latest_price","latest_usd","landed_usd",
    "min_price","second_lowest_price",
    "delta_vs_30d_pct","delta_vs_90d_pct","gap_to_second_lowest_pct",
]

# ───────────────────────── Helpers ─────────────────────────

def http_get(path_or_url: str, params: dict, label: str = "", prefer_count: str | None = None):
    """Lightweight GET. Only ask for count when we truly need it."""
    url = path_or_url if path_or_url.startswith("http") else f"{REST}/{path_or_url}"
    hdr = dict(AUTH_HDR)
    if prefer_count:  # 'exact' or 'planned'
        hdr["Prefer"] = f"count={prefer_count}"
    try:
        r = requests.get(url, params=params, headers=hdr, timeout=15)
        r.raise_for_status()
        return r.json(), r.headers.get("content-range")
    except requests.RequestException as e:
        LOG.log("http_error", {"label": label, "url": url, "params": params, "error": str(e)})
        st.warning(f"{label}: temporary data error. See logs.")
        return [], None

def pct_fmt(x):
    try: return f"{x*100:.1f}%"
    except: return "—"

def pg_in(values: list[str]) -> str:
    esc = ",".join([str(v).replace(",", " ") for v in values if v])
    return f"in.({esc})"

# ───────────────────────── Options (brands/categories) ─────────────────────────
# Old approach paged the big MV to dedupe distincts (slow). We now:
#   - cap at small max_rows (fast first paint)
#   - no count on these calls
# Ideal: replace with tiny ref tables/views (see “What else to do” below).

@st.cache_data(ttl=300)
def _scan_distinct_values_from_mv(col: str, page_size: int = 200, max_rows: int = 1200) -> list[str]:
    values, offset, seen = set(), 0, 0
    while True:
        data, _ = http_get(MV, {
            "select": col,
            f"{col}": "not.is.null",
            "order": f"{col}.asc",
            "limit": str(page_size), "offset": str(offset)
        }, label=f"scan:{col}")
        if not data: break
        for r in data:
            v = r.get(col)
            if v is not None: values.add(str(v))
        offset += len(data); seen += len(data)
        if len(values) >= max_rows or len(data) < page_size:
            break
    return sorted(values, key=lambda s: s.lower())

@st.cache_data(ttl=300)
def load_options_capped():
    brands = _scan_distinct_values_from_mv("brand")
    cats   = _scan_distinct_values_from_mv("category")
    # Quick min/max without count
    def minmax(col, cast=float):
        lo, _ = http_get(MV, {"select": col, "order": f"{col}.asc", "limit": "1"}, label=f"min_{col}")
        hi, _ = http_get(MV, {"select": col, "order": f"{col}.desc", "limit": "1"}, label=f"max_{col}")
        lo = cast(lo[0][col]) if lo and lo[0].get(col) is not None else (0 if cast is int else 0.0)
        hi = cast(hi[0][col]) if hi and hi[0].get(col) is not None else (0 if cast is int else 0.0)
        return lo, hi
    pmin, pmax = minmax("latest_price", float)
    hmin, hmax = minmax("min_hits", int)
    return brands, cats, float(pmin), float(pmax), int(hmin), int(hmax)

# ───────────────────────── Query params/state ─────────────────────────

def qp_get(): return st.query_params.to_dict()
def qp_set(**kwargs):
    qp = {**st.query_params, **{k: v for k, v in kwargs.items() if v is not None}}
    st.query_params.clear(); st.query_params.update(qp)

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

def _parse_range(s, cast=int, default=(0,0)):
    if not s: return default
    try: lo, hi = s.split("-", 1); return cast(lo), cast(hi)
    except: return default

def hydrate_from_qp(pmin, pmax, hmin, hmax):
    qp = qp_get()
    st.session_state.setdefault("w_order_by", qp.get("ob", "delta_vs_30d_pct"))
    st.session_state.setdefault("w_order_desc", qp.get("od", "1") == "1")
    st.session_state.setdefault("w_page", int(qp.get("page","0")) if qp.get("page") else 0)
    st.session_state.setdefault("w_page_size", int(qp.get("ps","24")) if qp.get("ps") else 24)
    st.session_state.setdefault("w_ship_usd", float(qp.get("ship", 7.0)))
    st.session_state.setdefault("w_margin_pct", float(qp.get("m", 25.0)))
    st.session_state.setdefault("w_cashback_pct", float(qp.get("cb", 0.0)))
    # Ranges from qp or defaults
    st.session_state.setdefault("w_price_range", _parse_range(qp.get("p"), int, (int(pmin), int(pmax))))
    st.session_state.setdefault("w_min_hits", _parse_range(qp.get("h"), int, (int(hmin), int(hmax))))
    st.session_state.setdefault("w_drop_prev", _parse_range(qp.get("d"), int, (-100, 100)))
    st.session_state.setdefault("w_drop_30", _parse_range(qp.get("d30"), int, (-100, 100)))
    st.session_state.setdefault("w_drop_90", (-100, 100))

# ───────────────────────── Data fetch ─────────────────────────

def build_params(flt: dict, limit: int, offset: int) -> dict:
    direction = "desc" if flt["order_desc"] else "asc"
    p = {
        "select": ",".join(SELECT_COLS),
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
def fetch_items(flt: dict, page: int, page_size: int):
    params = build_params(flt, page_size, page * page_size)
    data, cr = http_get(MV, params, label="fetch_items", prefer_count="planned")  # fast planner estimate
    total = None
    if cr and "/" in cr:
        try: total = int(cr.split("/")[-1])
        except: total = None
    return pd.DataFrame(data), total

MAX_POINTS = 200

@st.cache_data(ttl=300, max_entries=512)
def fetch_series(item_url: str, n: int = MAX_POINTS) -> pd.DataFrame:
    params_base = {"select": "timestamp,price", "url": f"eq.{item_url}", "order": "timestamp.desc", "limit": str(n)}
    for table in PRICES_TABLES:
        data, _ = http_get(table, params_base, label=f"fetch_series:{table}")
        df = pd.DataFrame(data)
        if not df.empty:
            break
    else:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Beirut")
    return (df[["timestamp","price"]]
              .dropna(subset=["timestamp"])
              .drop_duplicates(subset=["timestamp"], keep="last")
              .sort_values("timestamp")
              .reset_index(drop=True))

# ───────────────────────── Sidebar / Diagnostics ─────────────────────────

with st.sidebar:
    st.markdown("### Diagnostics")
    if st.button("Ping MV"):
        sample, cr = http_get(MV, {"select": "brand,latest_price", "limit": "1"}, label="ping_mv")
        st.success("MV reachable"); LOG.log("ping_mv_ok", {"cr": cr, "sample_rows": len(sample)})
    with st.expander("Logs", expanded=False): LOG.render()

# ───────────────────────── Load options (capped) & hydrate ─────────────────────────

brands, categories, pmin, pmax, hmin, hmax = load_options_capped()
hydrate_from_qp(pmin, pmax, hmin, hmax)

# ───────────────────────── Filters ─────────────────────────

def _number(key, label, default, **kw):
    return st.number_input(label, value=st.session_state.get(key, default), key=key, **kw)
def _slider(key, label, mn, mx, default, **kw):
    return st.slider(label, min_value=mn, max_value=mx, value=st.session_state.get(key, default), key=key, **kw)
def _select(key, label, options, default_idx=0, **kw):
    val = st.session_state.get(key, options[default_idx] if options else "")
    return st.selectbox(label, options=options, index=(options.index(val) if val in options else default_idx), key=key, **kw)
def _multi(key, label, options, default=None, **kw):
    return st.multiselect(label, options=options, default=st.session_state.get(key, default or []), key=key, **kw)
def _toggle(key, label, default=False, **kw):
    return st.toggle(label, value=st.session_state.get(key, default), key=key, **kw)
def _select_slider(key, label, options, default, **kw):
    return st.select_slider(label, options=options, value=st.session_state.get(key, default), key=key, **kw)

with st.form("filters"):
    st.subheader("Filters")

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_brands = _multi("w_brands", "Brands (capped list for speed)", brands, default=brands[:12])
    with c2:
        sel_cat = _select("w_category", "Category", [""] + categories, default_idx=0)
    with c3:
        price_range = _slider("w_price_range", "Price (AED)", int(pmin), int(pmax), (int(pmin), int(pmax)))

    c4, c5, c6 = st.columns(3)
    with c4:
        min_hits = _slider("w_min_hits", "Min Hits", int(hmin), max(100, int(hmax)), (int(hmin), int(hmax)))
    with c5:
        drop_prev = _slider("w_drop_prev", "Drop vs prev (%)", -100, 100, (-100, 100))
    with c6:
        drop_30 = _slider("w_drop_30", "Drop vs 30-day avg (%)", -100, 100, (-100, 100))

    st.markdown("---")
    cA, cB, cC = st.columns(3)
    with cA:
        order_by = _select("w_order_by", "Order by",
                           ["delta_vs_30d_pct","delta_vs_90d_pct","drop_pct_vs_prev",
                            "latest_price","min_hits","gap_to_second_lowest_pct","days_since_first_low"],
                           default_idx=0)
    with cB:
        order_desc = _toggle("w_order_desc", "Sort descending", True)
    with cC:
        page_size = _select_slider("w_page_size", "Page size", [12, 24, 48, 96], 24)

    cX, cY, cZ = st.columns(3)
    with cX:
        ship_usd = _number("w_ship_usd", "Shipping USD", 7.0, step=0.5)
    with cY:
        margin_pct = _number("w_margin_pct", "Margin %", 25.0, step=1.0)
    with cZ:
        cashback_pct = _number("w_cashback_pct", "Cashback %", 0.0, step=0.5)

    apply = st.form_submit_button("Apply", use_container_width=True)
    if apply:
        st.session_state["w_page"] = 0
        encode_filters_to_qp({
            "brands": sel_brands, "category": sel_cat,
            "price_range": price_range, "min_hits": min_hits,
            "drop_prev": drop_prev, "drop_30": drop_30, "drop_90": (-100, 100),
            "order_by": order_by, "order_desc": order_desc, "page_size": page_size,
            "ship_usd": ship_usd, "margin_pct": margin_pct, "cashback_pct": cashback_pct,
            "page": 0
        })
        st.rerun()

flt = {
    "brands": st.session_state.get("w_brands", []),
    "category": st.session_state.get("w_category", ""),
    "price_range": st.session_state.get("w_price_range", (int(pmin), int(pmax))),
    "min_hits": st.session_state.get("w_min_hits", (int(hmin), int(hmax))),
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

# ───────────────────────── Fetch & Render ─────────────────────────

df, total = fetch_items(flt, page=flt["page"], page_size=flt["page_size"])
count = len(df)

if count == 0:
    st.info("No items match the current filters.")
else:
    st.write(f"**{total or count} items** (showing {count})")

    st.markdown('<div class="card-grid">', unsafe_allow_html=True)

    for _, row in df.iterrows():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        link = row.get("link") or row.get("url") or "#"
        img  = row.get("image_url") or row.get("image") or ""
        brand = str(row.get("brand", "") or "")
        title = str(row.get("title", "") or "")

        if img:
            st.markdown(
                f'<a href="{link}" target="_blank"><img class="card-thumb" loading="lazy" src="{img}"/></a>',
                unsafe_allow_html=True
            )

        st.markdown(f'<div class="brand">{brand}</div>', unsafe_allow_html=True)
        st.markdown(f'<a href="{link}" target="_blank"><div class="title">{title}</div></a>', unsafe_allow_html=True)

        latest = row.get("latest_price")
        min_price = row.get("min_price")
        second_low = row.get("second_lowest_price")
        latest_usd = row.get("latest_usd")
        landed_usd = row.get("landed_usd")

        if latest is not None: st.markdown(f"**AED {latest:,.2f}**")
        if latest_usd is not None: st.markdown(f'<div class="usd-red">~ ${latest_usd:,.2f}</div>', unsafe_allow_html=True)
        if landed_usd is not None: st.markdown(f'<div class="usd-landed">Landed ~ ${landed_usd:,.2f}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        if min_price is not None and second_low is not None:
            st.markdown(f'<div class="small-cap">All-time low: AED {min_price:,.2f} • 2nd-lowest: AED {second_low:,.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-cap">30d Δ: {pct_fmt(row.get("delta_vs_30d_pct"))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-cap">90d Δ: {pct_fmt(row.get("delta_vs_90d_pct"))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-cap">2nd-lowest gap: {pct_fmt(row.get("gap_to_second_lowest_pct"))}</div>', unsafe_allow_html=True)

        with st.expander("📈 Price History (AED)", expanded=False):
            item_url = row.get("url")
            if not item_url:
                st.info("No URL available for time-series lookup.")
            else:
                ts = fetch_series(item_url, n=MAX_POINTS)
                if ts.empty:
                    st.info("No time-series data.")
                else:
                    # Lazy import plotly so first render is faster
                    import plotly.graph_objs as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts["timestamp"], y=ts["price"], mode="lines+markers", name="Price"))
                    fig.update_layout(xaxis_title="Date (Asia/Beirut)", yaxis_title="AED",
                                      margin=dict(l=10, r=10, t=30, b=10), height=280)
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)  # /card

    st.markdown('</div>', unsafe_allow_html=True)  # /grid

    # pager
    left, _, right = st.columns(3)
    with left:
        if st.button("⟵ Prev", disabled=flt["page"] <= 0, use_container_width=True):
            st.session_state["w_page"] = max(0, flt["page"] - 1); qp_set(page=str(st.session_state["w_page"])); st.rerun()
    with right:
        more = (total is None) or ((flt["page"] + 1) * flt["page_size"] < (total or 0))
        if st.button("Next ⟶", disabled=not more, use_container_width=True):
            st.session_state["w_page"] = flt["page"] + 1; qp_set(page=str(st.session_state["w_page"])); st.rerun()
