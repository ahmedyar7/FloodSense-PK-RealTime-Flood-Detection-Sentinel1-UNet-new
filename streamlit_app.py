import os
from datetime import datetime, timedelta
import io
import json
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import requests

import ee
import geopandas as gpd
import cv2
from shapely.geometry import shape

import matplotlib.pyplot as plt
import pydeck as pdk
from qdrant_client import QdrantClient
from rag import EmbeddingPipeline, ingest_documents, rag_query


def initialize_ee():
    try:
        if "gee" in st.secrets:
            import google.oauth2.service_account

            gee_creds = dict(st.secrets["gee"])

            # Ensure the private key has correct newline characters
            if "private_key" in gee_creds:
                # Handle both literal \n and real newlines
                gee_creds["private_key"] = gee_creds["private_key"].replace("\\n", "\n")

            # Modern Google OAuth2 credentials with explicit scope
            scopes = ["https://www.googleapis.com/auth/earthengine"]
            credentials = (
                google.oauth2.service_account.Credentials.from_service_account_info(
                    gee_creds, scopes=scopes
                )
            )
            ee.Initialize(credentials, project=gee_creds.get("project_id"))
        else:
            # Fallback for local development
            ee.Initialize()
    except Exception as e:
        st.error(f"EE initialization failed: {e}")
        st.info(
            "Please ensure your .streamlit/secrets.toml is correctly configured with [gee] credentials."
        )
        st.stop()


initialize_ee()

from scrapers.ffc_scraper import get_ffc_data
from engine.ai_alerts import FloodAI
from models.model_inference import load_flood_model, predict_flood, resolve_weights_path
from utils.ndwi import get_flood_mask, FLOOD_START, FLOOD_END
from utils.districts import (
    detect_name_column,
    shapely_to_ee,
    PRIORITY_DISTRICTS,
    flood_percent_for_district,
)

SHAPEFILE = "pakistan_districts.json"

st.set_page_config(page_title="FloodSense-PK Dashboard", layout="wide")


def inject_dark_theme():
    # Enhanced CSS for mobile responsiveness and modern UI
    st.markdown(
        """
        <style>
        /* Base container adjustments */
        .block-container { 
            padding-top: 2rem; 
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Mobile specific adjustments */
        @media (max-width: 768px) {
            .block-container { 
                padding-left: 1rem; 
                padding-right: 1rem; 
                padding-top: 1rem;
            }
            .header-title { font-size: 1.8rem !important; }
            .header-subtitle { font-size: 0.9rem !important; }
            .stMetric label { font-size: 0.8rem !important; }
            .stMetric div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
        }

        /* Metric card styling */
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
            height: auto;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(51, 102, 204, 0.15) !important;
            border-color: #3366cc !important;
        }

        /* Custom Header Classes */
        .custom-header {
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(90deg, rgba(51, 102, 204, 0.1) 0%, rgba(51, 102, 204, 0.02) 100%);
            border-left: 6px solid #3366cc;
            margin-bottom: 2rem;
        }
        .header-title {
            margin: 0;
            padding: 0;
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.2;
        }
        .header-subtitle {
            margin: 0.5rem 0 0 0;
            padding: 0;
            font-size: 1.2rem;
            color: #8aa5ff;
            font-weight: 400;
            letter-spacing: 0.5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_dark_theme()


def preprocess_image(tif_bytes):
    """
    Accepts raw GeoTIFF bytes (single-band VV from GEE).
    Returns (tensor [1,3,256,256], display_arr [256,256]).
    """
    with rasterio.open(io.BytesIO(tif_bytes)) as src:
        vv = src.read(1).astype(np.float32)  # (H, W) in dB, e.g. -25 to 0

    vv = cv2.resize(vv, (256, 256), interpolation=cv2.INTER_LINEAR)

    # ── Debug: verify real SAR data is coming through ──
    print(f"  SAR VV  min={vv.min():.2f}  max={vv.max():.2f}  mean={vv.mean():.2f}")

    # ── Display array for visualization: normalize [-25, 0] → [0, 1] ──
    display_arr = (np.clip(vv, -25, 0) + 25) / 25.0

    # ── Channel 0: normalized VV [-40, 10] → [0, 1] ──
    ch0 = (np.clip(vv, -40, 10) + 40) / 50.0

    # ── Channel 1: VH approximation (VH ≈ VV - 6dB empirically) ──
    vh_approx = vv - 6.0
    ch1 = (np.clip(vh_approx, -45, -5) + 45) / 40.0

    # ── Channel 2: VH/VV ratio in linear scale ──
    vv_lin = np.power(10.0, np.clip(vv, -40, 0) / 10.0)
    vh_lin = np.power(10.0, np.clip(vh_approx, -45, -5) / 10.0)
    ratio = np.clip(vh_lin / (vv_lin + 1e-8), 0.0, 2.0) / 2.0

    img_norm = np.stack([ch0, ch1, ratio], axis=-1).astype(np.float32)  # (256,256,3)

    print(f"  ch0={ch0.mean():.3f}  ch1={ch1.mean():.3f}  ratio={ratio.mean():.3f}")

    transform = A.Compose([ToTensorV2()])
    tensor = transform(image=img_norm)["image"].unsqueeze(0)  # (1, 3, 256, 256)

    return tensor, display_arr


def render_unet_overlay(display_arr, pred_mask):
    """
    Create one combined image: SAR grayscale + blue flood mask overlay.
    Works with any dimensions (256, 512, etc.)
    """
    h, w = display_arr.shape

    # Standardize SAR to uint8 RGB
    sar_uint8 = (np.clip(display_arr, 0, 1) * 255).astype(np.uint8)
    sar_rgb = np.stack([sar_uint8, sar_uint8, sar_uint8], axis=-1)

    blue = np.array([26, 95, 212], dtype=np.float32)
    out = sar_rgb.astype(np.float32)
    mask = pred_mask.astype(bool)

    # Alpha blend: flood pixels get blue overlay
    alpha = 0.65
    out[mask] = (1 - alpha) * out[mask] + alpha * blue

    return out.astype(np.uint8)


def render_sar_gray(display_arr):
    """Render the preprocessed SAR grayscale."""
    sar_uint8 = (np.clip(display_arr, 0, 1) * 255).astype(np.uint8)
    return sar_uint8


def render_prob_heatmap(pred_prob):
    """Render probability heatmap as an RGB image (0..1 -> blue-red)."""
    prob = np.clip(pred_prob, 0, 1)
    img = (prob * 255).astype(np.uint8)
    # Use a fixed colormap for readability
    color = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # BGR
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def render_unet_one_diagram(display_arr, pred_prob, pred_mask):
    """
    One combined diagram (dynamic sizing):
    - background: SAR grayscale
    - color: probability heatmap
    - highlight: predicted flood mask (blue overlay)
    """
    sar_gray = render_sar_gray(display_arr)
    sar_rgb = np.stack([sar_gray, sar_gray, sar_gray], axis=-1).astype(np.float32)

    prob = np.clip(pred_prob, 0, 1)
    prob_u8 = (prob * 255).astype(np.uint8)
    prob_color = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)  # BGR
    prob_rgb = cv2.cvtColor(prob_color, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Blend SAR + probability for context
    base = 0.45 * sar_rgb + 0.55 * prob_rgb

    # Highlight flood mask strongly
    mask = pred_mask.astype(bool)
    blue = np.array([26, 95, 212], dtype=np.float32)
    alpha = 0.70
    base[mask] = (1 - alpha) * base[mask] + alpha * blue

    return np.clip(base, 0, 255).astype(np.uint8)


@st.cache_resource
def load_model_cached():
    weights_path = resolve_weights_path()
    return load_flood_model(weights_path)


@st.cache_resource
def load_districts_cached():
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    name_col = detect_name_column(gdf)
    gdf = gdf.copy()
    gdf["__district_name__"] = gdf[name_col].astype(str)
    return gdf[["__district_name__", "geometry"]], name_col


@st.cache_resource
def init_rag_system():
    """Load and index all disaster knowledge documents into an in-memory Qdrant collection."""
    client = QdrantClient(":memory:")
    embedder = EmbeddingPipeline()
    ingest_documents(client, embedder)
    return client, embedder


@st.cache_resource
def get_flood_ai_cached():
    return FloodAI()


def match_station_to_district(district_name: str, river_flows: list[dict]):
    dn = district_name.lower().strip()
    # Prefer direct inclusion matches both ways.
    for row in river_flows:
        st_name = (row.get("station") or "").lower()
        if dn in st_name or st_name in dn:
            return row
    return None


def fetch_current_sar_image(bbox, date_start, date_end, size=256):
    if isinstance(date_start, str):
        date_start = datetime.strptime(date_start, "%Y-%m-%d")
    if isinstance(date_end, str):
        date_end = datetime.strptime(date_end, "%Y-%m-%d")

    region = ee.Geometry.Rectangle(bbox)

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(date_start.strftime("%Y-%m-%d"), date_end.strftime("%Y-%m-%d"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    if collection.size().getInfo() == 0:
        st.warning("No SAR data in selected range, falling back to 2024.")
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate("2024-01-01", "2024-12-31")
            .select("VV")
        )

    image = collection.median()
    elevation = ee.Image("USGS/SRTMGL1_003")
    slope_mask = ee.Terrain.slope(elevation).lt(15)
    masked_image = image.updateMask(slope_mask).unmask(0)

    # ✅ GeoTIFF — rasterio can read this properly
    url = masked_image.getDownloadURL(
        {
            "region": bbox,
            "dimensions": size,
            "format": "GEO_TIFF",
            "bands": ["VV"],
        }
    )

    # Increase timeout for large area exports
    res = requests.get(url, timeout=300)
    res.raise_for_status()

    print(f"  GeoTIFF downloaded: {len(res.content)} bytes")
    return res.content  # raw GeoTIFF bytes


def fetch_2010_mask_image(mask_ee, bbox, size=256):
    """
    Fetches the 2010 binary mask as a PNG for visual comparison.
    """
    # mask_ee is 0 or 1. We'll visualize it as Blue.
    vis_img = mask_ee.visualize(palette=["black", "blue"], min=0, max=1)

    url = vis_img.getDownloadURL(
        {
            "region": bbox,
            "dimensions": size,
            "format": "png",
        }
    )

    # Increase timeout for large area exports (e.g. National)
    res = requests.get(url, timeout=300)
    res.raise_for_status()
    return res.content


# ── Station Coordinate Mapping ──
STATION_COORDS = {
    "Tarbela Dam": [34.08, 72.69],
    "Nowshera": [34.01, 71.97],
    "Besham": [34.92, 72.88],
    "Kala Bagh": [32.96, 71.54],
    "Chashma": [32.44, 71.37],
    "Taunsa": [30.52, 70.84],
    "Guddu": [28.42, 69.70],
    "Sukkur": [27.71, 68.85],
    "Kotri": [25.37, 68.31],
    "Mangla Dam": [33.14, 73.64],
    "Marala": [32.67, 74.47],
    "Khanki": [32.41, 73.79],
    "Qadirabad": [32.31, 73.53],
    "Trimmu": [31.14, 72.15],
    "Punjnad": [29.35, 71.02],
    "Balloki": [31.22, 73.86],
    "Sidhnai": [30.57, 72.08],
    "Jassar": [32.11, 74.96],
    "Sulemanki": [30.37, 73.87],
    "Islam": [29.83, 72.55],
    "Ganda Singh Wala": [31.11, 74.46],
}

# ── River Flow Topology (Upstream -> Downstream) ──
RIVER_PATHS = [
    # Indus Main
    [
        "Tarbela Dam",
        "Besham",
        "Kala Bagh",
        "Chashma",
        "Taunsa",
        "Guddu",
        "Sukkur",
        "Kotri",
    ],
    # Kabul link
    ["Nowshera", "Kala Bagh"],
    # Jhelum link
    ["Mangla Dam", "Trimmu"],
    # Chenab link
    ["Marala", "Khanki", "Qadirabad", "Trimmu", "Punjnad"],
    # Ravi link
    ["Jassar", "Balloki", "Sidhnai", "Punjnad"],
    # Sutlej link
    ["Ganda Singh Wala", "Sulemanki", "Islam", "Punjnad"],
    # Panjnad to Indus
    ["Punjnad", "Guddu"],
]


def _rag_llm_fn(prompt: str) -> str:
    """Call Gemini (primary) or Groq (fallback) to answer a RAG-augmented prompt."""
    ai = get_flood_ai_cached()
    result = ai._try_gemini(prompt)
    if result:
        return result
    if ai.groq_enabled:
        try:
            resp = ai.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Groq error: {e}"
    return "AI service unavailable. Please configure GEMINI_API_KEY or GROQ_API_KEY."


@st.fragment
def render_rag_chatbot():
    st.divider()
    st.markdown("### Disaster Knowledge Assistant")
    st.caption(
        "Ask questions about Pakistan flood history, NDMA protocols, FFD river data, "
        "and disaster response — powered by a RAG knowledge base."
    )

    rag_client, rag_embedder = init_rag_system()

    if "rag_messages" not in st.session_state:
        st.session_state["rag_messages"] = []

    for msg in st.session_state["rag_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask about floods, NDMA protocols, or river systems..."):
        st.session_state["rag_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                response, source_docs = rag_query(
                    user_input, rag_client, rag_embedder, _rag_llm_fn
                )
            st.markdown(response)
            if source_docs:
                with st.expander("Sources", expanded=False):
                    for doc in source_docs:
                        st.caption(
                            f"**{doc.get('source', 'Unknown')}** — {doc.get('title', '')}"
                        )

        st.session_state["rag_messages"].append(
            {"role": "assistant", "content": response}
        )


def main():
    # ── Professional Executive Header ──
    st.markdown(
        """
        <div class="custom-header">
            <h1 class="header-title">FloodSense-PK</h1>
            <p class="header-subtitle">National Flood Intelligence & Early Warning System</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    gdf, _ = load_districts_cached()

    # Scale selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Analysis Scale")
    analysis_scale = st.sidebar.selectbox(
        "Select Scope",
        ["District", "Province", "National"],
        help="National/Province will provide a Low-Res Overview of the entire region.",
    )

    if analysis_scale == "National":
        district = "Pakistan"
    elif analysis_scale == "Province":
        prov_opts = sorted(
            [
                "Punjab",
                "Sindh",
                "Khyber Pakhtunkhwa",
                "Balochistan",
                "Azad Jammu & Kashmir",
                "Gilgit-Baltistan",
            ]
        )
        district = st.sidebar.selectbox("Select Province", prov_opts)
    else:
        # District picker
        priority_only = st.sidebar.checkbox("Use Priority Districts only", value=True)
        if priority_only:
            opts = sorted(
                {
                    x
                    for x in gdf["__district_name__"].tolist()
                    if any(p.lower() in x.lower() for p in PRIORITY_DISTRICTS)
                }
            )
        else:
            opts = sorted({x for x in gdf["__district_name__"].tolist()})
        district = st.sidebar.selectbox(
            "District (Pre-defined)", options=opts, index=0 if opts else 0
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced: Search Custom Area")
    search_name = st.sidebar.text_input(
        "Enter Area Name (Overrides Scale)",
        help="Type a city or tehsil name. If found, it will override the scale selection.",
    )

    # Date range
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=30)
    start_date = st.sidebar.date_input("Current SAR start date", value=default_start)
    end_date = st.sidebar.date_input("Current SAR end date", value=default_end)

    # Actions
    with st.sidebar:
        st.markdown("### Run")
        run = st.button("Run analysis", type="primary", use_container_width=True)

    if not run:
        st.info("Select a scale/district and press Run analysis.")
        return

    # ── Geometry Resolution ──
    geom = None
    display_name = district

    if search_name.strip():
        with st.spinner(
            f"Searching for '{search_name}' in Pakistan (Districts/Tehsils)..."
        ):
            # Use FAO GAUL Level 2, filtered specifically for Pakistan
            pakistan_fc = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
                ee.Filter.eq("ADM0_NAME", "Pakistan")
            )

            search_term = search_name.strip()
            # 1. Try exact match in ADM2_NAME (Districts/Sub-districts)
            match = pakistan_fc.filter(
                ee.Filter.eq("ADM2_NAME", search_term.capitalize())
            )

            # 2. Try partial match if exact fails
            if match.size().getInfo() == 0:
                match = pakistan_fc.filter(
                    ee.Filter.stringContains("ADM2_NAME", search_term)
                )

            # 3. Last fallback: search for Provinces/Regions
            if match.size().getInfo() == 0:
                prov_fc = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
                    ee.Filter.eq("ADM0_NAME", "Pakistan")
                )
                match = prov_fc.filter(
                    ee.Filter.stringContains("ADM1_NAME", search_term)
                )

            if match.size().getInfo() > 0:
                feat = match.first()
                props = feat.toDictionary().getInfo()
                # Get the official administrative name (Tehsil/District)
                official_name = (
                    props.get("ADM2_NAME") or props.get("ADM1_NAME") or search_term
                )

                geom = shape(feat.geometry().getInfo())
                display_name = f"{official_name} (Search)"
                st.sidebar.success(f"Verified Area: {official_name}")
            else:
                st.sidebar.warning(
                    f"'{search_name}' not found. Using selection instead."
                )

    if geom is None:
        if analysis_scale == "National":
            # Pakistan full boundary
            country = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
                ee.Filter.eq("country_na", "Pakistan")
            )
            if country.size().getInfo() > 0:
                geom = shape(country.first().geometry().getInfo())
                display_name = "Pakistan (National)"
            else:
                st.error("Could not fetch National boundary from GEE.")
        elif analysis_scale == "Province":
            # FAO Level 1 for provinces - use robust matching
            prov_fc = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
                ee.Filter.eq("ADM0_NAME", "Pakistan")
            )
            match = prov_fc.filter(ee.Filter.stringContains("ADM1_NAME", district))

            if match.size().getInfo() > 0:
                geom = shape(match.first().geometry().getInfo())
                display_name = f"{district} (Province)"
            else:
                st.error(f"Could not find Province boundary for '{district}' in GEE.")

        # Final fallback if GEE fails or for standard District mode
        if geom is None:
            # Dropdown district
            try:
                row = gdf[gdf["__district_name__"] == district].iloc[0]
                geom = row["geometry"]
            except:
                # Absolute fallback to first district if everything fails
                row = gdf.iloc[0]
                geom = row["geometry"]
                display_name = row["__district_name__"]

    ee_geom = shapely_to_ee(geom)
    district = display_name

    # Calculate bbox
    minx, miny, maxx, maxy = geom.bounds
    w_deg = maxx - minx
    h_deg = maxy - miny
    buffer_percent = 0.05 if analysis_scale in ["Province", "National"] else 0.15
    bbox = [
        minx - w_deg * buffer_percent,
        miny - h_deg * buffer_percent,
        maxx + w_deg * buffer_percent,
        maxy + h_deg * buffer_percent,
    ]

    # ── Dynamic Resolution Calculation ──
    # Target resolution: ~1000m for National/Province, ~80m for District
    target_res_m = 1000 if analysis_scale in ["Province", "National"] else 80

    pixels_w = int((w_deg * 1.1) * 111000 / target_res_m)
    pixels_h = int((h_deg * 1.1) * 111000 / target_res_m)

    # GEE Limits: 1024 is safe for memory and speed
    final_size = max(256, min(1024, max(pixels_w, pixels_h)))

    # 1) Historical 2010 mask
    if "hist_flood_mask" not in st.session_state:
        with st.spinner("Computing 2010 flood mask (Landsat via GEE)..."):
            pakistan_bbox = ee.Geometry.BBox(60.0, 23.0, 77.5, 37.5)
            hist_flood_mask, _, _ = get_flood_mask(pakistan_bbox)
            st.session_state["hist_flood_mask"] = hist_flood_mask

    with st.spinner("Computing 2010 flood %..."):
        pct_2010 = flood_percent_for_district(
            ee_geom, st.session_state["hist_flood_mask"]
        )

    # 2) Current (Sentinel-1 + U-Net)
    model = load_model_cached()
    with st.spinner(f"Fetching Sentinel-1 SAR ({final_size}px)..."):
        sar_bytes = fetch_current_sar_image(
            bbox=bbox,
            date_start=str(start_date),
            date_end=str(end_date),
            size=final_size,
        )

    with st.spinner("Running Tiled UNet analysis..."):
        unet_result = predict_flood(model, sar_bytes, district, bbox)

    pct_current = unet_result["water_coverage_pct"]
    risk_score = unet_result["risk_score"]
    settlement_risk = unet_result["settlement_risk"]

    # 3) River flows (FFC)
    @st.cache_data(ttl=300)
    def get_river_flows_cached():
        return get_ffc_data()

    with st.spinner("Scraping FFC river discharge data..."):
        river_flows = get_river_flows_cached()

    df_flows = pd.DataFrame(river_flows)
    matched_station = match_station_to_district(district, river_flows)

    # 4) Gemini AI insights
    with st.spinner("Generating evidence-based strategic insights..."):
        ai = FloodAI()
        # Calculate the defensible risk score using our new engine logic
        risk_score = ai.calculate_defensible_risk(
            {
                "flood_pct_current": pct_current,
                "flood_pct_2010": pct_2010,
                "river_status": (
                    matched_station["status"] if matched_station else "UNKNOWN"
                ),
            },
            river_flows,
        )

        summary_for_ai = [
            {
                "district": district,
                "flood_pct_current": pct_current,
                "flood_pct_2010": pct_2010,
                "risk_score": risk_score,
                "river_status": (
                    matched_station["status"] if matched_station else "UNKNOWN"
                ),
                "settlement_risk": settlement_risk,
            }
        ]
        insights = ai.generate_insights(summary_for_ai, river_flows)

    # 5) Visuals
    sar_gray = render_sar_gray(unet_result["display_arr"])
    overlay_img = render_unet_overlay(
        unet_result["display_arr"], unet_result["pred_mask"]
    )
    prob_heatmap = render_prob_heatmap(unet_result["pred_prob"])

    with st.spinner("Aligning 2010 historical visuals..."):
        hist_mask_bytes = fetch_2010_mask_image(
            st.session_state["hist_flood_mask"], bbox=bbox, size=final_size
        )

    st.divider()

    # Clean, non-overlapping UI using tabs
    t1, t2, t3, t4 = st.tabs(
        ["Overview", "Detection", "River Flows", "AI Intelligence"]
    )

    with t1:
        st.subheader("Flood Severity Comparison: 2010 vs. Current")
        st.markdown("""
        Evidence-based comparison between the historical maximum (2010) and current AI detection. 
        """)

        # ── Visual Comparison Section ──
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("#### **[A] 2010 Historical Baseline**")
            st.image(
                hist_mask_bytes,
                caption="Satellite: Landsat-5 | Method: MNDWI",
                width="stretch",
            )
            st.info("2010 Context: Peak flood footprint during the 2010 disaster.")
        with col_img2:
            st.markdown(f"#### **[B] Current Situation ({start_date})**")
            st.image(
                overlay_img,
                caption="Satellite: Sentinel-1 SAR | Method: UNet AI",
                use_container_width=True,
            )
            st.info("Live Status: Latest available radar-based water detection.")

        st.divider()

        # ── Metrics Section ──
        st.subheader("Statistical Analysis")
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "2010 HISTORICAL %",
            f"{pct_2010:.2f}%",
            help="Total district area flooded in 2010.",
        )
        c2.metric(
            "CURRENT FLOOD %",
            f"{pct_current:.2f}%",
            help="Current area flooded as detected by Sentinel-1/UNet.",
        )

        delta = float(pct_current) - float(pct_2010)
        c3.metric(
            "DELTA SEVERITY",
            f"{delta:+.2f}%",
            delta=f"{delta:+.2f}%",
            delta_color="inverse",
            help="Current % - 2010 %. Positive means worse than 2010.",
        )

        st.markdown(f"""
        ### **Risk Governance Summary**
        *   **Defensible Risk Score:** **{risk_score}/10** (Computed via Weighted Multi-Factor Analysis)
        *   **Comparative Severity:** Current flooding is **{ (pct_current / (pct_2010 if pct_2010 > 0 else 1)) * 100:.1f}%** of the 2010 benchmark.
        """)

        st.progress(min(1.0, risk_score / 10.0))

        st.divider()

        st.markdown("### Local River Monitoring")
        if matched_station:
            st.success(
                f"**Station:** {matched_station['station']} | **River:** {matched_station['river']} | **Status:** {matched_station['status']}"
            )
            # Add trend indicators
            itrend = matched_station.get("inflow_trend", "Steady")
            otrend = matched_station.get("outflow_trend", "Steady")
            i_icon = "↑" if "Rising" in itrend else "↓" if "Falling" in itrend else "→"
            o_icon = "↑" if "Rising" in otrend else "↓" if "Falling" in otrend else "→"

            st.caption(
                f"**Inflow:** {matched_station.get('inflow')} {i_icon} ({itrend}) | "
                f"**Outflow:** {matched_station.get('outflow')} {o_icon} ({otrend})"
            )
            st.caption(f"Last Updated: {matched_station.get('recorded', 'N/A')}")
        else:
            st.warning(
                "⚠️ **Data Gap:** No hydraulic monitoring station found for this district boundary. Confidence in river status is reduced."
            )

    with t2:
        st.subheader("UNet Deep Learning Analysis")
        st.caption("Detailed breakdown of AI model outputs and confidence levels.")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### **Unified Detection Mask**")
            one_diagram = render_unet_one_diagram(
                unet_result["display_arr"],
                unet_result["pred_prob"],
                unet_result["pred_mask"],
            )
            st.image(
                one_diagram,
                caption="Combined SAR + Probability + Mask",
                use_container_width=True,
            )

        with col_b:
            st.markdown("#### **Confidence Heatmap**")
            st.image(
                prob_heatmap,
                caption="AI Probability Score (0.0 to 1.0)",
                use_container_width=True,
            )

        st.divider()
        st.markdown("### Model Performance Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Pixel Water Coverage", f"{pct_current:.2f}%")
        m2.metric("Total Affected Area", f"{unet_result['affected_area_km2']:.1f} km²")
        m3.metric("AI Architecture", "ResNet34-UNet (Tiled)")

    with t3:
        st.subheader("Hydraulic Intelligence: FFC River Discharge")
        if df_flows.empty:
            st.warning("No river flows found. Check FFC data source.")
        else:
            # ── Status Distribution ──
            status_counts = df_flows["status"].value_counts(dropna=False).to_dict()
            st.markdown("#### **Network Status Overview**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("NORMAL", status_counts.get("NORMAL", 0))
            sc2.metric("HIGH", status_counts.get("HIGH", 0))
            sc3.metric("EXTREME", status_counts.get("EXTREME", 0))
            sc4.metric("DATA GAPS", status_counts.get("NOT_RECEIVED", 0))

            st.divider()

            # ── Charts Section ──
            st.markdown("#### **Top 10 Stations by Volume (Cusecs)**")
            df_plot = df_flows.copy()
            df_plot["inflow"] = pd.to_numeric(df_plot["inflow"], errors="coerce")
            df_plot["outflow"] = pd.to_numeric(df_plot["outflow"], errors="coerce")

            top_in = (
                df_plot.dropna(subset=["inflow"])
                .sort_values("inflow", ascending=False)
                .head(10)
            )
            top_out = (
                df_plot.dropna(subset=["outflow"])
                .sort_values("outflow", ascending=False)
                .head(10)
            )

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.caption("Top Inflow Stations")
                if not top_in.empty:
                    st.bar_chart(
                        top_in.set_index("station")["inflow"],
                        color="#3366cc",
                        use_container_width=True,
                    )
            with col_c2:
                st.caption("Top Outflow Stations")
                if not top_out.empty:
                    st.bar_chart(
                        top_out.set_index("station")["outflow"],
                        color="#dc3912",
                        use_container_width=True,
                    )

            st.divider()

            st.markdown("#### **Inflow vs. Outflow Correlation**")
            df_sc = df_plot.dropna(subset=["inflow", "outflow"]).copy()
            if len(df_sc) > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = {
                    "NORMAL": "#109618",
                    "HIGH": "#ff9900",
                    "EXTREME": "#ff0000",
                    "NOT_RECEIVED": "#7b8794",
                    "UNKNOWN": "#8aa5ff",
                }
                for status, g in df_sc.groupby(df_sc["status"].astype(str)):
                    ax.scatter(
                        g["inflow"],
                        g["outflow"],
                        s=45,
                        alpha=0.7,
                        label=status,
                        color=colors.get(status, "#8aa5ff"),
                        edgecolors="white",
                    )

                ax.set_xlabel("Inflow (Cusecs)", fontsize=9)
                ax.set_ylabel("Outflow (Cusecs)", fontsize=9)
                ax.legend(fontsize=8, title="Station Status")
                ax.grid(True, alpha=0.15)
                st.pyplot(fig, use_container_width=True)

            st.markdown("#### **Native Pakistan River Monitoring Map**")
            st.caption(
                "Interactive hydraulic network visualizing current barrage and dam status."
            )

            # Prepare data for Pydeck Map
            map_data = []
            for d in river_flows:
                # Use hardcoded coords if scraper lacks them
                coords = STATION_COORDS.get(d["station"])
                if coords:
                    color = [16, 150, 24, 200]  # Normal (Green)
                    if d["status"] == "HIGH":
                        color = [255, 153, 0, 200]  # Warning (Orange)
                    elif d["status"] == "EXTREME":
                        color = [255, 0, 0, 200]  # Critical (Red)

                    map_data.append(
                        {
                            "name": d["station"],
                            "river": d["river"],
                            "inflow": d["inflow"],
                            "outflow": d["outflow"],
                            "status": d["status"],
                            "latitude": coords[0],
                            "longitude": coords[1],
                            "color": color,
                        }
                    )

            if map_data:
                df_map = pd.DataFrame(map_data)

                # Convert RIVER_PATHS names to coordinate paths
                flow_paths = []
                for path_names in RIVER_PATHS:
                    coords = []
                    for name in path_names:
                        if name in STATION_COORDS:
                            # Pydeck expects [lon, lat]
                            c = STATION_COORDS[name]
                            coords.append([c[1], c[0]])
                    if len(coords) > 1:
                        flow_paths.append({"path": coords, "name": "River Segment"})

                # 1. Base Layer (Pakistan Outline)
                base_layer = pdk.Layer(
                    "GeoJsonLayer",
                    gdf,
                    opacity=0.05,
                    stroked=True,
                    filled=True,
                    get_fill_color=[150, 150, 150],
                    get_line_color=[255, 255, 255],
                )

                # 2. Path Layer (The Links)
                path_layer = pdk.Layer(
                    "PathLayer",
                    flow_paths,
                    width_min_pixels=3,
                    get_path="path",
                    get_color=[51, 102, 204, 180],  # Blue flow links
                    pickable=True,
                )

                # 3. Station Layer (The Points)
                station_layer = pdk.Layer(
                    "ScatterplotLayer",
                    df_map,
                    get_position=["longitude", "latitude"],
                    get_color="color",
                    get_radius=15000,
                    pickable=True,
                )

                view_state = pdk.ViewState(
                    latitude=30.0, longitude=70.0, zoom=5, pitch=0
                )

                st.pydeck_chart(
                    pdk.Deck(
                        map_style=None,
                        initial_view_state=view_state,
                        layers=[base_layer, path_layer, station_layer],
                        tooltip={
                            "text": "{name} ({river})\nStatus: {status}\nInflow: {inflow}\nOutflow: {outflow}"
                        },
                    ),
                    use_container_width=True,
                )
            else:
                st.warning("No station coordinates available to render the map.")

            st.markdown(
                f"""
            <div style='text-align: right; padding: 10px;'>
                <a href='https://ffd.pmd.gov.pk/river-state' target='_blank'>
                    <button style='background-color: transparent; color: #8aa5ff; padding: 5px 15px; border: 1px solid #8aa5ff; border-radius: 5px; cursor: pointer; font-size: 0.8em;'>
                        View FFD Source Page ↗
                    </button>
                </a>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.divider()
            st.markdown("#### **Raw Hydraulic Data Table**")
            st.dataframe(df_flows, use_container_width=True, height=400)

    with t4:
        st.subheader("Gemini AI: Tactical Intelligence Report")
        st.caption(
            "Evidence-based operational recommendations derived from spatial and hydraulic metrics."
        )

        # ── Risk Visual Overview ──
        c_risk1, c_risk2 = st.columns([1, 2])
        with c_risk1:
            st.metric("COMPOSITE RISK", f"{risk_score}/10")
            risk_color = (
                "red" if risk_score > 7 else "orange" if risk_score > 4 else "green"
            )
            st.markdown(
                f"<div style='height:15px; width:100%; background-color:{risk_color}; border-radius:10px;'></div>",
                unsafe_allow_html=True,
            )

        with c_risk2:
            conf_level = "HIGH" if matched_station else "MEDIUM (Data Gaps)"
            st.info(f"Report Fidelity: **{conf_level}**")

        st.divider()

        # ── Structured Insights ──
        if "[" in insights and "]" in insights:
            lines = insights.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if "[SITUATION" in line.upper():
                    st.markdown(f"#### 📡 Situation Summary")
                    st.info(line.split("]")[-1].strip())
                elif "[HYDRAULIC" in line.upper():
                    st.markdown(f"#### 🌊 Hydraulic Analysis")
                    st.write(line.split("]")[-1].strip())
                elif "[HISTORICAL" in line.upper():
                    st.markdown(f"#### 📜 Historical Benchmark")
                    st.write(line.split("]")[-1].strip())
                elif "[OPERATIONAL" in line.upper():
                    st.markdown(f"#### 🚨 Operational Actions")
                    st.warning(line.split("]")[-1].strip())
                elif "[CONFIDENCE" in line.upper():
                    st.caption(f"**Confidence Level:** {line.split(']')[-1].strip()}")
        else:
            st.markdown(insights)

        st.divider()
        st.caption(
            f"Governance: Weighted Risk Formula (Flood% 40, Delta 30, Hydraulic 30)"
        )

        render_rag_chatbot()


if __name__ == "__main__":
    main()
