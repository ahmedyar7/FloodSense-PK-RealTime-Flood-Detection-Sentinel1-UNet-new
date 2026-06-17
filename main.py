"""
main.py
-------
Unified Flood Detection Engine for Pakistan.
Combines:
1. Live River Flows (FFC Scraper)
2. Historical 2010 Flood Analysis (GEE Landsat)
3. Current Run-time Flood Prediction (GEE Sentinel-1 + UNet)
4. AI Strategic Insights (Groq / Gemini)
"""

import sys
import os
import argparse


def setup_console():
    """Windows cp1252 consoles crash on emoji unless UTF-8 is enabled."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


setup_console()

import json
import ee
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Heavy ML stack imported after console setup
from scrapers.ffc_scraper import get_ffc_data
from engine.ai_alerts import FloodAI
from engine.data_manager import save_json
from models.model_inference import load_flood_model, predict_flood, resolve_weights_path
from utils.ndwi import get_flood_mask
from utils.districts import detect_name_column, shapely_to_ee, PRIORITY_DISTRICTS, flood_percent_for_district
from utils.visualize import plot_static_map
import geopandas as gpd

# ── CONFIG ──────────────────────────────────────────────────────────────────
SHAPEFILE = "pakistan_districts.json"
PROJECT_ID = os.getenv("PROJECT_ID")
QUICK_DISTRICTS = ["Larkana", "Dera Ghazi Khan", "Nowshera", "Sukkur"]


def log(msg: str):
    print(msg, flush=True)


def init_gee():
    log("[GEE] Initializing Google Earth Engine...")
    try:
        if PROJECT_ID:
            ee.Initialize(project=PROJECT_ID)
        else:
            ee.Initialize()
        log("[GEE] Connected.")
    except Exception as e:
        log(f"[GEE] Initialization failed: {e}")
        log("      Run: earthengine authenticate")
        return False
    return True


def fetch_current_sar_image(bbox, date_start, date_end, size=256):
    """
    Fetches Sentinel-1 SAR tile (VV) for [date_start, date_end] and applies a slope mask.
    date_start/date_end can be `datetime` or 'YYYY-MM-DD' strings.
    """
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
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate("2024-01-01", "2024-12-31")
            .select("VV")
        )

    image = collection.median()

    elevation = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(elevation)
    slope_mask = slope.lt(15)
    masked_image = image.updateMask(slope_mask).unmask(0)

    url = masked_image.getThumbURL({
        "region": bbox,
        "dimensions": size,
        "format": "png",
        "min": -25,
        "max": 0,
    })

    res = requests.get(url, timeout=60)
    res.raise_for_status()
    return res.content


def parse_args():
    parser = argparse.ArgumentParser(description="FloodSense Pakistan pipeline")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Process only 4 demo districts (faster for testing)",
    )
    parser.add_argument(
        "--skip-historical",
        action="store_true",
        help="Skip 2010 Landsat baseline (GEE heavy); current flood % only",
    )
    parser.add_argument(
        "--current-start",
        type=str,
        default=None,
        help="Current SAR date range start (YYYY-MM-DD). Default: last 30 days.",
    )
    parser.add_argument(
        "--current-end",
        type=str,
        default=None,
        help="Current SAR date range end (YYYY-MM-DD). Default: today.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not init_gee():
        return

    date_end = datetime.now()
    if args.current_end:
        date_end = datetime.strptime(args.current_end, "%Y-%m-%d")

    date_start = date_end - timedelta(days=30)
    if args.current_start:
        date_start = datetime.strptime(args.current_start, "%Y-%m-%d")

    log("\n[MODEL] Loading U-Net weights...")
    weights_path = resolve_weights_path()
    log(f"        Using {weights_path}")
    model = load_flood_model(weights_path)

    log("\n[AI] Initializing Groq insights...")
    ai = FloodAI()

    log("\n[FFC] Fetching live river flows...")
    try:
        river_flows = get_ffc_data()
    except Exception as e:
        log(f"[FFC] Scraper failed ({e}). Check Data_URL in .env")
        river_flows = []
    save_json("river_flows.json", {
        "timestamp": datetime.now().strftime("%Y-%m-%d"),
        "data": river_flows,
    })
    log(f"        {len(river_flows)} stations loaded.")

    log(f"\n[GEO] Loading districts from {SHAPEFILE}...")
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    name_col = detect_name_column(gdf)
    priority = QUICK_DISTRICTS if args.quick else PRIORITY_DISTRICTS
    mask = gdf[name_col].apply(
        lambda x: any(p.lower() in str(x).lower() for p in priority)
    )
    gdf_priority = gdf[mask].copy()
    log(f"        Targeting {len(gdf_priority)} districts"
        + (" (quick mode)" if args.quick else "") + ".")

    if len(gdf_priority) == 0:
        log("[ERROR] No districts matched. Check pakistan_districts.json names.")
        return

    hist_flood_mask = None
    if not args.skip_historical:
        log("\n[HIST] Computing 2010 flood baseline (Landsat via GEE)...")
        log("       This can take several minutes for all of Pakistan.")
        pakistan_bbox = ee.Geometry.BBox(60.0, 23.0, 77.5, 37.5)
        hist_flood_mask, _, _ = get_flood_mask(pakistan_bbox)
    else:
        log("\n[HIST] Skipped (--skip-historical).")

    results = []
    total = len(gdf_priority)

    for n, (_, row) in enumerate(gdf_priority.iterrows(), start=1):
        name = row[name_col]
        log(f"\n[{n}/{total}] {name}")

        try:
            ee_geom = shapely_to_ee(row.geometry)
            bbox = list(row.geometry.bounds)

            if hist_flood_mask is not None:
                pct_2010 = flood_percent_for_district(ee_geom, hist_flood_mask)
            else:
                pct_2010 = None

            log("       Live SAR + U-Net...")
            sar_bytes = fetch_current_sar_image(bbox, date_start, date_end)
            unet_result = predict_flood(model, sar_bytes, name, bbox)
            pct_current = unet_result["water_coverage_pct"]

            flow_status = "UNKNOWN"
            for f in river_flows:
                if f["station"].lower() in name.lower() or name.lower() in f["station"].lower():
                    flow_status = f["status"]
                    break

            results.append({
                "district": name,
                "flood_pct_2010": pct_2010,
                "flood_pct_current": pct_current,
                "river_status": flow_status,
                "risk_score": unet_result["risk_score"],
                "geometry": row.geometry,
            })

            hist_str = f"{pct_2010}%" if pct_2010 is not None else "n/a"
            log(f"       Done: 2010={hist_str} | current={pct_current}% | river={flow_status}")

        except Exception as e:
            log(f"       [WARN] {name}: {e}")

    if not results:
        log("\n[ERROR] No districts processed successfully.")
        return

    log("\n[AI] Generating strategic insights (Groq)...")
    summary_for_ai = [{k: v for k, v in r.items() if k != "geometry"} for r in results]
    insights = ai.generate_insights(summary_for_ai, river_flows)
    save_json("ai_insights.json", {"content": insights})

    log("\n[VIZ] Building maps and dashboard...")
    df_results = pd.DataFrame(results)
    df_results["flood_pct"] = df_results["flood_pct_current"]

    os.makedirs("outputs", exist_ok=True)
    plot_static_map(df_results, out_path="outputs/flood_map.png")
    # Streamlit UI is the primary dashboard now (no extra HTML exports here).

    log("\n[DONE] Pipeline complete.")
    log("       Map:  outputs/flood_map.png")
    log("       UI:   use Streamlit: streamlit run streamlit_app.py")
    log("       Data: data/json/river_flows.json + data/json/ai_insights.json")
    log("\n       Serve locally:  python -m http.server 8000")
    log("       Then open:      http://localhost:8501 (Streamlit)")


if __name__ == "__main__":
    main()
