import os
import sys
import json
import ee
import requests
import geopandas as gpd
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from shapely.geometry import shape

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.ffc_scraper import get_ffc_data
from engine.ai_alerts import FloodAI
from models.model_inference import load_flood_model, predict_flood, resolve_weights_path
from utils.ndwi import get_flood_mask
from utils.districts import detect_name_column, shapely_to_ee, flood_percent_for_district

load_dotenv()

# --- CONFIG ---
SHAPEFILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pakistan_districts.json")
PROJECT_ID = os.getenv("PROJECT_ID")

# --- GEE INIT ---
def init_gee():
    try:
        if PROJECT_ID:
            ee.Initialize(project=PROJECT_ID)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        print(f"GEE init failed: {e}")
        return False

# --- SAR FETCH ---
def fetch_current_sar_image(bbox, date_start, date_end, size=256):
    """
    Fetches Sentinel-1 SAR GeoTIFF from GEE.
    """
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
    slope_mask = ee.Terrain.slope(elevation).lt(15)
    masked_image = image.updateMask(slope_mask).unmask(0)

    url = masked_image.getDownloadURL({
        "region": bbox,
        "dimensions": size,
        "format": "GEO_TIFF",
        "bands": ["VV"],
    })

    res = requests.get(url, timeout=300)
    res.raise_for_status()
    return res.content

def get_image_base64(display_arr, pred_mask):
    """
    Creates a PNG image with flood mask overlay and returns it as a base64 string.
    """
    sar_uint8 = (np.clip(display_arr, 0, 1) * 255).astype(np.uint8)
    sar_rgb = np.stack([sar_uint8, sar_uint8, sar_uint8], axis=-1)
    
    # Blue color for flood (RGB)
    blue = np.array([26, 95, 212], dtype=np.float32)
    out = sar_rgb.astype(np.float32)
    mask = pred_mask.astype(bool)
    
    alpha = 0.65
    out[mask] = (1 - alpha) * out[mask] + alpha * blue
    out = out.astype(np.uint8)
    
    # Encode as PNG (OpenCV uses BGR)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# --- ALERT ENGINE ---
def get_district_alert(district_name):
    """
    Full data pipeline: FFC River Flows + GEE 2010 Baseline + Sentinel-1/UNet Current Detection.
    """
    try:
        # 1. Init GEE
        if not init_gee():
            return {"error": "Failed to initialize Earth Engine."}

        # 2. Load Geometry
        gdf = gpd.read_file(SHAPEFILE)
        name_col = detect_name_column(gdf)
        
        # Match district
        mask = gdf[name_col].apply(lambda x: str(x).lower() == district_name.lower())
        if not mask.any():
            # Try partial match if exact fails
            mask = gdf[name_col].apply(lambda x: district_name.lower() in str(x).lower())
            
        if not mask.any():
            return {"error": f"District '{district_name}' not found in shapefile."}
            
        row = gdf[mask].iloc[0]
        actual_name = row[name_col]
        ee_geom = shapely_to_ee(row.geometry)
        bbox = list(row.geometry.bounds)

        # 3. Fetch River Data
        river_data = get_ffc_data()
        matched_station = None
        dn = actual_name.lower().strip()
        for station in river_data:
            st_name = (station.get("station") or "").lower()
            if dn in st_name or st_name in dn:
                matched_station = station
                break

        # 4. Historical 2010 Flood % (Landsat via GEE)
        hist_mask, _, _ = get_flood_mask(ee_geom)
        pct_2010 = flood_percent_for_district(ee_geom, hist_mask)

        # 5. Current Flood % (Sentinel-1 + UNet)
        date_end = datetime.now()
        date_start = date_end - timedelta(days=30)
        
        sar_bytes = fetch_current_sar_image(bbox, date_start, date_end, size=512)
        model = load_flood_model()
        unet_result = predict_flood(model, sar_bytes, actual_name, bbox)
        pct_current = unet_result["water_coverage_pct"]
        
        # Generate Flood Map Image
        map_base64 = get_image_base64(unet_result["display_arr"], unet_result["pred_mask"])

        # 6. Prepare data
        d_data = {
            "district": actual_name,
            "flood_pct_current": pct_current,
            "flood_pct_2010": pct_2010,
            "river_status": matched_station["status"] if matched_station else "UNKNOWN",
            "settlement_risk": unet_result["settlement_risk"]
        }
        
        # Risk Score (Manual calculation since AI is disabled)
        risk_score = min(10, max(1, round(pct_current / 5 + (3 if matched_station and matched_station['status'] == 'HIGH' else 0))))
        
        return {
            "risk_score": risk_score,
            "has_river": matched_station is not None,
            "status": d_data['river_status'],
            "station": matched_station['station'] if matched_station else "N/A",
            "inflow": matched_station['inflow'] if matched_station else "N/A",
            "outflow": matched_station['outflow'] if matched_station else "N/A",
            "pct_current": pct_current,
            "pct_2010": pct_2010,
            "map_image": map_base64,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}

if __name__ == "__main__":
    # Test
    print(get_district_alert("Swat"))
