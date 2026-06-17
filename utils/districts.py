"""
districts.py
------------
Loops over each district in the GeoJSON (or shapefile),
clips the flood mask, and calculates flood inundation %.
Supports: .json, .geojson, .shp
"""

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

from utils.ndwi import get_flood_mask


SCALE = 100         # Increased to 100m for faster processing of 148 districts
MAX_PIXELS = 1e10   # safety cap for ee.Reducer


def shapely_to_ee(geom):
    """Convert a Shapely geometry → ee.Geometry."""
    return ee.Geometry(mapping(geom))


def flood_percent_for_district(district_geom: ee.Geometry, flood_mask: ee.Image) -> float:
    """
    % of district area that is new flood water.
    = (flood pixels / total pixels) * 100
    """
    try:
        # Count flood pixels (1s)
        # We unmask to 0 so that masked areas (clouds) don't count as flood,
        # but we need to be careful with total_pixels.
        stats = flood_mask.reduceRegion(
            reducer   = ee.Reducer.sum().combine(
                reducer2 = ee.Reducer.count(),
                sharedInputs = True
            ),
            geometry  = district_geom,
            scale     = SCALE,
            maxPixels = MAX_PIXELS,
        )
        
        flood_pixels = stats.get("new_flood_sum")
        total_pixels = stats.get("new_flood_count")

        # Handle potential None values from GEE
        f_val = flood_pixels.getInfo() if flood_pixels is not None else 0
        t_val = total_pixels.getInfo() if total_pixels is not None else 0
        
        f_val = float(f_val) if f_val is not None else 0.0
        t_val = float(t_val) if t_val is not None else 0.0
        
        if f_val > 0:
            print(f"    📈 Found {f_val} flood pixels out of {t_val}")

        # Avoid division by zero
        if t_val > 0:
            pct = (f_val / t_val) * 100
        else:
            pct = 0.0
        
        return round(pct, 4)
    except Exception as e:
        print(f"    ⚠ Error computing pct: {e}")
        return 0.0


def detect_name_column(gdf: gpd.GeoDataFrame) -> str:
    """
    Auto-detects which column holds district names.
    Tries common column names from Kaggle / HDX / GADM datasets.
    """
    candidates = [
        "districts", "DISTRICTS", "district", "DISTRICT",
        "NAME_2", "name_2", "NAME", "name", "ADM2_EN", "adm2_en",
        "DIST_NAME", "dist_name", "District_N"
    ]
    for col in candidates:
        if col in gdf.columns:
            print(f"  ✅ Using column '{col}' for district names")
            return col

    # fallback: use first non-geometry string column
    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == object:
            print(f"  ⚠ Guessing column '{col}' for district names")
            return col

    return None


# ── Priority Districts ──────────────────────────────────────────────────────
# These are the districts most affected by floods in Pakistan
PRIORITY_DISTRICTS = [
    "Swat", "Shangla", "Kanju", "Mingora", "Kalam", "Behrain", # North
    "Charsadda", "Nowshera", "Peshawar", "Dera Ismail Khan",    # KP
    "Rajanpur", "Dera Ghazi Khan", "Muzaffargarh", "Layyah",    # Punjab
    "Sukkur", "Larkana", "Shikarpur", "Jacobabad", "Kashmore",  # Sindh
    "Jafferabad", "Naseerabad"                                  # Balochistan
]

def compute_district_flood(file_path: str) -> pd.DataFrame:
    """
    Reads a GeoJSON (.json / .geojson) or Shapefile (.shp) and returns:
      district | flood_pct | geometry
    """
    print(f"📂 Loading file: {file_path}")
    gdf = gpd.read_file(file_path)   
    print(f"   Total districts in file: {len(gdf)}")

    # Make sure CRS is WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Auto-detect name column
    name_col = detect_name_column(gdf)
    
    # Filter for Priority Districts
    if name_col:
        # Case-insensitive partial matching to be safe
        mask = gdf[name_col].apply(lambda x: any(p.lower() in str(x).lower() for p in PRIORITY_DISTRICTS))
        gdf = gdf[mask].copy()
        print(f"🎯 Filtered to {len(gdf)} priority districts.")

    # Full Pakistan bounding box for image filtering
    pakistan_bbox = ee.Geometry.BBox(60.0, 23.0, 77.5, 37.5)
    flood_mask, _, _ = get_flood_mask(pakistan_bbox)

    records = []
    total = len(gdf)

    for i, row in gdf.iterrows():
        name = row[name_col] if name_col else str(i)
        print(f"[{i+1}/{total}] Processing: {name}")

        try:
            ee_geom = shapely_to_ee(row.geometry)
            pct     = flood_percent_for_district(ee_geom, flood_mask)
            print(f"  → flood: {pct:.4f}%")
        except Exception as e:
            print(f"  ⚠ Skipped {name}: {e}")
            pct = None

        records.append({
            "district"  : name,
            "flood_pct" : pct,
            "geometry"  : row.geometry,
        })

    df = pd.DataFrame(records)
    return df