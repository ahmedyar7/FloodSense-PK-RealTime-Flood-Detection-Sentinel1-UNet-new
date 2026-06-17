"""
ndwi.py
-------
Loads Landsat 5 imagery from Google Earth Engine for:
  - Baseline : July 2009  (normal water bodies)
  - Flood    : August 2010 (peak flood)

Computes NDWI and returns binary water masks.
"""

import ee

# ── MNDWI threshold ─────────────────────────────────────────────────────────
# MNDWI (Green - SWIR1) is better at detecting water in turbid conditions.
# -0.1 is a VERY sensitive threshold to catch any hint of moisture.
WATER_THRESHOLD = -0.1

# ── Date ranges ─────────────────────────────────────────────────────────────
BASELINE_START = "2009-01-01"   
BASELINE_END   = "2009-12-31"   
FLOOD_START    = "2010-07-28"   
FLOOD_END      = "2010-09-30"   


def scale_landsat(image):
    """
    Correctly scale optical SR bands (SR_B1–SR_B7).
    Scale: 0.0000275, Offset: -0.2
    """
    optical = image.select("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7") \
                   .multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True)


def mask_clouds(image):
    """
    Uses the QA_PIXEL band to mask clouds and cloud shadows.
    """
    qa    = image.select("QA_PIXEL")
    # Bit 3: Cloud Shadow, Bit 4: Cloud
    mask = qa.bitwiseAnd(1 << 3).eq(0) \
             .And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)


def compute_mndwi(image):
    """
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    Landsat 5 TM C2 L2: Green = SR_B2, SWIR1 = SR_B5
    """
    green = image.select("SR_B2")
    swir1 = image.select("SR_B5")
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI")
    return mndwi


def get_water_mask(start_date: str, end_date: str, region: ee.Geometry,
                   cloud_pct: int = 80, use_max: bool = False) -> ee.Image:
    """
    Returns a binary water mask (1=water, 0=land) for the given period.
    """
    collection = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
        .map(mask_clouds)
        .map(scale_landsat)
    )

    count = int(collection.size().getInfo())
    print(f"  📸 Images found ({start_date} → {end_date}): {count}")

    if count == 0:
        return ee.Image.constant(0).rename("water").clip(region)

    # For flood, max() helps capture peak water even between clouds
    # For baseline, median() is more stable
    composite = collection.max() if use_max else collection.median()

    index      = compute_mndwi(composite)
    water     = index.gt(WATER_THRESHOLD).rename("water")
    return water


def get_flood_mask(region: ee.Geometry):
    """
    Returns only NEW flood water:
      new_flood = (water in flood period) AND NOT (water in baseline)
    """
    print(f"🛰  Loading baseline ({BASELINE_START} → {BASELINE_END})...")
    baseline = get_water_mask(BASELINE_START, BASELINE_END, region, cloud_pct=70, use_max=False)

    print(f"🛰  Loading flood period ({FLOOD_START} → {FLOOD_END})...")
    # Use max() for flood to capture the extent
    flood    = get_water_mask(FLOOD_START, FLOOD_END, region, cloud_pct=90, use_max=True)

    # Use unmask(0) to ensure boolean logic works even where pixels are masked
    # We want: (Water in 2010) AND (NOT Water in 2009)
    # If a pixel was cloudy in 2010, we can't claim it's flooded (stay 0).
    # If a pixel was cloudy in 2009, we assume it wasn't water (so if it's water in 2010, it's "new").
    new_flood = flood.unmask(0).And(baseline.unmask(0).Not()).rename("new_flood")

    return new_flood, baseline, flood