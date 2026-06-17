"""
visualize.py
------------
Creates two outputs:
  1. Static choropleth PNG  (matplotlib)
  2. Interactive HTML map   (folium)
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.features import GeoJsonTooltip
import os


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Static PNG ────────────────────────────────────────────────────────────

def plot_static_map(df: pd.DataFrame, out_path: str = "outputs/flood_map.png"):
    """
    Choropleth: white → light red → deep red based on flood_pct.
    """
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf["flood_pct"] = gdf["flood_pct"].fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    gdf.plot(
        column     = "flood_pct",
        ax         = ax,
        cmap       = "YlOrRd",
        linewidth  = 0.4,
        edgecolor  = "#333333",
        legend     = True,
        legend_kwds= {
            "label"      : "Flood Inundation (%)",
            "orientation": "vertical",
            "shrink"     : 0.6,
        },
        missing_kwds= {"color": "#1e2530", "label": "No data"},
    )

    ax.set_title(
        "Pakistan — 2010 Flood Inundation by District\n(% of area flooded vs July 2009 baseline)",
        color    = "white",
        fontsize = 15,
        pad      = 14,
        fontweight = "bold",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] Static map saved -> {out_path}", flush=True)
    plt.close()


# ── 2. Interactive Folium map ────────────────────────────────────────────────

def plot_interactive_map(df: pd.DataFrame, out_path: str = "outputs/dashboard.html"):
    """
    Folium choropleth with hover tooltips showing district name, current flood %, 
    2010 flood %, and river status.
    """
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf["flood_pct_current"] = gdf["flood_pct_current"].fillna(0).round(2)
    gdf["flood_pct_2010"] = pd.to_numeric(gdf["flood_pct_2010"], errors="coerce").fillna(0).round(2)
    gdf["flood_pct"] = gdf["flood_pct_current"] # for color scale

    # Centre map on Pakistan
    m = folium.Map(location=[30.5, 69.0], zoom_start=6, tiles="CartoDB dark_matter")

    # Add Title
    title_html = '''
             <h3 align="center" style="font-size:20px; color:#ffffff; background-color:#1e2530; margin:0; padding:10px;">
             <b>Pakistan Flood Intelligence Dashboard (Sentinel-1 + Landsat + FFC)</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Colour scale
    vmax = gdf["flood_pct_current"].max()
    if pd.isna(vmax) or vmax <= 0:
        vmax = 10.0

    colormap = folium.LinearColormap(
        colors = ["#fff5f0", "#fc8d59", "#d73027"],
        vmin   = 0,
        vmax   = vmax,
        caption= "Current Flood Inundation (%)",
    )

    folium.GeoJson(
        gdf,
        style_function = lambda feature: {
            "fillColor"  : colormap(feature["properties"]["flood_pct_current"]),
            "color"      : "#555555",
            "weight"     : 0.5,
            "fillOpacity": 0.75,
        },
        tooltip = GeoJsonTooltip(
            fields   = ["district", "flood_pct_current", "flood_pct_2010", "river_status", "risk_score"],
            aliases  = ["District:", "Current Flood %:", "2010 Flood %:", "River Status:", "Risk Score (1-10):"],
            localize = True,
        ),
    ).add_to(m)

    colormap.add_to(m)

    m.save(out_path)
    print(f"[OK] Interactive map saved -> {out_path}", flush=True)
