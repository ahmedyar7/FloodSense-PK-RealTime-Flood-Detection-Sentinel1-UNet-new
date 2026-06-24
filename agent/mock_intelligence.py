"""
Mock intelligence streams for the Disaster Intelligence Agent.

These simulate the upstream producers (Sentinel-1 U-Net, FFD hydraulics, and the
Landsat-5 2010 benchmark) so the agent can be exercised end-to-end without the
live data pipelines.
"""

from .schemas import (
    SatelliteIntelligence,
    HydraulicIntelligence,
    HistoricalIntelligence,
    RAGContext,
    RiverTrend,
    RiverStatus,
)

# Charsadda — a high-risk scenario (mirrors pr2.md Test 1: 31% coverage,
# rising rivers, 8% above the 2010 benchmark).
MOCK_SATELLITE = SatelliteIntelligence(
    district="Charsadda",
    flood_extent_percentage=31.0,
    affected_area_km2=303.5,
)

MOCK_HYDRAULIC = HydraulicIntelligence(
    station="Nowshera Gauge (Kabul River)",
    river_discharge_cusecs=95_000,
    inflow_cusecs=95_000,
    outflow_cusecs=70_000,
    trend=RiverTrend.RISING,
    status=RiverStatus.HIGH,
)

MOCK_HISTORICAL = HistoricalIntelligence(
    benchmark_year=2010,
    benchmark_flood_percentage=23.0,
)

MOCK_RAG_CONTEXT = RAGContext(
    context=(
        "Charsadda lies in the floodplain of the Kabul River and is particularly "
        "vulnerable to rapid inundation when embankments fail. The 2010 event "
        "displaced residents across the entire district."
    ),
    sources=["FFD River Stage Report", "NDMA Flood Report 2010"],
)
