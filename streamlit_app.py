import os
from concurrent.futures import ThreadPoolExecutor
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
import plotly.graph_objects as go
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
from agent import (
    RiskLevel,
    FloodState,
    evaluate_safe_zones,
    recommend_safe_zone,
    send_flood_alert,
    EmailConfigError,
    fetch_osm_safe_zones,
    build_safe_zone_candidates,
    straight_line_route,
    # Full disaster-intelligence workflow orchestrator + input schemas
    run_pipeline,
    SatelliteIntelligence,
    HydraulicIntelligence,
    HistoricalIntelligence,
    RAGContext,
    RiverTrend,
    RiverStatus,
)

SHAPEFILE = "pakistan_districts.json"

# Rough rural population density (people/km²) used to estimate population-at-risk
# for the authority-style situation summary in the alert email.
ESTIMATED_POP_DENSITY_PER_KM2 = 350


def _risk_level_from_score(score: float) -> RiskLevel:
    """Map the 1–10 defensible risk score onto the platform's risk bands."""
    if score >= 7:
        return RiskLevel.HIGH_RISK
    if score >= 4:
        return RiskLevel.MODERATE_RISK
    return RiskLevel.LOW_RISK


def _map_river_status(status: str) -> RiverStatus:
    """Map an FFC status string onto the agent's RiverStatus enum."""
    s = (status or "").upper()
    if "EXTREME" in s:
        return RiverStatus.EXTREME
    if "HIGH" in s:
        return RiverStatus.HIGH
    return RiverStatus.NORMAL


def _map_river_trend(trend: str) -> RiverTrend:
    """Map an FFC inflow-trend string onto the agent's RiverTrend enum."""
    t = (trend or "").lower()
    if "ris" in t:
        return RiverTrend.RISING
    if "fall" in t:
        return RiverTrend.FALLING
    return RiverTrend.STABLE


def _district_area_km2(geom) -> float | None:
    """Equal-area district size (km²) for the simulation's coverage maths.

    Used only as a fallback when current coverage is ~0% and the area cannot be
    derived from the flooded extent. Returns ``None`` if it cannot be computed.
    """
    try:
        # EPSG:6933 (World Cylindrical Equal Area) gives areas in m².
        area_m2 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0]
        return float(area_m2) / 1_000_000 if area_m2 > 0 else None
    except Exception:
        return None


def _build_rag_context(district: str) -> RAGContext:
    """Retrieve a little knowledge-base context for the district (best-effort)."""
    try:
        from rag import retrieve, build_context

        client, embedder = init_rag_system()
        docs = retrieve(f"flood risk and history for {district}", client, embedder, top_k=3)
        if not docs:
            return RAGContext()
        sources = sorted({d.get("source", "") for d in docs if d.get("source")})
        return RAGContext(context=build_context(docs), sources=sources)
    except Exception as e:  # RAG is optional context; never block the workflow
        print(f"RAG context unavailable for workflow: {e}")
        return RAGContext()


def render_disaster_workflow(
    district: str,
    geom,
    pct_current: float,
    pct_2010: float,
    affected_area_km2: float,
    matched_station: dict | None,
):
    """Run and visualise the end-to-end FloodSense-PK orchestration.

    Satellite + River + Historical + RAG
      → Disaster Intelligence Agent → Simulation Agent → Response & Communication.
    """
    st.subheader("Agentic Command Center")
    st.caption(
        "Four autonomous agents run in sequence — each consumes the previous "
        "agent's structured output: data fusion, risk classification, "
        "flood simulation, and response generation."
    )

    # ── Assemble the four input streams from the dashboard's live data ──
    satellite = SatelliteIntelligence(
        district=district,
        flood_extent_percentage=min(float(pct_current), 100.0),
        affected_area_km2=float(affected_area_km2),
    )

    if matched_station:
        hydraulic = HydraulicIntelligence(
            station=matched_station.get("station", district),
            river_discharge_cusecs=float(matched_station.get("inflow", 0) or 0),
            inflow_cusecs=float(matched_station.get("inflow", 0) or 0),
            outflow_cusecs=float(matched_station.get("outflow", 0) or 0),
            trend=_map_river_trend(matched_station.get("inflow_trend")),
            status=_map_river_status(matched_station.get("status")),
        )
    else:
        st.info(
            "No hydraulic station matched this area — simulating with a neutral "
            "river (no net flux). Connect a station for a dynamic projection."
        )
        hydraulic = HydraulicIntelligence(
            station=f"{district} (no gauge)",
            river_discharge_cusecs=0.0,
            inflow_cusecs=0.0,
            outflow_cusecs=0.0,
            trend=RiverTrend.STABLE,
            status=RiverStatus.NORMAL,
        )

    historical = HistoricalIntelligence(
        benchmark_year=2010,
        benchmark_flood_percentage=min(float(pct_2010), 100.0),
    )

    population_at_risk = int(float(affected_area_km2) * ESTIMATED_POP_DENSITY_PER_KM2)
    # Only needed when current coverage is ~0% (area can't be derived from extent).
    district_area = None if float(pct_current) > 0.01 else _district_area_km2(geom)

    # ── Live agent activity log ──
    with st.status("Orchestrating the four-agent pipeline…", expanded=True) as status:
        st.write("**Data Fusion Agent** — merging satellite, hydraulic and historical streams…")
        st.write("**Data Fusion Agent** — querying RAG knowledge base for district context…")
        rag_context = _build_rag_context(district)
        st.write(
            f"**Data Fusion Agent** — done. Fused 4 streams "
            f"({len(rag_context.sources)} knowledge sources grounded)."
        )
        st.write("**Disaster Intelligence Agent** — classifying multi-factor risk…")
        st.write("**Simulation Agent** — projecting flood progression across horizons…")
        st.write("**Response Agent** — drafting citizen and authority communications…")
        result = run_pipeline(
            satellite,
            hydraulic,
            historical,
            rag_context,
            population_at_risk=population_at_risk,
            district_area_km2=district_area,
        )
        st.write("**Pipeline** — all agents completed.")
        status.update(
            label="Agent pipeline complete — 4 of 4 agents finished",
            state="complete",
            expanded=False,
        )

    assessment = result.assessment
    progression = result.progression

    # ── Pipeline flow banner ──
    _agent_cards = [
        ("01", "Data Fusion", "Satellite · River · Historical · RAG"),
        ("02", "Intelligence", "Risk classification"),
        ("03", "Simulation", "Flood progression"),
        ("04", "Response", "Alerts & evacuation"),
    ]
    _cards_html = "<div class='agent-flow'>"
    for i, (step, name, role) in enumerate(_agent_cards):
        if i:
            _cards_html += "<div class='agent-arrow'>&#8594;</div>"
        _cards_html += (
            "<div class='agent-card'>"
            f"<div class='agent-step'>STAGE {step}</div>"
            f"<div class='agent-name'>{name} Agent</div>"
            f"<div class='agent-role'>{role}</div>"
            "<span class='agent-status'>COMPLETE</span>"
            "</div>"
        )
    _cards_html += "</div>"
    st.markdown(_cards_html, unsafe_allow_html=True)

    # ── Per-agent result tabs (one per pipeline stage) ──
    tab_fusion, tab_intel, tab_sim, tab_resp = st.tabs(
        [
            "Data Fusion Agent",
            "Intelligence Agent",
            "Simulation Agent",
            "Response Agent",
        ]
    )

    # ── 0 · Data Fusion Agent ──
    with tab_fusion:
        st.markdown("#### Fused input streams")
        st.caption(
            "The Data Fusion Agent merges four live data streams into one "
            "structured payload that every downstream agent consumes."
        )
        in1, in2, in3, in4 = st.columns(4)
        in1.markdown(
            "<div class='stream-chip'><span class='chip-label'>Satellite</span>"
            f"<div class='chip-value'>{satellite.flood_extent_percentage:.2f}%</div>"
            f"<div class='chip-sub'>{satellite.affected_area_km2:,.0f} km² flooded</div></div>",
            unsafe_allow_html=True,
        )
        in2.markdown(
            "<div class='stream-chip'><span class='chip-label'>Hydraulic</span>"
            f"<div class='chip-value'>{hydraulic.status.value}</div>"
            f"<div class='chip-sub'>{hydraulic.station} · {hydraulic.trend.value}</div></div>",
            unsafe_allow_html=True,
        )
        in3.markdown(
            "<div class='stream-chip'><span class='chip-label'>Historical</span>"
            f"<div class='chip-value'>{historical.benchmark_flood_percentage:.2f}%</div>"
            f"<div class='chip-sub'>{historical.benchmark_year} benchmark</div></div>",
            unsafe_allow_html=True,
        )
        in4.markdown(
            "<div class='stream-chip'><span class='chip-label'>RAG Knowledge</span>"
            f"<div class='chip-value'>{len(rag_context.sources)} sources</div>"
            f"<div class='chip-sub'>{', '.join(rag_context.sources[:2]) or 'no grounding'}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("##### Hydraulic detail")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Field": "Station", "Value": hydraulic.station},
                        {"Field": "Inflow (cusecs)", "Value": f"{hydraulic.inflow_cusecs:,.0f}"},
                        {"Field": "Outflow (cusecs)", "Value": f"{hydraulic.outflow_cusecs:,.0f}"},
                        {"Field": "Trend", "Value": hydraulic.trend.value},
                        {"Field": "Status", "Value": hydraulic.status.value},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with f2:
            st.markdown("##### Knowledge grounding")
            if rag_context.sources:
                for src in rag_context.sources:
                    st.caption(f"• {src}")
                if rag_context.context:
                    with st.expander("Retrieved context passages", expanded=False):
                        st.text(rag_context.context)
            else:
                st.caption(
                    "No knowledge-base passages matched this district — agents "
                    "ran on live sensor data only."
                )

    risk_palette = {
        RiskLevel.HIGH_RISK: "#e74c3c",
        RiskLevel.MODERATE_RISK: "#e67e22",
        RiskLevel.LOW_RISK: "#2ecc71",
    }
    risk_color = risk_palette[assessment.risk_level]

    # ── 1 · Disaster Intelligence Agent ──
    with tab_intel:
        g1, g2 = st.columns([1, 1.4])
        with g1:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=assessment.flood_coverage_percentage,
                    number={"suffix": "%", "font": {"size": 42, "color": "#ffffff"}},
                    title={
                        "text": f"<b>{assessment.risk_level.value}</b><br>"
                        "<span style='font-size:0.75em;color:#93a1c9'>Flood coverage driving the assessment</span>",
                        "font": {"size": 16, "color": risk_color},
                    },
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#93a1c9"},
                        "bar": {"color": risk_color, "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 10], "color": "rgba(46,204,113,0.18)"},
                            {"range": [10, 30], "color": "rgba(230,126,34,0.18)"},
                            {"range": [30, 100], "color": "rgba(231,76,60,0.18)"},
                        ],
                        "threshold": {
                            "line": {"color": risk_color, "width": 4},
                            "thickness": 0.85,
                            "value": assessment.flood_coverage_percentage,
                        },
                    },
                )
            )
            gauge.update_layout(
                height=300,
                margin=dict(l=30, r=30, t=70, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#dbe4ff"},
            )
            st.plotly_chart(gauge, use_container_width=True)
        with g2:
            st.markdown("#### Risk Assessment")
            st.write(assessment.explanation)
            st.warning(f"**Recommended action:** {assessment.recommended_action}")
            if rag_context.sources:
                st.caption("Knowledge grounding: " + ", ".join(rag_context.sources))

    # ── 2 · Simulation Agent ──
    with tab_sim:
        s1, s2, s3 = st.columns(3)
        s1.metric("Current Coverage", f"{progression.initial_coverage_percentage:.1f}%")
        s2.metric(
            "Projected Peak",
            f"{progression.peak_coverage_percentage:.1f}%",
            delta=f"{progression.peak_coverage_percentage - progression.initial_coverage_percentage:+.1f}%",
            delta_color="inverse",
        )
        s3.metric("Trajectory", "Expanding" if progression.expanding else "Receding")

        # Projection curve (start point + each horizon) as a gradient area chart.
        hours = [0] + [p.horizon_hours for p in progression.projections]
        coverage = [progression.initial_coverage_percentage] + [
            p.projected_coverage_percentage for p in progression.projections
        ]
        peak_i = int(max(range(len(coverage)), key=coverage.__getitem__))

        proj_fig = go.Figure()
        proj_fig.add_trace(
            go.Scatter(
                x=hours,
                y=coverage,
                mode="lines+markers",
                line={"color": "#3d7bff", "width": 3, "shape": "spline"},
                marker={"size": 9, "color": "#8aa5ff", "line": {"color": "#3d7bff", "width": 2}},
                fill="tozeroy",
                fillcolor="rgba(61,123,255,0.15)",
                name="Projected coverage",
                hovertemplate="+%{x} h → %{y:.2f}%<extra></extra>",
            )
        )
        proj_fig.add_annotation(
            x=hours[peak_i],
            y=coverage[peak_i],
            text=f"Peak {coverage[peak_i]:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e67e22",
            font={"color": "#e67e22", "size": 13},
            yshift=8,
        )
        proj_fig.update_layout(
            title="Projected flood coverage",
            height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#dbe4ff"},
            xaxis={"title": "Hours ahead", "gridcolor": "rgba(255,255,255,0.06)"},
            yaxis={"title": "Coverage %", "gridcolor": "rgba(255,255,255,0.06)"},
            showlegend=False,
        )
        st.plotly_chart(proj_fig, use_container_width=True)

        c_ds, c_tbl = st.columns([1, 1.2])
        with c_ds:
            st.markdown("##### Downstream flood-wave propagation")
            if progression.downstream_districts_at_risk:
                ds = progression.downstream_districts_at_risk
                ds_fig = go.Figure(
                    go.Bar(
                        x=[d.eta_hours for d in ds],
                        y=[d.district for d in ds],
                        orientation="h",
                        marker={
                            "color": [d.eta_hours for d in ds],
                            "colorscale": [[0, "#e74c3c"], [1, "#f4d03f"]],
                            "reversescale": True,
                        },
                        text=[f"{d.eta_hours} h" for d in ds],
                        textposition="outside",
                        hovertemplate="%{y}: wave ETA %{x} h<extra></extra>",
                    )
                )
                ds_fig.update_layout(
                    height=90 + 45 * len(ds),
                    margin=dict(l=10, r=40, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#dbe4ff"},
                    xaxis={"title": "Wave ETA (hours)", "gridcolor": "rgba(255,255,255,0.06)"},
                    yaxis={"autorange": "reversed"},
                )
                st.plotly_chart(ds_fig, use_container_width=True)
            else:
                st.caption("No downstream propagation expected under current river conditions.")
        with c_tbl:
            st.markdown("##### Horizon detail")
            detail_df = pd.DataFrame(
                [
                    {
                        "Horizon (h)": p.horizon_hours,
                        "Coverage %": p.projected_coverage_percentage,
                        "Area (km²)": p.projected_affected_area_km2,
                        "Δ Area (km²)": p.net_area_change_km2,
                        "Population at risk": p.projected_population_at_risk,
                    }
                    for p in progression.projections
                ]
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

        st.info(progression.summary)

    # ── 3 · Response & Communication Agent ──
    with tab_resp:
        if result.recommended_safe_zone is not None:
            sz = result.recommended_safe_zone
            z1, z2, z3 = st.columns(3)
            z1.metric("Safe Zone", sz.name)
            if result.evacuation_route is not None:
                rt = result.evacuation_route
                z2.metric("Distance", f"{rt.distance_km:g} km")
                z3.metric("Travel Time", f"~{rt.estimated_travel_time_min} min")
                st.success(
                    f"**Evacuation route:** {' → '.join(rt.path)}  \n"
                    f"Destination coordinates: `{sz.latitude:.4f}, {sz.longitude:.4f}` · "
                    f"[Open in Google Maps](https://www.google.com/maps/dir/?api=1&destination={sz.latitude},{sz.longitude})"
                )
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(
                "<div class='alert-box citizen'><span class='alert-tag'>Citizen Alert</span>"
                f"{result.citizen_alert}</div>",
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                "<div class='alert-box authority'><span class='alert-tag'>Authority Situation Report</span>"
                f"{result.authority_alert}</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Safe-zone discovery and routing use live OpenStreetMap data in the "
            "Personal Flood Alert email path (sidebar)."
        )


def _sample_elevations(points):
    """Return SRTM ground elevation (m) for each ``(lat, lon)`` point.

    Uses one Earth Engine ``reduceRegions`` call over USGS SRTM. Returns a list
    parallel to ``points``; an entry is ``None`` when the sample is unavailable
    (the caller then falls back to a safe default elevation).
    """
    if not points:
        return []
    try:
        feats = [
            ee.Feature(ee.Geometry.Point([lon, lat]), {"idx": i})
            for i, (lat, lon) in enumerate(points)
        ]
        fc = ee.FeatureCollection(feats)
        srtm = ee.Image("USGS/SRTMGL1_003")
        sampled = srtm.reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=30
        ).getInfo()
        elev_by_idx = {}
        for f in sampled.get("features", []):
            props = f.get("properties", {})
            elev_by_idx[props.get("idx")] = props.get("first")
        return [elev_by_idx.get(i) for i in range(len(points))]
    except Exception as e:  # elevation is best-effort; never block the alert
        print(f"Elevation sampling failed: {e}")
        return [None] * len(points)


def _maybe_send_flood_alert(
    recipient_email,
    always_email,
    district,
    risk_score,
    coverage_pct,
    affected_area_km2,
    geom,
    bbox,
    pred_mask,
):
    """Email a personalised evacuation alert when the area is in danger.

    Auto-sends on HIGH risk; the ``always_email`` opt-in forces a send at any
    level. Safe zones are real shelters fetched live from OpenStreetMap around the
    selected area's centroid, with flood-zone/route checks derived from the UNet
    flood mask and elevation sampled from GEE SRTM.
    """
    recipient_email = (recipient_email or "").strip()
    if not recipient_email:
        return

    risk_level = _risk_level_from_score(risk_score)
    if risk_level != RiskLevel.HIGH_RISK and not always_email:
        st.info(
            f"No alert email sent — current risk for {district} is "
            f"**{risk_level.value}**. Tick *“Email me regardless of risk level”* "
            "in the sidebar to receive it anyway."
        )
        return

    # Origin proxy: centroid of the area actually under analysis.
    centroid = geom.centroid
    origin_lat, origin_lon = float(centroid.y), float(centroid.x)

    with st.spinner("Finding real safe zones near this area (OpenStreetMap)…"):
        try:
            raw_shelters = fetch_osm_safe_zones(origin_lat, origin_lon, radius_m=20000)
        except Exception as e:  # Overpass timeout / network / HTTP error
            st.error(f"Could not query OpenStreetMap for safe zones: {e}")
            return

    if not raw_shelters:
        st.warning(
            "No schools, hospitals, or shelters are mapped in OpenStreetMap near "
            f"**{district}** — no shelter-specific alert was sent."
        )
        return

    elevations = _sample_elevations(
        [(r["latitude"], r["longitude"]) for r in raw_shelters]
    )
    candidates = build_safe_zone_candidates(
        raw_shelters,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        mask=pred_mask,
        bbox=bbox,
        elevations=elevations,
    )

    safe_zone = recommend_safe_zone(candidates)
    if safe_zone is None:
        # No candidate passed every check; recommend the nearest one that is at
        # least outside the flood zone rather than sending nothing.
        outside = [c for c in candidates if not c.in_flood_zone]
        if not outside:
            st.error(
                "Every nearby shelter sits inside the detected flood zone; "
                "no safe destination to recommend."
            )
            return
        safe_zone = min(
            outside, key=lambda c: c.distance_km if c.distance_km is not None else 1e9
        )
        st.warning(
            "No shelter passed every safety check; recommending the nearest one "
            f"outside the flood zone: **{safe_zone.name}**."
        )

    route = straight_line_route(
        f"{district} (your area)", origin_lat, origin_lon, safe_zone
    )
    flood_state = FloodState(
        district=district,
        flood_coverage_percentage=min(float(coverage_pct), 100.0),
        affected_area_km2=float(affected_area_km2),
        population_at_risk=int(float(affected_area_km2) * ESTIMATED_POP_DENSITY_PER_KM2),
        available_shelters=len(evaluate_safe_zones(candidates)),
    )

    try:
        with st.spinner(f"Sending flood alert to {recipient_email}…"):
            send_flood_alert(
                recipient=recipient_email,
                risk_level=risk_level,
                flood_state=flood_state,
                safe_zone=safe_zone,
                route=route,
            )
        st.success(
            f"Flood alert emailed to **{recipient_email}** — directing them to "
            f"**{safe_zone.name}** "
            f"({safe_zone.latitude:.5f}, {safe_zone.longitude:.5f}), "
            f"{route.distance_km:g} km, ~{route.estimated_travel_time_min} min."
        )
    except EmailConfigError as e:
        st.error(
            f"Email not configured, alert not sent: {e}"
        )
    except Exception as e:  # network/auth/SMTP errors must not crash the dashboard
        st.error(f"Failed to send alert email to {recipient_email}: {e}")

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
            max-width: 1600px;
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

        /* ── Agentic pipeline banner ── */
        .agent-flow {
            display: flex;
            align-items: stretch;
            gap: 0;
            margin: 0.5rem 0 1.25rem 0;
            flex-wrap: wrap;
        }
        .agent-card {
            flex: 1 1 0;
            min-width: 150px;
            background: linear-gradient(160deg, rgba(51,102,204,0.12) 0%, rgba(255,255,255,0.02) 100%);
            border: 1px solid rgba(138,165,255,0.25);
            border-radius: 12px;
            padding: 0.9rem 0.8rem;
            text-align: center;
            position: relative;
        }
        .agent-card .agent-step {
            font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
            color: #8aa5ff;
        }
        .agent-card .agent-name {
            font-size: 1rem; font-weight: 700; color: #ffffff;
            margin-top: 0.35rem; letter-spacing: 0.3px;
        }
        .agent-card .agent-role {
            font-size: 0.78rem; color: #aab6d8; margin-top: 0.2rem;
        }
        .agent-card .agent-status {
            display: inline-block; margin-top: 0.55rem;
            font-size: 0.68rem; font-weight: 700; letter-spacing: 1px;
            padding: 3px 12px; border-radius: 999px;
            background: rgba(46, 204, 113, 0.15); color: #2ecc71;
            border: 1px solid rgba(46, 204, 113, 0.4);
        }
        .agent-arrow {
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem; color: #8aa5ff; padding: 0 6px;
        }
        @media (max-width: 768px) {
            .agent-flow { flex-direction: column; }
            .agent-arrow { transform: rotate(90deg); padding: 2px 0; }
        }

        /* ── Stream (input) chips ── */
        .stream-chip {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            height: 100%;
        }
        .stream-chip .chip-label {
            font-size: 0.7rem; letter-spacing: 1.2px; color: #aab6d8;
            text-transform: uppercase; font-weight: 700;
        }
        .stream-chip .chip-value {
            font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 2px;
        }
        .stream-chip .chip-sub { font-size: 0.78rem; color: #8aa5ff; margin-top: 2px; }

        /* ── Chat hero ── */
        .chat-hero {
            padding: 1.1rem 1.4rem;
            border-radius: 14px;
            background: linear-gradient(120deg, rgba(51,102,204,0.18) 0%, rgba(102,51,204,0.10) 100%);
            border: 1px solid rgba(138,165,255,0.3);
            margin-bottom: 1rem;
        }
        .chat-hero .chat-title {
            font-size: 1.35rem; font-weight: 700; color: #ffffff; margin: 0;
        }
        .chat-hero .chat-sub { font-size: 0.85rem; color: #b9c6f2; margin: 0.3rem 0 0 0; }
        .kb-badge {
            display: inline-block; margin-top: 0.55rem; margin-right: 0.4rem;
            font-size: 0.68rem; font-weight: 600;
            padding: 3px 12px; border-radius: 999px;
            background: rgba(46, 204, 113, 0.12); color: #2ecc71;
            border: 1px solid rgba(46, 204, 113, 0.35);
        }
        .kb-badge.dim {
            background: rgba(255,255,255,0.05); color: #93a1c9;
            border-color: rgba(255,255,255,0.15);
        }

        /* ── Alert preview boxes ── */
        .alert-box {
            border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.02);
            font-size: 0.9rem; line-height: 1.6; color: #e8ecf8;
            white-space: pre-wrap;
        }
        .alert-box.citizen { border-left: 5px solid #e67e22; }
        .alert-box.authority { border-left: 5px solid #3366cc; }
        .alert-box .alert-tag {
            font-size: 0.66rem; font-weight: 700; letter-spacing: 1px;
            text-transform: uppercase; color: #93a1c9; display: block;
            margin-bottom: 0.45rem;
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
    """Connect to Qdrant and ensure the disaster-knowledge collection is populated.

    Prefers the persistent Docker Qdrant on ``localhost:6333`` (so the collection
    shows up in the dashboard and survives restarts), falling back to an in-memory
    instance if Docker isn't reachable. Ingestion only runs when the collection is
    missing or empty, so restarts don't re-embed or create duplicate points.
    """
    from rag.ingest import COLLECTION_NAME

    embedder = EmbeddingPipeline()
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=5)
        client.get_collections()  # probe: raises if Docker Qdrant isn't up
    except Exception:
        client = QdrantClient(":memory:")

    collections = {c.name for c in client.get_collections().collections}
    populated = (
        COLLECTION_NAME in collections
        and client.count(COLLECTION_NAME).count > 0
    )
    if not populated:
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


# GEE single-request download cap (pixels per side) — larger images are tiled.
GEE_MAX_TILE_PX = 1024


def _bbox_pixel_dims(bbox, size):
    """Pixel width/height for a bbox where ``size`` is the long-side pixels."""
    w_deg = bbox[2] - bbox[0]
    h_deg = bbox[3] - bbox[1]
    if w_deg >= h_deg:
        px_w = int(size)
        px_h = max(1, int(round(size * h_deg / w_deg)))
    else:
        px_h = int(size)
        px_w = max(1, int(round(size * w_deg / h_deg)))
    return px_w, px_h


def _download_sar_tile(masked_image, tile_bbox, px_w, px_h):
    """Download one GeoTIFF tile from GEE and return it as a float32 array."""
    url = masked_image.getDownloadURL(
        {
            "region": tile_bbox,
            "dimensions": f"{px_w}x{px_h}",
            "format": "GEO_TIFF",
            "bands": ["VV"],
        }
    )
    res = requests.get(url, timeout=300)
    res.raise_for_status()
    with rasterio.open(io.BytesIO(res.content)) as src:
        return src.read(1).astype(np.float32)


def fetch_current_sar_image(bbox, date_start, date_end, size=256):
    """Fetch the Sentinel-1 VV composite for ``bbox`` as GeoTIFF bytes.

    Requests up to ``GEE_MAX_TILE_PX`` are a single download (fast path).
    Larger ``size`` values — needed to keep full ~80 m resolution over big
    districts — are split into a grid of aligned sub-bbox tiles, downloaded in
    parallel, stitched into one array, and re-encoded as a single GeoTIFF so
    every downstream consumer (rasterio / tiled UNet) works unchanged.
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

    px_w, px_h = _bbox_pixel_dims(bbox, size)

    # ── Fast path: fits in a single GEE request ──
    if max(px_w, px_h) <= GEE_MAX_TILE_PX:
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

    # ── Tiled path: full-resolution mosaic of the whole district ──
    minx, miny, maxx, maxy = bbox
    w_deg, h_deg = maxx - minx, maxy - miny
    tiles_x = -(-px_w // GEE_MAX_TILE_PX)  # ceil division
    tiles_y = -(-px_h // GEE_MAX_TILE_PX)

    # Global pixel edges keep every tile perfectly aligned on the shared grid.
    x_edges = [round(i * px_w / tiles_x) for i in range(tiles_x + 1)]
    y_edges = [round(i * px_h / tiles_y) for i in range(tiles_y + 1)]

    jobs = []  # (row, col, tile_bbox, tile_px_w, tile_px_h)
    for iy in range(tiles_y):
        for ix in range(tiles_x):
            tile_bbox = [
                minx + (x_edges[ix] / px_w) * w_deg,
                maxy - (y_edges[iy + 1] / px_h) * h_deg,
                minx + (x_edges[ix + 1] / px_w) * w_deg,
                maxy - (y_edges[iy] / px_h) * h_deg,
            ]
            jobs.append(
                (
                    iy,
                    ix,
                    tile_bbox,
                    x_edges[ix + 1] - x_edges[ix],
                    y_edges[iy + 1] - y_edges[iy],
                )
            )

    print(
        f"  Tiled SAR fetch: {px_w}x{px_h}px as {tiles_x}x{tiles_y} grid "
        f"({len(jobs)} tiles)"
    )

    canvas = np.zeros((px_h, px_w), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_download_sar_tile, masked_image, tb, tw, th): (iy, ix, tw, th)
            for iy, ix, tb, tw, th in jobs
        }
        for future in futures:
            iy, ix, tw, th = futures[future]
            tile_arr = future.result()  # propagate download errors
            y0, x0 = y_edges[iy], x_edges[ix]
            canvas[y0 : y0 + th, x0 : x0 + tw] = tile_arr[:th, :tw]
            print(f"  tile ({iy},{ix}) done: {tw}x{th}px")

    # Re-encode the stitched mosaic as one georeferenced GeoTIFF.
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    with rasterio.io.MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=px_h,
            width=px_w,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(canvas, 1)
        stitched = mem.read()

    print(f"  Stitched GeoTIFF: {len(stitched)} bytes ({px_w}x{px_h}px)")
    return stitched


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


RAG_SUGGESTED_QUESTIONS = [
    "What happened in the 2010 Pakistan floods?",
    "What are NDMA's flood evacuation protocols?",
    "Which rivers and barrages are most flood-prone?",
    "How should districts prepare before monsoon season?",
]


@st.fragment
def render_rag_chatbot():
    rag_client, rag_embedder = init_rag_system()

    # Knowledge-base status badges for the hero card.
    from rag.ingest import COLLECTION_NAME

    try:
        kb_points = rag_client.count(COLLECTION_NAME).count
    except Exception:
        kb_points = 0
    backend_cls = type(getattr(rag_client, "_client", rag_client)).__name__
    kb_backend = (
        "Qdrant · persistent" if "Remote" in backend_cls else "Qdrant · in-memory"
    )

    st.markdown(
        "<div class='chat-hero'>"
        "<p class='chat-title'>Disaster Knowledge Assistant</p>"
        "<p class='chat-sub'>Ask about Pakistan flood history, NDMA protocols, FFD "
        "river data and disaster response — answers are grounded in the RAG "
        "knowledge base with cited sources.</p>"
        f"<span class='kb-badge'>{kb_points} knowledge chunks indexed</span>"
        f"<span class='kb-badge dim'>{kb_backend}</span>"
        "<span class='kb-badge dim'>Gemini with Groq fallback</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    if "rag_messages" not in st.session_state:
        st.session_state["rag_messages"] = []

    # Suggested starter questions (only while the chat is empty).
    pending = None
    if not st.session_state["rag_messages"]:
        st.markdown("**Suggested questions**")
        sugg_cols = st.columns(2)
        for i, q in enumerate(RAG_SUGGESTED_QUESTIONS):
            if sugg_cols[i % 2].button(q, key=f"rag_sugg_{i}", use_container_width=True):
                pending = q
    else:
        if st.button("Clear conversation", key="rag_clear"):
            st.session_state["rag_messages"] = []
            st.rerun(scope="fragment")

    for msg in st.session_state["rag_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources", expanded=False):
                    for src in msg["sources"]:
                        st.caption(src)

    user_input = st.chat_input("Ask about floods, NDMA protocols, or river systems...")
    query = user_input or pending
    if query:
        st.session_state["rag_messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                response, source_docs = rag_query(
                    query, rag_client, rag_embedder, _rag_llm_fn
                )
            st.markdown(response)
            source_lines = []
            for doc in source_docs:
                line = f"**{doc.get('source', 'Unknown')}**"
                if doc.get("section"):
                    line += f" — {doc['section']}"
                if doc.get("page_number"):
                    line += f" (p. {doc['page_number']})"
                source_lines.append(line)
            if source_lines:
                with st.expander("Sources", expanded=False):
                    for src in source_lines:
                        st.caption(src)

        st.session_state["rag_messages"].append(
            {"role": "assistant", "content": response, "sources": source_lines}
        )
        st.rerun(scope="fragment")


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

    # Personal flood-alert subscription
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Personal Flood Alert")
    recipient_email = st.sidebar.text_input(
        "Your email (optional)",
        help=(
            "If this area is in danger, we'll email you the situation, the nearest "
            "safe zone with exact coordinates, and the estimated travel time."
        ),
    )
    always_email = st.sidebar.checkbox(
        "Email me regardless of risk level[FOR TESTING ONLY]",
        value=False,
        help="Otherwise you are only emailed when the risk is HIGH.",
    )

    # Actions
    with st.sidebar:
        st.markdown("### Run")
        run = st.button("Run analysis", type="primary", use_container_width=True)

    # Results persist in session state, so widget clicks (view toggles, chat)
    # re-render the last analysis instead of wiping the page.
    if not run and "analysis" not in st.session_state:
        st.info("Select a scale/district and press Run analysis.")
        # Knowledge assistant is available even before an analysis run.
        render_rag_chatbot()
        return

    if run:
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

        # Full-district resolution: sizes beyond 1024 px are fetched as an aligned
        # tile mosaic (see fetch_current_sar_image), so big districts keep ~80 m/px.
        # 4096 px caps memory (~64 MB float32) and UNet inference time.
        final_size = max(256, min(4096, max(pixels_w, pixels_h)))

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

        # 4b) Personal flood-alert email (Response & Communication Agent)
        _maybe_send_flood_alert(
            recipient_email=recipient_email,
            always_email=always_email,
            district=district,
            risk_score=risk_score,
            coverage_pct=pct_current,
            affected_area_km2=unet_result.get("affected_area_km2", 0.0),
            geom=geom,
            bbox=bbox,
            pred_mask=unet_result.get("pred_mask"),
        )

        # 5) Visuals
        sar_gray = render_sar_gray(unet_result["display_arr"])
        overlay_img = render_unet_overlay(
            unet_result["display_arr"], unet_result["pred_mask"]
        )
        prob_heatmap = render_prob_heatmap(unet_result["pred_prob"])

        with st.spinner("Aligning 2010 historical visuals..."):
            # PNG visualisation stays a single GEE request — cap at 1024 px.
            hist_mask_bytes = fetch_2010_mask_image(
                st.session_state["hist_flood_mask"], bbox=bbox, size=min(final_size, 1024)
            )

        # Persist everything the render section needs so widget interactions
        # (view toggles, chat, etc.) don't wipe the page on rerun.
        st.session_state["analysis"] = {
            "district": district,
            "geom": geom,
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "pct_2010": pct_2010,
            "pct_current": pct_current,
            "risk_score": risk_score,
            "settlement_risk": settlement_risk,
            "unet_result": unet_result,
            "matched_station": matched_station,
            "df_flows": df_flows,
            "river_flows": river_flows,
            "insights": insights,
            "sar_gray": sar_gray,
            "overlay_img": overlay_img,
            "prob_heatmap": prob_heatmap,
            "hist_mask_bytes": hist_mask_bytes,
        }

    # ── Render from persisted results (survives widget-triggered reruns) ──
    if "analysis" not in st.session_state:
        return
    a = st.session_state["analysis"]
    district = a["district"]
    geom = a["geom"]
    bbox = a["bbox"]
    start_date = a["start_date"]
    pct_2010 = a["pct_2010"]
    pct_current = a["pct_current"]
    risk_score = a["risk_score"]
    settlement_risk = a["settlement_risk"]
    unet_result = a["unet_result"]
    matched_station = a["matched_station"]
    df_flows = a["df_flows"]
    river_flows = a["river_flows"]
    insights = a["insights"]
    sar_gray = a["sar_gray"]
    overlay_img = a["overlay_img"]
    prob_heatmap = a["prob_heatmap"]
    hist_mask_bytes = a["hist_mask_bytes"]

    st.divider()

    # Clean, non-overlapping UI using tabs
    t1, t2, t3, t4, t5, t6 = st.tabs(
        [
            "Overview",
            "Detection",
            "River Flows",
            "AI Intelligence",
            "Agentic Workflow",
            "Knowledge Assistant",
        ]
    )

    with t1:
        st.subheader("Flood Severity Comparison: 2010 vs. Current")
        st.markdown("""
        Evidence-based comparison between the historical maximum (2010) and current AI detection. 
        """)

        # ── Visual Comparison Section ──
        view_mode = st.radio(
            "Image view",
            ["Side by side", "Large view"],
            horizontal=True,
            label_visibility="collapsed",
            key="overview_view_mode",
        )
        if view_mode == "Side by side":
            col_img1, col_img2 = st.columns(2)
        else:
            # Full-width stacked images — each uses the entire page width.
            col_img1 = col_img2 = st.container()
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
                "**Data Gap:** No hydraulic monitoring station found for this district boundary. Confidence in river status is reduced."
            )

    with t2:
        st.subheader("UNet Deep Learning Analysis")
        st.caption("Detailed breakdown of AI model outputs and confidence levels.")

        det_view = st.radio(
            "Image view",
            ["Side by side", "Large view"],
            horizontal=True,
            label_visibility="collapsed",
            key="detection_view_mode",
        )
        if det_view == "Side by side":
            col_a, col_b = st.columns(2)
        else:
            col_a = col_b = st.container()

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
                    st.markdown(f"#### Situation Summary")
                    st.info(line.split("]")[-1].strip())
                elif "[HYDRAULIC" in line.upper():
                    st.markdown(f"#### Hydraulic Analysis")
                    st.write(line.split("]")[-1].strip())
                elif "[HISTORICAL" in line.upper():
                    st.markdown(f"#### Historical Benchmark")
                    st.write(line.split("]")[-1].strip())
                elif "[OPERATIONAL" in line.upper():
                    st.markdown(f"#### Operational Actions")
                    st.warning(line.split("]")[-1].strip())
                elif "[CONFIDENCE" in line.upper():
                    st.caption(f"**Confidence Level:** {line.split(']')[-1].strip()}")
        else:
            st.markdown(insights)

        st.divider()
        st.caption(
            f"Governance: Weighted Risk Formula (Flood% 40, Delta 30, Hydraulic 30)"
        )

    with t5:
        render_disaster_workflow(
            district=district,
            geom=geom,
            pct_current=pct_current,
            pct_2010=pct_2010,
            affected_area_km2=unet_result.get("affected_area_km2", 0.0),
            matched_station=matched_station,
        )

    with t6:
        render_rag_chatbot()


if __name__ == "__main__":
    main()
