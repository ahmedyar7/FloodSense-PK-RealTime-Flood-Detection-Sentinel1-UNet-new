"""
Tests for the end-to-end FloodSense-PK orchestrator (``agent.pipeline``).

Verifies the complete workflow from ``final-workflow.md`` runs as a single call:

    Satellite + River + Historical + RAG
      → Disaster Intelligence Agent
      → Simulation Agent
      → Response & Communication Agent
      → safe-zone recommendation, evacuation route, citizen + authority alerts
"""

from agent import (
    GraphEdge,
    LocationType,
    PipelineResult,
    RiskLevel,
    SafeZoneCandidate,
    run_pipeline,
)
from agent.mock_intelligence import (
    MOCK_HISTORICAL,
    MOCK_HYDRAULIC,
    MOCK_RAG_CONTEXT,
    MOCK_SATELLITE,
)


def _safe_zones() -> list[SafeZoneCandidate]:
    return [
        SafeZoneCandidate(
            name="Government High School #2",
            location_type=LocationType.SCHOOL,
            latitude=34.1719,
            longitude=71.7440,
            in_flood_zone=False,
            elevation_m=40.0,
            road_accessible=True,
            route_crosses_flood=False,
            distance_km=3.2,
        ),
        SafeZoneCandidate(
            name="Flooded School",
            location_type=LocationType.SCHOOL,
            latitude=34.1463,
            longitude=71.7308,
            in_flood_zone=True,
            elevation_m=8.0,
            road_accessible=True,
            route_crosses_flood=True,
            distance_km=1.0,
        ),
    ]


def _road_graph() -> list[GraphEdge]:
    return [
        GraphEdge(source="Origin", target="Government High School #2", distance_km=2.5, flooded=True),
        GraphEdge(source="Origin", target="Junction A", distance_km=1.4),
        GraphEdge(source="Junction A", target="Government High School #2", distance_km=1.8),
    ]


def test_full_pipeline_produces_all_stages():
    result = run_pipeline(
        MOCK_SATELLITE,
        MOCK_HYDRAULIC,
        MOCK_HISTORICAL,
        MOCK_RAG_CONTEXT,
        safe_zone_candidates=_safe_zones(),
        road_graph=_road_graph(),
        origin_node="Origin",
        population_at_risk=120_000,
    )

    assert isinstance(result, PipelineResult)

    # Stage 1 — Disaster Intelligence Agent.
    assert result.assessment.risk_level == RiskLevel.HIGH_RISK

    # Stage 2 — Simulation Agent.
    assert result.progression.district == MOCK_SATELLITE.district
    assert result.progression.expanding is True
    assert result.progression.projections, "Simulation must yield projections"
    assert any(
        r.district == "Nowshera" for r in result.progression.downstream_districts_at_risk
    )

    # Stage 3 — Response & Communication Agent.
    assert result.recommended_safe_zone.name == "Government High School #2"
    assert result.evacuation_route is not None
    assert result.evacuation_route.path == ["Origin", "Junction A", "Government High School #2"]
    assert result.evacuation_route.distance_km == 3.2
    assert result.evacuation_route.estimated_travel_time_min == 11

    # Outputs — citizen + authority communications.
    assert "Government High School #2" in result.citizen_alert
    assert "Please follow the highlighted evacuation route immediately." in result.citizen_alert
    assert "AUTHORITY FLOOD SITUATION REPORT — Charsadda" in result.authority_alert
    assert "Estimated Population at Risk:" in result.authority_alert


def test_pipeline_communicates_projected_peak_not_just_current():
    # The authority alert must reflect the simulated peak coverage, which for a
    # rising river is strictly greater than the current coverage.
    result = run_pipeline(
        MOCK_SATELLITE,
        MOCK_HYDRAULIC,
        MOCK_HISTORICAL,
        MOCK_RAG_CONTEXT,
        population_at_risk=120_000,
    )
    peak = result.progression.peak_coverage_percentage
    assert peak > MOCK_SATELLITE.flood_extent_percentage
    assert f"Current Flood Coverage: {peak:g}%" in result.authority_alert


def test_pipeline_runs_without_response_inputs():
    # With no safe-zone candidates or road graph, the pipeline still completes
    # the intelligence + simulation stages and emits best-effort alerts.
    result = run_pipeline(
        MOCK_SATELLITE,
        MOCK_HYDRAULIC,
        MOCK_HISTORICAL,
        MOCK_RAG_CONTEXT,
    )
    assert result.recommended_safe_zone is None
    assert result.evacuation_route is None
    assert "FLOOD EVACUATION ALERT" in result.citizen_alert
    assert result.progression.projections
