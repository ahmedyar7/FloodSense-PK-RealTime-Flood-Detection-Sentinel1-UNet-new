"""
End-to-end FloodSense-PK orchestrator.

Wires the complete disaster-intelligence workflow described in
``final-workflow.md`` into a single call:

    Satellite + River + Historical + RAG knowledge
      → Disaster Intelligence Agent   (risk assessment)
      → Simulation Agent              (flood progression)
      → Response & Communication Agent(safe zones, routes, alerts)
      → Safe-zone recommendation, evacuation route, citizen + authority alerts

Every stage is deterministic and rule-based; an optional ``llm_fn`` only enriches
the natural-language narratives produced along the way.
"""

from typing import Callable, Optional, Sequence

from pydantic import BaseModel, Field

from .disaster_agent import assess_risk
from .response_agent import (
    evaluate_safe_zones,
    generate_authority_alert,
    generate_citizen_alert,
    plan_route,
    recommend_safe_zone,
)
from .response_schemas import (
    EvacuationRoute,
    FloodState,
    GraphEdge,
    SafeZoneCandidate,
)
from .schemas import (
    HistoricalIntelligence,
    HydraulicIntelligence,
    RAGContext,
    RiskAssessment,
    SatelliteIntelligence,
)
from .simulation_agent import DEFAULT_HORIZONS_HOURS, simulate_from_assessment
from .simulation_schemas import FloodProgression


class PipelineResult(BaseModel):
    """Consolidated output of the full FloodSense-PK workflow."""

    assessment: RiskAssessment = Field(
        ..., description="Disaster Intelligence Agent risk assessment."
    )
    progression: FloodProgression = Field(
        ..., description="Simulation Agent flood progression projection."
    )
    recommended_safe_zone: Optional[SafeZoneCandidate] = Field(
        None, description="Best validated safe zone, if any candidates were supplied."
    )
    evacuation_route: Optional[EvacuationRoute] = Field(
        None, description="Planned evacuation route, if a road graph was supplied."
    )
    citizen_alert: str = Field(..., description="Public-facing evacuation alert.")
    authority_alert: str = Field(..., description="Authority situation report.")


def run_pipeline(
    satellite: SatelliteIntelligence,
    hydraulic: HydraulicIntelligence,
    historical: HistoricalIntelligence,
    rag_context: Optional[RAGContext] = None,
    *,
    safe_zone_candidates: Optional[list[SafeZoneCandidate]] = None,
    road_graph: Optional[list[GraphEdge]] = None,
    origin_node: Optional[str] = None,
    population_at_risk: int = 0,
    district_area_km2: Optional[float] = None,
    horizons_hours: Sequence[float] = DEFAULT_HORIZONS_HOURS,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> PipelineResult:
    """
    Run the four-stage workflow and return a consolidated :class:`PipelineResult`.

    The response stage is best-effort: safe-zone and routing outputs are only
    produced when the corresponding inputs (candidates / road graph + origin) are
    supplied, so the orchestrator works whether or not live OSM data is wired in.
    """
    # --- Stage 1: Disaster Intelligence Agent ------------------------------- #
    assessment = assess_risk(
        satellite, hydraulic, historical, rag_context, llm_fn=llm_fn
    )

    # --- Stage 2: Simulation Agent ----------------------------------------- #
    progression = simulate_from_assessment(
        assessment,
        satellite,
        hydraulic,
        population_at_risk=population_at_risk or None,
        district_area_km2=district_area_km2,
        horizons_hours=horizons_hours,
        llm_fn=llm_fn,
    )

    # --- Stage 3: Response & Communication Agent --------------------------- #
    valid_zones: list[SafeZoneCandidate] = (
        evaluate_safe_zones(safe_zone_candidates) if safe_zone_candidates else []
    )
    recommended = recommend_safe_zone(safe_zone_candidates) if safe_zone_candidates else None

    route: Optional[EvacuationRoute] = None
    if road_graph and origin_node and recommended is not None:
        try:
            route = plan_route(road_graph, origin=origin_node, destination=recommended.name)
        except ValueError:
            # No dry route available; leave the route unset rather than failing.
            route = None

    # The response agent communicates the *projected peak* situation so citizens
    # and authorities act on where the flood is heading, not just where it is.
    peak_coverage = progression.peak_coverage_percentage
    peak_area = max(
        (p.projected_affected_area_km2 for p in progression.projections),
        default=satellite.affected_area_km2,
    )
    peak_population = max(
        (
            p.projected_population_at_risk
            for p in progression.projections
            if p.projected_population_at_risk is not None
        ),
        default=population_at_risk,
    )

    flood_state = FloodState(
        district=satellite.district,
        flood_coverage_percentage=peak_coverage,
        affected_area_km2=peak_area,
        population_at_risk=peak_population,
        available_shelters=len(valid_zones),
    )
    authority_alert = generate_authority_alert(flood_state)

    shelter_name = recommended.name if recommended is not None else "nearest designated relief camp"
    distance_km = route.distance_km if route is not None else (
        recommended.distance_km if recommended is not None and recommended.distance_km is not None else 0.0
    )
    travel_time_min = route.estimated_travel_time_min if route is not None else 0
    citizen_alert = generate_citizen_alert(
        district=satellite.district,
        risk_level=assessment.risk_level,
        shelter_name=shelter_name,
        distance_km=distance_km,
        travel_time_min=travel_time_min,
    )

    return PipelineResult(
        assessment=assessment,
        progression=progression,
        recommended_safe_zone=recommended,
        evacuation_route=route,
        citizen_alert=citizen_alert,
        authority_alert=authority_alert,
    )


__all__ = ["PipelineResult", "run_pipeline"]
