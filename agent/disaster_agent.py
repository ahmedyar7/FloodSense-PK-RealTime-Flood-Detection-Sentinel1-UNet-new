"""
Disaster Intelligence Agent — the central reasoning node of the platform.

It fuses four intelligence streams (satellite, hydraulic, historical, and RAG
context) into a single structured ``RiskAssessment``. Risk *classification* is
deterministic and rule-based so it is testable and reproducible without an LLM;
the natural-language *explanation* is optionally enriched by an LLM through the
same Gemini→Groq pipeline used by the rest of the project, with a deterministic
fallback when no model is available.
"""

from typing import Callable, Optional

from .schemas import (
    SatelliteIntelligence,
    HydraulicIntelligence,
    HistoricalIntelligence,
    RAGContext,
    RiskAssessment,
    RiskLevel,
    RiverTrend,
    RiverStatus,
)

# Classification thresholds (district flood coverage, %).
HIGH_COVERAGE_THRESHOLD = 20.0
MODERATE_COVERAGE_THRESHOLD = 5.0
# Coverage above which a worsening hydraulic/historical signal escalates to HIGH.
ESCALATION_COVERAGE_THRESHOLD = 10.0

RECOMMENDED_ACTIONS = {
    RiskLevel.HIGH_RISK: (
        "Evacuation preparation: issue evacuation orders for flood-prone union "
        "councils, pre-position rescue boats and relief supplies, and prioritise "
        "vulnerable populations per NDMA Phase 2 SOPs."
    ),
    RiskLevel.MODERATE_RISK: (
        "Heightened readiness: activate NDMA Phase 1 alert, stage rescue assets, "
        "and monitor river gauges and satellite extent at increased frequency."
    ),
    RiskLevel.LOW_RISK: (
        "Routine monitoring: maintain standard surveillance of river gauges and "
        "satellite imagery; no immediate field action required."
    ),
}


def classify_risk(
    satellite: SatelliteIntelligence,
    hydraulic: HydraulicIntelligence,
    historical: HistoricalIntelligence,
) -> RiskLevel:
    """Deterministic, defensible risk classification from the numeric streams."""
    coverage = satellite.flood_extent_percentage
    delta = coverage - historical.benchmark_flood_percentage
    worsening_hydraulics = (
        hydraulic.trend == RiverTrend.RISING or hydraulic.status == RiverStatus.EXTREME
    )

    # Severe absolute inundation, an extreme river stage, or above-benchmark
    # flooding with rising rivers all warrant the highest risk band.
    if (
        coverage >= HIGH_COVERAGE_THRESHOLD
        or hydraulic.status == RiverStatus.EXTREME
        or (
            coverage >= ESCALATION_COVERAGE_THRESHOLD
            and delta > 0
            and worsening_hydraulics
        )
    ):
        return RiskLevel.HIGH_RISK

    if (
        coverage >= MODERATE_COVERAGE_THRESHOLD
        or hydraulic.status == RiverStatus.HIGH
        or worsening_hydraulics
        or delta > 0
    ):
        return RiskLevel.MODERATE_RISK

    return RiskLevel.LOW_RISK


def _build_explanation(
    risk_level: RiskLevel,
    satellite: SatelliteIntelligence,
    hydraulic: HydraulicIntelligence,
    historical: HistoricalIntelligence,
    rag_context: RAGContext,
) -> str:
    """Deterministic, evidence-grounded explanation referencing every stream."""
    coverage = satellite.flood_extent_percentage
    delta = coverage - historical.benchmark_flood_percentage
    direction = "above" if delta >= 0 else "below"

    lines = [
        f"{satellite.district} is classified as {risk_level.value}.",
        (
            f"Satellite (Sentinel-1/U-Net): {coverage:.1f}% of the district is "
            f"inundated, covering {satellite.affected_area_km2:.1f} km²."
        ),
        (
            f"Hydraulic (FFD @ {hydraulic.station}): discharge "
            f"{hydraulic.river_discharge_cusecs:,.0f} cusecs, inflow "
            f"{hydraulic.inflow_cusecs:,.0f} / outflow {hydraulic.outflow_cusecs:,.0f} "
            f"cusecs, status {hydraulic.status.value}, trend {hydraulic.trend.value}."
        ),
        (
            f"Historical (Landsat-5 {historical.benchmark_year} benchmark): current "
            f"extent is {abs(delta):.1f}% {direction} the "
            f"{historical.benchmark_flood_percentage:.1f}% benchmark."
        ),
    ]
    if rag_context.context.strip():
        srcs = ", ".join(rag_context.sources) if rag_context.sources else "knowledge base"
        lines.append(f"Knowledge context ({srcs}) corroborates this assessment.")
    return " ".join(lines)


def assess_risk(
    satellite: SatelliteIntelligence,
    hydraulic: HydraulicIntelligence,
    historical: HistoricalIntelligence,
    rag_context: Optional[RAGContext] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> RiskAssessment:
    """
    Disaster Intelligence Agent node.

    Synthesises the four intelligence streams into a ``RiskAssessment``. The
    ``risk_level`` and ``recommended_action`` are derived deterministically; when
    ``llm_fn`` is supplied it is used to produce a richer narrative explanation,
    falling back to the deterministic explanation if the call fails or returns
    nothing.
    """
    rag_context = rag_context or RAGContext()

    risk_level = classify_risk(satellite, hydraulic, historical)
    recommended_action = RECOMMENDED_ACTIONS[risk_level]
    explanation = _build_explanation(
        risk_level, satellite, hydraulic, historical, rag_context
    )

    if llm_fn is not None:
        prompt = (
            "You are the Disaster Intelligence Agent for Pakistan flood response. "
            "Using ONLY the evidence below, write a concise 3-4 sentence operational "
            "explanation justifying the stated risk level. Reference concrete numbers.\n\n"
            f"Stated risk level: {risk_level.value}\n"
            f"Evidence:\n{explanation}\n\n"
            f"Retrieved context:\n{rag_context.context}\n\n"
            "Explanation:"
        )
        try:
            llm_text = llm_fn(prompt)
            if llm_text and llm_text.strip():
                explanation = llm_text.strip()
        except Exception:
            # Keep the deterministic explanation on any LLM failure.
            pass

    return RiskAssessment(
        risk_level=risk_level,
        flood_coverage_percentage=satellite.flood_extent_percentage,
        explanation=explanation,
        recommended_action=recommended_action,
    )
