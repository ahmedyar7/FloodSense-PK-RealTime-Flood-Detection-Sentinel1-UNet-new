"""
Tests for the Disaster Intelligence Agent (PR 2).

Test 1 (High Risk Classification) is the mandated case from pr2.md:
31% district coverage, rising river levels, and flood extent 8% above the
2010 benchmark must yield risk_level "HIGH RISK" with an "Evacuation
preparation" recommendation.
"""

from agent import (
    assess_risk,
    SatelliteIntelligence,
    HydraulicIntelligence,
    HistoricalIntelligence,
    RAGContext,
    RiskLevel,
    RiverTrend,
    RiverStatus,
)


def _high_risk_inputs():
    satellite = SatelliteIntelligence(
        district="Charsadda",
        flood_extent_percentage=31.0,  # 31% district coverage
        affected_area_km2=303.5,
    )
    hydraulic = HydraulicIntelligence(
        station="Nowshera Gauge (Kabul River)",
        river_discharge_cusecs=95_000,
        inflow_cusecs=95_000,
        outflow_cusecs=70_000,
        trend=RiverTrend.RISING,  # rising river levels
        status=RiverStatus.HIGH,
    )
    historical = HistoricalIntelligence(
        benchmark_year=2010,
        benchmark_flood_percentage=23.0,  # 31% is 8% above the 2010 benchmark
    )
    rag_context = RAGContext(
        context="Charsadda lies in the floodplain of the Kabul River.",
        sources=["FFD River Stage Report"],
    )
    return satellite, hydraulic, historical, rag_context


def test_high_risk_classification():
    satellite, hydraulic, historical, rag_context = _high_risk_inputs()

    assessment = assess_risk(satellite, hydraulic, historical, rag_context)

    assert assessment.risk_level == RiskLevel.HIGH_RISK
    assert assessment.risk_level.value == "HIGH RISK"
    assert "Evacuation preparation" in assessment.recommended_action
    assert assessment.flood_coverage_percentage == 31.0


def test_assessment_explanation_references_all_streams():
    satellite, hydraulic, historical, rag_context = _high_risk_inputs()

    assessment = assess_risk(satellite, hydraulic, historical, rag_context)
    explanation = assessment.explanation

    assert "31.0%" in explanation  # satellite
    assert hydraulic.station in explanation  # hydraulic
    assert "2010" in explanation  # historical benchmark
    assert "8.0% above" in explanation  # delta vs benchmark


def test_low_risk_classification():
    satellite = SatelliteIntelligence(
        district="Multan", flood_extent_percentage=1.0, affected_area_km2=4.0
    )
    hydraulic = HydraulicIntelligence(
        station="Trimmu Headworks",
        river_discharge_cusecs=120_000,
        inflow_cusecs=120_000,
        outflow_cusecs=120_000,
        trend=RiverTrend.FALLING,
        status=RiverStatus.NORMAL,
    )
    historical = HistoricalIntelligence(benchmark_flood_percentage=18.0)

    assessment = assess_risk(satellite, hydraulic, historical)

    assert assessment.risk_level == RiskLevel.LOW_RISK
    assert "Routine monitoring" in assessment.recommended_action


def test_moderate_risk_classification():
    satellite = SatelliteIntelligence(
        district="Nowshera", flood_extent_percentage=7.0, affected_area_km2=40.0
    )
    hydraulic = HydraulicIntelligence(
        station="Nowshera Gauge",
        river_discharge_cusecs=60_000,
        inflow_cusecs=60_000,
        outflow_cusecs=58_000,
        trend=RiverTrend.STABLE,
        status=RiverStatus.HIGH,
    )
    historical = HistoricalIntelligence(benchmark_flood_percentage=23.0)

    assessment = assess_risk(satellite, hydraulic, historical)

    assert assessment.risk_level == RiskLevel.MODERATE_RISK
    assert "readiness" in assessment.recommended_action.lower()


def test_extreme_river_status_forces_high_risk():
    """Even with low coverage, an EXTREME river stage escalates to HIGH RISK."""
    satellite = SatelliteIntelligence(
        district="Sukkur", flood_extent_percentage=3.0, affected_area_km2=20.0
    )
    hydraulic = HydraulicIntelligence(
        station="Sukkur Barrage",
        river_discharge_cusecs=510_000,
        inflow_cusecs=510_000,
        outflow_cusecs=480_000,
        trend=RiverTrend.RISING,
        status=RiverStatus.EXTREME,
    )
    historical = HistoricalIntelligence(benchmark_flood_percentage=2.0)

    assessment = assess_risk(satellite, hydraulic, historical)

    assert assessment.risk_level == RiskLevel.HIGH_RISK


def test_llm_fn_enriches_explanation_with_fallback():
    satellite, hydraulic, historical, rag_context = _high_risk_inputs()

    def llm_fn(prompt: str) -> str:
        assert "Disaster Intelligence Agent" in prompt
        assert "HIGH RISK" in prompt
        return "LLM narrative: evacuation should begin immediately."

    assessment = assess_risk(satellite, hydraulic, historical, rag_context, llm_fn=llm_fn)
    assert assessment.explanation == "LLM narrative: evacuation should begin immediately."

    # A failing LLM must not break the assessment; it falls back deterministically.
    def broken_llm(prompt: str) -> str:
        raise RuntimeError("model unavailable")

    fallback = assess_risk(satellite, hydraulic, historical, rag_context, llm_fn=broken_llm)
    assert fallback.risk_level == RiskLevel.HIGH_RISK
    assert "Charsadda" in fallback.explanation
