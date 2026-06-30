"""
Tests for the Simulation Agent — the flood-progression node that sits between
the Disaster Intelligence Agent and the Response & Communication Agent.

Covers the projection dynamics (expansion, recession, capping), the exact
volume-balance maths, downstream propagation, the bridge from a Disaster
Intelligence ``RiskAssessment``, and optional LLM narrative enrichment.
"""

import pytest

from agent import (
    DownstreamRisk,
    FloodProgression,
    RiskLevel,
    RiverStatus,
    RiverTrend,
    assess_risk,
    downstream_districts,
    simulate_from_assessment,
    simulate_progression,
    spilled_area_change_km2,
)
from agent.mock_intelligence import (
    MOCK_HISTORICAL,
    MOCK_HYDRAULIC,
    MOCK_RAG_CONTEXT,
    MOCK_SATELLITE,
)


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
def test_rising_river_expands_flood_monotonically():
    prog = simulate_progression(
        district="Charsadda",
        current_coverage_percentage=31.0,
        current_affected_area_km2=303.5,
        inflow_cusecs=95_000,
        outflow_cusecs=70_000,
        trend=RiverTrend.RISING,
        status=RiverStatus.HIGH,
    )

    assert isinstance(prog, FloodProgression)
    assert prog.expanding is True

    coverages = [p.projected_coverage_percentage for p in prog.projections]
    # Strictly increasing across the (sorted) horizons, and above the start.
    assert coverages == sorted(coverages)
    assert coverages[-1] > prog.initial_coverage_percentage
    # Peak occurs at the furthest horizon when the river keeps rising.
    assert prog.peak_horizon_hours == prog.projections[-1].horizon_hours
    assert all(p.net_area_change_km2 > 0 for p in prog.projections)


def test_draining_river_recedes_flood():
    # Outflow exceeds inflow → net flux negative → flooded area shrinks.
    prog = simulate_progression(
        district="Sukkur",
        current_coverage_percentage=20.0,
        current_affected_area_km2=200.0,
        inflow_cusecs=80_000,
        outflow_cusecs=120_000,
        trend=RiverTrend.FALLING,
        status=RiverStatus.NORMAL,
    )

    assert prog.expanding is False
    assert prog.projections[-1].projected_coverage_percentage < prog.initial_coverage_percentage
    assert all(p.net_area_change_km2 < 0 for p in prog.projections)


def test_coverage_is_capped_at_100_percent():
    # A massive sustained net flux must not push coverage above 100%.
    prog = simulate_progression(
        district="Jacobabad",
        current_coverage_percentage=60.0,
        current_affected_area_km2=600.0,
        inflow_cusecs=1_500_000,
        outflow_cusecs=0,
        trend=RiverTrend.RISING,
        status=RiverStatus.EXTREME,
        horizons_hours=(6, 12, 24, 48),
    )

    for p in prog.projections:
        assert 0.0 <= p.projected_coverage_percentage <= 100.0
        assert p.projected_affected_area_km2 <= prog.district_area_km2 + 1e-6
    assert prog.peak_coverage_percentage == 100.0


def test_projection_uses_exact_volume_balance_maths():
    # The projected area must equal current area + the published flux formula.
    inflow, outflow = 95_000, 70_000
    delta_24h = spilled_area_change_km2(
        inflow, outflow, RiverTrend.RISING, RiverStatus.HIGH, 24
    )
    prog = simulate_progression(
        district="Charsadda",
        current_coverage_percentage=31.0,
        current_affected_area_km2=303.5,
        inflow_cusecs=inflow,
        outflow_cusecs=outflow,
        trend=RiverTrend.RISING,
        status=RiverStatus.HIGH,
        horizons_hours=(24,),
    )
    projected = prog.projections[0].projected_affected_area_km2
    assert projected == pytest.approx(round(303.5 + delta_24h, 2), abs=0.01)


def test_population_at_risk_scales_with_area():
    prog = simulate_progression(
        district="Charsadda",
        current_coverage_percentage=31.0,
        current_affected_area_km2=303.5,
        inflow_cusecs=95_000,
        outflow_cusecs=70_000,
        trend=RiverTrend.RISING,
        status=RiverStatus.HIGH,
        population_at_risk=120_000,
    )
    pops = [p.projected_population_at_risk for p in prog.projections]
    assert all(isinstance(v, int) for v in pops)
    # Population grows with the expanding flooded area, starting above current.
    assert pops == sorted(pops)
    assert pops[0] >= 120_000


def test_zero_coverage_requires_district_area():
    with pytest.raises(ValueError):
        simulate_progression(
            district="Multan",
            current_coverage_percentage=0.0,
            current_affected_area_km2=0.0,
            inflow_cusecs=50_000,
            outflow_cusecs=40_000,
            trend=RiverTrend.RISING,
            status=RiverStatus.HIGH,
        )
    # Supplying the district area resolves the ambiguity.
    prog = simulate_progression(
        district="Multan",
        current_coverage_percentage=0.0,
        current_affected_area_km2=0.0,
        inflow_cusecs=50_000,
        outflow_cusecs=40_000,
        trend=RiverTrend.RISING,
        status=RiverStatus.HIGH,
        district_area_km2=3_720.0,
    )
    assert prog.district_area_km2 == 3_720.0


# --------------------------------------------------------------------------- #
# Downstream propagation
# --------------------------------------------------------------------------- #
def test_downstream_propagation_follows_river_chain():
    risks = downstream_districts(
        "Charsadda", RiverTrend.RISING, RiverStatus.HIGH, max_horizon_hours=48
    )
    assert all(isinstance(r, DownstreamRisk) for r in risks)
    names = [r.district for r in risks]
    assert names[0] == "Nowshera"
    # ETAs are cumulative and strictly increasing downstream.
    etas = [r.eta_hours for r in risks]
    assert etas == sorted(etas)
    assert risks[0].eta_hours == 6.0


def test_no_downstream_when_river_calm():
    # A falling, normal river is not pushing a wave downstream.
    assert (
        downstream_districts(
            "Charsadda", RiverTrend.FALLING, RiverStatus.NORMAL, max_horizon_hours=48
        )
        == []
    )


def test_downstream_respects_horizon_window():
    # Only Nowshera (eta 6h) fits inside a 6h window; Attock (14h) does not.
    risks = downstream_districts(
        "Charsadda", RiverTrend.RISING, RiverStatus.HIGH, max_horizon_hours=6
    )
    assert [r.district for r in risks] == ["Nowshera"]


# --------------------------------------------------------------------------- #
# Workflow bridge + LLM enrichment
# --------------------------------------------------------------------------- #
def test_simulate_from_assessment_uses_disaster_outputs():
    assessment = assess_risk(
        MOCK_SATELLITE, MOCK_HYDRAULIC, MOCK_HISTORICAL, MOCK_RAG_CONTEXT
    )
    prog = simulate_from_assessment(
        assessment, MOCK_SATELLITE, MOCK_HYDRAULIC, population_at_risk=120_000
    )
    assert assessment.risk_level == RiskLevel.HIGH_RISK
    assert prog.district == MOCK_SATELLITE.district
    assert prog.initial_coverage_percentage == MOCK_SATELLITE.flood_extent_percentage
    assert prog.expanding is True
    # The Charsadda → Nowshera propagation is surfaced.
    assert any(r.district == "Nowshera" for r in prog.downstream_districts_at_risk)


def test_llm_fn_enriches_summary_with_fallback():
    def llm_fn(prompt: str) -> str:
        assert "Simulation Agent" in prompt
        return "LLM narrative: the flood crest reaches Nowshera within six hours."

    prog = simulate_progression(
        district="Charsadda",
        current_coverage_percentage=31.0,
        current_affected_area_km2=303.5,
        inflow_cusecs=95_000,
        outflow_cusecs=70_000,
        trend=RiverTrend.RISING,
        status=RiverStatus.HIGH,
        llm_fn=llm_fn,
    )
    assert prog.summary == "LLM narrative: the flood crest reaches Nowshera within six hours."

    def broken_llm(prompt: str) -> str:
        raise RuntimeError("model unavailable")

    fallback = simulate_progression(
        district="Charsadda",
        current_coverage_percentage=31.0,
        current_affected_area_km2=303.5,
        inflow_cusecs=95_000,
        outflow_cusecs=70_000,
        trend=RiverTrend.RISING,
        status=RiverStatus.HIGH,
        llm_fn=broken_llm,
    )
    # A failing LLM must not break the simulation; the deterministic summary stands.
    assert "Charsadda" in fallback.summary
    assert fallback.expanding is True
