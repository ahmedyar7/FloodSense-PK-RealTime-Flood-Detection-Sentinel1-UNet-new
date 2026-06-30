"""
Simulation Agent — predicts how a flood will progress over time.

This is the third node of the FloodSense-PK workflow:

    Disaster Intelligence Agent → **Simulation Agent** → Response & Communication

Given the *current* flood state (from Sentinel-1/U-Net) and the hydraulic drivers
(from the FFD river gauges), it projects future flood extent at a set of time
horizons and identifies the downstream districts the flood wave will threaten.

The dynamics are a transparent, physically-grounded **volume-balance** model so
the projection is deterministic and fully testable without an LLM:

* The net river flux (inflow − outflow) that spills overbank accumulates as
  floodwater volume over time.
* That volume spreads across the district floodplain at an assumed average
  inundation depth, growing (or, when outflow dominates, shrinking) the flooded
  area.
* River *trend* and *status* scale the spill magnitude (a rising/extreme river
  pushes more water overbank through weakened embankments).

An optional ``llm_fn`` enriches the natural-language ``summary`` only; the
numeric projection is never delegated to a model.
"""

from typing import Callable, Optional, Sequence

from .schemas import (
    HydraulicIntelligence,
    RiskAssessment,
    RiverStatus,
    RiverTrend,
    SatelliteIntelligence,
)
from .simulation_schemas import DownstreamRisk, FloodProgression, FloodProjection

# --- Physical constants -----------------------------------------------------#
# 1 cusec = 1 ft³/s; convert the cubic-foot component to cubic metres.
CUSEC_TO_M3_PER_S = 0.0283168
SECONDS_PER_HOUR = 3600

# Fraction of the net river flux assumed to spill overbank into the assessed
# district's floodplain (the rest stays in-channel or passes downstream).
SPILL_FRACTION = 0.15
# Assumed average depth (m) of new inundation when spreading spilled volume.
AVG_INUNDATION_DEPTH_M = 1.5

# Magnitude modifiers — a rising and/or extreme river drives proportionally more
# water overbank (embankment overtopping and breaches) than a falling/normal one.
TREND_MOMENTUM = {
    RiverTrend.RISING: 1.15,
    RiverTrend.STABLE: 1.0,
    RiverTrend.FALLING: 0.9,
}
STATUS_MOMENTUM = {
    RiverStatus.NORMAL: 0.85,
    RiverStatus.HIGH: 1.0,
    RiverStatus.EXTREME: 1.25,
}

DEFAULT_HORIZONS_HOURS: tuple[float, ...] = (6.0, 12.0, 24.0, 48.0)

# Downstream river ordering for the priority districts, with approximate
# flood-wave travel times (hours). Walking this chain propagates the threat from
# an upstream district to the settlements below it on the same river system.
# Grounded in the Kabul and Indus orderings used throughout the project.
DOWNSTREAM_CHAIN: dict[str, tuple[str, float]] = {
    # Kabul River (KP)
    "charsadda": ("Nowshera", 6.0),
    "nowshera": ("Attock", 8.0),
    # Indus (north → south)
    "dera ismail khan": ("Dera Ghazi Khan", 18.0),
    "dera ghazi khan": ("Rajanpur", 12.0),
    "rajanpur": ("Kashmore", 14.0),
    "kashmore": ("Jacobabad", 10.0),
    "jacobabad": ("Shikarpur", 8.0),
    "shikarpur": ("Sukkur", 6.0),
    "sukkur": ("Larkana", 16.0),
}


def spilled_area_change_km2(
    inflow_cusecs: float,
    outflow_cusecs: float,
    trend: RiverTrend,
    status: RiverStatus,
    hours: float,
) -> float:
    """
    Change in flooded area (km²) over ``hours`` from the net river flux.

    Positive when the river is gaining (inflow > outflow), negative when it is
    draining. This is the single source of truth for the simulation dynamics so
    tests can reproduce it exactly.
    """
    net_flux_cusecs = inflow_cusecs - outflow_cusecs
    effective_cusecs = (
        net_flux_cusecs
        * SPILL_FRACTION
        * TREND_MOMENTUM[trend]
        * STATUS_MOMENTUM[status]
    )
    volume_m3 = effective_cusecs * CUSEC_TO_M3_PER_S * SECONDS_PER_HOUR * hours
    area_m2 = volume_m3 / AVG_INUNDATION_DEPTH_M
    return area_m2 / 1_000_000  # → km²


def _resolve_district_area(
    coverage_percentage: float,
    affected_area_km2: float,
    district_area_km2: Optional[float],
) -> float:
    """Determine the total district area used for coverage maths."""
    if district_area_km2 is not None:
        return district_area_km2
    if coverage_percentage > 0:
        return affected_area_km2 / (coverage_percentage / 100.0)
    raise ValueError(
        "district_area_km2 is required when current coverage is 0% "
        "(cannot derive district area from a 0% extent)."
    )


def downstream_districts(
    district: str,
    trend: RiverTrend,
    status: RiverStatus,
    max_horizon_hours: float,
) -> list[DownstreamRisk]:
    """
    Walk the river chain below ``district`` and list threatened settlements.

    Downstream propagation is only flagged when the situation is actually
    worsening (a rising river or a HIGH/EXTREME stage); a falling, normal river
    is not pushing a wave downstream.
    """
    worsening = trend == RiverTrend.RISING or status in (
        RiverStatus.HIGH,
        RiverStatus.EXTREME,
    )
    if not worsening:
        return []

    risks: list[DownstreamRisk] = []
    current = district.lower()
    cumulative = 0.0
    seen = {current}
    while current in DOWNSTREAM_CHAIN:
        nxt, lag = DOWNSTREAM_CHAIN[current]
        cumulative += lag
        if cumulative > max_horizon_hours:
            break
        risks.append(DownstreamRisk(district=nxt, eta_hours=cumulative))
        current = nxt.lower()
        if current in seen:
            break
        seen.add(current)
    return risks


def simulate_progression(
    district: str,
    current_coverage_percentage: float,
    current_affected_area_km2: float,
    inflow_cusecs: float,
    outflow_cusecs: float,
    trend: RiverTrend,
    status: RiverStatus,
    *,
    population_at_risk: Optional[int] = None,
    district_area_km2: Optional[float] = None,
    horizons_hours: Sequence[float] = DEFAULT_HORIZONS_HOURS,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> FloodProgression:
    """
    Project the flood's progression across the requested time horizons.

    Returns a :class:`FloodProgression` with one :class:`FloodProjection` per
    horizon, the projected peak, and the downstream districts at risk.
    """
    if not horizons_hours:
        raise ValueError("At least one projection horizon is required.")

    horizons = sorted(float(h) for h in horizons_hours)
    total_area = _resolve_district_area(
        current_coverage_percentage, current_affected_area_km2, district_area_km2
    )
    # Population density over the currently flooded area, for scaling forward.
    density = (
        population_at_risk / current_affected_area_km2
        if population_at_risk is not None and current_affected_area_km2 > 0
        else None
    )

    projections: list[FloodProjection] = []
    for hours in horizons:
        delta_area = spilled_area_change_km2(
            inflow_cusecs, outflow_cusecs, trend, status, hours
        )
        projected_area = min(
            max(current_affected_area_km2 + delta_area, 0.0), total_area
        )
        projected_coverage = min(max(projected_area / total_area * 100.0, 0.0), 100.0)
        projected_population = (
            int(round(density * projected_area)) if density is not None else None
        )
        projections.append(
            FloodProjection(
                horizon_hours=hours,
                projected_coverage_percentage=round(projected_coverage, 2),
                projected_affected_area_km2=round(projected_area, 2),
                net_area_change_km2=round(projected_area - current_affected_area_km2, 2),
                projected_population_at_risk=projected_population,
            )
        )

    # Peak is evaluated across the current state plus every projection.
    peak = max(
        projections,
        key=lambda p: p.projected_coverage_percentage,
    )
    if current_coverage_percentage >= peak.projected_coverage_percentage:
        peak_coverage = round(current_coverage_percentage, 2)
        peak_horizon = 0.0
    else:
        peak_coverage = peak.projected_coverage_percentage
        peak_horizon = peak.horizon_hours

    expanding = projections[-1].projected_affected_area_km2 > current_affected_area_km2 + 1e-9

    downstream = downstream_districts(district, trend, status, horizons[-1])

    summary = _build_summary(
        district,
        current_coverage_percentage,
        projections,
        expanding,
        peak_coverage,
        peak_horizon,
        downstream,
    )
    if llm_fn is not None:
        summary = _maybe_llm_summary(llm_fn, summary) or summary

    return FloodProgression(
        district=district,
        initial_coverage_percentage=round(current_coverage_percentage, 2),
        initial_affected_area_km2=round(current_affected_area_km2, 2),
        district_area_km2=round(total_area, 2),
        expanding=expanding,
        peak_coverage_percentage=peak_coverage,
        peak_horizon_hours=peak_horizon,
        projections=projections,
        downstream_districts_at_risk=downstream,
        summary=summary,
    )


def simulate_from_assessment(
    assessment: RiskAssessment,
    satellite: SatelliteIntelligence,
    hydraulic: HydraulicIntelligence,
    *,
    population_at_risk: Optional[int] = None,
    district_area_km2: Optional[float] = None,
    horizons_hours: Sequence[float] = DEFAULT_HORIZONS_HOURS,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> FloodProgression:
    """
    Workflow bridge: run the simulation directly from the Disaster Intelligence
    Agent's outputs and the originating satellite/hydraulic streams.
    """
    return simulate_progression(
        district=satellite.district,
        current_coverage_percentage=assessment.flood_coverage_percentage,
        current_affected_area_km2=satellite.affected_area_km2,
        inflow_cusecs=hydraulic.inflow_cusecs,
        outflow_cusecs=hydraulic.outflow_cusecs,
        trend=hydraulic.trend,
        status=hydraulic.status,
        population_at_risk=population_at_risk,
        district_area_km2=district_area_km2,
        horizons_hours=horizons_hours,
        llm_fn=llm_fn,
    )


def _build_summary(
    district: str,
    initial_coverage: float,
    projections: list[FloodProjection],
    expanding: bool,
    peak_coverage: float,
    peak_horizon: float,
    downstream: list[DownstreamRisk],
) -> str:
    """Deterministic, evidence-grounded progression narrative."""
    last = projections[-1]
    direction = "expand" if expanding else "recede"
    lines = [
        f"{district} flood is projected to {direction} from "
        f"{initial_coverage:.1f}% to {last.projected_coverage_percentage:.1f}% "
        f"coverage over the next {last.horizon_hours:.0f} hours.",
        (
            f"Peak coverage of {peak_coverage:.1f}% is expected "
            + (
                "now (already at or past peak)."
                if peak_horizon == 0
                else f"around +{peak_horizon:.0f}h."
            )
        ),
    ]
    if downstream:
        names = ", ".join(f"{d.district} (~{d.eta_hours:.0f}h)" for d in downstream)
        lines.append(f"Downstream districts at risk as the wave propagates: {names}.")
    else:
        lines.append("No downstream propagation expected under current conditions.")
    return " ".join(lines)


def _maybe_llm_summary(llm_fn: Callable[[str], str], evidence: str) -> Optional[str]:
    """Enrich the summary via the LLM, falling back silently on any failure."""
    prompt = (
        "You are the Simulation Agent for Pakistan flood response. Using ONLY the "
        "projection evidence below, write a concise 2-3 sentence narrative of how "
        "the flood will progress. Reference concrete numbers.\n\n"
        f"Projection evidence:\n{evidence}\n\n"
        "Narrative:"
    )
    try:
        text = llm_fn(prompt)
        if text and text.strip():
            return text.strip()
    except Exception:
        return None
    return None
