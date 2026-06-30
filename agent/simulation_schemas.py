"""
Pydantic schemas for the Simulation Agent.

The Simulation Agent sits between the Disaster Intelligence Agent and the
Response & Communication Agent in the FloodSense-PK workflow. It takes the
*current* flood situation plus the hydraulic drivers and projects how the flood
will **progress over time** — both in the assessed district and downstream along
the river system — producing the forward-looking state the response agent needs.
"""

from typing import Optional

from pydantic import BaseModel, Field


class FloodProjection(BaseModel):
    """Projected flood state at a single future time horizon."""

    horizon_hours: float = Field(
        ..., gt=0, description="Hours into the future this projection is for."
    )
    projected_coverage_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Projected percentage of district area under water.",
    )
    projected_affected_area_km2: float = Field(
        ..., ge=0, description="Projected absolute flooded area in square kilometres."
    )
    net_area_change_km2: float = Field(
        ...,
        description=(
            "Change in flooded area versus now (positive = expanding, "
            "negative = receding)."
        ),
    )
    projected_population_at_risk: Optional[int] = Field(
        None, ge=0, description="Projected population in the flooded area, if known."
    )


class DownstreamRisk(BaseModel):
    """A downstream district expected to be threatened as the flood propagates."""

    district: str = Field(..., description="Downstream district name.")
    eta_hours: float = Field(
        ..., gt=0, description="Estimated flood-wave arrival time, in hours."
    )


class FloodProgression(BaseModel):
    """Structured output of the Simulation Agent."""

    district: str = Field(..., description="District under simulation.")
    initial_coverage_percentage: float = Field(
        ..., ge=0, le=100, description="Current district flood coverage."
    )
    initial_affected_area_km2: float = Field(
        ..., ge=0, description="Current flooded area in square kilometres."
    )
    district_area_km2: float = Field(
        ..., gt=0, description="Total district area used for coverage maths."
    )
    expanding: bool = Field(
        ..., description="True if the flood is projected to grow over the horizon."
    )
    peak_coverage_percentage: float = Field(
        ..., ge=0, le=100, description="Maximum projected coverage across all horizons."
    )
    peak_horizon_hours: float = Field(
        ..., ge=0, description="Horizon (hours) at which peak coverage occurs."
    )
    projections: list[FloodProjection] = Field(
        ..., description="Ordered per-horizon projections."
    )
    downstream_districts_at_risk: list[DownstreamRisk] = Field(
        default_factory=list,
        description="Downstream districts threatened as the wave propagates.",
    )
    summary: str = Field(
        ..., description="Human-readable narrative of the projected progression."
    )


__all__ = [
    "FloodProjection",
    "DownstreamRisk",
    "FloodProgression",
]
