"""
Pydantic schemas for the Response & Communication Agent (PR 3).

This agent answers the final operational question — *"Where should I go and how
do I get there safely?"* — by combining safe-zone evaluation, evacuation routing,
and public communication. Each input/output is a validated, typed model so the
evaluator, router, and alert generator receive well-structured data.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Re-use the project-wide risk classification produced by the Disaster
# Intelligence Agent (PR 2) rather than redefining it.
from .schemas import RiskLevel


class LocationType(str, Enum):
    """OpenStreetMap amenity classes we treat as potential safe zones."""

    SCHOOL = "school"
    HOSPITAL = "hospital"
    RELIEF_CAMP = "relief_camp"


class SafeZoneCandidate(BaseModel):
    """A candidate shelter sourced from (mock) OpenStreetMap data.

    A candidate is only a valid safe zone when it is *outside* the flood zone, at
    a *safe elevation*, *accessible via roads*, and reachable *without crossing
    flooded areas* (``route_crosses_flood`` is then ``False``).
    """

    name: str = Field(..., description="Human-readable shelter name.")
    location_type: LocationType = Field(..., description="OSM amenity class.")
    latitude: float = Field(
        ..., ge=-90, le=90, description="Shelter latitude (WGS84 decimal degrees)."
    )
    longitude: float = Field(
        ..., ge=-180, le=180, description="Shelter longitude (WGS84 decimal degrees)."
    )
    in_flood_zone: bool = Field(
        ..., description="True if the location itself sits inside the flood extent."
    )
    elevation_m: float = Field(
        ..., description="Ground elevation above sea level, in metres."
    )
    road_accessible: bool = Field(
        ..., description="True if the location is reachable by the road network."
    )
    route_crosses_flood: bool = Field(
        ...,
        description=(
            "True if the only approach road crosses a flooded area "
            "(i.e. the route is partially or fully blocked)."
        ),
    )
    distance_km: Optional[float] = Field(
        None,
        ge=0,
        description="Optional straight-line distance from the origin, for ranking.",
    )


class GraphEdge(BaseModel):
    """An undirected road segment in the routing graph."""

    source: str = Field(..., description="Origin node id.")
    target: str = Field(..., description="Destination node id.")
    distance_km: float = Field(..., ge=0, description="Segment length in kilometres.")
    flooded: bool = Field(
        False, description="True if the segment is impassable due to flooding."
    )


class EvacuationRoute(BaseModel):
    """Output of the routing engine — the optimal path to a safe zone."""

    path: list[str] = Field(
        ..., description="Ordered node ids from origin to safe zone."
    )
    distance_km: float = Field(
        ..., ge=0, description="Total route length in kilometres."
    )
    estimated_travel_time_min: int = Field(
        ..., ge=0, description="Estimated travel time in whole minutes."
    )


class FloodState(BaseModel):
    """Current macro flood situation, driving the authority alert."""

    district: str = Field(..., description="District under assessment.")
    flood_coverage_percentage: float = Field(
        ..., ge=0, le=100, description="Percentage of district area under water."
    )
    affected_area_km2: float = Field(
        ..., ge=0, description="Absolute flooded area in square kilometres."
    )
    population_at_risk: int = Field(
        ..., ge=0, description="Estimated population in the affected area."
    )
    available_shelters: int = Field(
        ..., ge=0, description="Number of validated safe zones available."
    )


__all__ = [
    "RiskLevel",
    "LocationType",
    "SafeZoneCandidate",
    "GraphEdge",
    "EvacuationRoute",
    "FloodState",
]
