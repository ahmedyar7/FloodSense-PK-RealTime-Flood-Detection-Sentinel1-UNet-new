"""
Pydantic schemas for the Disaster Intelligence Agent.

The agent synthesises four independent intelligence streams into a single
"Risk Assessment". Each input stream has its own typed schema so the agent
node receives validated, well-structured data.
"""

from enum import Enum
from pydantic import BaseModel, Field


class RiverTrend(str, Enum):
    RISING = "RISING"
    FALLING = "FALLING"
    STABLE = "STABLE"


class RiverStatus(str, Enum):
    """FFD river-stage classification (mirrors the thresholds in mock_documents)."""

    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class RiskLevel(str, Enum):
    LOW_RISK = "LOW RISK"
    MODERATE_RISK = "MODERATE RISK"
    HIGH_RISK = "HIGH RISK"


class SatelliteIntelligence(BaseModel):
    """Sentinel-1 SAR / U-Net flood-segmentation output."""

    district: str = Field(..., description="District under assessment.")
    flood_extent_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of district area currently under water (U-Net mask).",
    )
    affected_area_km2: float = Field(
        ..., ge=0, description="Absolute flooded area in square kilometres."
    )


class HydraulicIntelligence(BaseModel):
    """Federal Flood Division (FFD) river-stage and discharge data."""

    station: str = Field(..., description="Nearest gauge / barrage station.")
    river_discharge_cusecs: float = Field(
        ..., ge=0, description="Current river discharge in cusecs."
    )
    inflow_cusecs: float = Field(..., ge=0, description="Inflow at the station.")
    outflow_cusecs: float = Field(..., ge=0, description="Outflow at the station.")
    trend: RiverTrend = Field(..., description="Rising / falling / stable river level.")
    status: RiverStatus = Field(
        RiverStatus.NORMAL, description="FFD stage classification."
    )


class HistoricalIntelligence(BaseModel):
    """Landsat-5 historical benchmark (default 2010 super-flood)."""

    benchmark_year: int = Field(2010, description="Reference flood year.")
    benchmark_flood_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="District flood coverage during the benchmark event.",
    )


class RAGContext(BaseModel):
    """Knowledge retrieved from the RAG module built in PR 1."""

    context: str = Field("", description="Formatted context passages.")
    sources: list[str] = Field(
        default_factory=list, description="Source labels for the retrieved passages."
    )


class RiskAssessment(BaseModel):
    """Structured output of the Disaster Intelligence Agent."""

    risk_level: RiskLevel = Field(..., description="Overall classified risk level.")
    flood_coverage_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current district flood coverage driving the assessment.",
    )
    explanation: str = Field(
        ..., description="Human-readable justification synthesising all streams."
    )
    recommended_action: str = Field(
        ..., description="Concrete recommended operational action."
    )
