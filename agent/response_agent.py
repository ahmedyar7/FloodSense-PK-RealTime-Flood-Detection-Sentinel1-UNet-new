"""
Response & Communication Agent — the final operational node of the platform.

It answers *"Where should I go and how do I get there safely?"* in three stages:

1. **Safe-zone evaluation** — filter (mock OSM) candidate shelters down to those
   that are outside the flood zone, at a safe elevation, road-accessible, and
   reachable without crossing flooded areas.
2. **Evacuation routing** — compute the optimal route to the chosen safe zone
   with Dijkstra's shortest-path algorithm over the road graph, skipping flooded
   segments.
3. **Alert generation** — produce a *citizen* alert (immediate action, nearest
   shelter, distance, time) and an *authority* alert (macro statistics).

All logic is deterministic and rule-based so it is fully testable without an LLM.
"""

import heapq
from typing import Optional

from .response_schemas import (
    EvacuationRoute,
    FloodState,
    GraphEdge,
    RiskLevel,
    SafeZoneCandidate,
)

# Minimum ground elevation (m) we consider safe from inundation.
SAFE_ELEVATION_THRESHOLD_M = 15.0

# Average evacuation speed (km/h) in congested, flood-affected conditions; used to
# turn a route distance into an estimated travel time.
AVERAGE_EVACUATION_SPEED_KMH = 18.0


# --------------------------------------------------------------------------- #
# 1. Safe-zone evaluation
# --------------------------------------------------------------------------- #
def is_safe_zone(candidate: SafeZoneCandidate) -> bool:
    """A candidate is valid only if it passes *all four* safety checks."""
    return (
        not candidate.in_flood_zone
        and candidate.elevation_m >= SAFE_ELEVATION_THRESHOLD_M
        and candidate.road_accessible
        and not candidate.route_crosses_flood
    )


def evaluate_safe_zones(
    candidates: list[SafeZoneCandidate],
) -> list[SafeZoneCandidate]:
    """Return only the candidates that pass every safety check."""
    return [c for c in candidates if is_safe_zone(c)]


def recommend_safe_zone(
    candidates: list[SafeZoneCandidate],
) -> Optional[SafeZoneCandidate]:
    """Pick the best safe zone: the nearest valid candidate, or ``None``.

    Candidates without a known distance are ranked last so any candidate with a
    concrete distance is preferred.
    """
    valid = evaluate_safe_zones(candidates)
    if not valid:
        return None
    return min(
        valid,
        key=lambda c: c.distance_km if c.distance_km is not None else float("inf"),
    )


# --------------------------------------------------------------------------- #
# 2. Evacuation routing (Dijkstra)
# --------------------------------------------------------------------------- #
def plan_route(
    edges: list[GraphEdge],
    origin: str,
    destination: str,
    speed_kmh: float = AVERAGE_EVACUATION_SPEED_KMH,
) -> EvacuationRoute:
    """Shortest evacuation route from ``origin`` to ``destination``.

    Roads are bidirectional and flooded segments are impassable, so they are
    excluded from the graph. Raises ``ValueError`` when no dry route exists.
    """
    # Build an undirected adjacency list from the non-flooded segments.
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        if edge.flooded:
            continue
        adjacency.setdefault(edge.source, []).append((edge.target, edge.distance_km))
        adjacency.setdefault(edge.target, []).append((edge.source, edge.distance_km))

    # Classic Dijkstra with a binary heap.
    best_distance: dict[str, float] = {origin: 0.0}
    previous: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(0.0, origin)]
    visited: set[str] = set()

    while queue:
        distance, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            break
        for neighbour, weight in adjacency.get(node, []):
            candidate = distance + weight
            if candidate < best_distance.get(neighbour, float("inf")):
                best_distance[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))

    if destination not in best_distance:
        raise ValueError(f"No dry evacuation route from {origin!r} to {destination!r}.")

    # Reconstruct the path from destination back to origin.
    path = [destination]
    while path[-1] != origin:
        path.append(previous[path[-1]])
    path.reverse()

    # Round to one decimal so the reported distance is stable against float noise.
    total_distance = round(best_distance[destination], 1)
    travel_time = round(total_distance / speed_kmh * 60)

    return EvacuationRoute(
        path=path,
        distance_km=total_distance,
        estimated_travel_time_min=travel_time,
    )


# --------------------------------------------------------------------------- #
# 3. Alert generation
# --------------------------------------------------------------------------- #
def generate_citizen_alert(
    district: str,
    risk_level: RiskLevel,
    shelter_name: str,
    distance_km: float,
    travel_time_min: int,
) -> str:
    """Public-facing citizen alert focused on immediate evacuation action."""
    return (
        f"🚨 FLOOD EVACUATION ALERT — {district} ({risk_level.value})\n"
        f"Nearest Safe Shelter: {shelter_name}\n"
        f"Distance: {distance_km:g} km "
        f"(approx. {travel_time_min} minutes travel time)\n"
        f"Please follow the highlighted evacuation route immediately."
    )


def generate_authority_alert(flood_state: FloodState) -> str:
    """Authority-facing alert focused on macro flood statistics."""
    return (
        f"AUTHORITY FLOOD SITUATION REPORT — {flood_state.district}\n"
        f"Current Flood Coverage: {flood_state.flood_coverage_percentage:g}%\n"
        f"Affected Area: {flood_state.affected_area_km2:g} sq km\n"
        f"Estimated Population at Risk: {flood_state.population_at_risk:,}\n"
        f"Available Shelters: {flood_state.available_shelters}"
    )
