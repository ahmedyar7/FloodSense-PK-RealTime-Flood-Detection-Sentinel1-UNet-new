"""
Email delivery for the Response & Communication Agent.

When the platform detects danger for a citizen who subscribed with their email at
the area-selection screen, this module composes and sends a personalised flood
alert: the current situation of the affected area, the recommended safe zone with
its exact coordinates, and the estimated travel time to reach it.

Transport is plain SMTP (Python's standard library), so it works with any SMTP
provider. Credentials are read from the environment (see ``.env``):

    EMAIL_SENDER          sender address (e.g. a Gmail address)
    EMAIL_APP_PASSWORD    SMTP password / Gmail App Password
    SMTP_HOST             SMTP host   (default: smtp.gmail.com)
    SMTP_PORT             SMTP port   (default: 587, STARTTLS)

``build_alert_email`` is pure and side-effect free so it is fully unit-testable;
``send_flood_alert`` performs the actual network send.
"""

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional

from .response_agent import generate_citizen_alert
from .response_schemas import (
    EvacuationRoute,
    FloodState,
    RiskLevel,
    SafeZoneCandidate,
)


class EmailConfigError(RuntimeError):
    """Raised when SMTP credentials are missing or incomplete."""


@dataclass(frozen=True)
class SMTPConfig:
    """SMTP connection settings, normally loaded from the environment."""

    sender: str
    password: str
    host: str = "smtp.gmail.com"
    port: int = 587

    @classmethod
    def from_env(cls) -> "SMTPConfig":
        # Ensure .env is loaded even when this module is used standalone.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        sender = os.getenv("EMAIL_SENDER", "").strip()
        password = os.getenv("EMAIL_APP_PASSWORD", "").strip()
        if not sender or not password:
            raise EmailConfigError(
                "Email is not configured. Set EMAIL_SENDER and EMAIL_APP_PASSWORD "
                "in your .env (see agent/email_notifier.py for details)."
            )
        return cls(
            sender=sender,
            password=password,
            host=os.getenv("SMTP_HOST", "smtp.gmail.com").strip(),
            port=int(os.getenv("SMTP_PORT", "587")),
        )


def maps_link(latitude: float, longitude: float) -> str:
    """Google Maps directions link to the given coordinates."""
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&destination={latitude},{longitude}"
    )


def build_alert_email(
    recipient: str,
    risk_level: RiskLevel,
    flood_state: FloodState,
    safe_zone: SafeZoneCandidate,
    route: EvacuationRoute,
) -> EmailMessage:
    """Compose the personalised flood-alert email (no network I/O).

    The body combines the citizen action message, the current situation of the
    affected area, and precise directions to the recommended safe zone.
    """
    district = flood_state.district
    citizen_message = generate_citizen_alert(
        district=district,
        risk_level=risk_level,
        shelter_name=safe_zone.name,
        distance_km=route.distance_km,
        travel_time_min=route.estimated_travel_time_min,
    )

    body = (
        f"{citizen_message}\n\n"
        "──────────────────────────────────────────\n"
        f"CURRENT SITUATION — {district}\n"
        "──────────────────────────────────────────\n"
        f"Risk level:                {risk_level.value}\n"
        f"Flood coverage:            {flood_state.flood_coverage_percentage:g}% of the district\n"
        f"Affected area:             {flood_state.affected_area_km2:g} sq km\n"
        f"Estimated population at risk: {flood_state.population_at_risk:,}\n"
        f"Available safe shelters:   {flood_state.available_shelters}\n\n"
        "──────────────────────────────────────────\n"
        "WHERE TO GO\n"
        "──────────────────────────────────────────\n"
        f"Safe zone:                 {safe_zone.name} ({safe_zone.location_type.value})\n"
        f"Exact coordinates:         {safe_zone.latitude:.5f}, {safe_zone.longitude:.5f}\n"
        f"Open directions:           {maps_link(safe_zone.latitude, safe_zone.longitude)}\n"
        f"Distance:                  {route.distance_km:g} km\n"
        f"Estimated travel time:     {route.estimated_travel_time_min} minutes\n"
        f"Evacuation route:          {' → '.join(route.path)}\n\n"
        "Move now and avoid flooded roads. This is an automated alert from "
        "FloodSense PK.\n"
    )

    message = EmailMessage()
    message["Subject"] = (
        f"🚨 FLOOD ALERT [{risk_level.value}] — {district}: evacuate to {safe_zone.name}"
    )
    message["To"] = recipient
    message.set_content(body)
    return message


def send_flood_alert(
    recipient: str,
    risk_level: RiskLevel,
    flood_state: FloodState,
    safe_zone: SafeZoneCandidate,
    route: EvacuationRoute,
    config: Optional[SMTPConfig] = None,
) -> EmailMessage:
    """Build and send the flood-alert email over SMTP.

    Returns the sent ``EmailMessage`` (handy for logging/UI). Raises
    ``EmailConfigError`` if SMTP credentials are missing.
    """
    config = config or SMTPConfig.from_env()
    message = build_alert_email(recipient, risk_level, flood_state, safe_zone, route)
    message["From"] = config.sender

    with smtplib.SMTP(config.host, config.port) as server:
        server.starttls()
        server.login(config.sender, config.password)
        server.send_message(message)

    return message
