"""
Tests for the email alert notifier (PR 3 — email integration).

These cover the pure composition logic and verify that ``send_flood_alert``
drives SMTP correctly without opening a real network connection (SMTP is mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import (
    EmailConfigError,
    EvacuationRoute,
    FloodState,
    LocationType,
    RiskLevel,
    SMTPConfig,
    SafeZoneCandidate,
    build_alert_email,
    maps_link,
    send_flood_alert,
)


def _scenario():
    flood_state = FloodState(
        district="Charsadda",
        flood_coverage_percentage=31,
        affected_area_km2=140,
        population_at_risk=120_000,
        available_shelters=8,
    )
    safe_zone = SafeZoneCandidate(
        name="Government High School #2",
        location_type=LocationType.SCHOOL,
        latitude=34.17190,
        longitude=71.74400,
        in_flood_zone=False,
        elevation_m=40.0,
        road_accessible=True,
        route_crosses_flood=False,
        distance_km=3.2,
    )
    route = EvacuationRoute(
        path=["Origin", "Junction A", "Government High School #2"],
        distance_km=3.2,
        estimated_travel_time_min=11,
    )
    return flood_state, safe_zone, route


def test_build_alert_email_contains_situation_location_and_eta():
    flood_state, safe_zone, route = _scenario()

    message = build_alert_email(
        recipient="citizen@example.com",
        risk_level=RiskLevel.HIGH_RISK,
        flood_state=flood_state,
        safe_zone=safe_zone,
        route=route,
    )

    assert message["To"] == "citizen@example.com"
    assert "Charsadda" in message["Subject"]
    assert "Government High School #2" in message["Subject"]

    body = message.get_content()
    # Current situation of the affected area.
    assert "Flood coverage:            31% of the district" in body
    assert "120,000" in body
    # Exact coordinates of the recommended safe location.
    assert "34.17190, 71.74400" in body
    assert maps_link(34.17190, 71.74400) in body
    # Estimated time taken to reach the safe zone.
    assert "Estimated travel time:     11 minutes" in body
    # Mandated citizen-action phrasing is carried through.
    assert "Nearest Safe Shelter: Government High School #2" in body
    assert "Please follow the highlighted evacuation route immediately." in body


def test_send_flood_alert_drives_smtp_without_network():
    flood_state, safe_zone, route = _scenario()
    config = SMTPConfig(sender="alerts@floodsense.pk", password="app-pass")

    fake_server = MagicMock()
    with patch("agent.email_notifier.smtplib.SMTP") as smtp_cls:
        smtp_cls.return_value.__enter__.return_value = fake_server

        message = send_flood_alert(
            recipient="citizen@example.com",
            risk_level=RiskLevel.HIGH_RISK,
            flood_state=flood_state,
            safe_zone=safe_zone,
            route=route,
            config=config,
        )

    smtp_cls.assert_called_once_with("smtp.gmail.com", 587)
    fake_server.starttls.assert_called_once()
    fake_server.login.assert_called_once_with("alerts@floodsense.pk", "app-pass")
    fake_server.send_message.assert_called_once()
    assert message["From"] == "alerts@floodsense.pk"


def test_smtp_config_from_env_requires_credentials(monkeypatch):
    # Blank credentials (present but empty) must be treated as "not configured".
    # Setting them empty also stops load_dotenv() from repopulating from .env.
    monkeypatch.setenv("EMAIL_SENDER", "")
    monkeypatch.setenv("EMAIL_APP_PASSWORD", "")
    with pytest.raises(EmailConfigError):
        SMTPConfig.from_env()


def test_smtp_config_from_env_reads_values(monkeypatch):
    monkeypatch.setenv("EMAIL_SENDER", "alerts@floodsense.pk")
    monkeypatch.setenv("EMAIL_APP_PASSWORD", "secret-pass")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")

    config = SMTPConfig.from_env()

    assert config.sender == "alerts@floodsense.pk"
    assert config.password == "secret-pass"
    assert config.host == "smtp.example.com"
    assert config.port == 2525
