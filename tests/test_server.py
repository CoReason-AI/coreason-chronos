import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from coreason_chronos.server import app
from coreason_chronos.schemas import TemporalEvent, TemporalGranularity

# Mock data
MOCK_EVENT = TemporalEvent(
    id=uuid4(),
    description="Test Event",
    timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
    granularity=TemporalGranularity.PRECISE,
    source_snippet="Test Event at 10am"
)

@pytest.fixture
def client():
    # Patch ChronosTimekeeperAsync to avoid loading model
    with patch("coreason_chronos.server.ChronosTimekeeperAsync") as MockTimekeeper:
        # Mock instance
        mock_instance = MockTimekeeper.return_value

        # Mock async methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)

        mock_instance.extract_from_text = AsyncMock(return_value=[MOCK_EVENT])

        mock_instance.forecast_series = AsyncMock(return_value=MagicMock(
            median=[10.0],
            lower_bound=[5.0],
            upper_bound=[15.0],
            confidence_level=0.9,
            model_dump=lambda **kwargs: {
                "median": [10.0],
                "lower_bound": [5.0],
                "upper_bound": [15.0],
                "confidence_level": 0.9
            }
        ))

        with TestClient(app) as c:
            yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "amazon/chronos-t5-tiny",
        "device": "cpu"
    }

def test_extract_endpoint(client):
    response = client.post("/extract", json={
        "text": "Test text",
        "ref_date": "2024-01-01T00:00:00Z"
    })
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["description"] == "Test Event"
    # Check serialization
    assert "timestamp" in data[0]

def test_extract_endpoint_no_date(client):
    response = client.post("/extract", json={
        "text": "Test text"
    })
    assert response.status_code == 200

def test_forecast_endpoint(client):
    response = client.post("/forecast", json={
        "history": [1, 2, 3],
        "prediction_length": 1,
        "confidence_level": 0.9
    })
    assert response.status_code == 200
    data = response.json()
    assert data["median"] == [10.0]

def test_forecast_endpoint_invalid(client):
    response = client.post("/forecast", json={
        "history": [], # Invalid
        "prediction_length": 1,
        "confidence_level": 0.9
    })
    assert response.status_code == 422 # Validation error

def test_extract_endpoint_error(client):
    # Mock extract to raise exception
    with patch("coreason_chronos.server.app.state.timekeeper.extract_from_text", side_effect=Exception("Simulated error")):
        response = client.post("/extract", json={"text": "Fail me"})
        assert response.status_code == 500
        assert "Simulated error" in response.json()["detail"]

def test_forecast_endpoint_error(client):
    # Mock forecast to raise exception
    with patch("coreason_chronos.server.app.state.timekeeper.forecast_series", side_effect=Exception("Simulated forecast error")):
        response = client.post("/forecast", json={
            "history": [1, 2],
            "prediction_length": 1,
            "confidence_level": 0.9
        })
        assert response.status_code == 500
        assert "Simulated forecast error" in response.json()["detail"]
