from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from coreason_chronos.schemas import TemporalEvent, TemporalGranularity
from coreason_chronos.server import app, lifespan

# Mock data
MOCK_EVENT = TemporalEvent(
    id=uuid4(),
    description="Test Event",
    timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
    granularity=TemporalGranularity.PRECISE,
    source_snippet="Test Event at 10am",
)


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    # Patch ChronosTimekeeperAsync to avoid loading model
    with patch("coreason_chronos.server.ChronosTimekeeperAsync") as MockTimekeeper:
        # Mock instance
        mock_instance = MockTimekeeper.return_value

        # Mock async methods
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)

        mock_instance.extract_from_text = AsyncMock(return_value=[MOCK_EVENT])

        mock_instance.forecast_series = AsyncMock(
            return_value=MagicMock(
                median=[10.0],
                lower_bound=[5.0],
                upper_bound=[15.0],
                confidence_level=0.9,
                model_dump=lambda **kwargs: {
                    "median": [10.0],
                    "lower_bound": [5.0],
                    "upper_bound": [15.0],
                    "confidence_level": 0.9,
                },
            )
        )

        # Manually set state to bypass lifespan execution during test
        app.state.timekeeper = mock_instance

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

        # Cleanup
        if hasattr(app.state, "timekeeper"):
            del app.state.timekeeper


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "amazon/chronos-t5-tiny",
        "device": "cpu",
    }


@pytest.mark.asyncio
async def test_extract_endpoint(client: AsyncClient) -> None:
    response = await client.post("/extract", json={"text": "Test text", "ref_date": "2024-01-01T00:00:00Z"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["description"] == "Test Event"
    # Check serialization
    assert "timestamp" in data[0]


@pytest.mark.asyncio
async def test_extract_endpoint_no_date(client: AsyncClient) -> None:
    response = await client.post("/extract", json={"text": "Test text"})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_forecast_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/forecast",
        json={"history": [1, 2, 3], "prediction_length": 1, "confidence_level": 0.9},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["median"] == [10.0]


@pytest.mark.asyncio
async def test_forecast_endpoint_invalid(client: AsyncClient) -> None:
    response = await client.post(
        "/forecast",
        json={
            "history": [],  # Invalid
            "prediction_length": 1,
            "confidence_level": 0.9,
        },
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_extract_endpoint_error(client: AsyncClient) -> None:
    # Mock extract to raise exception
    app.state.timekeeper.extract_from_text.side_effect = Exception("Simulated error")

    try:
        response = await client.post("/extract", json={"text": "Fail me"})
        assert response.status_code == 500
        assert "Simulated error" in response.json()["detail"]
    finally:
        # Reset side effect
        app.state.timekeeper.extract_from_text.side_effect = None


@pytest.mark.asyncio
async def test_forecast_endpoint_error(client: AsyncClient) -> None:
    # Mock forecast to raise exception
    app.state.timekeeper.forecast_series.side_effect = Exception("Simulated forecast error")

    try:
        response = await client.post(
            "/forecast",
            json={"history": [1, 2], "prediction_length": 1, "confidence_level": 0.9},
        )
        assert response.status_code == 500
        assert "Simulated forecast error" in response.json()["detail"]
    finally:
        app.state.timekeeper.forecast_series.side_effect = None


@pytest.mark.asyncio
async def test_lifespan() -> None:
    # Test the lifespan context manager
    with patch("coreason_chronos.server.ChronosTimekeeperAsync") as MockTimekeeper:
        mock_instance = MockTimekeeper.return_value
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)

        # Test the lifespan
        async with lifespan(app):
            assert hasattr(app.state, "timekeeper")
            assert app.state.timekeeper == mock_instance
            mock_instance.__aenter__.assert_called_once()

        mock_instance.__aexit__.assert_called_once()
