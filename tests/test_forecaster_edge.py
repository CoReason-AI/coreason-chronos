from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError

from coreason_chronos.forecaster import ChronosForecaster
from coreason_chronos.schemas import ForecastRequest


@pytest.fixture  # type: ignore
def mock_pipeline_class() -> Generator[MagicMock, None, None]:
    with patch("coreason_chronos.forecaster.ChronosPipeline") as mock_class:
        pipeline_instance = MagicMock()
        mock_class.from_pretrained.return_value = pipeline_instance
        yield mock_class


def test_validation_empty_history() -> None:
    """Test that empty history raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        ForecastRequest(history=[], prediction_length=5, confidence_level=0.9)
    assert "history must not be empty" in str(excinfo.value)


def test_validation_nan_history() -> None:
    """Test that history with NaN raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        ForecastRequest(history=[1.0, float("nan"), 3.0], prediction_length=5, confidence_level=0.9)
    assert "history must not contain NaN or Inf values" in str(excinfo.value)


def test_validation_inf_history() -> None:
    """Test that history with Inf raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        ForecastRequest(history=[1.0, float("inf"), 3.0], prediction_length=5, confidence_level=0.9)
    assert "history must not contain NaN or Inf values" in str(excinfo.value)


def test_single_point_history(mock_pipeline_class: MagicMock) -> None:
    """Test forecasting with a single data point."""
    mock_instance = mock_pipeline_class.from_pretrained.return_value
    mock_instance.predict.return_value = torch.rand(1, 20, 5)

    forecaster = ChronosForecaster()
    request = ForecastRequest(history=[10.0], prediction_length=5, confidence_level=0.9)

    result = forecaster.forecast(request)
    assert len(result.median) == 5


def test_constant_history(mock_pipeline_class: MagicMock) -> None:
    """Test forecasting with constant history (logic check, though mocked)."""
    mock_instance = mock_pipeline_class.from_pretrained.return_value
    # If history is constant, model usually predicts constant or slightly noisy constant.
    # Since we mock the pipeline, we are testing that the code doesn't crash.
    mock_instance.predict.return_value = torch.ones(1, 20, 5) * 10.0

    forecaster = ChronosForecaster()
    request = ForecastRequest(history=[10.0] * 20, prediction_length=5, confidence_level=0.9)

    result = forecaster.forecast(request)
    # If all samples are 10.0, median should be 10.0
    assert result.median == [10.0] * 5
    assert result.lower_bound == [10.0] * 5
    assert result.upper_bound == [10.0] * 5


@pytest.mark.live  # type: ignore
def test_real_model_complex_cases() -> None:
    """
    Test real model with edge cases.
    """
    try:
        forecaster = ChronosForecaster(model_name="amazon/chronos-t5-tiny", device="cpu")
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    # Case 1: Minimal history (1 point)
    req_single = ForecastRequest(history=[100.0], prediction_length=3, confidence_level=0.9)
    res_single = forecaster.forecast(req_single)
    assert len(res_single.median) == 3

    # Case 2: Constant history
    req_const = ForecastRequest(history=[50.0] * 20, prediction_length=3, confidence_level=0.9)
    res_const = forecaster.forecast(req_const)
    # Check bounds - reasonable for tiny model?
    # It might drift, but shouldn't be crazy.
    assert 40.0 < res_const.median[0] < 60.0
