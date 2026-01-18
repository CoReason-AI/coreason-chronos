from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
from coreason_chronos.forecaster import ChronosForecaster
from coreason_chronos.schemas import ForecastRequest, ForecastResult


@pytest.fixture
def mock_pipeline_class() -> Generator[MagicMock, None, None]:
    with patch("coreason_chronos.forecaster.ChronosPipeline") as mock_class:
        pipeline_instance = MagicMock()
        mock_class.from_pretrained.return_value = pipeline_instance
        yield mock_class


def test_initialization(mock_pipeline_class: MagicMock) -> None:
    forecaster = ChronosForecaster(model_name="test-model", device="cpu")

    # Assert pipeline is the return value of from_pretrained
    assert forecaster.pipeline == mock_pipeline_class.from_pretrained.return_value

    # Verify from_pretrained was called correctly
    mock_pipeline_class.from_pretrained.assert_called_with("test-model", device_map="cpu", torch_dtype=torch.float32)


def test_initialization_with_quantization(mock_pipeline_class: MagicMock) -> None:
    # Test INT8 quantization initialization
    _ = ChronosForecaster(model_name="test-model", device="cpu", quantization="int8")

    # Assert load_in_8bit=True is passed
    mock_pipeline_class.from_pretrained.assert_called_with("test-model", device_map="cpu", load_in_8bit=True)
    # torch_dtype should NOT be passed when load_in_8bit is True (or handled by accelerate)
    # Our implementation logic excludes torch_dtype if quantization="int8"
    assert "torch_dtype" not in mock_pipeline_class.from_pretrained.call_args.kwargs


def test_initialization_unsupported_quantization(mock_pipeline_class: MagicMock) -> None:
    """Test that ValueError is raised for unsupported quantization modes."""
    with pytest.raises(ValueError, match="Unsupported quantization mode: float16"):
        ChronosForecaster(model_name="test-model", device="cpu", quantization="float16")


def test_initialization_int8_on_cpu(mock_pipeline_class: MagicMock) -> None:
    """
    Test that we can request int8 even on CPU.
    The underlying library might fail, but our wrapper should attempt to pass the config.
    """
    _ = ChronosForecaster(model_name="test-model", device="cpu", quantization="int8")
    mock_pipeline_class.from_pretrained.assert_called_with("test-model", device_map="cpu", load_in_8bit=True)


def test_forecast_happy_path(mock_pipeline_class: MagicMock) -> None:
    # Setup mock instance
    mock_instance = mock_pipeline_class.from_pretrained.return_value

    # Mock prediction output
    # Shape: [num_series=1, num_samples=20, prediction_length=3]
    prediction_length = 3
    num_samples = 20
    # Create random samples
    mock_samples = torch.rand(1, num_samples, prediction_length)
    mock_instance.predict.return_value = mock_samples

    forecaster = ChronosForecaster()

    request = ForecastRequest(
        history=[1.0, 2.0, 3.0, 4.0],
        prediction_length=prediction_length,
        confidence_level=0.90,
    )

    result = forecaster.forecast(request)

    assert isinstance(result, ForecastResult)
    assert len(result.median) == prediction_length
    assert len(result.lower_bound) == prediction_length
    assert len(result.upper_bound) == prediction_length
    assert result.confidence_level == 0.90

    # Verify pipeline.predict call
    call_args = mock_instance.predict.call_args
    assert call_args is not None
    # Check context tensor
    tensor_arg = call_args[0][0]
    assert torch.equal(tensor_arg, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert call_args[1]["prediction_length"] == prediction_length


def test_forecast_logic_quantiles(mock_pipeline_class: MagicMock) -> None:
    """
    Test that quantiles are calculated correctly from the samples.
    """
    mock_instance = mock_pipeline_class.from_pretrained.return_value

    forecaster = ChronosForecaster()

    # Create deterministic samples to verify quantile calculation
    # Let's say we have 5 samples: [10, 20, 30, 40, 50] at each step
    # prediction_length = 1
    # samples = [[10], [20], [30], [40], [50]] -> shape [1, 5, 1]

    samples_data = torch.tensor([[[10.0], [20.0], [30.0], [40.0], [50.0]]])
    mock_instance.predict.return_value = samples_data

    # Request P80 confidence (alpha = 0.1, so 10th and 90th percentile)
    # With 5 samples:
    # 0% -> 10, 25% -> 20, 50% -> 30, 75% -> 40, 100% -> 50
    # numpy quantile interpolation='linear' (default)
    # 0.1 quantile of [10, 20, 30, 40, 50]
    # index = 0.1 * (5-1) = 0.4 -> 10 + 0.4*(20-10) = 14.0
    # 0.5 quantile = 30.0
    # 0.9 quantile -> index = 0.9 * 4 = 3.6 -> 40 + 0.6*(50-40) = 46.0

    request = ForecastRequest(
        history=[1.0],
        prediction_length=1,
        confidence_level=0.80,  # 10% - 90%
    )

    result = forecaster.forecast(request)

    assert result.median[0] == 30.0
    assert result.lower_bound[0] == 14.0
    assert result.upper_bound[0] == 46.0


def test_forecast_covariates_warning(mock_pipeline_class: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    """
    Test that a warning is logged when covariates are provided.
    """
    mock_instance = mock_pipeline_class.from_pretrained.return_value
    mock_instance.predict.return_value = torch.rand(1, 20, 3)

    forecaster = ChronosForecaster()

    request = ForecastRequest(history=[1.0, 2.0], prediction_length=3, confidence_level=0.90, covariates=[1, 0, 1])

    with patch("coreason_chronos.forecaster.logger") as mock_logger:
        forecaster.forecast(request)
        mock_logger.warning.assert_called_with(
            "Covariates were provided but are not supported by the current Chronos implementation."
            " They will be ignored."
        )


def test_forecast_no_warning_when_covariates_none_or_empty(
    mock_pipeline_class: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test that NO warning is logged when covariates are None or empty.
    """
    mock_instance = mock_pipeline_class.from_pretrained.return_value
    mock_instance.predict.return_value = torch.rand(1, 20, 3)

    forecaster = ChronosForecaster()

    # Case 1: None
    request_none = ForecastRequest(history=[1.0], prediction_length=3, confidence_level=0.90, covariates=None)

    with patch("coreason_chronos.forecaster.logger") as mock_logger:
        forecaster.forecast(request_none)
        mock_logger.warning.assert_not_called()

    # Case 2: Empty List
    request_empty = ForecastRequest(history=[1.0], prediction_length=3, confidence_level=0.90, covariates=[])

    with patch("coreason_chronos.forecaster.logger") as mock_logger:
        forecaster.forecast(request_empty)
        mock_logger.warning.assert_not_called()


@pytest.mark.live
def test_forecast_integration_real_model() -> None:
    """
    Integration test using the real model.
    """
    pytest.skip("Skipping heavy model test in limited environment")
    try:
        forecaster = ChronosForecaster(model_name="amazon/chronos-t5-tiny", device="cpu")
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    history = [float(i) for i in range(20)]  # Linear trend
    request = ForecastRequest(
        history=history,
        prediction_length=5,
        confidence_level=0.90,
    )

    result = forecaster.forecast(request)

    assert len(result.median) == 5
    assert not all(v == 0 for v in result.median)
