from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from coreason_chronos.schemas import ForecastRequest, ForecastResult
from coreason_chronos.visualizer import plot_forecast

# Force non-interactive backend for tests
matplotlib.use("Agg")


@pytest.fixture
def sample_request() -> ForecastRequest:
    return ForecastRequest(
        history=[10.0, 12.0, 15.0, 14.0, 16.0],
        prediction_length=3,
        confidence_level=0.9,
    )


@pytest.fixture
def sample_result() -> ForecastResult:
    return ForecastResult(
        median=[17.0, 18.0, 19.0],
        lower_bound=[16.0, 17.0, 18.0],
        upper_bound=[18.0, 19.0, 20.0],
        confidence_level=0.9,
    )


def test_plot_forecast_returns_figure(sample_request: ForecastRequest, sample_result: ForecastResult) -> None:
    """
    Test that plot_forecast returns a matplotlib Figure object.
    """
    fig = plot_forecast(sample_request, sample_result)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_forecast_content(sample_request: ForecastRequest, sample_result: ForecastResult) -> None:
    """
    Test that the figure contains the expected elements.
    """
    fig = plot_forecast(sample_request, sample_result, title="Test Plot")
    ax = fig.gca()

    # Check title
    assert ax.get_title() == "Test Plot"

    # Check lines: History and Median
    lines = ax.get_lines()
    # We expect 2 lines: 1 for history, 1 for median forecast
    assert len(lines) == 2

    # Verify History Line
    hist_line = lines[0]
    # Cast to ndarray to satisfy mypy
    hist_x = cast(np.ndarray, hist_line.get_xdata())
    hist_y = cast(np.ndarray, hist_line.get_ydata())
    assert len(hist_x) == len(sample_request.history)
    np.testing.assert_array_equal(hist_y, sample_request.history)

    # Verify Forecast Line (Median)
    # We prepended the last history point, so length should be prediction_length + 1
    forecast_line = lines[1]
    forecast_x = cast(np.ndarray, forecast_line.get_xdata())
    forecast_y = cast(np.ndarray, forecast_line.get_ydata())
    assert len(forecast_x) == sample_request.prediction_length + 1
    # First point should be last history point
    assert forecast_y[0] == sample_request.history[-1]
    # Rest should be median
    np.testing.assert_array_equal(forecast_y[1:], sample_result.median)

    # Check collections (fill_between for confidence interval)
    collections = ax.collections
    # We expect 1 collection
    assert len(collections) == 1

    plt.close(fig)


def test_plot_forecast_labels(sample_request: ForecastRequest, sample_result: ForecastResult) -> None:
    """
    Test that custom labels are applied.
    """
    fig = plot_forecast(sample_request, sample_result, ylabel="Revenue")
    ax = fig.gca()
    assert ax.get_ylabel() == "Revenue"
    plt.close(fig)


def test_plot_visual_continuity(sample_request: ForecastRequest, sample_result: ForecastResult) -> None:
    """
    Ensure the forecast line visually connects to the history line.
    """
    fig = plot_forecast(sample_request, sample_result)
    ax = fig.gca()
    lines = ax.get_lines()

    hist_line = lines[0]
    forecast_line = lines[1]

    # Cast to ndarray
    hist_x = cast(np.ndarray, hist_line.get_xdata())
    forecast_x = cast(np.ndarray, forecast_line.get_xdata())
    hist_y = cast(np.ndarray, hist_line.get_ydata())
    forecast_y = cast(np.ndarray, forecast_line.get_ydata())

    # Last X of history == First X of forecast
    # Indices: History [0, 1, 2, 3, 4] -> Last index 4
    # Forecast starts at 4
    assert hist_x[-1] == forecast_x[0]

    # Last Y of history == First Y of forecast
    assert hist_y[-1] == forecast_y[0]

    plt.close(fig)
