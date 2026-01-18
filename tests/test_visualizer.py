from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from coreason_chronos.schemas import ForecastRequest, ForecastResult
from coreason_chronos.visualizer import plot_forecast
from matplotlib.figure import Figure

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


def test_plot_single_history_point() -> None:
    """
    Test behavior when history contains only a single data point.
    """
    request = ForecastRequest(
        history=[100.0],
        prediction_length=2,
        confidence_level=0.9,
    )
    result = ForecastResult(
        median=[101.0, 102.0],
        lower_bound=[100.0, 101.0],
        upper_bound=[102.0, 103.0],
        confidence_level=0.9,
    )

    fig = plot_forecast(request, result)
    ax = fig.gca()
    lines = ax.get_lines()

    # Verify history line (len 1)
    hist_line = lines[0]
    hist_x = cast(np.ndarray, hist_line.get_xdata())
    assert len(hist_x) == 1
    assert hist_x[0] == 0

    # Verify forecast line starts at 0
    forecast_line = lines[1]
    forecast_x = cast(np.ndarray, forecast_line.get_xdata())
    assert forecast_x[0] == 0
    assert len(forecast_x) == 3  # [0, 1, 2]

    plt.close(fig)


def test_plot_large_dataset() -> None:
    """
    Test handling of large history and forecast arrays (performance/stability check).
    """
    history = [float(i) for i in range(1000)]
    forecast_len = 500
    median = [float(i) for i in range(1000, 1500)]
    bounds = median  # Collapsed CI for simplicity

    request = ForecastRequest(
        history=history,
        prediction_length=forecast_len,
        confidence_level=0.9,
    )
    result = ForecastResult(
        median=median,
        lower_bound=bounds,
        upper_bound=bounds,
        confidence_level=0.9,
    )

    fig = plot_forecast(request, result)
    ax = fig.gca()
    lines = ax.get_lines()

    hist_line = lines[0]
    forecast_line = lines[1]

    # Check data lengths
    hist_x = cast(np.ndarray, hist_line.get_xdata())
    assert len(hist_x) == 1000

    forecast_x = cast(np.ndarray, forecast_line.get_xdata())
    assert len(forecast_x) == 501  # 500 + 1 anchor

    plt.close(fig)


def test_plot_zero_variance() -> None:
    """
    Test plotting when median equals bounds (flat line, collapsed confidence interval).
    """
    request = ForecastRequest(
        history=[10.0, 10.0, 10.0],
        prediction_length=3,
        confidence_level=0.9,
    )
    result = ForecastResult(
        median=[10.0, 10.0, 10.0],
        lower_bound=[10.0, 10.0, 10.0],
        upper_bound=[10.0, 10.0, 10.0],
        confidence_level=0.9,
    )

    fig = plot_forecast(request, result)
    ax = fig.gca()
    collections = ax.collections
    assert len(collections) == 1  # Should still exist even if width is 0
    plt.close(fig)


def test_plot_nan_values_in_forecast() -> None:
    """
    Test resilience when forecast result contains NaNs.
    """
    request = ForecastRequest(
        history=[1.0, 2.0, 3.0],
        prediction_length=3,
        confidence_level=0.9,
    )
    result = ForecastResult(
        median=[4.0, float("nan"), 6.0],
        lower_bound=[3.0, float("nan"), 5.0],
        upper_bound=[5.0, float("nan"), 7.0],
        confidence_level=0.9,
    )

    # Should not raise exception
    fig = plot_forecast(request, result)
    ax = fig.gca()
    lines = ax.get_lines()

    forecast_line = lines[1]
    forecast_y = cast(np.ndarray, forecast_line.get_ydata())

    assert np.isnan(forecast_y[2])  # index 0 is anchor, 1 is 4.0, 2 is nan
    plt.close(fig)


def test_plot_confidence_label_formatting() -> None:
    """
    Test label generation for non-integer confidence levels.
    """
    request = ForecastRequest(
        history=[1.0, 2.0],
        prediction_length=1,
        confidence_level=0.955,
    )
    result = ForecastResult(
        median=[3.0],
        lower_bound=[2.0],
        upper_bound=[4.0],
        confidence_level=0.955,
    )

    fig = plot_forecast(request, result)
    ax = fig.gca()
    # Check legend text
    legend = ax.get_legend()
    # If legend is not None, find the text
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]

    # logic: int(0.955 * 100) = 95
    assert any("95%" in t for t in texts)

    plt.close(fig)
