
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from coreason_chronos.schemas import ForecastRequest, ForecastResult
from coreason_chronos.utils.logger import logger


def plot_forecast(
    request: ForecastRequest, result: ForecastResult, title: str = "Forecast", ylabel: str = "Value"
) -> Figure:
    """
    Generates a matplotlib Figure showing the history and the forecast with confidence intervals.

    Args:
        request: The forecast request containing historical data.
        result: The forecast result containing predictions and confidence intervals.
        title: Title of the plot.
        ylabel: Label for the Y-axis.

    Returns:
        A matplotlib Figure object.
    """
    logger.debug(f"Generating forecast plot: {title}")

    # Ensure non-interactive backend is used if not already configured
    # (Though typically this is handled by environment configuration)
    # logic: We create a new figure explicitly.

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # 1. Plot History
    # History indices: 0 to N-1
    history_len = len(request.history)
    history_x = np.arange(history_len)
    ax.plot(history_x, request.history, label="History", color="black", linestyle="-")

    # 2. Plot Forecast
    # Forecast starts from the last history point?
    # Chronos forecasts the *next* steps.
    # To make the plot continuous, we can prepend the last history point to the forecast arrays.
    # Or strictly plot forecast from N to N+K-1.
    # Let's plot strictly from N.

    # But visual continuity is nice.
    # Let's check if the last history point should be the anchor.
    # If history is [A, B, C] (indices 0, 1, 2)
    # Forecast is for index 3, 4, 5...
    # Visually, there is a gap between 2 and 3 if we use line plots.
    # So we should include the last history point in the forecast plot arrays.

    last_hist_val = request.history[-1]
    last_hist_idx = history_len - 1

    forecast_len = len(result.median)
    # Indices for forecast: last_hist_idx to last_hist_idx + forecast_len
    # Wait, if we prepend, length is +1.
    forecast_x = np.arange(last_hist_idx, last_hist_idx + forecast_len + 1)

    # Prepend last history value to forecast data
    median = [last_hist_val] + result.median
    lower = [last_hist_val] + result.lower_bound
    upper = [last_hist_val] + result.upper_bound

    # Plot Median
    ax.plot(forecast_x, median, label="Median Forecast", color="blue", linestyle="--")

    # Plot Confidence Interval
    ax.fill_between(
        forecast_x,
        lower,
        upper,
        color="blue",
        alpha=0.2,
        label=f"Confidence Interval ({int(result.confidence_level * 100)}%)",
    )

    # Styling
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Clean layout
    fig.tight_layout()

    return fig
