from typing import Any, Dict, Optional

import numpy as np
import torch
from chronos import ChronosPipeline

from coreason_chronos.schemas import ForecastRequest, ForecastResult
from coreason_chronos.utils.logger import logger

DEFAULT_CHRONOS_MODEL = "amazon/chronos-t5-tiny"


class ChronosForecaster:
    """
    The Oracle: Forecasting engine using Amazon Chronos T5 model.

    This class wraps the Foundation Time-Series Model to provide zero-shot forecasting capabilities.
    It handles tokenization, inference, and quantile estimation to generate probabilistic forecasts.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CHRONOS_MODEL,
        device: str = "cpu",
        quantization: Optional[str] = None,
    ) -> None:
        """
        Initialize the Chronos pipeline.

        Args:
            model_name: The HuggingFace model identifier (e.g., "amazon/chronos-t5-small").
            device: Device to run the model on ('cpu' or 'cuda').
            quantization: Quantization mode (e.g., 'int8').
                          If 'int8', uses `load_in_8bit=True` (requires bitsandbytes).

        Raises:
            ValueError: If an unsupported quantization mode is provided.
        """
        logger.info(
            f"Initializing ChronosForecaster with model '{model_name}' on {device} (Quantization: {quantization})"
        )

        kwargs: Dict[str, Any] = {
            "device_map": device,
        }

        # Handle torch_dtype and quantization logic
        if quantization == "int8":
            # 8-bit quantization typically requires bitsandbytes and CUDA,
            # but usually 'load_in_8bit=True' handles the config.
            kwargs["load_in_8bit"] = True
            # When using load_in_8bit, torch_dtype is often inferred or set to float16 automatically
            # by accelerate/bitsandbytes
        elif quantization is not None:
            raise ValueError(f"Unsupported quantization mode: {quantization}")
        else:
            # Default behavior
            if device == "cpu":
                kwargs["torch_dtype"] = torch.float32
            else:
                kwargs["torch_dtype"] = torch.bfloat16  # pragma: no cover

        self.pipeline = ChronosPipeline.from_pretrained(model_name, **kwargs)

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """
        Generate a probabilistic forecast based on the request.

        Args:
            request: The forecasting request containing history, horizon, and confidence level.

        Returns:
            ForecastResult containing median prediction and confidence intervals.
        """
        logger.debug(f"Received forecast request for {len(request.history)} history points.")

        if request.covariates:
            logger.warning(
                "Covariates were provided but are not supported by the current Chronos implementation."
                " They will be ignored."
            )

        # Convert history to tensor
        context = torch.tensor(request.history)

        # Predict
        # pipeline.predict returns shape [num_series, num_samples, prediction_length]
        # We process one series at a time here (ForecastRequest is single series).
        forecast_tensor = self.pipeline.predict(
            context,
            prediction_length=request.prediction_length,
            num_samples=20,  # Default samples for distribution
        )

        # Calculate quantiles
        # confidence_level e.g. 0.90 means we want the middle 90%.
        # So we want (1 - 0.90) / 2 = 0.05 and 1 - 0.05 = 0.95
        alpha = (1.0 - request.confidence_level) / 2.0
        low_q = alpha
        high_q = 1.0 - alpha

        # Extract numpy array from tensor [1, num_samples, prediction_length] -> [num_samples, prediction_length]
        forecast_samples = forecast_tensor[0].numpy()

        low, median, high = np.quantile(forecast_samples, [low_q, 0.5, high_q], axis=0)

        result = ForecastResult(
            median=median.tolist(),
            lower_bound=low.tolist(),
            upper_bound=high.tolist(),
            confidence_level=request.confidence_level,
        )

        logger.info("Forecast generated successfully.")
        return result
