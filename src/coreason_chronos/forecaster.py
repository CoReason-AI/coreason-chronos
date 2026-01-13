import numpy as np
import torch
from chronos import ChronosPipeline

from coreason_chronos.schemas import ForecastRequest, ForecastResult
from coreason_chronos.utils.logger import logger


class ChronosForecaster:
    """
    Forecasting engine using Amazon Chronos T5 model.
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-tiny", device: str = "cpu") -> None:
        """
        Initialize the Chronos pipeline.

        Args:
            model_name: The HuggingFace model identifier.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        logger.info(f"Initializing ChronosForecaster with model '{model_name}' on {device}")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        )

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """
        Generate a forecast based on the request.

        Args:
            request: The forecasting request containing history and parameters.

        Returns:
            ForecastResult containing median and confidence intervals.
        """
        logger.debug(f"Received forecast request for {len(request.history)} history points.")

        if request.covariates:
            logger.warning(
                "Covariates were provided but are not supported by the current Chronos implementation. They will be ignored."
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
