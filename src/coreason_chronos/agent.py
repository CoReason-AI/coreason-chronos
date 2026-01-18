# Prosperity Public License 3.0
from datetime import datetime
from typing import List, Optional

from coreason_chronos.causality import CausalityEngine
from coreason_chronos.forecaster import DEFAULT_CHRONOS_MODEL, ChronosForecaster
from coreason_chronos.schemas import ComplianceResult, ForecastRequest, ForecastResult, TemporalEvent
from coreason_chronos.timeline_extractor import TimelineExtractor
from coreason_chronos.utils.logger import logger
from coreason_chronos.validator import ValidationRule


class ChronosTimekeeper:
    """
    The main agent orchestration class for the Extract-Align-Forecast loop.
    Acts as a facade over the specialized components.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CHRONOS_MODEL,
        device: str = "cpu",
        quantization: Optional[str] = None,
    ) -> None:
        """
        Initialize the Timekeeper with its sub-components.

        Args:
            model_name: Name of the Chronos model to load.
            device: Device for the model ('cpu' or 'cuda').
            quantization: Quantization mode (e.g. 'int8') passed to Forecaster.
        """
        logger.info(
            f"Initializing ChronosTimekeeper (Model: {model_name}, Device: {device}, Quantization: {quantization})"
        )
        self.extractor = TimelineExtractor()
        self.forecaster = ChronosForecaster(model_name=model_name, device=device, quantization=quantization)
        self.causality = CausalityEngine()

    def extract_from_text(self, text: str, reference_date: datetime) -> List[TemporalEvent]:
        """
        Extracts a timeline of events from unstructured text.

        Args:
            text: The unstructured text to process.
            reference_date: The anchor date for relative time calculations.

        Returns:
            A list of resolved TemporalEvents.
        """
        logger.info(f"Agent: Extracting timeline from text (Length: {len(text)} chars)")
        return self.extractor.extract_events(text, reference_date)

    def forecast_series(
        self, history: List[float], prediction_length: int, confidence_level: float = 0.9
    ) -> ForecastResult:
        """
        Generates a forecast for the given time series history.

        Args:
            history: List of historical values.
            prediction_length: Number of steps to forecast.
            confidence_level: Probability for the prediction interval (default 0.9).

        Returns:
            ForecastResult containing predictions and intervals.
        """
        logger.info(f"Agent: Forecasting series (History: {len(history)}, Horizon: {prediction_length})")
        request = ForecastRequest(
            history=history,
            prediction_length=prediction_length,
            confidence_level=confidence_level,
        )
        return self.forecaster.forecast(request)

    def check_compliance(
        self, target: TemporalEvent, reference: TemporalEvent, rule: ValidationRule
    ) -> ComplianceResult:
        """
        Validates a compliance rule between two events.

        Args:
            target: The target event (e.g., Report Submission).
            reference: The reference event (e.g., Adverse Event Occurrence).
            rule: The validation rule to apply (e.g., MaxDelayRule).

        Returns:
            ComplianceResult indicating pass/fail and drift.
        """
        logger.info(
            f"Agent: Checking compliance '{rule.__class__.__name__}' "
            f"between '{target.description}' and '{reference.description}'"
        )
        return rule.validate(target.timestamp, reference.timestamp)

    def analyze_causality(self, cause: TemporalEvent, effect: TemporalEvent) -> bool:
        """
        Determines if a causal relationship is temporally plausible.

        Args:
            cause: The potential cause event.
            effect: The potential effect event.

        Returns:
            True if the cause plausibly precedes or overlaps the effect, False otherwise.
        """
        logger.info(f"Agent: Analyzing causality between '{cause.description}' and '{effect.description}'")
        return self.causality.is_plausible_cause(cause, effect)
