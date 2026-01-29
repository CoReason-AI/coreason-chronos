# Prosperity Public License 3.0
from datetime import datetime
from functools import partial
from typing import Any, List, Optional, cast

import anyio
import httpx
from coreason_identity.models import UserContext

from coreason_chronos.causality import CausalityEngine
from coreason_chronos.forecaster import DEFAULT_CHRONOS_MODEL, ChronosForecaster
from coreason_chronos.schemas import ComplianceResult, ForecastRequest, ForecastResult, TemporalEvent
from coreason_chronos.timeline_extractor import TimelineExtractor
from coreason_chronos.utils.logger import logger
from coreason_chronos.validator import ValidationRule


class ChronosTimekeeperAsync:
    """
    The Timekeeper (Async): Orchestrates the Extract-Align-Forecast loop.

    This agent serves as the central facade for the library, integrating:
    - Timeline Extraction (The Historian)
    - Forecasting (The Oracle)
    - Causal Analysis (The Sequencer)
    - Compliance Checking (The Compliance Clock)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CHRONOS_MODEL,
        device: str = "cpu",
        quantization: Optional[str] = None,
        extractor: Optional[TimelineExtractor] = None,
        forecaster: Optional[ChronosForecaster] = None,
        causality: Optional[CausalityEngine] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """
        Initialize the Timekeeper with its sub-components.

        Args:
            model_name: Name of the Chronos model to load (if forecaster is not injected).
            device: Device for the model ('cpu' or 'cuda').
            quantization: Quantization mode (e.g. 'int8') passed to Forecaster.
            extractor: Optional injected TimelineExtractor instance.
            forecaster: Optional injected ChronosForecaster instance.
            causality: Optional injected CausalityEngine instance.
            client: Optional injected httpx.AsyncClient for network operations.
        """
        logger.info(
            f"Initializing ChronosTimekeeperAsync (Model: {model_name}, Device: {device}, Quantization: {quantization})"
        )
        self.extractor = extractor or TimelineExtractor()
        self.forecaster = forecaster or ChronosForecaster(
            model_name=model_name, device=device, quantization=quantization
        )
        self.causality = causality or CausalityEngine()

        # Resource management
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

    async def __aenter__(self) -> "ChronosTimekeeperAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._client.aclose()

    async def extract_from_text(
        self, text: str, reference_date: datetime, *, context: UserContext
    ) -> List[TemporalEvent]:
        """
        Extracts a timeline of events from unstructured text (Longitudinal Reconstruction).

        Args:
            text: The unstructured text to process.
            reference_date: The anchor date for relative time calculations.
            context: The user context for identity verification.

        Returns:
            A list of resolved TemporalEvents, sorted chronologically.
        """
        logger.info(
            f"Agent: Extracting timeline from text (Length: {len(text)} chars)",
            user_id=context.user_id,
        )
        result = await anyio.to_thread.run_sync(self.extractor.extract_events, text, reference_date)
        return result

    async def forecast_series(
        self,
        history: List[float],
        prediction_length: int,
        confidence_level: float = 0.9,
        *,
        context: UserContext,
    ) -> ForecastResult:
        """
        Generates a forecast for the given time series history (Zero-Shot Prediction).

        Args:
            history: List of historical values.
            prediction_length: Number of steps to forecast.
            confidence_level: Probability for the prediction interval (default 0.9).
            context: The user context for identity verification.

        Returns:
            ForecastResult containing predictions and intervals.
        """
        logger.info(
            f"Agent: Forecasting series (History: {len(history)}, Horizon: {prediction_length})",
            user_id=context.user_id,
        )
        request = ForecastRequest(
            history=history,
            prediction_length=prediction_length,
            confidence_level=confidence_level,
        )
        result = await anyio.to_thread.run_sync(self.forecaster.forecast, request)
        return result

    async def check_compliance(
        self, target: TemporalEvent, reference: TemporalEvent, rule: ValidationRule, *, context: UserContext
    ) -> ComplianceResult:
        """
        Validates a compliance rule between two events.

        Args:
            target: The target event (e.g., Report Submission).
            reference: The reference event (e.g., Adverse Event Occurrence).
            rule: The validation rule to apply (e.g., MaxDelayRule).
            context: The user context for identity verification.

        Returns:
            ComplianceResult indicating pass/fail and drift.
        """
        logger.info(
            f"Agent: Checking compliance '{rule.__class__.__name__}' "
            f"between '{target.description}' and '{reference.description}'",
            user_id=context.user_id,
        )
        return rule.validate(target.timestamp, reference.timestamp)

    async def analyze_causality(self, cause: TemporalEvent, effect: TemporalEvent, *, context: UserContext) -> bool:
        """
        Determines if a causal relationship is temporally plausible.

        Args:
            cause: The potential cause event.
            effect: The potential effect event.
            context: The user context for identity verification.

        Returns:
            True if the cause plausibly precedes or overlaps the effect, False otherwise.
        """
        logger.info(
            f"Agent: Analyzing causality between '{cause.description}' and '{effect.description}'",
            user_id=context.user_id,
        )
        return self.causality.is_plausible_cause(cause, effect)


class ChronosTimekeeper:
    """
    The Timekeeper (Sync Facade): Wraps ChronosTimekeeperAsync for synchronous usage.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CHRONOS_MODEL,
        device: str = "cpu",
        quantization: Optional[str] = None,
        extractor: Optional[TimelineExtractor] = None,
        forecaster: Optional[ChronosForecaster] = None,
        causality: Optional[CausalityEngine] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._async = ChronosTimekeeperAsync(
            model_name=model_name,
            device=device,
            quantization=quantization,
            extractor=extractor,
            forecaster=forecaster,
            causality=causality,
            client=client,
        )

    def __enter__(self) -> "ChronosTimekeeper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        anyio.run(self._async.__aexit__, exc_type, exc_val, exc_tb)

    def extract_from_text(self, text: str, reference_date: datetime, *, context: UserContext) -> List[TemporalEvent]:
        """Synchronous wrapper for extract_from_text."""
        return anyio.run(partial(self._async.extract_from_text, text, reference_date, context=context))

    def forecast_series(
        self, history: List[float], prediction_length: int, confidence_level: float = 0.9, *, context: UserContext
    ) -> ForecastResult:
        """Synchronous wrapper for forecast_series."""
        return anyio.run(
            partial(
                self._async.forecast_series,
                history,
                prediction_length,
                confidence_level,
                context=context,
            )
        )

    def check_compliance(
        self, target: TemporalEvent, reference: TemporalEvent, rule: ValidationRule, *, context: UserContext
    ) -> ComplianceResult:
        """Synchronous wrapper for check_compliance."""
        return anyio.run(partial(self._async.check_compliance, target, reference, rule, context=context))

    def analyze_causality(self, cause: TemporalEvent, effect: TemporalEvent, *, context: UserContext) -> bool:
        """Synchronous wrapper for analyze_causality."""
        return anyio.run(partial(self._async.analyze_causality, cause, effect, context=context))

    # Expose underlying components for compatibility if needed, or deprecate direct access.
    # The existing tests access agent.forecaster directly.
    # We should expose properties that delegate to self._async.
    @property
    def forecaster(self) -> ChronosForecaster:
        return self._async.forecaster

    @property
    def extractor(self) -> TimelineExtractor:
        return self._async.extractor

    @property
    def causality(self) -> CausalityEngine:
        return self._async.causality
