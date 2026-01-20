# Prosperity Public License 3.0
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import click
from dateparser import parse
from pydantic import ValidationError

from coreason_chronos.agent import ChronosTimekeeper
from coreason_chronos.schemas import TemporalEvent
from coreason_chronos.utils.logger import logger
from coreason_chronos.validator import MaxDelayRule


@click.group()
def cli() -> None:
    """Coreason Chronos: Temporal Reasoning & Forecasting CLI"""
    pass


@cli.command()
@click.argument("input_text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Path to text file containing the narrative.")
@click.option(
    "--ref-date",
    "-d",
    help="Reference date (ISO 8601). Defaults to now (UTC).",
    default=lambda: datetime.now(timezone.utc).isoformat(),
)
def extract(input_text: Optional[str], file: Optional[str], ref_date: str) -> None:
    """
    Extract temporal events from text or file.

    Performs longitudinal reconstruction by parsing absolute and relative dates relative to a reference date.
    """
    if file:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
    elif input_text:
        text = input_text
    else:
        click.echo("Error: Must provide INPUT_TEXT argument or --file option.", err=True)
        sys.exit(1)

    # Parse reference date
    parsed_date = parse(ref_date)
    if not parsed_date:
        click.echo(f"Error: Could not parse reference date '{ref_date}'", err=True)
        sys.exit(1)

    # Ensure timezone awareness
    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
    else:
        parsed_date = parsed_date.astimezone(timezone.utc)

    logger.info(f"Extracting events relative to {parsed_date}")

    with ChronosTimekeeper(device="cpu") as agent:  # CLI defaults to CPU for now
        events = agent.extract_from_text(text, parsed_date)

    # Output JSON
    output = [event.model_dump(mode="json") for event in events]
    click.echo(json.dumps(output, indent=2))


@cli.command()
@click.argument("history_str", metavar="HISTORY")
@click.option("--steps", "-s", default=12, help="Number of steps to forecast.")
@click.option("--confidence", "-c", default=0.9, help="Confidence level (0.0 - 1.0).")
@click.option("--model", "-m", default="amazon/chronos-t5-tiny", help="HuggingFace model ID.")
@click.option("--quantization", "-q", help="Quantization mode (e.g. 'int8').")
@click.option("--plot-output", "-p", type=click.Path(writable=True), help="Path to save the forecast plot.")
def forecast(
    history_str: str,
    steps: int,
    confidence: float,
    model: str,
    quantization: Optional[str],
    plot_output: Optional[str],
) -> None:
    """
    Forecast future values based on history using SOTA foundation models.
    HISTORY should be a comma-separated list of numbers (e.g., "10,20,30").
    """
    try:
        history = [float(x.strip()) for x in history_str.split(",")]
    except ValueError:
        click.echo("Error: History must be a comma-separated list of numbers.", err=True)
        sys.exit(1)

    try:
        with ChronosTimekeeper(model_name=model, device="cpu", quantization=quantization) as agent:
            result = agent.forecast_series(history, steps, confidence)
    except (ValueError, ValidationError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(json.dumps(result.model_dump(mode="json"), indent=2))

    if plot_output:
        try:
            import matplotlib.pyplot as plt

            from coreason_chronos.schemas import ForecastRequest
            from coreason_chronos.visualizer import plot_forecast

            # Reconstruct request object since Agent consumes it internally
            req = ForecastRequest(history=history, prediction_length=steps, confidence_level=confidence)
            fig = plot_forecast(req, result, title="Chronos Forecast")
            fig.savefig(plot_output)
            plt.close(fig)
            logger.info(f"Forecast plot saved to {plot_output}")
        except OSError as e:
            click.echo(f"Error: Failed to save plot. {e}", err=True)
            sys.exit(1)


@cli.command()
@click.argument("target_time")
@click.argument("reference_time")
@click.option("--max-delay-hours", "-h", type=float, required=True, help="Maximum allowed delay in hours.")
def validate(target_time: str, reference_time: str, max_delay_hours: float) -> None:
    """
    Check if Target Time is within Max Delay of Reference Time (GxP Compliance).
    """
    t_time = parse(target_time)
    r_time = parse(reference_time)

    if not t_time or not r_time:
        click.echo("Error: Could not parse dates.", err=True)
        sys.exit(1)

    # Ensure TZ
    if t_time.tzinfo is None:
        t_time = t_time.replace(tzinfo=timezone.utc)
    if r_time.tzinfo is None:
        r_time = r_time.replace(tzinfo=timezone.utc)

    rule = MaxDelayRule(max_delay=timedelta(hours=max_delay_hours))

    # We don't need full agent for this simple check, but consistency suggests usage.
    # However, Agent.check_compliance requires TemporalEvent objects.
    # We can just use the rule directly or mock events.
    # Let's use Rule directly as it's cleaner for CLI.

    # Wait, the task is "expose capabilities of ChronosTimekeeper".
    # Agent.check_compliance takes TemporalEvent.
    # So I should create dummy events.
    from uuid import uuid4

    from coreason_chronos.schemas import TemporalGranularity

    t_event = TemporalEvent(
        id=uuid4(), description="Target", timestamp=t_time, granularity=TemporalGranularity.PRECISE, source_snippet=""
    )
    r_event = TemporalEvent(
        id=uuid4(),
        description="Reference",
        timestamp=r_time,
        granularity=TemporalGranularity.PRECISE,
        source_snippet="",
    )

    with ChronosTimekeeper(device="cpu") as agent:
        result = agent.check_compliance(t_event, r_event, rule)

    click.echo(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    cli()  # pragma: no cover
