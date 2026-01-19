import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from coreason_chronos.main import cli


def test_extract_command_text() -> None:
    runner = CliRunner()
    text = "Start on Jan 1st 2024. Event 2 days later."
    # Pass input_text as argument
    result = runner.invoke(cli, ["extract", text])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) >= 2


def test_extract_command_file() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("story.txt", "w") as f:
            f.write("Start on Jan 1st 2024.")

        result = runner.invoke(cli, ["extract", "--file", "story.txt"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert "Jan 1st" in data[0]["source_snippet"]


def test_extract_missing_args() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["extract"])
    # Should fail because input_text is optional but logic enforces one or the other
    assert result.exit_code != 0
    assert "Error: Must provide INPUT_TEXT argument or --file option" in result.output


def test_extract_invalid_ref_date() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["extract", "Hello", "--ref-date", "NotADate"])
    assert result.exit_code != 0
    assert "Error: Could not parse reference date" in result.output


def test_extract_naive_ref_date() -> None:
    # default is aware (now UTC).
    # providing a naive string e.g. "2024-01-01"
    # dateparser usually handles TZ if configured, but here we just check coverage
    runner = CliRunner()
    result = runner.invoke(cli, ["extract", "Hello", "--ref-date", "2024-01-01 10:00"])
    assert result.exit_code == 0
    # It parses as naive (if no settings) then we force UTC.
    # The code: if parsed_date.tzinfo is None: ...
    # This covers that branch.


def test_extract_aware_ref_date() -> None:
    # Explicit TZ in string
    runner = CliRunner()
    result = runner.invoke(cli, ["extract", "Hello", "--ref-date", "2024-01-01 10:00+05:00"])
    assert result.exit_code == 0
    # Code: else: parsed_date.astimezone(timezone.utc)
    # Covers else branch.


def test_forecast_command() -> None:
    # Mock the heavy ChronosForecaster
    with patch("coreason_chronos.agent.ChronosForecaster") as MockForecaster:
        mock_instance = MockForecaster.return_value
        mock_instance.forecast.return_value = MagicMock(
            median=[110.0, 112.0],
            lower_bound=[100.0, 102.0],
            upper_bound=[120.0, 122.0],
            confidence_level=0.9,
            model_dump=lambda: {
                "median": [110.0, 112.0],
                "lower_bound": [100.0, 102.0],
                "upper_bound": [120.0, 122.0],
                "confidence_level": 0.9,
            },
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["forecast", "10,20,30", "--steps", "2"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["median"] == [110.0, 112.0]


def test_validate_command() -> None:
    runner = CliRunner()
    # 2024-01-01 10:00 vs 2024-01-01 12:00. Delay 2 hours. Max 3 hours. -> Compliant.
    result = runner.invoke(cli, ["validate", "2024-01-01T12:00:00", "2024-01-01T10:00:00", "--max-delay-hours", "3"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["is_compliant"] is True

    # Fail case
    result_fail = runner.invoke(
        cli, ["validate", "2024-01-01T14:00:00", "2024-01-01T10:00:00", "--max-delay-hours", "3"]
    )
    assert result_fail.exit_code == 0
    data_fail = json.loads(result_fail.output)
    assert data_fail["is_compliant"] is False


def test_validate_invalid_dates() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "BadDate", "2024-01-01", "-h", "3"])
    assert result.exit_code != 0
    assert "Error: Could not parse dates" in result.output


def test_extract_file_not_found() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["extract", "--file", "nonexistent.txt"])
    assert result.exit_code != 0


def test_forecast_invalid_input() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["forecast", "10,foo,30"])
    assert result.exit_code == 1
    assert "Error: History must be a comma-separated list" in result.output
