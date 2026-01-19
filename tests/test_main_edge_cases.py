from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from coreason_chronos.main import cli


def test_forecast_invalid_quantization() -> None:
    """
    Test that invalid quantization mode is handled gracefully.
    """
    with patch("coreason_chronos.agent.ChronosForecaster") as MockForecaster:
        # Mock initialization to raise ValueError
        MockForecaster.side_effect = ValueError("Unsupported quantization mode: invalid")

        runner = CliRunner()
        result = runner.invoke(cli, ["forecast", "10,20", "--quantization", "invalid"])

        assert result.exit_code != 0
        assert "Error: Unsupported quantization mode: invalid" in result.output


def test_forecast_plot_file_error() -> None:
    """
    Test that file write errors during plotting are handled gracefully.
    """
    with patch("coreason_chronos.agent.ChronosForecaster") as MockForecaster:
        mock_instance = MockForecaster.return_value
        mock_instance.forecast.return_value = MagicMock(
            median=[110.0],
            lower_bound=[100.0],
            upper_bound=[120.0],
            confidence_level=0.9,
            model_dump=lambda **kwargs: {},
        )

        with patch("coreason_chronos.visualizer.plot_forecast") as mock_plot:
            mock_fig = MagicMock()
            mock_plot.return_value = mock_fig
            # Simulate OSError when saving (e.g. permission denied or directory not found)
            mock_fig.savefig.side_effect = OSError("Permission denied")

            with patch("matplotlib.pyplot.close"):
                runner = CliRunner()
                # Providing a path that would cause error if not mocked, but mock handles it
                result = runner.invoke(cli, ["forecast", "10,20", "--plot-output", "/invalid/path.png"])

                # Should fail gracefully
                assert result.exit_code != 0
                assert "Error: Failed to save plot" in result.output
                assert "Permission denied" in result.output


def test_forecast_invalid_steps() -> None:
    """
    Test that invalid prediction length (steps) is handled gracefully.
    """
    # We patch ChronosForecaster so Agent initialization succeeds.
    with patch("coreason_chronos.agent.ChronosForecaster"):
        runner = CliRunner()
        result = runner.invoke(cli, ["forecast", "10,20", "--steps", "-5"])

        # It should exit with non-zero
        assert result.exit_code != 0
        # The output should contain the pydantic error details
        assert "prediction_length must be positive" in result.output


def test_forecast_complex_success() -> None:
    """
    Complex scenario: All options enabled.
    """
    with patch("coreason_chronos.agent.ChronosForecaster") as MockForecaster:
        mock_instance = MockForecaster.return_value
        mock_instance.forecast.return_value = MagicMock(
            median=[110.0, 112.0],
            lower_bound=[100.0, 102.0],
            upper_bound=[120.0, 122.0],
            confidence_level=0.95,
            model_dump=lambda **kwargs: {"median": [110.0, 112.0], "confidence_level": 0.95},
        )

        with patch("coreason_chronos.visualizer.plot_forecast") as mock_plot:
            mock_fig = MagicMock()
            mock_plot.return_value = mock_fig

            with patch("matplotlib.pyplot.close") as mock_close:
                runner = CliRunner()
                with runner.isolated_filesystem():
                    # Invoke with all options
                    result = runner.invoke(
                        cli,
                        [
                            "forecast",
                            "10,20,30,40,50",
                            "--steps",
                            "2",
                            "--confidence",
                            "0.95",
                            "--quantization",
                            "int8",
                            "--model",
                            "test/model",
                            "--plot-output",
                            "out.png",
                        ],
                    )

                    assert result.exit_code == 0

                    # Verify initialization with all params
                    MockForecaster.assert_called_with(model_name="test/model", device="cpu", quantization="int8")

                    # Verify forecast call
                    args, _ = mock_instance.forecast.call_args
                    req = args[0]
                    assert req.history == [10.0, 20.0, 30.0, 40.0, 50.0]
                    assert req.prediction_length == 2
                    assert req.confidence_level == 0.95

                    # Verify plot saved
                    mock_fig.savefig.assert_called_once_with("out.png")
                    mock_close.assert_called_once()
