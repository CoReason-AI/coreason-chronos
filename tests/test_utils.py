# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_chronos

from pathlib import Path
from unittest.mock import MagicMock, patch

from coreason_chronos.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    # Since the logger is initialized on import, we check side effects

    # Check if logs directory creation is handled
    log_path = Path("logs")
    assert log_path.exists()
    assert log_path.is_dir()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None


def test_logger_dir_creation() -> None:
    """Test that the logs directory is created if it doesn't exist."""
    import importlib

    from coreason_chronos.utils import logger as logger_module

    # We patch pathlib.Path so that when the module imports it (or uses it), it gets our mock.
    # Since we are reloading, the module will re-import Path.
    # If we patch 'pathlib.Path', the re-import will grab the mock.

    with patch("pathlib.Path") as MockPath:
        # The code does: log_path = Path("logs")
        # So we need MockPath("logs") to return our mock instance.

        mock_path_instance = MagicMock()
        MockPath.return_value = mock_path_instance

        # Simulate directory does NOT exist
        mock_path_instance.exists.return_value = False

        # Reload the module to trigger the top-level code again
        importlib.reload(logger_module)

        # Verify mkdir was called
        mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

    # Restore the module to normal state
    importlib.reload(logger_module)
