# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_chronos

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from coreason_chronos.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    # Since the logger is initialized on import, we check side effects

    # Check if logs directory creation is handled
    # Note: running this test might actually create the directory in the test environment
    # if it doesn't exist.

    log_path = Path("logs")
    assert log_path.exists()
    assert log_path.is_dir()


def test_logger_directory_creation() -> None:
    """Test that the logger creation logic handles directory creation."""
    # To avoid PermissionError on Windows (locked log file), we use mocks
    # to simulate the directory not existing and verify mkdir is called.

    import coreason_chronos.utils.logger

    # We patch pathlib.Path because reload will re-import it.
    with patch("pathlib.Path") as mock_path_cls:
        # Setup the mock to simulate "logs" directory check
        mock_path_instance = MagicMock()

        # When Path("logs") is called, return our mock instance
        # We need to ensure we only intercept the specific call we care about if possible,
        # or just make sure our mock behaves like a path for everything else.
        # But for this simple test, returning a mock is fine.
        mock_path_cls.return_value = mock_path_instance

        # Simulate that the directory does NOT exist
        mock_path_instance.exists.return_value = False

        # Reload the module to trigger the top-level code
        importlib.reload(coreason_chronos.utils.logger)

        # Verify logic
        # Check if Path("logs") was called
        mock_path_cls.assert_any_call("logs")

        # Check if mkdir was called on the instance returned by Path("logs")
        mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

    # Reload the real logger to restore state for other tests and ensure side effects (real dir creation) happen
    importlib.reload(coreason_chronos.utils.logger)


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None
