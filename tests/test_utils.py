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

import coreason_chronos.utils.logger as logger_module
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


def test_logger_mkdir_logic() -> None:
    """
    Test that the logger module attempts to create the directory if it doesn't exist.
    We patch pathlib.Path because the module imports it.
    """
    with patch("pathlib.Path") as mock_path_cls:
        # Setup the mock to mimic Path("logs") behavior
        mock_path_instance = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        # When checking for logs/app.log, we want to allow that too, or just focus on the first call
        # The module calls Path("logs") then checks exists().

        # Simulate directory does NOT exist
        mock_path_instance.exists.return_value = False

        # Reload the module to trigger the logic with the mocked Path
        importlib.reload(logger_module)

        # Check if mkdir was called
        # Verify Path("logs") was called
        mock_path_cls.assert_any_call("logs")

        # Verify mkdir called on that instance
        mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

    # Reload again to restore normal state using real Path
    importlib.reload(logger_module)
