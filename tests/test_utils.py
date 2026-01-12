# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_chronos

import shutil
from pathlib import Path

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
    # We can't easily re-import the module to trigger the code,
    # but we can verify the logic by simulating the condition if we extracted the setup function.
    # However, since it's global scope code, we can just ensure the directory exists.

    # To hit the coverage line `log_path.mkdir`, we would need to delete the directory
    # and re-import or reload the module.

    # Let's try reloading the module after deleting the directory.
    import importlib

    import coreason_chronos.utils.logger

    log_path = Path("logs")
    if log_path.exists():
        shutil.rmtree(log_path)

    assert not log_path.exists()

    importlib.reload(coreason_chronos.utils.logger)

    assert log_path.exists()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None
