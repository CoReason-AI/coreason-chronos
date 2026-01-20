# coreason-chronos

Domain: Temporal Reasoning, Time-Series Forecasting, & Longitudinal Reconstruction

[![License](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason_chronos)
[![CI](https://github.com/CoReason-AI/coreason_chronos/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_chronos/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/CoReason-AI/coreason_chronos)

## Overview

**coreason-chronos** is the specialized temporal processing unit of the CoReason ecosystem. It enables agents to reason about **When**, **How Long**, and **What's Next**.

Core Philosophy: *"Semantic time is fuzzy. Symbolic time is exact. We need both."*

It provides three critical capabilities:

1.  **Longitudinal Reconstruction:** Extracting events from unstructured text (e.g., "Patient started Taxol 3 weeks after surgery") and mapping them to a normalized absolute timeline.
2.  **Probabilistic Forecasting:** Predicting future events (e.g., "Clinical Trial Enrollment will hit 100% in 4.5 months +/- 2 weeks") using SOTA Time-Series Foundation Models.
3.  **Temporal Logic Validation:** Ensuring GxP compliance by validating temporal constraints (e.g., "Was the adverse event reported within the 24-hour statutory window?").

## Features

-   **The Timeline Extractor (The Historian):**
    -   Converts relative dates ("2 weeks later") to absolute timestamps.
    -   Resolves "anchored" events based on semantic proximity to reference events.
    -   Outputs structured `EventSeries` JSON.

-   **The Oracle (The Forecaster):**
    -   Leverages Foundation Time-Series Models (Amazon Chronos-T5).
    -   Zero-shot prediction capability for metrics like Patient Enrollment or Drug Inventory.
    -   Provides probabilistic forecasts (P90 confidence intervals).

-   **The Compliance Clock (The Validator):**
    -   Symbolic Rule Engine for regulatory checks.
    -   Validates constraints like `Report_Time <= Event_Time + 24_Hours`.

-   **The Causality Engine (The Sequencer):**
    -   Uses **Allen's Interval Algebra** to determine temporal plausibility of causal relationships.

## Installation

```bash
pip install coreason-chronos
```

## Usage

### 1. Initialize the Timekeeper

The `ChronosTimekeeper` is the main entry point for the library.

```python
from datetime import datetime, timezone
from coreason_chronos.agent import ChronosTimekeeper

# Initialize the agent
agent = ChronosTimekeeper()
```

### 2. Longitudinal Reconstruction (Timeline Extraction)

```python
text = "Patient was admitted on 2024-01-01. Symptoms started 2 days later."
reference_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

events = agent.extract_from_text(text, reference_date)

for event in events:
    print(f"{event.timestamp}: {event.description}")
# Output:
# 2024-01-01 00:00:00+00:00: Patient was admitted...
# 2024-01-03 00:00:00+00:00: Derived from anchor...
```

### 3. Forecasting

```python
history = [10, 15, 20, 25, 30]  # Weekly enrollment
forecast = agent.forecast_series(history, prediction_length=5)

print(f"Median Forecast: {forecast.median}")
print(f"90% Confidence Interval: {forecast.lower_bound} - {forecast.upper_bound}")
```

### 4. Compliance Check

```python
from datetime import timedelta
from coreason_chronos.validator import MaxDelayRule

# Check if reporting was done within 24 hours
rule = MaxDelayRule(max_delay=timedelta(hours=24))

# ... (assuming target_event and ref_event are TemporalEvent objects)
result = agent.check_compliance(target_event, ref_event, rule)

if not result.is_compliant:
    print(f"Compliance Violation! Drift: {result.drift}")
```
