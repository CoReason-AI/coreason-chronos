# The Architecture and Utility of coreason-chronos

### 1. The Philosophy (The Why)

In the rigorous domains of GxP (Good Practice) compliance and clinical pharmacology, time is rarely simple. It is a duality: **Semantic Time** ("Patient experienced nausea 3 days after the second infusion") versus **Symbolic Time** ("2024-06-12T14:30:00Z").

Standard software libraries excel at the latter but fail spectacularly at the former. Traditional regex-based parsers cannot reliably resolve "3 days later" because they lack the context of *what* happened 3 days earlier. Furthermore, in the realm of forecasting, the legacy approach of training individual ARIMA or Prophet models for every single metric (e.g., enrollment rates for 50 different clinical sites) is operationally unscalable.

**coreason-chronos** exists to bridge this gap. It is the specialized temporal processing unit of the CoReason ecosystem, designed to reason about **When**, **How Long**, and **What's Next**.

Its architecture is driven by three core insights:
1.  **Context is King:** Resolving relative time requires a directed acyclic graph (DAG) of events, linking anchors (causes) to targets (effects) through semantic fuzzy matching.
2.  **Zero-Shot is Essential:** By leveraging Foundation Time-Series Models (specifically Amazon's Chronos), we can forecast complex logistical and clinical trends without the fragility of custom model training.
3.  **Strict Compliance:** While the inputs (text) are fuzzy, the outputs (validations) must be exact. The system uses strict symbolic algebra for date math to ensure audit-ready compliance checks.

### 2. Under the Hood (The Dependencies & Logic)

The package's "weight" and capabilities are defined by a carefully selected stack that balances deep learning power with symbolic precision:

*   **`transformers` & `torch`**: These provide the backbone for the **Oracle** (the Forecaster). By wrapping the `amazon/chronos-t5` family of models, the package transforms time-series data into a token sequence, allowing it to "generate" future values just as an LLM generates text.
*   **`rapidfuzz`**: This powers the **Timeline Extractor's** ability to semantically link events. It allows the system to understand that "after the surgery" refers to the "Appendectomy" event mentioned paragraphs earlier, effectively solving the "Anchor Resolution" problem.
*   **`dateparser` & `dateutil`**: While neural models handle the complex logic, these libraries provide the bedrock for parsing standard date formats and performing accurate calendar arithmetic (handling leap years, varying month lengths, etc.).
*   **`pydantic`**: Acts as the strict contract layer, ensuring that all data flowing in and out—whether extracted events or forecast requests—adheres to a rigorous schema (ISO 8601 compliance, strict typing).

**The Logic Flow:**
The primary orchestration happens in the `ChronosTimekeeper` class. For timeline extraction, it employs a multi-pass strategy:
1.  **Scan:** Identify absolute dates and simple relative terms (e.g., "last Friday").
2.  **Anchor:** Detect complex relative durations (e.g., "2 weeks after...").
3.  **Resolve:** Use fuzzy logic to link these anchored durations to their parent events, iteratively filling in the timeline until all resolvable dates are grounded in absolute time.

### 3. In Practice (The How)

The following examples demonstrate the `ChronosTimekeeper` acting as the central facade for the library's capabilities.

#### A. Longitudinal Reconstruction
Extracting a structured timeline from a messy clinical note.

```python
from datetime import datetime, timezone
from coreason_chronos.agent import ChronosTimekeeper

# The "Timekeeper" orchestrates the extraction logic
agent = ChronosTimekeeper()

narrative = """
The patient was admitted on Jan 5th, 2024.
Observations were stable initially.
Severe nausea began 2 days after admission.
"""

# We ground all relative dates to a known reference point (usually the doc date)
ref_date = datetime(2024, 1, 10, tzinfo=timezone.utc)

events = agent.extract_from_text(narrative, reference_date=ref_date)

for event in events:
    print(f"{event.timestamp.date()}: {event.description}")

# Output:
# 2024-01-05: The patient was admitted...
# 2024-01-07: Derived from anchor '2 days after admission'...
```

#### B. Zero-Shot Forecasting
Predicting future trends without training a model.

```python
from coreason_chronos.agent import ChronosTimekeeper

# Initialize with a specific Foundation Model
agent = ChronosTimekeeper(model_name="amazon/chronos-t5-tiny", device="cpu")

# Historical data (e.g., weekly enrollment numbers)
history = [10.0, 12.0, 15.0, 18.0, 22.0, 25.0, 30.0]

# Forecast the next 3 weeks with 90% confidence
result = agent.forecast_series(history, prediction_length=3, confidence_level=0.9)

print(f"Median Forecast: {result.median}")
print(f"90% Confidence Interval: {result.lower_bound} - {result.upper_bound}")

# Output will contain the predicted values for the next 3 steps,
# derived from the model's understanding of general time-series patterns.
```
