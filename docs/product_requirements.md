# Product Requirements Document: coreason-chronos

Domain: Temporal Reasoning, Time-Series Forecasting, & Longitudinal Reconstruction
Architectural Role: The "Timekeeper" / The Forecaster
Core Philosophy: "Semantic time is fuzzy. Symbolic time is exact. We need both."
Dependencies: coreason-refinery (Data Source), coreason-graph-nexus (Event Linking), gluonts / autogluon (Forecasting Models)

## ---

**1. Executive Summary**

coreason-chronos is the specialized temporal processing unit of the CoReason ecosystem. It enables agents to reason about **When**, **How Long**, and **What's Next**.

It provides three critical capabilities:

1. **Longitudinal Reconstruction:** Extracting events from unstructured text (e.g., "Patient started Taxol 3 weeks after surgery") and mapping them to a normalized absolute timeline.
2. **Probabilistic Forecasting:** Predicting future events (e.g., "Clinical Trial Enrollment will hit 100% in 4.5 months +/- 2 weeks") using SOTA Time-Series Foundation Models.
3. **Temporal Logic Validation:** Ensuring GxP compliance by validating temporal constraints (e.g., "Was the adverse event reported within the 24-hour statutory window?").

## **2. Functional Philosophy**

The agent must implement the **Extract-Align-Forecast Loop**:

1. **Relative-to-Absolute Grounding:** LLMs see "Next Tuesday." Chronos converts this to 2025-10-14T00:00:00Z based on the document's metadata date. It uses symbolic algebra, not token prediction, for date math.
2. **Zero-Shot Forecasting (SOTA):** We leverage **Foundation Time-Series Models** (like Amazon's Chronos or Salesforce's MOIRAI). Unlike traditional ARIMA models that need retraining for every specific metric, these models can zero-shot predict "Patient Enrollment" or "Drug Inventory" based on universal patterns learned from billions of data points.
3. **The Patient Journey (The Timeline):** In Pharma, the sequence matters. Did the Liver Failure happen *before* or *after* the drug administration? Chronos builds a **Directed Acyclic Graph (DAG)** of events to prove causality.

## ---

**3. Core Functional Requirements (Component Level)**

### **3.1 The Timeline Extractor (The Historian)**

**Concept:** Turns text into a timeline.

* **Mechanism:**
  * **NER for Dates:** Identifies absolute dates ("Jan 1st"), relative dates ("2 weeks later"), and durations ("for 3 months").
  * **Anchor Resolution:** "3 days after admission" -> Finds "Admission Date" -> Adds 3 days.
* **Output:** A structured EventSeries JSON.
  * [{event: "Dose 1", timestamp: "2025-01-01"}, {event: "Adverse Effect", timestamp: "2025-01-03"}]

### **3.2 The Oracle (The Forecaster)**

**Concept:** Predicts the future using SOTA Foundation Models.

* **Model:** **Chronos-T5 (Small/Base)**. This transforms time series data into a token sequence and uses a Transformer to predict the next "tokens" (values).
* **Action:**
  * Input: [10, 15, 20, 25] (Weekly Enrollment).
  * Task: "Forecast next 12 weeks."
  * Output: [30, 35, ...] with P90 confidence intervals.
* **Value:** Allows the coreason-economist to predict budget burn rates or supply shortages without training custom ML models.

### **3.3 The Compliance Clock (The Validator)**

**Concept:** Validates regulatory time constraints.

* **Logic:** Symbolic Rule Engine.
  * *Rule:* Safety_Report_Window = Event_Time + 24_Hours.
  * *Check:* Report_Time <= Safety_Report_Window.
* **Output:** Boolean (Compliant/Non-Compliant) + Drift ("Reported 2 hours late").

### **3.4 The Causality Engine (The Sequencer)**

**Concept:** Determines "Before/After" relationships for safety signals.

* **Algorithm:** Uses **Allen's Interval Algebra**.
  * Relationships: X before Y, X overlaps Y, X during Y.
* **Use Case:** Determining if a drug *could* have caused a side effect (Temporal Plausibility).

## ---

**4. Integration Requirements**

* **coreason-refinery:**
  * Feeds unstructured text to Chronos for Timeline Extraction.
* **coreason-graph-nexus:**
  * Chronos populates the (Event)-[:HAPPENED_AT]->(Time) nodes in the knowledge graph.
  * Enables queries like: "Show all patients who had Event A followed by Event B within 14 days."
* **coreason-connect:**
  * Used to trigger external scheduling actions based on forecasts (e.g., "Forecast predicts stockout in 2 weeks" -> connect.sap.order_supply()).

## ---

**5. User Stories**

### **Story A: The "Patient Journey" (Timeline Extraction)**

Context: A Clinical Scientist is reviewing a PDF Case Report Form (CRF) with messy notes. "Patient felt nausea 2 days after the second infusion."
Action: Chronos scans the text.
Logic:

1. Finds "Second Infusion Date" = 2024-06-10.
2. Calculates 2024-06-10 + 2 days = 2024-06-12.
   Result: Creates a structured timeline entry: Event: Nausea | Date: 2024-06-12 | Source: Page 4.

### **Story B: The "Enrollment Forecast" (Zero-Shot Prediction)**

Context: Trial Manager asks: "When will we reach 500 patients?"
Input: Historical enrollment data for the last 3 months (vector of integers).
Chronos: Feeds vector to Chronos-T5 Model.
Result: "Based on current acceleration, you will hit 500 patients on Oct 14th (± 5 days)."
Visual: Generates a plot showing the cone of uncertainty.

### **Story C: The "Compliance Check" (Safety)**

Context: An Adverse Event (SAE) occurred on Friday 23:00. It was reported to the FDA on Monday 09:00.
Rule: "Must report within 24 calendar hours."
Chronos:

1. Deadline = Friday 23:00 + 24h = Saturday 23:00.
2. Report Time = Monday 09:00.
3. Monday > Saturday.
   Result: Flags "Compliance Violation: Reporting delayed by 34 hours."

## ---

**6. Data Schema**

### **TemporalEvent**

```python
class TemporalGranularity(str, Enum):
    PRECISE = "PRECISE"       # "2024-01-01 10:00"
    DATE_ONLY = "DATE_ONLY"   # "2024-01-01"
    FUZZY = "FUZZY"           # "Early January 2024"

class TemporalEvent(BaseModel):
    id: UUID
    description: str          # "Headache onset"
    timestamp: datetime
    granularity: TemporalGranularity

    # Allen's Algebra
    duration_minutes: Optional[int]
    ends_at: Optional[datetime]

    source_snippet: str       # "onset 2 hours later"
```

### **ForecastRequest**

```python
class ForecastRequest(BaseModel):
    history: List[float]      # [10, 12, 15, 18...]
    prediction_length: int    # 12 steps
    confidence_level: float   # 0.90 (P90)

    # SOTA: Contextual Covariates
    # e.g., "Is this a holiday?"
    covariates: Optional[List[int]]
```

## ---

**7. Implementation Directives for the Coding Agent**

1. **Library Selection:**
   * **Forecasting:** Use **autogluon.timeseries** or HuggingFace's **transformers** (for Chronos-T5). These provide high-level abstractions over the complex models.
   * **Parsing:** Use **dateparser** for robust relative date parsing ("3 weeks ago").
2. **Model Quantization:** Time-series foundation models can be large. Use quantized versions (INT8) where possible to keep inference latency low.
3. **Strict Math:** Never use an LLM to calculate "Date A + 15 days." Use Python's datetime and relativedelta. LLMs are bad at arithmetic; Python is perfect at it.
4. **Serialization:** Store all dates in **ISO 8601 UTC** format internally (YYYY-MM-DDTHH:MM:SSZ). Only convert to local time for the User Interface.

## **Final Prompt Instructions**

### **Mandatory Code Quality & CI Rules**

You **must** strictly follow this workflow before producing the final answer:

1. **After all code changes are made, run:**

    `ruff format .`

`ruff check --fix .`

       **2. Then run full pre-commit validation:**

	`pre-commit run --all-files`

    `3.` **If ANY files are modified by these commands:**

* You **must** stage and commit those changes.

  * Then repeat step 2.

**4. The task is NOT complete until:**

	`pre-commit run --all-files`

 finishes with:

* **No file modifications**

  * **No hook failures**

  * **No mypy errors**

  5. The final branch **must** pass all pre-commit hooks without making changes.
