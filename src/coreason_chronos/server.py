from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from coreason_chronos.agent import ChronosTimekeeperAsync
from coreason_chronos.schemas import ForecastRequest, ForecastResult
from coreason_chronos.forecaster import DEFAULT_CHRONOS_MODEL
from coreason_identity.models import UserContext

# Define request model for extraction
class ExtractionRequest(BaseModel):
    text: str
    ref_date: Optional[datetime] = None

# Default context for API operations
API_USER_CONTEXT = UserContext(
    user_id="api-user",
    email="api@coreason.ai",
    groups=[],
    scopes=[],
    claims={}
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the timekeeper on startup
    # Using default model (amazon/chronos-t5-tiny) and cpu device
    timekeeper = ChronosTimekeeperAsync(
        model_name=DEFAULT_CHRONOS_MODEL,
        device="cpu",
    )

    # Enter async context (this initializes client and potentially other resources)
    await timekeeper.__aenter__()

    app.state.timekeeper = timekeeper
    yield

    # Cleanup
    await timekeeper.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan, title="Temporal Intelligence Microservice")

@app.post("/extract")
async def extract_endpoint(request: ExtractionRequest) -> List[Any]:
    timekeeper: ChronosTimekeeperAsync = app.state.timekeeper

    # Default ref_date to now(UTC) if not provided, ensuring timezone awareness
    ref_date = request.ref_date or datetime.now(timezone.utc)

    try:
        events = await timekeeper.extract_from_text(
            text=request.text,
            reference_date=ref_date,
            context=API_USER_CONTEXT
        )
        # Return serialized events as requested
        return [event.model_dump(mode="json") for event in events]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastResult)
async def forecast_endpoint(request: ForecastRequest):
    timekeeper: ChronosTimekeeperAsync = app.state.timekeeper

    try:
        result = await timekeeper.forecast_series(
            history=request.history,
            prediction_length=request.prediction_length,
            confidence_level=request.confidence_level,
            context=API_USER_CONTEXT
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "ready",
        "model": DEFAULT_CHRONOS_MODEL,
        "device": "cpu"
    }
