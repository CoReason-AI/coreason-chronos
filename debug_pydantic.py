from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, ConfigDict

class TestModel(BaseModel):
    dt: datetime
    td: timedelta

m = TestModel(
    dt=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    td=timedelta(days=1, hours=2)
)

print(f"JSON Mode Dump: {m.model_dump(mode='json')}")
print(f"JSON Dump: {m.model_dump_json()}")
