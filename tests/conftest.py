import pytest
from coreason_identity.models import UserContext

@pytest.fixture
def user_context() -> UserContext:
    return UserContext(
        user_id="test-user",
        email="test@coreason.ai",
        groups=["tester"],
        claims={"source": "test"},
    )
