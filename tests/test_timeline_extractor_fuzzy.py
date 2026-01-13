from datetime import datetime, timezone
from coreason_chronos.timeline_extractor import TimelineExtractor

def test_story_a_fuzzy_match() -> None:
    """
    Test Story A: "Patient felt nausea 2 days after the second infusion."
    Event: "Second Infusion Date: 2024-06-10"

    The anchor "the second infusion" does not appear verbatim in the reference event text.
    It requires matching "the second infusion" to "Second Infusion Date".
    """
    text = "Clinical Notes. Second Infusion Date: 2024-06-10. Patient felt nausea 2 days after the second infusion."
    ref_date = datetime(2024, 6, 20, tzinfo=timezone.utc) # Ref date after events

    extractor = TimelineExtractor()
    events = extractor.extract_events(text, ref_date)

    # We expect 2 events:
    # 1. 2024-06-10 (The infusion)
    # 2. 2024-06-12 (The nausea, 2 days after)

    assert len(events) >= 2

    # Sort by time
    events.sort(key=lambda x: x.timestamp)

    infusion = events[0]
    nausea = events[1]

    assert infusion.timestamp == datetime(2024, 6, 10, tzinfo=timezone.utc)
    assert nausea.timestamp == datetime(2024, 6, 12, tzinfo=timezone.utc)
    assert "nausea" in nausea.description or "nausea" in nausea.source_snippet or "derived" in nausea.description.lower()

def test_fuzzy_match_anchor_before_event() -> None:
    """
    Test where the anchor appears before the event definition in the text.
    "Symptoms started 2 days before surgery. Surgery was on Jan 5."
    """
    text = "Symptoms started 2 days before surgery. Surgery was on Jan 5."
    # Use ref_date after Jan 5 so Jan 5 is considered 'past' (2024)
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)

    extractor = TimelineExtractor()
    events = extractor.extract_events(text, ref_date)

    assert len(events) >= 2
    events.sort(key=lambda x: x.timestamp)

    symptoms = events[0]
    surgery = events[1]

    assert surgery.timestamp == datetime(2024, 1, 5, tzinfo=timezone.utc)
    assert symptoms.timestamp == datetime(2024, 1, 3, tzinfo=timezone.utc)
