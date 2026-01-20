from datetime import datetime, timezone

from coreason_chronos.timeline_extractor import TimelineExtractor


def test_ambiguous_fuzzy_match_prioritizes_score() -> None:
    """
    Test that a high-score match (exact words) is preferred over a low-score match (partial words),
    even if the low-score match is closer in the text.

    Scenario:
    Event A: "Second Infusion" (Target, 100% match, Farther)
    Event B: "Third Infusion" (Distractor, 50% match, Closer)
    Anchor: "the second infusion"
    """
    # Layout: [Second Infusion] ... [Third Infusion] ... [Anchor "after second infusion"]
    # Distance: Anchor -> Third is smaller than Anchor -> Second.
    # Logic must pick Second because "second" matches "second", but "third" does not.

    text = (
        "History: Second Infusion on Jan 10. Current: Third Infusion on Jan 20. "
        "Reaction 2 days after the second infusion."
    )
    # Use ref_date in Feb so Jan dates are treated as current year (2024)
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)

    extractor = TimelineExtractor()
    events = extractor.extract_events(text, ref_date)

    # Events:
    # 1. Jan 10 (Second)
    # 2. Jan 20 (Third)
    # 3. Reaction. Should be Jan 10 + 2 = Jan 12.

    assert len(events) >= 3

    # Identify the reaction event robustly using source_snippet
    reaction = next(e for e in events if "after the second infusion" in e.source_snippet)

    # Expectation: Jan 12
    assert reaction.timestamp == datetime(2024, 1, 12, tzinfo=timezone.utc)


def test_chained_fuzzy_events() -> None:
    """
    Test chained dependencies with fuzzy matching.
    Event A: "Surgery on Jan 1"
    Event B: "Discharge 3 days after surgery" (Fuzzy link to A)
    Event C: "Follow-up 1 week after discharge" (Fuzzy link to B)
    """
    text = "Surgery on Jan 1. Discharge 3 days after surgery. Follow-up 1 week after discharge."
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)

    extractor = TimelineExtractor()
    events = extractor.extract_events(text, ref_date)

    # 1. Surgery: Jan 1
    # 2. Discharge: Jan 4
    # 3. Follow-up: Jan 11

    events.sort(key=lambda x: x.timestamp)
    assert len(events) == 3

    surgery = events[0]
    discharge = events[1]
    followup = events[2]

    assert surgery.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert discharge.timestamp == datetime(2024, 1, 4, tzinfo=timezone.utc)
    assert followup.timestamp == datetime(2024, 1, 11, tzinfo=timezone.utc)


def test_fuzzy_match_far_away() -> None:
    """
    Test fuzzy match where the anchor is far away from the event, ensuring no context overlap.
    """
    # Event "Second Infusion" is at start.
    # Anchor "after second infusion" is > 50 chars away.
    padding = " " * 100
    text = f"Second Infusion on Jan 10.{padding}Reaction 2 days after the second infusion."
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)

    extractor = TimelineExtractor()
    events = extractor.extract_events(text, ref_date)

    assert len(events) >= 2
    reaction = next(e for e in events if "after the second infusion" in e.source_snippet)

    # Should resolve to Jan 12
    assert reaction.timestamp == datetime(2024, 1, 12, tzinfo=timezone.utc)
