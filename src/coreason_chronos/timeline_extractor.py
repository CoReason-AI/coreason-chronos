# Prosperity Public License 3.0
import re
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from dateparser.search import search_dates

from coreason_chronos.schemas import TemporalEvent, TemporalGranularity
from coreason_chronos.utils.logger import logger

# Regex to identify snippets that are likely pure durations (e.g., "3 months")
# and not absolute or relative points in time (e.g., "3 months ago").
DURATION_REGEX = re.compile(r"^\d+\s+(year|month|week|day|hour|minute|second)s?$", re.IGNORECASE)


class TimelineExtractor:
    """
    Extracts timeline events from text, converting relative dates to absolute ones
    based on a reference date.
    """

    def __init__(self) -> None:
        pass

    def extract_events(self, text: str, reference_date: datetime) -> List[TemporalEvent]:
        """
        Extracts events from the given text.

        Args:
            text: The unstructured text containing temporal information.
            reference_date: The document's metadata date, used as a base for relative calculations.
                            Must be timezone-aware.

        Returns:
            A list of TemporalEvent objects found in the text.
        """
        if reference_date.tzinfo is None:
            raise ValueError("reference_date must be timezone-aware")

        # dateparser expects settings to configure the base date
        # "RELATIVE_BASE" allows "yesterday" to be relative to reference_date
        settings = {
            "RELATIVE_BASE": reference_date.replace(tzinfo=None),  # dateparser prefers naive for base in some versions?
            # Actually dateparser handles TZAware, but let's be careful.
            # Upon checking docs, RELATIVE_BASE should be datetime.
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "past",  # Default assumption? Maybe 'future' is better for "next week"?
            # Let's check logic. If doc date is 2020, and text says "next week", it should be 2020 + 7 days.
            # dateparser default is current time if not specified.
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        }

        # If the reference_date has a timezone, we want the output to match it or be UTC.
        # Directives say: Store all dates in ISO 8601 UTC format internally.

        extracted_dates = search_dates(text, languages=["en"], settings=settings)

        events: List[TemporalEvent] = []

        if not extracted_dates:
            logger.info("No dates found in text.")
            return events

        for source_snippet, date_obj in extracted_dates:
            if not date_obj:
                continue

            # Filter out pure durations/numbers interpreted as dates (e.g., "50 years" -> 1974)
            if DURATION_REGEX.match(source_snippet):
                logger.debug(f"Discarding potential duration/age misidentified as date: '{source_snippet}'")
                continue

            # Ensure date_obj is timezone aware. dateparser might return naive if not configured well.
            if date_obj.tzinfo is None:
                # If naive, assume it's in the same timezone as reference_date (or UTC if we enforce it)
                # But the requirement says "Store all dates in ISO 8601 UTC".
                # If we parsed "Jan 1st", it's ambiguous.
                # Let's assume UTC for simplicity unless specified.
                date_obj = date_obj.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC
                date_obj = date_obj.astimezone(timezone.utc)

            # Determine Granularity
            # dateparser doesn't explicitly return granularity.
            # Heuristic: If time is 00:00:00 and source didn't specify time, it might be DATE_ONLY.
            # For now, let's look at the source snippet.
            # If "at 10pm" in snippet -> PRECISE.
            # If just "Jan 1st" -> DATE_ONLY.
            # This is hard to perfect without deep parsing.
            # For this iteration, let's default to PRECISE if it has time components other than 0,
            # or DATE_ONLY otherwise.
            # Actually, dateparser returns a datetime.

            granularity = TemporalGranularity.PRECISE
            if date_obj.hour == 0 and date_obj.minute == 0 and date_obj.second == 0 and "00:00" not in source_snippet:
                # This is a weak heuristic but a start.
                granularity = TemporalGranularity.DATE_ONLY

            event = TemporalEvent(
                id=uuid4(),
                description=f"Event associated with '{source_snippet}'",
                timestamp=date_obj,
                granularity=granularity,
                source_snippet=source_snippet,
            )
            events.append(event)

        logger.info(f"Extracted {len(events)} events from text.")
        return events
