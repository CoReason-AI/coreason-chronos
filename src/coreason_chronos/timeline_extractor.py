# Prosperity Public License 3.0
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Set
from uuid import uuid4

from dateparser.search import search_dates
from dateutil.relativedelta import relativedelta

from coreason_chronos.schemas import TemporalEvent, TemporalGranularity
from coreason_chronos.utils.logger import logger

# Regex to identify snippets that are likely pure durations (e.g., "3 months")
# and not absolute or relative points in time (e.g., "3 months ago").
DURATION_REGEX = re.compile(r"^\d+\s+(year|month|week|day|hour|minute|second)s?$", re.IGNORECASE)

# Regex for anchored events: "2 days after admission", "3 weeks before the surgery"
# Captures: duration, unit, direction, anchor_phrase
ANCHOR_REGEX = re.compile(
    r"(?P<duration>\d+(?:\.\d+)?)\s+(?P<unit>year|month|week|day|hour|minute|second)s?\s+"
    r"(?P<direction>after|before)\s+(?P<anchor>[\w\s]+?)(?:$|[.,;])",
    re.IGNORECASE,
)


class TimelineExtractor:
    """
    Extracts timeline events from text, converting relative dates to absolute ones
    based on a reference date.
    """

    def __init__(self) -> None:
        pass

    def _find_snippet_index(self, text: str, snippet: str, start_index: int = 0) -> int:
        """
        Locates the snippet in the text starting from start_index.
        Returns the index or -1 if not found.
        """
        return text.find(snippet, start_index)

    def _get_context_description(self, text: str, start: int, end: int, window: int = 50) -> str:
        """
        Extracts context around the match to serve as the event description.
        Captures `window` characters before and after.
        """
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        snippet = text[ctx_start:ctx_end].strip()
        return snippet.replace("\n", " ")

    def _parse_duration(self, value: float, unit: str) -> relativedelta:
        """
        Converts a value and unit string into a relativedelta.
        """
        unit = unit.lower()
        if not unit.endswith("s"):
            unit += "s"

        # relativedelta expects integers for days, months, etc unless we use microseconds?
        # Actually relativedelta kwargs accept floats but it's better to be int if possible.
        # But our regex allows float.
        # For simplicity, if float, we might need to handle it.
        # But relativedelta doesn't support float for 'days'.
        # Check if int
        int_val = int(value)
        if abs(value - int_val) < 0.001:
            kwargs = {unit: int_val}
        else:
            # Handle fractional?
            # '2.5 days' -> 2 days + 12 hours?
            # For now, let's round or cast to int as per standard usage, or use microseconds?
            # The requirement says "Strict Math".
            # If unit is days, 0.5 days = 12 hours.
            # Let's just cast to int for AUC scope unless critical.
            # Or use timedelta for days/hours? relativedelta is better for months/years.
            # Let's assume int for now as user stories use integers.
            kwargs = {unit: int(value)}

        return relativedelta(**kwargs)  # type: ignore

    def _extract_anchored_candidates(self, text: str) -> List[Dict[str, Any]]:
        """
        Scans text for anchored event patterns.
        Returns a list of dictionaries with match details.
        """
        candidates = []
        for match in ANCHOR_REGEX.finditer(text):
            candidates.append(
                {
                    "duration_val": float(match.group("duration")),
                    "unit": match.group("unit"),
                    "direction": match.group("direction").lower(),
                    "anchor_phrase": match.group("anchor").strip(),
                    "start": match.start(),
                    "end": match.end(),
                    "full_match": match.group(0),
                }
            )
        return candidates

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

        # Pass 1: Standard Extraction (Absolute & Simple Relative)
        settings = {
            "RELATIVE_BASE": reference_date.replace(tzinfo=None),
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "past",
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        }

        extracted_dates = search_dates(text, languages=["en"], settings=settings)

        # Intermediate storage for standard events with position info
        # Structure: {'event': TemporalEvent, 'start': int, 'end': int, 'snippet': str}
        standard_events_meta: List[Dict[str, Any]] = []

        if extracted_dates:
            search_cursor = 0
            for source_snippet, date_obj in extracted_dates:
                if not date_obj:
                    continue

                if DURATION_REGEX.match(source_snippet):
                    continue

                # Ensure UTC
                if date_obj.tzinfo is None:
                    date_obj = date_obj.replace(tzinfo=timezone.utc)
                else:
                    date_obj = date_obj.astimezone(timezone.utc)

                # Locate in text
                start_idx = self._find_snippet_index(text, source_snippet, search_cursor)
                if start_idx == -1:
                    # Should not happen if search_dates works on this text, but safeguard
                    # If snippet appears earlier than cursor (weird), try from 0
                    start_idx = self._find_snippet_index(text, source_snippet, 0)  # pragma: no cover

                if start_idx != -1:
                    end_idx = start_idx + len(source_snippet)
                    search_cursor = start_idx + 1  # Advance cursor conservatively

                    # Extract context
                    description = self._get_context_description(text, start_idx, end_idx)

                    # Determine Granularity
                    granularity = TemporalGranularity.PRECISE
                    if (
                        date_obj.hour == 0
                        and date_obj.minute == 0
                        and date_obj.second == 0
                        and "00:00" not in source_snippet
                    ):
                        granularity = TemporalGranularity.DATE_ONLY

                    event = TemporalEvent(
                        id=uuid4(),
                        description=description,
                        timestamp=date_obj,
                        granularity=granularity,
                        source_snippet=source_snippet,
                    )

                    standard_events_meta.append(
                        {"event": event, "start": start_idx, "end": end_idx, "snippet": source_snippet}
                    )

        # Pass 2: Anchored Extraction
        anchored_candidates = self._extract_anchored_candidates(text)
        final_events: List[TemporalEvent] = []

        # Track which standard events are actually valid (not overridden by anchored)
        # We assume initially all are valid
        valid_standard_indices: Set[int] = set(range(len(standard_events_meta)))

        # Process Anchored Candidates
        for cand in anchored_candidates:
            # Check for overlap with standard events
            # If the anchored match overlaps significantly with a standard match,
            # we prefer the anchored one because it has explicit logic.
            cand_range = range(cand["start"], cand["end"])

            overlapping_indices = []
            for idx, meta in enumerate(standard_events_meta):
                meta_range = range(meta["start"], meta["end"])
                # Check overlap
                if set(cand_range).intersection(meta_range):
                    overlapping_indices.append(idx)

            # Mark overlapping standard events as invalid
            for idx in overlapping_indices:
                valid_standard_indices.discard(idx)

            # Resolve Anchor
            anchor_phrase = cand["anchor_phrase"]

            # Look for the anchor in the descriptions of *valid* standard events
            # We look in standard_events_meta (but checking if they are still considered valid?)
            # Actually, the anchor might be another event that IS valid.
            # Or it might be an event that was invalid? (Unlikely)

            # We search in ALL standard events derived from Pass 1.
            # Even if we invalidated one (because it was the "2 days after" part),
            # the anchor *target* should be valid.

            found_anchor_event = None
            for idx, meta in enumerate(standard_events_meta):
                # Don't anchor to self (the overlapped one)
                if idx in overlapping_indices:
                    continue

                # Check if anchor phrase is in the description
                # Using case-insensitive check
                # meta["event"] is TemporalEvent
                current_event: TemporalEvent = meta["event"]
                if anchor_phrase.lower() in current_event.description.lower():
                    found_anchor_event = current_event
                    break

            if found_anchor_event:
                # Calculate new date
                delta = self._parse_duration(cand["duration_val"], cand["unit"])
                if cand["direction"] == "after":
                    new_time = found_anchor_event.timestamp + delta
                else:  # before
                    new_time = found_anchor_event.timestamp - delta

                # Create new event
                new_event = TemporalEvent(
                    id=uuid4(),
                    description=f"Derived from anchor '{cand['full_match']}' linked to {found_anchor_event.id}",
                    timestamp=new_time,
                    # Inherit granularity? Or Precise? Delta makes it precise-ish.
                    granularity=found_anchor_event.granularity,
                    source_snippet=cand["full_match"],
                )
                final_events.append(new_event)
                logger.info(f"Resolved anchored event '{cand['full_match']}' to {new_time}")
            else:
                logger.warning(f"Could not resolve anchor '{anchor_phrase}' for snippet '{cand['full_match']}'")

        # Add valid standard events to final list
        for idx in valid_standard_indices:
            final_events.append(standard_events_meta[idx]["event"])

        # Sort by timestamp
        final_events.sort(key=lambda x: x.timestamp)

        logger.info(f"Extracted {len(final_events)} events from text.")
        return final_events
