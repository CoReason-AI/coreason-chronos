# Prosperity Public License 3.0
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
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

        int_val = int(value)
        if abs(value - int_val) < 0.001:
            kwargs = {unit: int_val}
        else:
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

    def _find_closest_event(
        self, target_start: int, target_end: int, events_meta: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Finds the event in events_meta that is closest to the target interval [target_start, target_end].
        """
        best_event = None
        min_dist = float("inf")

        for meta in events_meta:
            # Calculate gap between [target_start, target_end] and [meta["start"], meta["end"]]
            e_start, e_end = meta["start"], meta["end"]

            # Overlap check
            if max(target_start, e_start) < min(target_end, e_end):
                dist = 0
            else:
                # Gap is distance between closest edges
                # Case 1: Target before Event
                if target_end <= e_start:
                    dist = e_start - target_end
                # Case 2: Event before Target
                else:  # e_end <= target_start
                    dist = target_start - e_end

            if dist < min_dist:
                min_dist = dist
                best_event = meta

        return best_event

    def extract_events(self, text: str, reference_date: datetime) -> List[TemporalEvent]:
        """
        Extracts events from the given text.
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

        # Storage for all resolved events (Standard + Anchored) with metadata
        # We start with Standard events
        resolved_events_meta: List[Dict[str, Any]] = []

        if extracted_dates:
            search_cursor = 0
            for source_snippet, date_obj in extracted_dates:
                if not date_obj:
                    continue

                if DURATION_REGEX.match(source_snippet):
                    continue

                if date_obj.tzinfo is None:
                    date_obj = date_obj.replace(tzinfo=timezone.utc)
                else:
                    date_obj = date_obj.astimezone(timezone.utc)

                start_idx = self._find_snippet_index(text, source_snippet, search_cursor)
                if start_idx == -1:
                    start_idx = self._find_snippet_index(text, source_snippet, 0)  # pragma: no cover

                if start_idx != -1:
                    end_idx = start_idx + len(source_snippet)
                    search_cursor = start_idx + 1

                    description = self._get_context_description(text, start_idx, end_idx)

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

                    resolved_events_meta.append(
                        {
                            "event": event,
                            "start": start_idx,
                            "end": end_idx,
                            "snippet": source_snippet,
                            "is_anchored": False,
                        }
                    )

        # Pass 2: Identify Anchored Candidates
        anchored_candidates = self._extract_anchored_candidates(text)

        # 2a. Prune Standard events that overlap with Anchored candidates (prefer Anchor logic)
        # However, if we prune them, we remove them from resolved_events_meta.
        # But we want to prune them *only if* the Anchor logic is actually used?
        # No, if the regex matches, it's likely a better match than dateparser's fuzzy relative.
        # So we identify indices to remove from resolved_events_meta first.

        indices_to_remove = set()
        for cand in anchored_candidates:
            cand_range = range(cand["start"], cand["end"])
            for idx, meta in enumerate(resolved_events_meta):
                meta_range = range(meta["start"], meta["end"])
                if set(cand_range).intersection(meta_range):
                    indices_to_remove.add(idx)

        # Rebuild resolved_events_meta without the overlapping standard events
        # Use list comprehension to filter, but we need to track indices or just rebuild
        resolved_events_meta = [meta for idx, meta in enumerate(resolved_events_meta) if idx not in indices_to_remove]

        # 2b. Iterative Resolution Loop
        # We need to resolve anchors against `resolved_events_meta`.
        # When an anchor is resolved, it is added to `resolved_events_meta` so subsequent anchors can find it.

        unresolved_candidates = list(anchored_candidates)
        max_iterations = len(anchored_candidates) + 1  # Safe upper bound

        for _ in range(max_iterations):
            progress_made = False
            still_unresolved = []

            for cand in unresolved_candidates:
                anchor_phrase = cand["anchor_phrase"]
                cand_start = cand["start"]
                cand_end = cand["end"]

                # Search for occurrences of anchor_phrase in text
                # We want occurrences that are NOT inside the candidate string itself?
                # Actually, "anchor_phrase" is a substring of "full_match" (candidate string).
                # We want to find *other* occurrences of "anchor_phrase" in the text,
                # OR we rely on the fact that an event should be *at* that location.

                # Wait. "Middle 2 days after Start."
                # Anchor: "Start".
                # We want to find the event associated with "Start".
                # There is an event at "Start on Jan 1".
                # The word "Start" is at index 0.
                # The event "Jan 1" is at index 9.
                # The word "Start" is near the event "Jan 1".

                # Strategy:
                # 1. Find all occurrences of anchor_phrase in text.
                # 2. Filter out the one that is inside the candidate's own span (e.g. "after Start").
                # 3. For each valid occurrence, find the closest event in `resolved_events_meta`.
                # 4. Pick the global closest match.

                best_match_event = None
                min_dist_global = float("inf")

                # Find occurrences
                pattern = re.escape(anchor_phrase)
                # We iterate manually to get indices
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    occ_start = m.start()
                    occ_end = m.end()

                    # Check if this occurrence overlaps with the candidate's own span (the "after X" part)
                    # cand["start"] is start of "2 days after Start"
                    # cand["end"] is end of "2 days after Start"
                    # The anchor phrase IS inside the candidate span.
                    # Wait. "2 days after Start". The word "Start" is at the end of this string.
                    # We are looking for the *target* "Start".
                    # If the text says: "Start on Jan 1. Middle 2 days after Start."
                    # Occurrences of "Start":
                    # 1. Index 0 ("Start on Jan 1") -> Linked to Event "Jan 1"
                    # 2. Index 37 ("... after Start") -> Inside candidate.

                    # So we exclude occurrences inside [cand_start, cand_end].
                    if max(cand_start, occ_start) < min(cand_end, occ_end):
                        continue

                    # Valid occurrence. Find closest event.
                    match_meta = self._find_closest_event(occ_start, occ_end, resolved_events_meta)
                    if match_meta:
                        # Re-calculate distance logic (duplicated from _find_closest_event for global comparison)
                        e_start, e_end = match_meta["start"], match_meta["end"]

                        if max(occ_start, e_start) < min(occ_end, e_end):
                            dist = 0
                        elif occ_end <= e_start:
                            dist = e_start - occ_end
                        else:
                            dist = occ_start - e_end

                        if dist < min_dist_global:
                            min_dist_global = dist
                            best_match_event = match_meta["event"]

                # If we found a match, resolve
                if best_match_event:
                    delta = self._parse_duration(cand["duration_val"], cand["unit"])
                    if cand["direction"] == "after":
                        new_time = best_match_event.timestamp + delta
                    else:
                        new_time = best_match_event.timestamp - delta

                    new_event = TemporalEvent(
                        id=uuid4(),
                        description=f"Derived from anchor '{cand['full_match']}' linked to {best_match_event.id}",
                        timestamp=new_time,
                        granularity=best_match_event.granularity,
                        source_snippet=cand["full_match"],
                    )

                    resolved_events_meta.append(
                        {
                            "event": new_event,
                            "start": cand_start,
                            "end": cand_end,
                            "snippet": cand["full_match"],
                            "is_anchored": True,
                        }
                    )
                    logger.info(f"Resolved anchored event '{cand['full_match']}' to {new_time}")
                    progress_made = True
                else:
                    still_unresolved.append(cand)

            unresolved_candidates = still_unresolved
            if not progress_made or not unresolved_candidates:
                break

        # Log remaining unresolved
        for cand in unresolved_candidates:
            logger.warning(f"Could not resolve anchor '{cand['anchor_phrase']}' for snippet '{cand['full_match']}'")

        final_events = [meta["event"] for meta in resolved_events_meta]
        final_events.sort(key=lambda x: x.timestamp)

        logger.info(f"Extracted {len(final_events)} events from text.")
        return final_events
