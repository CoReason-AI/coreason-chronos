# Prosperity Public License 3.0
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from dateparser.search import search_dates
from dateutil.relativedelta import relativedelta
from rapidfuzz import fuzz

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
    The Historian: Turns text into a timeline.

    Implements a multi-pass strategy to extract events:
    1.  Absolute/Simple Relative extraction (using dateparser).
    2.  Anchored extraction (using regex and fuzzy matching for "2 days after X").
    3.  Anchor resolution logic to link relative events to established absolute timestamps.
    """

    def __init__(self) -> None:
        pass

    def _find_snippet_index(self, text: str, snippet: str, start_index: int = 0) -> int:
        """
        Locates the snippet in the text starting from start_index.

        Args:
            text: The full text.
            snippet: The substring to find.
            start_index: The index to start searching from.

        Returns:
            The start index of the snippet, or -1 if not found.
        """
        return text.find(snippet, start_index)

    def _get_context_description(self, text: str, start: int, end: int, window: int = 50) -> str:
        """
        Extracts context around a match to serve as the event description.

        Args:
            text: The full source text.
            start: Start index of the match.
            end: End index of the match.
            window: Number of characters to capture before and after.

        Returns:
            A clean string containing the surrounding context.
        """
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        snippet = text[ctx_start:ctx_end].strip()
        return snippet.replace("\n", " ")

    def _parse_duration(self, value: float, unit: str) -> relativedelta:
        """
        Converts a value and unit string into a relativedelta object.

        Args:
            value: The numeric duration value (e.g., 2.5).
            unit: The time unit (e.g., "days").

        Returns:
            A relativedelta representing the duration.
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
        Scans text for anchored event patterns (e.g., "2 days after admission").

        Args:
            text: The text to scan.

        Returns:
            A list of candidate dictionaries containing match details.
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

    def _clean_text_for_matching(self, s: str) -> str:
        """
        Cleans text for semantic matching (lowercase, remove punctuation, remove stop words).

        Args:
            s: The input string.

        Returns:
            The cleaned string.
        """
        stop_words = {"the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by"}
        # lower, remove non-word chars except spaces
        s = re.sub(r"[^\w\s]", "", s.lower())
        tokens = s.split()
        tokens = [t for t in tokens if t not in stop_words]
        return " ".join(tokens)

    def _calculate_semantic_score(self, anchor: str, target_text: str) -> float:
        """
        Calculates a semantic match score between the anchor phrase and the target text.
        Uses RapidFuzz for standard, efficient matching.

        Args:
            anchor: The anchor phrase to match (e.g., "admission").
            target_text: The target text to search in (e.g., "Patient admission date").

        Returns:
            A float score between 0.0 and 1.0.
        """
        anchor_clean = self._clean_text_for_matching(anchor)
        target_clean = self._clean_text_for_matching(target_text)

        if not anchor_clean:
            return 0.0

        score = fuzz.token_set_ratio(anchor_clean, target_clean)
        logger.debug(f"Fuzzy Match: '{anchor_clean}' vs '{target_clean}' -> {score}")
        return float(score) / 100.0

    def _create_temporal_event(
        self,
        text: str,
        start_idx: int,
        end_idx: int,
        date_obj: datetime,
        source_snippet: str,
    ) -> TemporalEvent:
        """
        Helper to instantiate a TemporalEvent with proper context and granularity.
        """
        description = self._get_context_description(text, start_idx, end_idx)

        granularity = TemporalGranularity.PRECISE
        if date_obj.hour == 0 and date_obj.minute == 0 and date_obj.second == 0 and "00:00" not in source_snippet:
            granularity = TemporalGranularity.DATE_ONLY

        return TemporalEvent(
            id=uuid4(),
            description=description,
            timestamp=date_obj,
            granularity=granularity,
            source_snippet=source_snippet,
        )

    def _extract_standard_events(self, text: str, reference_date: datetime) -> List[Dict[str, Any]]:
        """
        Pass 1: Extracts absolute and simple relative dates using dateparser.

        Args:
            text: The text to parse.
            reference_date: The base date for relative calculations (e.g., "today").

        Returns:
            A list of metadata dictionaries containing resolved events and their positions.
        """
        settings = {
            "RELATIVE_BASE": reference_date.replace(tzinfo=None),
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "past",
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        }

        extracted_dates = search_dates(text, languages=["en"], settings=settings)
        resolved_events_meta: List[Dict[str, Any]] = []

        if not extracted_dates:
            return resolved_events_meta

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

                event = self._create_temporal_event(text, start_idx, end_idx, date_obj, source_snippet)

                resolved_events_meta.append(
                    {
                        "event": event,
                        "start": start_idx,
                        "end": end_idx,
                        "snippet": source_snippet,
                        "is_anchored": False,
                    }
                )

        return resolved_events_meta

    def _find_best_anchor_match(
        self,
        anchor_phrase: str,
        cand_start: int,
        cand_end: int,
        text: str,
        resolved_events_meta: List[Dict[str, Any]],
    ) -> Optional[TemporalEvent]:
        """
        Finds the best matching event for a given anchor phrase based on semantic score and proximity.

        Args:
            anchor_phrase: The phrase identifying the anchor (e.g., "surgery").
            cand_start: Start index of the anchored phrase.
            cand_end: End index of the anchored phrase.
            text: Full text.
            resolved_events_meta: Pool of already resolved events.

        Returns:
            The best matching TemporalEvent, or None if no suitable match found.
        """
        fuzzy_candidates = []

        for meta in resolved_events_meta:
            evt = meta["event"]

            # Mask anchor text from description to avoid self-matching
            window = 50
            d_start = max(0, meta["start"] - window)
            d_end = min(len(text), meta["end"] + window)

            # If overlap between event context and candidate anchor context
            if max(d_start, cand_start) < min(d_end, cand_end):
                p1_end = max(d_start, cand_start)
                part1 = text[d_start:p1_end]

                p2_start = min(d_end, cand_end)
                part2 = text[p2_start:d_end]

                masked_desc = (part1 + " " + part2).strip().replace("\n", " ")
            else:
                masked_desc = evt.description

            # Calculate max score from description or snippet
            score_desc = self._calculate_semantic_score(anchor_phrase, masked_desc)
            score_snip = self._calculate_semantic_score(anchor_phrase, evt.source_snippet)
            score = max(score_desc, score_snip)

            # Threshold for fuzzy match
            if score >= 0.5:
                # Calculate distance
                if cand_start > meta["start"]:
                    dist = cand_start - meta["end"]
                else:
                    dist = meta["start"] - cand_end
                dist = max(0, dist)

                fuzzy_candidates.append({"event": evt, "score": score, "dist": dist})

        if fuzzy_candidates:
            # Sort by Score DESC, then Distance ASC
            fuzzy_candidates.sort(key=lambda x: (-x["score"], x["dist"]))
            return cast(TemporalEvent, fuzzy_candidates[0]["event"])

        return None

    def _resolve_anchored_events(
        self,
        text: str,
        anchored_candidates: List[Dict[str, Any]],
        resolved_events_meta: List[Dict[str, Any]],
    ) -> None:
        """
        Iteratively resolves anchored candidates against the pool of resolved events.
        Modifies resolved_events_meta in place.

        Args:
            text: Full text.
            anchored_candidates: List of detected anchor patterns.
            resolved_events_meta: List of currently resolved events.
        """
        unresolved_candidates = list(anchored_candidates)
        max_iterations = len(anchored_candidates) + 1  # Safe upper bound

        for _ in range(max_iterations):
            progress_made = False
            still_unresolved = []

            for cand in unresolved_candidates:
                best_match_event = self._find_best_anchor_match(
                    cand["anchor_phrase"],
                    cand["start"],
                    cand["end"],
                    text,
                    resolved_events_meta,
                )

                if best_match_event:
                    delta = self._parse_duration(cand["duration_val"], cand["unit"])
                    if cand["direction"] == "after":
                        new_time = best_match_event.timestamp + delta
                    else:
                        new_time = best_match_event.timestamp - delta

                    new_event = TemporalEvent(
                        id=uuid4(),
                        description=f"Derived from anchor '{cand['full_match']}' linked to {best_match_event.description[:20]}...",  # noqa: E501
                        timestamp=new_time,
                        granularity=best_match_event.granularity,
                        source_snippet=cand["full_match"],
                    )

                    resolved_events_meta.append(
                        {
                            "event": new_event,
                            "start": cand["start"],
                            "end": cand["end"],
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

    def extract_events(self, text: str, reference_date: datetime) -> List[TemporalEvent]:
        """
        Main entry point: Extracts events from the given text relative to a reference date.

        Args:
            text: The unstructured text containing temporal information.
            reference_date: The anchor date (usually document metadata date) for interpreting relative terms
                            like "today".

        Returns:
            A list of TemporalEvent objects, sorted chronologically.

        Raises:
            ValueError: If reference_date is not timezone-aware.
        """
        if reference_date.tzinfo is None:
            raise ValueError("reference_date must be timezone-aware")

        # Pass 1: Standard Extraction
        resolved_events_meta = self._extract_standard_events(text, reference_date)

        # Pass 2: Identify Anchored Candidates
        anchored_candidates = self._extract_anchored_candidates(text)

        # Filter out standard events that overlap with anchored candidates
        # (This handles cases where dateparser might partially pick up an anchor phrase)
        indices_to_remove = set()
        for cand in anchored_candidates:
            cand_range = range(cand["start"], cand["end"])
            for idx, meta in enumerate(resolved_events_meta):
                meta_range = range(meta["start"], meta["end"])
                if set(cand_range).intersection(meta_range):
                    indices_to_remove.add(idx)

        resolved_events_meta = [meta for idx, meta in enumerate(resolved_events_meta) if idx not in indices_to_remove]

        # Pass 3: Resolve Anchors
        self._resolve_anchored_events(text, anchored_candidates, resolved_events_meta)

        final_events = [meta["event"] for meta in resolved_events_meta]
        final_events.sort(key=lambda x: x.timestamp)

        logger.info(f"Extracted {len(final_events)} events from text.")
        return final_events
