# Prosperity Public License 3.0
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
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

# Regex to extract duration context associated with an event
# e.g. "for 3 days", "lasting 2 weeks", "spanning 10 hours"
DURATION_CONTEXT_REGEX = re.compile(
    r"(?:for|lasting|spanning)\s+(?P<value>\d+(?:\.\d+)?)\s+(?P<unit>year|month|week|day|hour|minute|second)s?",
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

    def _calculate_total_minutes_and_delta(self, value: float, unit: str) -> Tuple[int, Any]:
        """
        Calculates total minutes and appropriate delta (timedelta or relativedelta).
        Handles fractional values for fixed units (days, hours, minutes).
        """
        unit = unit.lower()
        if not unit.endswith("s"):
            unit += "s"

        fixed_units = {"days": 24 * 60, "hours": 60, "minutes": 1, "weeks": 7 * 24 * 60, "seconds": 1 / 60}

        if unit in fixed_units:
            minutes = int(value * fixed_units[unit])
            # Use timedelta for fixed units to allow fractional precision mapping
            td_kwargs = {unit: value}
            return minutes, timedelta(**td_kwargs)  # pragma: no cover

        # Variable units (months, years) - Fallback to relativedelta (integer based mostly)
        delta = self._parse_duration(value, unit)
        # Approx minutes
        base = datetime(2000, 1, 1)
        end = base + delta
        diff = end - base
        minutes = int(diff.total_seconds() // 60)
        return minutes, delta

    def _resolve_duration(
        self, text: str, start_idx: int, end_idx: int, forbidden_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[Optional[int], Optional[Any]]:
        """
        Scans the text around the event location (start_idx, end_idx) for duration patterns.
        Returns (duration_minutes, delta).

        Prioritizes the closest match to the event snippet.
        Ensures the match does not fall into any forbidden_ranges (e.g., other events or anchors).
        """
        # Look in a window around the event
        window = 50
        ctx_start = max(0, start_idx - window)
        ctx_end = min(len(text), end_idx + window)
        context_text = text[ctx_start:ctx_end]

        matches = list(DURATION_CONTEXT_REGEX.finditer(context_text))
        if not matches:
            return None, None

        # Find best match (closest distance to snippet range)
        # snippet range in context_text coordinates:
        snip_start_local = start_idx - ctx_start
        snip_end_local = end_idx - ctx_start

        best_match = None
        min_dist = float("inf")

        for m in matches:
            m_start_global = ctx_start + m.start()
            m_end_global = ctx_start + m.end()

            # Check overlap with forbidden ranges
            is_forbidden = False
            if forbidden_ranges:
                for f_start, f_end in forbidden_ranges:
                    # Check for intersection
                    if max(m_start_global, f_start) < min(m_end_global, f_end):
                        is_forbidden = True  # pragma: no cover
                        break

            if is_forbidden:
                continue

            # Check for "intervening" ranges between snippet and match
            # If the match is separated from the snippet by a forbidden range, it's likely belonging to that range
            has_intervening = False
            if forbidden_ranges:
                for f_start, f_end in forbidden_ranges:
                    # Case 1: Snippet < Forbidden < Match
                    if end_idx <= f_start and f_end <= m_start_global:
                        has_intervening = True
                        break
                    # Case 2: Match < Forbidden < Snippet
                    if m_end_global <= f_start and f_end <= start_idx:
                        has_intervening = True
                        break

            if has_intervening:
                continue

            m_start = m.start()
            m_end = m.end()

            # Calculate distance
            if m_start >= snip_end_local:
                dist = m_start - snip_end_local
            elif m_end <= snip_start_local:
                dist = snip_start_local - m_end
            else:
                dist = 0

            if dist < min_dist:
                min_dist = dist
                best_match = m

        if best_match:
            val = float(best_match.group("value"))
            unit = best_match.group("unit")
            return self._calculate_total_minutes_and_delta(val, unit)

        return None, None

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

    def _calculate_semantic_score(self, anchor: str, target_text: str) -> float:
        """
        Calculates a semantic match score between the anchor phrase and the target text.
        Uses RapidFuzz for standard, efficient matching.
        """
        # Pre-process to remove stop words to help token set ratio
        stop_words = {"the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by"}

        def clean(s: str) -> str:
            # lower, remove non-word chars except spaces
            s = re.sub(r"[^\w\s]", "", s.lower())
            tokens = s.split()
            tokens = [t for t in tokens if t not in stop_words]
            return " ".join(tokens)

        anchor_clean = clean(anchor)
        target_clean = clean(target_text)

        if not anchor_clean:
            return 0.0

        score = fuzz.token_set_ratio(anchor_clean, target_clean)
        logger.debug(f"Fuzzy Match: '{anchor_clean}' vs '{target_clean}' -> {score}")
        return float(score) / 100.0

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

        # Identify Anchored Candidates Early to use as Forbidden Ranges for Standard Durations
        anchored_candidates = self._extract_anchored_candidates(text)

        # Pre-process dates to find ranges for Forbidden calculation
        extracted_dates_raw = search_dates(text, languages=["en"], settings=settings) or []

        date_ranges = []
        search_cursor = 0
        for source_snippet, _ in extracted_dates_raw:
            if DURATION_REGEX.match(source_snippet):
                continue  # pragma: no cover

            idx = self._find_snippet_index(text, source_snippet, search_cursor)
            if idx == -1:
                idx = self._find_snippet_index(text, source_snippet, 0)

            if idx != -1:
                end = idx + len(source_snippet)
                date_ranges.append((idx, end))
                search_cursor = idx + 1

        anchor_ranges = [(c["start"], c["end"]) for c in anchored_candidates]
        all_forbidden_ranges = date_ranges + anchor_ranges

        # Storage for all resolved events (Standard + Anchored) with metadata
        # We start with Standard events
        resolved_events_meta: List[Dict[str, Any]] = []

        if extracted_dates_raw:
            search_cursor = 0
            for source_snippet, date_obj in extracted_dates_raw:
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

                    # Extract Duration
                    # Don't forbid *this* event snippet (start_idx, end_idx) from being near itself
                    current_forbidden = [r for r in all_forbidden_ranges if r != (start_idx, end_idx)]

                    duration_minutes, duration_delta = self._resolve_duration(
                        text, start_idx, end_idx, current_forbidden
                    )
                    ends_at = date_obj + duration_delta if duration_delta is not None else None

                    if ends_at and ends_at <= date_obj:
                        ends_at = None
                        duration_minutes = None

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
                        duration_minutes=duration_minutes,
                        ends_at=ends_at,
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
        # (Already calculated anchored_candidates above)

        indices_to_remove = set()
        for cand in anchored_candidates:
            cand_range = range(cand["start"], cand["end"])
            for idx, meta in enumerate(resolved_events_meta):
                meta_range = range(meta["start"], meta["end"])
                if set(cand_range).intersection(meta_range):
                    indices_to_remove.add(idx)

        # Rebuild resolved_events_meta without the overlapping standard events
        resolved_events_meta = [meta for idx, meta in enumerate(resolved_events_meta) if idx not in indices_to_remove]

        # 2b. Iterative Resolution Loop
        unresolved_candidates = list(anchored_candidates)
        max_iterations = len(anchored_candidates) + 1  # Safe upper bound

        for _ in range(max_iterations):
            progress_made = False
            still_unresolved = []

            for cand in unresolved_candidates:
                anchor_phrase = cand["anchor_phrase"]
                cand_start = cand["start"]
                cand_end = cand["end"]

                best_match_event = None

                # Strategy B: Semantic/Fuzzy Match against resolved events
                # We prioritize semantic match over raw proximity now with RapidFuzz
                fuzzy_candidates = []

                for meta in resolved_events_meta:
                    evt = meta["event"]

                    # Mask anchor text from description
                    window = 50
                    d_start = max(0, meta["start"] - window)
                    d_end = min(len(text), meta["end"] + window)

                    # Anchor span
                    c_start, c_end = cand_start, cand_end

                    # If overlap
                    if max(d_start, c_start) < min(d_end, c_end):
                        # Construct a masked text snippet
                        p1_end = max(d_start, c_start)
                        part1 = text[d_start:p1_end]

                        p2_start = min(d_end, c_end)
                        part2 = text[p2_start:d_end]

                        masked_desc = (part1 + " " + part2).strip().replace("\n", " ")
                    else:
                        masked_desc = evt.description

                    # Calculate max score from description or snippet
                    score_desc = self._calculate_semantic_score(anchor_phrase, masked_desc)
                    score_snip = self._calculate_semantic_score(anchor_phrase, evt.source_snippet)
                    score = max(score_desc, score_snip)

                    # Threshold for fuzzy match
                    # Adjusted threshold after adding stopword removal
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
                    best_candidate = fuzzy_candidates[0]
                    best_match_event = best_candidate["event"]

                # If we found a match, resolve
                if best_match_event:
                    delta = self._parse_duration(cand["duration_val"], cand["unit"])
                    if cand["direction"] == "after":
                        new_time = best_match_event.timestamp + delta
                    else:
                        new_time = best_match_event.timestamp - delta

                    # Extract Duration for the new event
                    current_forbidden = [f for f in all_forbidden_ranges if f != (cand_start, cand_end)]

                    duration_minutes, duration_delta = self._resolve_duration(
                        text, cand_start, cand_end, current_forbidden
                    )
                    ends_at = new_time + duration_delta if duration_delta is not None else None

                    if ends_at and ends_at <= new_time:
                        ends_at = None
                        duration_minutes = None

                    new_event = TemporalEvent(
                        id=uuid4(),
                        description=f"Derived from anchor '{cand['full_match']}' linked to {best_match_event.description[:20]}...",  # noqa: E501
                        timestamp=new_time,
                        granularity=best_match_event.granularity,
                        source_snippet=cand["full_match"],
                        duration_minutes=duration_minutes,
                        ends_at=ends_at,
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
