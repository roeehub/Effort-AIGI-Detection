"""
Speaker Tracker for audio chunk naming.

Tracks speaking state per participant and calculates dominant speaker
for each audio chunk based on speaking duration and threshold.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class SpeakingPeriod:
    """Represents a period when a participant was speaking."""
    participant_name: str
    start_ms: int
    end_ms: Optional[int] = None  # None if still speaking


@dataclass
class SpeakerTrackerConfig:
    """Configuration for speaker chunk naming."""
    enabled: bool = True
    threshold_proportion: float = 0.80
    include_in_filename: bool = True
    include_in_metadata: bool = True


class SpeakerTracker:
    """
    Tracks speaking state per participant and determines dominant speaker for chunks.

    The tracker maintains speaking periods across chunks and calculates
    which participant (if any) should be assigned to a given audio chunk
    based on the threshold proportion.
    """

    def __init__(self, config: SpeakerTrackerConfig):
        """
        Initialize speaker tracker.

        Args:
            config: Configuration for speaker chunk naming
        """
        self.config = config

        # Current speaking state per participant: name -> is_speaking
        self._speaking_state: Dict[str, bool] = {}

        # Active speaking periods (participant -> start_ms when currently speaking)
        self._active_speaking_start: Dict[str, int] = {}

        # Completed speaking periods for calculation
        self._speaking_periods: List[SpeakingPeriod] = []

        # Track the last processed timestamp for cleanup
        self._last_chunk_end_ms: int = 0

    def process_speaker_events(self, events: List[Dict]) -> None:
        """
        Process incoming speaker events to update speaking state.

        Args:
            events: List of speaker event dicts with:
                - participant_name: str
                - is_speaking: bool
                - timestamp_ms: int
                - chunk_sequence: int
        """
        for event in events:
            participant = event.get('participant_name', '')
            is_speaking = event.get('is_speaking', False)
            timestamp_ms = event.get('timestamp_ms', 0)

            if not participant:
                continue

            prev_state = self._speaking_state.get(participant, False)

            if is_speaking and not prev_state:
                # Started speaking
                self._speaking_state[participant] = True
                self._active_speaking_start[participant] = timestamp_ms

            elif not is_speaking and prev_state:
                # Stopped speaking
                self._speaking_state[participant] = False
                start_ms = self._active_speaking_start.pop(participant, timestamp_ms)

                # Record the completed speaking period
                self._speaking_periods.append(SpeakingPeriod(
                    participant_name=participant,
                    start_ms=start_ms,
                    end_ms=timestamp_ms
                ))

    def calculate_dominant_speaker(
        self,
        chunk_start_ms: int,
        chunk_duration_ms: int
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Calculate the dominant speaker for a given audio chunk.

        Args:
            chunk_start_ms: Start timestamp of the chunk in milliseconds
            chunk_duration_ms: Duration of the chunk in milliseconds

        Returns:
            Tuple of:
                - Dominant speaker name (or None if no dominant speaker)
                - Dict of participant -> proportion of chunk they were speaking
        """
        if not self.config.enabled:
            return None, {}

        chunk_end_ms = chunk_start_ms + chunk_duration_ms

        # Calculate speaking duration per participant within this chunk
        speaking_durations: Dict[str, int] = {}

        # Process completed speaking periods
        for period in self._speaking_periods:
            # Calculate overlap between speaking period and chunk
            overlap_start = max(period.start_ms, chunk_start_ms)
            overlap_end = min(period.end_ms or chunk_end_ms, chunk_end_ms)

            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaking_durations[period.participant_name] = \
                    speaking_durations.get(period.participant_name, 0) + duration

        # Process active speaking periods (still speaking)
        for participant, start_ms in self._active_speaking_start.items():
            # Calculate overlap with chunk
            overlap_start = max(start_ms, chunk_start_ms)
            overlap_end = chunk_end_ms

            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaking_durations[participant] = \
                    speaking_durations.get(participant, 0) + duration

        # Calculate proportions
        proportions: Dict[str, float] = {}
        for participant, duration in speaking_durations.items():
            proportions[participant] = duration / chunk_duration_ms

        # Determine dominant speaker based on threshold
        threshold = self.config.threshold_proportion
        exceeding_threshold = [
            (participant, prop)
            for participant, prop in proportions.items()
            if prop >= threshold
        ]

        dominant_speaker = None

        if len(exceeding_threshold) == 1:
            # Exactly one participant exceeds threshold -> assign to them
            dominant_speaker = exceeding_threshold[0][0]
        elif len(exceeding_threshold) > 1:
            # Multiple participants exceed threshold -> ambiguous, no assignment
            dominant_speaker = None
        else:
            # No participant exceeds threshold -> no assignment
            dominant_speaker = None

        # Cleanup old speaking periods (keep only recent ones)
        self._cleanup_old_periods(chunk_end_ms)
        self._last_chunk_end_ms = chunk_end_ms

        return dominant_speaker, proportions

    def _cleanup_old_periods(self, current_time_ms: int, retention_ms: int = 30000) -> None:
        """
        Remove old speaking periods to prevent memory growth.

        Args:
            current_time_ms: Current timestamp
            retention_ms: How long to keep old periods (default 30 seconds)
        """
        cutoff = current_time_ms - retention_ms
        self._speaking_periods = [
            p for p in self._speaking_periods
            if (p.end_ms or current_time_ms) > cutoff
        ]

    def get_speaking_state(self) -> Dict[str, bool]:
        """Get current speaking state for all tracked participants."""
        return self._speaking_state.copy()

    def reset(self) -> None:
        """Reset all tracking state."""
        self._speaking_state.clear()
        self._active_speaking_start.clear()
        self._speaking_periods.clear()
        self._last_chunk_end_ms = 0
