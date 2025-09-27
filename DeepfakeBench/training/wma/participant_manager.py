# participant_manager.py

"""
Manages stateful analysis for individual participants to provide stable verdicts.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np

# Import protobuf enum values for banner levels
import wma_streaming_pb2 as pb2

# --- Configuration Constants ---
HISTORY_WINDOW_SIZE = 150  # Total number of probabilities to keep per participant
ACTIVE_TEST_WINDOW = 75  # Number of recent probabilities to use for verdict calculation
DEFAULT_START_PROB = 0.2  # Assumed probability for a new, unseen participant
MIN_RESPONSE_INTERVAL = 5  # Send a response every N batches even if verdict is stable
RESET_AFTER_INACTIVE_MIN = 2.0  # Forget a participant after this many minutes of inactivity


# --- Data Structure for Participant State ---
@dataclass
class ParticipantState:
    """Holds the state for a single participant."""
    history: deque = field(default_factory=lambda: deque(
        [DEFAULT_START_PROB] * HISTORY_WINDOW_SIZE, maxlen=HISTORY_WINDOW_SIZE
    ))
    current_verdict: int = pb2.GREEN
    batch_counter: int = 0
    last_seen_ts: float = field(default_factory=time.time)


# --- The Main Manager Class ---
class ParticipantManager:
    """
    Manages the state and verdict logic for all participants in a thread-safe manner.
    """

    def __init__(self, threshold: float, margin: float):
        """
        Initializes the manager.
        Args:
            threshold: The base threshold for FAKE vs REAL decision.
            margin: The margin around the threshold to create the YELLOW band.
        """
        self.participants: Dict[str, ParticipantState] = {}
        self.lock = threading.Lock()
        self.threshold = threshold
        self.margin = margin
        print(f"[ParticipantManager] Initialized with threshold={threshold:.2f}, margin={margin:.2f}")

    def _calculate_band_level(self, mean_prob: float) -> int:
        """Maps a mean probability score to a GREEN, YELLOW, or RED verdict."""
        if mean_prob >= self.threshold + self.margin:
            return pb2.RED
        elif mean_prob >= self.threshold - self.margin:
            return pb2.YELLOW
        else:
            return pb2.GREEN

    def _cleanup_inactive_participants(self):
        """Removes participants who have not been seen for a configured duration."""
        now = time.time()
        inactive_threshold_sec = RESET_AFTER_INACTIVE_MIN * 60
        inactive_pids = [
            pid for pid, state in self.participants.items()
            if now - state.last_seen_ts > inactive_threshold_sec
        ]
        if inactive_pids:
            for pid in inactive_pids:
                del self.participants[pid]
            print(f"[ParticipantManager] Cleaned up inactive participants: {inactive_pids}")

    def _get_or_create_state(self, participant_id: str) -> ParticipantState:
        """Retrieves or creates the state for a given participant."""
        if participant_id not in self.participants:
            print(f"[ParticipantManager] New participant seen: {participant_id}. Creating initial state.")
            self.participants[participant_id] = ParticipantState()
        return self.participants[participant_id]

    def process_and_decide(self, participant_id: str, new_probs: List[float]) -> Optional[Tuple[int, float]]:
        """
        Processes new probabilities and decides if a banner should be sent.

        Args:
            participant_id: The unique ID of the participant.
            new_probs: A list of new fake probabilities from the latest frame batch.

        Returns:
            A tuple of (verdict, confidence_score) if a response should be sent, otherwise None.
        """
        if not new_probs:
            return None

        with self.lock:
            # 1. Periodically clean up old participants to prevent memory leaks
            self._cleanup_inactive_participants()

            # 2. Get the current state for the participant
            state = self._get_or_create_state(participant_id)

            # 3. Update history: add new probabilities to the front of the deque
            state.history.extendleft(reversed(new_probs))

            # 4. Calculate a new verdict based on the active window
            active_window_probs = [state.history[i] for i in range(min(len(state.history), ACTIVE_TEST_WINDOW))]
            mean_prob = float(np.mean(active_window_probs)) if active_window_probs else DEFAULT_START_PROB
            new_verdict = self._calculate_band_level(mean_prob)

            # 5. Update state counters
            state.batch_counter += 1
            state.last_seen_ts = time.time()

            # 6. Decide if a response should be triggered
            verdict_changed = new_verdict != state.current_verdict
            interval_reached = state.batch_counter >= MIN_RESPONSE_INTERVAL

            print(
                f"[ParticipantManager] ID: {participant_id}, MeanProb: {mean_prob:.3f}, NewVerdict: {pb2.BannerLevel.Name(new_verdict)}, "
                f"OldVerdict: {pb2.BannerLevel.Name(state.current_verdict)}, Changed: {verdict_changed}, "
                f"Counter: {state.batch_counter}/{MIN_RESPONSE_INTERVAL}")

            if verdict_changed or interval_reached:
                print(
                    f"[ParticipantManager] TRIGGER! Sending verdict for {participant_id}. Reason: {'Change' if verdict_changed else 'Interval'}.")
                state.current_verdict = new_verdict
                state.batch_counter = 0
                # Return both the verdict and the mean probability (confidence score)
                return new_verdict, mean_prob
            else:
                # Suppress the response, as the verdict is stable and the interval is not met
                return None

    def reset_all(self):
        """Clears the state of all participants."""
        with self.lock:
            if self.participants:
                print(f"[ParticipantManager] RESETTING ALL {len(self.participants)} PARTICIPANT STATES.")
                self.participants.clear()
            else:
                print("[ParticipantManager] Reset requested, but no active participants.")
