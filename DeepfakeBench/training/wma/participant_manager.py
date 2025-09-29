# participant_manager.py

"""
Manages stateful analysis for individual participants to provide stable verdicts.
Also manages audio sliding window for stable audio verdicts.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

# Import protobuf enum values for banner levels
import wma_streaming_pb2 as pb2

# --- Configuration Constants ---
HISTORY_WINDOW_SIZE = 96  # Total number of probabilities to keep per participant
ACTIVE_TEST_WINDOW = 48  # Number of recent probabilities to use for verdict calculation
DEFAULT_START_PROB = 0.2  # Assumed probability for a new, unseen participant
MIN_RESPONSE_INTERVAL = 4  # Send a response every N batches even if verdict is stable
RESET_AFTER_INACTIVE_MIN = 2.0  # Forget a participant after this many minutes of inactivity


# --- Data Structure for Participant State ---
# Add 'is_new' flag to ParticipantState
@dataclass
class ParticipantState:
    """Holds the state for a single participant."""
    history: deque = field(default_factory=lambda: deque(
        [DEFAULT_START_PROB] * HISTORY_WINDOW_SIZE, maxlen=HISTORY_WINDOW_SIZE
    ))
    current_verdict: int = pb2.GREEN
    batch_counter: int = 0
    last_seen_ts: float = field(default_factory=time.time)
    is_new: bool = True  # Flag to track if this participant is new


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
            is_new_participant = state.is_new

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
                f"Counter: {state.batch_counter}/{MIN_RESPONSE_INTERVAL}, IsNew: {is_new_participant}")

            # Always send a banner for new participants
            if is_new_participant or verdict_changed or interval_reached:
                reason = "New" if is_new_participant else "Change" if verdict_changed else "Interval"
                print(
                    f"[ParticipantManager] TRIGGER! Sending verdict for {participant_id}. Reason: {reason}.")
                state.current_verdict = new_verdict
                state.batch_counter = 0
                state.is_new = False  # Mark participant as no longer new
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


# --- Audio Sliding Window Manager ---
class AudioWindowManager:
    """
    Manages a sliding window of audio analysis results to provide stable audio verdicts.
    """
    
    AUDIO_WINDOW_SIZE = 5  # Keep track of 5 audio datapoints
    RED_THRESHOLD = 4  # 4/5 red datapoints trigger red banner
    
    def __init__(self):
        """Initialize the audio window manager."""
        self.lock = threading.Lock()
        # Start with 5 green datapoints (prediction=True means real/green)
        self.audio_window = deque([True] * self.AUDIO_WINDOW_SIZE, maxlen=self.AUDIO_WINDOW_SIZE)
        print(f"[AudioWindowManager] Initialized with {self.AUDIO_WINDOW_SIZE} green datapoints")
    
    def process_audio_result(self, api_result: Dict) -> Optional[int]:
        """
        Process an audio API result and decide if a banner should be sent.
        
        Args:
            api_result: Dictionary containing ASV API response with 'probs' and 'prediction' keys
            
        Returns:
            Banner level (pb2.GREEN or pb2.RED) if a verdict change occurred, None otherwise
        """
        if not api_result or 'probs' not in api_result or 'prediction' not in api_result:
            return None
            
        probs = api_result['probs']
        if not probs or not isinstance(probs, list):
            return None
            
        # Check if audio is silent (negative probability)
        prob_value = probs[0]  # First (and usually only) probability value
        if prob_value < 0:
            print(f"[AudioWindowManager] Ignoring silent audio with prob={prob_value:.3f}")
            return None
        
        # Get the current prediction (True=real/green, False=fake/red)
        current_prediction = api_result['prediction']
        
        with self.lock:
            # Remember the previous verdict
            previous_verdict = self._calculate_verdict()
            
            # Add new prediction to the sliding window
            self.audio_window.append(current_prediction)
            
            # Calculate new verdict
            new_verdict = self._calculate_verdict()
            
            print(f"!******** WINDOW ********!")
            print(f"[AudioWindowManager] Audio prob={prob_value:.3f}, prediction={current_prediction}, "
                  f"window={list(self.audio_window)}, verdict={pb2.BannerLevel.Name(new_verdict)}")
            
            # Return verdict only if it changed
            if new_verdict != previous_verdict:
                print(f"!******** AUDIO VERDICT CHANGE - WILL SEND RESPONSE ********!")
                print(f"[AudioWindowManager] VERDICT CHANGED: {pb2.BannerLevel.Name(previous_verdict)} -> {pb2.BannerLevel.Name(new_verdict)}")
                print(f"!**********************************************************!")
                return new_verdict
            
            return None
    
    def _calculate_verdict(self) -> int:
        """
        Calculate the current verdict based on the sliding window.
        
        Returns:
            pb2.GREEN or pb2.RED based on the 4/5 threshold
        """
        # Count red predictions (False means fake/red)
        red_count = sum(1 for prediction in self.audio_window if not prediction)
        
        # If 4 or more out of 5 are red, return red verdict
        if red_count >= self.RED_THRESHOLD:
            return pb2.RED
        else:
            return pb2.GREEN
    
    def reset(self):
        """Reset the audio window to all green datapoints."""
        with self.lock:
            self.audio_window.clear()
            self.audio_window.extend([True] * self.AUDIO_WINDOW_SIZE)
            print(f"[AudioWindowManager] Reset to {self.AUDIO_WINDOW_SIZE} green datapoints")
