"""
Queue Manager for WMA Streaming Server.

Implements per-participant frame queues and audio batch queues
to enable parallel processing with guaranteed data freshness.
"""

import asyncio
import time
import logging
from collections import deque
from typing import Optional, Dict, Any, List


class ParticipantFrameQueue:
    """
    Bounded queue that maintains the most recent N frames for a participant.
    
    Automatically drops oldest frames when full to guarantee freshness.
    Queue size equals batch size to ensure full batches are always fresh.
    """
    
    def __init__(self, participant_id: str, max_size: int = 32):
        """
        Initialize participant frame queue.
        
        Args:
            participant_id: Unique identifier for the participant
            max_size: Maximum frames to keep (equals batch size)
                     Default 32 frames = ~3.2 seconds at 10 fps
        """
        self.participant_id = participant_id
        self.frames = deque(maxlen=max_size)  # Auto-drops oldest when full
        self.lock = asyncio.Lock()
        self.new_frame_event = asyncio.Event()
        
        # Metrics
        self.total_received = 0
        self.total_dropped = 0
        self.created_at = time.time()
        self.last_frame_at = None
    
    async def add_frame(self, frame_data: bytes, metadata: Dict[str, Any]):
        """
        Add frame to queue, automatically dropping oldest if full.
        
        Args:
            frame_data: Raw frame bytes (JPEG encoded)
            metadata: Frame metadata (participant_id, timestamp, sequence, etc.)
        """
        async with self.lock:
            # Track if we're dropping a frame
            if len(self.frames) >= self.frames.maxlen:
                self.total_dropped += 1
                dropped_frame = self.frames[0]  # The one about to be evicted
                logging.debug(
                    f"[Queue {self.participant_id}] Dropping stale frame "
                    f"seq={dropped_frame.get('sequence', '?')} "
                    f"age={time.time() - dropped_frame['timestamp']:.2f}s"
                )
            
            # Add new frame
            self.frames.append({
                'data': frame_data,
                'metadata': metadata,
                'timestamp': time.time(),
                'sequence': metadata.get('sequence_number', -1)
            })
            
            self.total_received += 1
            self.last_frame_at = time.time()
            self.new_frame_event.set()
    
    async def get_full_batch(self) -> List[Dict[str, Any]]:
        """
        Get all frames currently in the queue and clear it.
        
        Returns the freshest N frames where N = current queue size.
        After extraction, queue is cleared for next batch accumulation.
        
        Returns:
            List of frame dictionaries with 'data', 'metadata', 'timestamp', 'sequence'
        """
        async with self.lock:
            if len(self.frames) == 0:
                return []
            
            # Extract all frames (they're already the freshest N)
            batch = list(self.frames)
            
            # Clear queue for next batch
            self.frames.clear()
            self.new_frame_event.clear()
            
            logging.debug(
                f"[Queue {self.participant_id}] Extracted batch of {len(batch)} frames, "
                f"queue cleared"
            )
            
            return batch
    
    async def wait_for_batch(self, min_size: int = 16, timeout: float = 2.0) -> bool:
        """
        Wait until queue has at least min_size frames, or timeout expires.
        
        Args:
            min_size: Minimum frames needed for a batch (default 16)
            timeout: Max seconds to wait (default 2.0)
        
        Returns:
            True if batch is ready (>= min_size frames), False if timeout
        """
        try:
            # Use asyncio.timeout for Python 3.11+, or asyncio.wait_for for older
            async with asyncio.timeout(timeout):
                while len(self.frames) < min_size:
                    self.new_frame_event.clear()
                    await self.new_frame_event.wait()
                return True
        except (asyncio.TimeoutError, TimeoutError):
            # Timeout expired, check if we have any frames
            return len(self.frames) > 0
    
    def get_size(self) -> int:
        """Get current number of frames in queue (thread-safe read)."""
        return len(self.frames)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics for monitoring.
        
        Returns:
            Dictionary with queue metrics
        """
        return {
            'participant_id': self.participant_id,
            'current_size': len(self.frames),
            'max_size': self.frames.maxlen,
            'total_received': self.total_received,
            'total_dropped': self.total_dropped,
            'drop_rate': (
                self.total_dropped / self.total_received 
                if self.total_received > 0 else 0
            ),
            'age_seconds': time.time() - self.created_at,
            'last_frame_age': (
                time.time() - self.last_frame_at 
                if self.last_frame_at else None
            )
        }
    
    def __repr__(self) -> str:
        return (
            f"ParticipantFrameQueue(id={self.participant_id}, "
            f"size={len(self.frames)}/{self.frames.maxlen}, "
            f"received={self.total_received}, dropped={self.total_dropped})"
        )


class AudioBatchQueue:
    """
    Global queue for audio batches (audio processing is not per-participant).
    
    Maintains recent audio batches with automatic staleness eviction.
    """
    
    def __init__(self, max_size: int = 50):
        """
        Initialize audio batch queue.
        
        Args:
            max_size: Maximum number of audio batches to keep
        """
        self.batches = deque(maxlen=max_size)
        self.lock = asyncio.Lock()
        self.new_batch_event = asyncio.Event()
        
        # Metrics
        self.total_received = 0
        self.total_dropped = 0
        self.created_at = time.time()
        self.last_batch_at = None
    
    async def add_batch(self, audio_data: bytes, metadata: Dict[str, Any]):
        """
        Add audio batch to queue.
        
        Args:
            audio_data: Raw audio batch bytes
            metadata: Audio metadata (timestamp, format, etc.)
        """
        async with self.lock:
            # Track drops
            if len(self.batches) >= self.batches.maxlen:
                self.total_dropped += 1
            
            self.batches.append({
                'data': audio_data,
                'metadata': metadata,
                'timestamp': time.time()
            })
            
            self.total_received += 1
            self.last_batch_at = time.time()
            self.new_batch_event.set()
    
    async def get_batch(self) -> Optional[Dict[str, Any]]:
        """
        Get oldest audio batch from queue (FIFO).
        
        Returns:
            Audio batch dictionary or None if queue is empty
        """
        async with self.lock:
            if len(self.batches) == 0:
                return None
            
            batch = self.batches.popleft()
            
            if len(self.batches) == 0:
                self.new_batch_event.clear()
            
            return batch
    
    async def wait_for_batch(self, timeout: float = 1.0) -> bool:
        """
        Wait until at least one batch is available, or timeout expires.
        
        Args:
            timeout: Max seconds to wait
        
        Returns:
            True if batch is available, False if timeout
        """
        try:
            async with asyncio.timeout(timeout):
                while len(self.batches) == 0:
                    self.new_batch_event.clear()
                    await self.new_batch_event.wait()
                return True
        except (asyncio.TimeoutError, TimeoutError):
            return len(self.batches) > 0
    
    def get_size(self) -> int:
        """Get current number of batches in queue."""
        return len(self.batches)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics for monitoring."""
        return {
            'current_size': len(self.batches),
            'max_size': self.batches.maxlen,
            'total_received': self.total_received,
            'total_dropped': self.total_dropped,
            'drop_rate': (
                self.total_dropped / self.total_received 
                if self.total_received > 0 else 0
            ),
            'age_seconds': time.time() - self.created_at,
            'last_batch_age': (
                time.time() - self.last_batch_at 
                if self.last_batch_at else None
            )
        }
    
    def __repr__(self) -> str:
        return (
            f"AudioBatchQueue(size={len(self.batches)}/{self.batches.maxlen}, "
            f"received={self.total_received}, dropped={self.total_dropped})"
        )
