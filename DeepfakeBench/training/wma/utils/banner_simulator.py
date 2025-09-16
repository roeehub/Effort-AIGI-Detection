"""
Banner simulation logic for stub backend.

Randomly generates banner actions for testing purposes.
Simulates realistic backend behavior for validating Service 5 communication.
"""

import random
import time
import uuid
from typing import Optional, List
from wma_streaming_pb2 import ScreenBanner, BannerLevel, BannerScope


class BannerSimulator:
    """Simulates banner generation for testing backend responses."""
    
    def __init__(self, banner_probability: float = 0.1, per_person_probability: float = 0.2):
        """
        Initialize banner simulator.
        
        Args:
            banner_probability: Probability of generating a global banner (0.0-1.0)
            per_person_probability: Probability of generating per-person banner for each participant (0.0-1.0)
        """
        self.banner_probability = banner_probability
        self.per_person_probability = per_person_probability
        self.last_banner_time = 0
        self.last_per_person_banner_times = {}  # Track per-participant banner times
        self.min_banner_interval_ms = 1000  # Minimum 1 second between banners FOR TESTING
        self.min_per_person_interval_ms = 2000  # Minimum 2 seconds between per-person banners for same participant FOR TESTING
        
        # Banner level distribution (weighted probabilities)
        self.banner_levels = [
            (BannerLevel.GREEN, 0.4),    # 40% green
            (BannerLevel.YELLOW, 0.35),  # 35% yellow  
            (BannerLevel.RED, 0.25)      # 25% red
        ]
        
        # TTL ranges for different levels (in milliseconds)
        self.ttl_ranges = {
            BannerLevel.GREEN: (1500, 3000),   # 1.5-3 seconds
            BannerLevel.YELLOW: (2000, 4000),  # 2-4 seconds
            BannerLevel.RED: (2500, 5000)      # 2.5-5 seconds
        }
        
        # Banner types for image selection
        self.banner_types = [
            "warning",    # Warning icon
            "alert",      # Alert icon
            "info",       # Information icon
            "attention",  # Attention icon
            "caution"     # Caution icon
        ]
    
    def should_generate_banner(self) -> bool:
        """
        Determine if a banner should be generated based on probability and timing.
        
        Returns:
            True if a banner should be generated
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Check minimum interval
        if current_time - self.last_banner_time < self.min_banner_interval_ms:
            return False
        
        # Check probability
        return random.random() < self.banner_probability
    
    def generate_banner(self) -> Optional[ScreenBanner]:
        """
        Generate a random banner if conditions are met.
        
        Returns:
            ScreenBanner message or None if no banner should be generated
        """
        if not self.should_generate_banner():
            return None
        
        # Update last banner time
        self.last_banner_time = time.time() * 1000
        
        # Select random banner level based on weighted distribution
        level = self._select_weighted_level()
        
        # Generate TTL based on level
        ttl_ms = random.randint(*self.ttl_ranges[level])
        
        # Create banner message
        banner = ScreenBanner()
        banner.level = level
        banner.ttl_ms = ttl_ms
        banner.placement = "TopCenter"  # Default placement as specified
        banner.action_id = f"act-{uuid.uuid4().hex[:8]}"
        banner.scope = "global"
        
        return banner
    
    def _select_weighted_level(self) -> BannerLevel:
        """
        Select a banner level using weighted random selection.
        
        Returns:
            Selected BannerLevel
        """
        # Create cumulative weights
        cumulative_weights = []
        total_weight = 0
        
        for level, weight in self.banner_levels:
            total_weight += weight
            cumulative_weights.append((level, total_weight))
        
        # Select random point
        random_point = random.random() * total_weight
        
        # Find corresponding level
        for level, cumulative_weight in cumulative_weights:
            if random_point <= cumulative_weight:
                return level
        
        # Fallback to GREEN
        return BannerLevel.GREEN
    
    def force_banner(self, level: BannerLevel, ttl_ms: Optional[int] = None) -> ScreenBanner:
        """
        Force generation of a specific banner for testing.
        
        Args:
            level: Banner level to generate
            ttl_ms: Custom TTL, or None for random TTL based on level
            
        Returns:
            Generated ScreenBanner
        """
        if ttl_ms is None:
            ttl_ms = random.randint(*self.ttl_ranges[level])
        
        banner = ScreenBanner()
        banner.level = level
        banner.ttl_ms = ttl_ms
        banner.placement = "TopCenter"
        banner.action_id = f"act-{uuid.uuid4().hex[:8]}"
        banner.scope = "global"
        
        return banner
    
    def should_generate_per_person_banner(self, participant_id: str) -> bool:
        """
        Determine if a per-person banner should be generated for a participant.
        
        Args:
            participant_id: Participant identifier
            
        Returns:
            True if a per-person banner should be generated
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Check minimum interval for this participant
        if participant_id in self.last_per_person_banner_times:
            last_time = self.last_per_person_banner_times[participant_id]
            if current_time - last_time < self.min_per_person_interval_ms:
                return False
        
        # Check probability
        return random.random() < self.per_person_probability
    
    def generate_per_person_banner(self, participant_id: str) -> Optional[ScreenBanner]:
        """
        Generate a per-person banner for a specific participant.
        
        Args:
            participant_id: Participant identifier
            
        Returns:
            ScreenBanner message or None if no banner should be generated
        """
        if not self.should_generate_per_person_banner(participant_id):
            return None
        
        # Update last banner time for this participant
        self.last_per_person_banner_times[participant_id] = time.time() * 1000
        
        # Select random banner level based on weighted distribution
        level = self._select_weighted_level()
        
        # Generate TTL based on level
        ttl_ms = random.randint(*self.ttl_ranges[level])
        
        # Select random banner type for image
        banner_type = random.choice(self.banner_types)
        
        # Calculate expiry timestamp
        current_time_ms = int(time.time() * 1000)
        expiry_timestamp_ms = current_time_ms + ttl_ms
        
        # Create banner message
        banner = ScreenBanner()
        banner.level = level
        banner.ttl_ms = ttl_ms
        banner.placement = "TopRight"  # Per-person banners at top-right of participant
        banner.action_id = f"act-{uuid.uuid4().hex[:8]}"
        banner.scope = "participant"
        banner.scope_enum = BannerScope.SCOPE_PARTICIPANT
        banner.participant_id = participant_id
        banner.banner_type = banner_type
        banner.expiry_timestamp_ms = expiry_timestamp_ms
        
        return banner
    
    def generate_participant_banners(self, participant_ids: List[str]) -> List[ScreenBanner]:
        """
        Generate per-person banners for a list of participants.
        
        Args:
            participant_ids: List of participant identifiers
            
        Returns:
            List of generated ScreenBanner messages
        """
        banners = []
        for participant_id in participant_ids:
            banner = self.generate_per_person_banner(participant_id)
            if banner:
                banners.append(banner)
        return banners
    
    def get_banner_stats(self) -> dict:
        """
        Get banner simulation statistics.
        
        Returns:
            Dictionary with simulation stats
        """
        return {
            "banner_probability": self.banner_probability,
            "per_person_probability": self.per_person_probability,
            "min_interval_ms": self.min_banner_interval_ms,
            "min_per_person_interval_ms": self.min_per_person_interval_ms,
            "last_banner_time": self.last_banner_time,
            "per_person_banner_count": len(self.last_per_person_banner_times),
            "level_distribution": {
                BannerLevel.Name(level): weight 
                for level, weight in self.banner_levels
            }
        }