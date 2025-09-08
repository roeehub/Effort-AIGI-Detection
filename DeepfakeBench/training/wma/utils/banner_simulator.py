"""
Banner simulation logic for stub backend.

Randomly generates banner actions for testing purposes.
Simulates realistic backend behavior for validating Service 5 communication.
"""

import random
import time
import uuid
from typing import Optional
from wma_streaming_pb2 import ScreenBanner, BannerLevel


class BannerSimulator:
    """Simulates banner generation for testing backend responses."""
    
    def __init__(self, banner_probability: float = 0.1):
        """
        Initialize banner simulator.
        
        Args:
            banner_probability: Probability of generating a banner (0.0-1.0)
        """
        self.banner_probability = banner_probability
        self.last_banner_time = 0
        self.min_banner_interval_ms = 5000  # Minimum 5 seconds between banners
        
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
    
    def get_banner_stats(self) -> dict:
        """
        Get banner simulation statistics.
        
        Returns:
            Dictionary with simulation stats
        """
        return {
            "banner_probability": self.banner_probability,
            "min_interval_ms": self.min_banner_interval_ms,
            "last_banner_time": self.last_banner_time,
            "level_distribution": {
                BannerLevel.Name(level): weight 
                for level, weight in self.banner_levels
            }
        }