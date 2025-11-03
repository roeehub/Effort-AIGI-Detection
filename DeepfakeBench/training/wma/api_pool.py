"""
API Pool Manager for WMA Streaming Server.

Implements client-side load balancing across multiple API servers
with health tracking and automatic failover.
"""

import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Any, Optional


class VideoAPIPool:
    """
    Manages a pool of video inference API servers with health tracking
    and intelligent load distribution.
    
    Features:
    - Round-robin load balancing
    - Health tracking and automatic failover
    - Connection pooling for efficiency
    - Retry logic with exponential backoff
    """
    
    def __init__(self, api_urls: List[str], timeout: float = 5.0, 
                 threshold: float = 0.75, yolo_conf_threshold: float = 0.80):
        """
        Initialize API pool.
        
        Args:
            api_urls: List of API endpoints, e.g.:
                ['http://server1:8999/check_frame_batch',
                 'http://server2:8999/check_frame_batch']
            timeout: Request timeout in seconds (default 5.0)
            threshold: Inference threshold for the model (default 0.75)
            yolo_conf_threshold: YOLO confidence threshold (default 0.80)
        """
        if not api_urls:
            raise ValueError("At least one API URL must be provided")
        
        self.api_urls = api_urls
        self.timeout = timeout
        self.threshold = threshold
        self.yolo_conf_threshold = yolo_conf_threshold
        self.current_index = 0
        self.lock = asyncio.Lock()
        
        # Health tracking for each API
        self.api_health = {url: {
            'available': True,
            'error_count': 0,
            'last_error': None,
            'last_error_time': None,
            'total_requests': 0,
            'total_errors': 0,
            'avg_response_time': 0.0,
            'last_success_time': None
        } for url in api_urls}
        
        # HTTP session with connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        
        logging.info(
            f"[VideoAPIPool] Initialized with {len(api_urls)} API server(s): "
            f"{', '.join(api_urls)}"
        )
        logging.info(
            f"[VideoAPIPool] API Parameters: model_type=custom, "
            f"threshold={threshold}, yolo_conf_threshold={yolo_conf_threshold}"
        )
    
    async def initialize(self):
        """Initialize HTTP session (call this after event loop is running)."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(
                    limit_per_host=10,  # Max 10 concurrent connections per host
                    force_close=False,  # Keep connections alive
                    enable_cleanup_closed=True
                )
            )
            logging.info("[VideoAPIPool] HTTP session initialized")
    
    async def get_next_available_url(self) -> str:
        """
        Get next API URL using round-robin among healthy servers.
        Falls back to unhealthy servers if all are down.
        
        Returns:
            API URL to use for next request
        """
        async with self.lock:
            # Try to find a healthy server
            attempts = 0
            while attempts < len(self.api_urls):
                url = self.api_urls[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.api_urls)
                
                if self.api_health[url]['available']:
                    return url
                
                attempts += 1
            
            # All servers marked unhealthy, return next one anyway
            # (maybe they recovered, worth trying)
            url = self.api_urls[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_urls)
            
            logging.warning(
                f"[VideoAPIPool] All API servers unhealthy, trying {url} anyway"
            )
            return url
    
    async def infer_batch(self, api_url: str, image_bytes_list: List[bytes]) -> Dict[str, Any]:
        """
        Send batch of frames to specified API server.
        
        Args:
            api_url: The API endpoint to use
            image_bytes_list: List of 16-32 JPEG-encoded frames
        
        Returns:
            API response with inference results
            
        Raises:
            Exception: If API call fails after retries
        """
        if self.session is None:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare multipart form data
            # NOTE: API expects all frames under 'files' field name (not unique names)
            data = aiohttp.FormData()
            for idx, img_bytes in enumerate(image_bytes_list):
                data.add_field(
                    'files',  # API expects 'files' for all frames
                    img_bytes,
                    filename=f'frame_{idx}.jpg',
                    content_type='image/jpeg'
                )
            
            # CRITICAL: API parameters for custom model
            params = {
                'model_type': 'custom',
                'threshold': str(self.threshold),
                'debug': 'false',
                'yolo_conf_threshold': str(self.yolo_conf_threshold)
            }
            
            # Make request with parameters
            async with self.session.post(api_url, data=data, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"API returned status {response.status}: {error_text[:200]}"
                    )
                
                result = await response.json()
                
                # Update health metrics
                elapsed = time.time() - start_time
                await self._update_health_success(api_url, elapsed)
                
                logging.debug(
                    f"[VideoAPIPool] {api_url} processed {len(image_bytes_list)} frames "
                    f"in {elapsed:.2f}s"
                )
                
                return result
        
        except Exception as e:
            elapsed = time.time() - start_time
            
            # Update health metrics
            await self._update_health_error(api_url, str(e))
            
            logging.error(
                f"[VideoAPIPool] {api_url} failed after {elapsed:.2f}s: {e}"
            )
            raise
    
    async def _update_health_success(self, api_url: str, response_time: float):
        """Record successful API call."""
        async with self.lock:
            health = self.api_health[api_url]
            health['total_requests'] += 1
            health['error_count'] = 0  # Reset consecutive error count
            health['available'] = True
            health['last_success_time'] = time.time()
            
            # Update rolling average response time (exponential moving average)
            alpha = 0.2  # Smoothing factor (0 = no change, 1 = replace entirely)
            if health['avg_response_time'] == 0:
                health['avg_response_time'] = response_time
            else:
                health['avg_response_time'] = (
                    alpha * response_time + 
                    (1 - alpha) * health['avg_response_time']
                )
    
    async def _update_health_error(self, api_url: str, error: str):
        """Record failed API call and potentially mark server as unhealthy."""
        async with self.lock:
            health = self.api_health[api_url]
            health['total_requests'] += 1
            health['total_errors'] += 1
            health['error_count'] += 1
            health['last_error'] = error
            health['last_error_time'] = time.time()
            
            # Mark as unavailable after 3 consecutive errors
            if health['error_count'] >= 3:
                health['available'] = False
                logging.warning(
                    f"[VideoAPIPool] API {api_url} marked UNHEALTHY after "
                    f"{health['error_count']} consecutive errors. Last error: {error[:100]}"
                )
    
    async def mark_api_error(self, api_url: str):
        """External method for workers to report errors."""
        await self._update_health_error(api_url, "Worker reported error")
    
    def get_health_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current health statistics for all API servers.
        
        Returns:
            Dictionary mapping API URL to health stats
        """
        return {
            url: {
                'available': health['available'],
                'total_requests': health['total_requests'],
                'total_errors': health['total_errors'],
                'consecutive_errors': health['error_count'],
                'error_rate': (
                    health['total_errors'] / health['total_requests']
                    if health['total_requests'] > 0 else 0
                ),
                'avg_response_time_ms': health['avg_response_time'] * 1000,
                'last_error': health['last_error'],
                'last_error_time': health['last_error_time'],
                'last_success_time': health['last_success_time']
            }
            for url, health in self.api_health.items()
        }
    
    async def close(self):
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logging.info("[VideoAPIPool] HTTP session closed")
    
    def __repr__(self) -> str:
        healthy = sum(1 for h in self.api_health.values() if h['available'])
        return (
            f"VideoAPIPool({len(self.api_urls)} servers, "
            f"{healthy} healthy)"
        )
