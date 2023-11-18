import time
import threading
from typing import Optional

from ..hardware.bandwidth_estimator import BandwidthEstimator


class TokenBucket:
    
    def __init__(self, min_delay: float = 10e-6, max_delay: float = 200e-6, 
                 target_bandwidth_ratio: float = 0.85):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.target_bandwidth_ratio = target_bandwidth_ratio
        
        # Token bucket parameters
        self.bucket_size = 1024 * 1024  # 1MB burst capacity
        self.tokens = self.bucket_size
        self.last_refill = time.time()
        
        # Rate limiting state
        self.enabled = False
        self.current_rate = 0.0  # tokens per second
        self.adaptive_rate = True
        
        # Hardware profiling
        self.bandwidth_estimator = BandwidthEstimator()
        
        self._lock = threading.Lock()
        self._update_rate()
    
    def enable_pacing(self) -> None:
        with self._lock:
            self.enabled = True
            self._update_rate()
    
    def disable_pacing(self) -> None:
        with self._lock:
            self.enabled = False
    
    def acquire_tokens(self, chunk_size: int) -> float:
        if not self.enabled:
            return 0.0
        
        with self._lock:
            self._refill_tokens()
            
            if self.tokens >= chunk_size:
                self.tokens -= chunk_size
                return 0.0  # No delay needed
            
            # Calculate delay needed
            tokens_needed = chunk_size - self.tokens
            delay = tokens_needed / self.current_rate
            
            # Clamp delay to bounds
            delay = max(self.min_delay, min(delay, self.max_delay))
            
            # Simulate token consumption after delay
            future_tokens = self.tokens + (delay * self.current_rate)
            self.tokens = max(0, future_tokens - chunk_size)
            
            return delay
    
    def pace_submission(self, chunk_size: int) -> None:
        delay = self.acquire_tokens(chunk_size)
        if delay > 0:
            time.sleep(delay)
    
    def get_current_rate(self) -> float:
        return self.current_rate
    
    def get_effective_bandwidth(self) -> float:
        link_bandwidth = self.bandwidth_estimator.get_link_bandwidth()
        return link_bandwidth * self.target_bandwidth_ratio
    
    def _refill_tokens(self):
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            new_tokens = elapsed * self.current_rate
            self.tokens = min(self.bucket_size, self.tokens + new_tokens)
            self.last_refill = now
    
    def _update_rate(self):
        # Set rate based on target bandwidth utilization
        link_bw = self.bandwidth_estimator.get_link_bandwidth()
        self.current_rate = link_bw * self.target_bandwidth_ratio
    
    def update_network_conditions(self):
        with self._lock:
            self.bandwidth_estimator.update_measurements()
            self._update_rate()
    
    def get_stats(self) -> dict:
        with self._lock:
            return {
                'enabled': self.enabled,
                'current_rate_gbps': self.current_rate / (1e9),
                'tokens_available': self.tokens,
                'bucket_utilization': 1.0 - (self.tokens / self.bucket_size),
                'target_bandwidth_ratio': self.target_bandwidth_ratio
            }