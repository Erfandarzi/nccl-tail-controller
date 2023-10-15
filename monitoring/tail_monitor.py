import time
import threading
from typing import List, Optional, Dict
from collections import deque

from ..algorithms.tdigest import TDigest


class TailMonitor:
    
    def __init__(self, window_size: int = 48, quantiles: List[float] = None):
        self.window_size = window_size
        self.quantiles = quantiles or [0.99, 0.999, 0.9999]
        
        self.latencies = deque(maxlen=window_size)
        self.tdigest = TDigest(delta=0.01)
        self.sample_count = 0
        
        self._lock = threading.Lock()
        self._last_update = time.time()
        
        # Stability tracking
        self._stable_windows = 0
        self._stability_threshold = 5
        
    def record_latency(self, latency: float) -> None:
        with self._lock:
            self.latencies.append(latency)
            self.tdigest.update(latency)
            self.sample_count += 1
            self._last_update = time.time()
            
            self._update_stability_metrics()
    
    def get_percentile(self, percentile: float) -> float:
        with self._lock:
            if len(self.latencies) == 0:
                return 0.0
            
            return self.tdigest.quantile(percentile)
    
    def get_all_percentiles(self) -> Dict[str, float]:
        with self._lock:
            result = {}
            for q in self.quantiles:
                result[f'p{int(q*10000)}'] = self.tdigest.quantile(q)
            return result
    
    def is_stable(self) -> bool:
        with self._lock:
            return self._stable_windows >= self._stability_threshold
    
    def get_sample_count(self) -> int:
        return self.sample_count
    
    def get_window_stats(self) -> Dict:
        with self._lock:
            if len(self.latencies) == 0:
                return {
                    'mean': 0.0, 'min': 0.0, 'max': 0.0, 
                    'count': 0, 'variance': 0.0
                }
            
            latencies_list = list(self.latencies)
            mean_lat = sum(latencies_list) / len(latencies_list)
            variance = sum((x - mean_lat) ** 2 for x in latencies_list) / len(latencies_list)
            
            return {
                'mean': mean_lat,
                'min': min(latencies_list),
                'max': max(latencies_list),
                'count': len(latencies_list),
                'variance': variance
            }
    
    def _update_stability_metrics(self):
        if len(self.latencies) < self.window_size:
            return
        
        current_p99 = self.tdigest.quantile(0.99)
        
        # Simple stability check: p99 hasn't changed dramatically
        if hasattr(self, '_last_p99'):
            change_ratio = abs(current_p99 - self._last_p99) / max(self._last_p99, 1e-6)
            if change_ratio < 0.1:  # Less than 10% change
                self._stable_windows += 1
            else:
                self._stable_windows = 0
        
        self._last_p99 = current_p99
    
    def reset_window(self):
        with self._lock:
            self.latencies.clear()
            self.tdigest = TDigest(delta=0.01)
            self._stable_windows = 0