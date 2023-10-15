import random
import time
import threading
from typing import Dict, List, Optional

try:
    import pycuda.driver as cuda
    from pycuda import gpuarray
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class CudaStagger:
    
    def __init__(self, max_stagger_us: int = 100):
        self.max_stagger_us = max_stagger_us
        self.enabled = False
        
        # CUDA event management
        self.cuda_available = CUDA_AVAILABLE
        self.events: Dict[int, 'cuda.Event'] = {}
        
        # Fallback timing for systems without CUDA
        self.fallback_offsets: Dict[int, float] = {}
        
        self._lock = threading.Lock()
        
        if self.cuda_available:
            self._initialize_cuda()
        
        # Generate per-peer phase offsets
        self._generate_peer_offsets()
    
    def enable_staggering(self) -> None:
        with self._lock:
            self.enabled = True
    
    def disable_staggering(self) -> None:
        with self._lock:
            self.enabled = False
    
    def apply_stagger(self, peer_id: int = 0) -> None:
        if not self.enabled:
            return
        
        offset = self._get_peer_offset(peer_id)
        
        if self.cuda_available and peer_id in self.events:
            self._cuda_stagger(peer_id, offset)
        else:
            self._fallback_stagger(offset)
    
    def _cuda_stagger(self, peer_id: int, offset_us: float) -> None:
        try:
            # Use CUDA events to create precise timing delays
            event = self.events.get(peer_id)
            if event is None:
                event = cuda.Event()
                self.events[peer_id] = event
            
            # Record event and create delay
            event.record()
            
            # Busy wait for the offset duration
            start = time.perf_counter()
            while (time.perf_counter() - start) < (offset_us / 1e6):
                cuda.Context.synchronize()
                
        except Exception:
            # Fall back to sleep-based staggering
            self._fallback_stagger(offset_us)
    
    def _fallback_stagger(self, offset_us: float) -> None:
        # High-precision sleep using busy wait for microsecond delays
        if offset_us > 1000:  # > 1ms, use regular sleep
            time.sleep(offset_us / 1e6)
        else:
            # Busy wait for sub-millisecond precision
            start = time.perf_counter()
            while (time.perf_counter() - start) < (offset_us / 1e6):
                pass
    
    def _get_peer_offset(self, peer_id: int) -> float:
        if peer_id not in self.fallback_offsets:
            self.fallback_offsets[peer_id] = random.uniform(0, self.max_stagger_us)
        
        return self.fallback_offsets[peer_id]
    
    def _initialize_cuda(self):
        if not self.cuda_available:
            return
        
        try:
            cuda.init()
            # We don't create a context here as it should be managed by the application
        except Exception:
            self.cuda_available = False
    
    def _generate_peer_offsets(self, num_peers: int = 16):
        # Pre-generate offsets for common peer counts
        for i in range(num_peers):
            self.fallback_offsets[i] = random.uniform(0, self.max_stagger_us)
    
    def set_peer_offset(self, peer_id: int, offset_us: float):
        with self._lock:
            self.fallback_offsets[peer_id] = min(offset_us, self.max_stagger_us)
    
    def get_peer_offset(self, peer_id: int) -> float:
        return self.fallback_offsets.get(peer_id, 0.0)
    
    def reset_peer_offsets(self):
        with self._lock:
            self.fallback_offsets.clear()
            self._generate_peer_offsets()
    
    def get_stats(self) -> Dict:
        return {
            'enabled': self.enabled,
            'cuda_available': self.cuda_available,
            'max_stagger_us': self.max_stagger_us,
            'active_peers': len(self.fallback_offsets),
            'average_offset_us': sum(self.fallback_offsets.values()) / len(self.fallback_offsets) if self.fallback_offsets else 0
        }
    
    def cleanup(self):
        if self.cuda_available:
            for event in self.events.values():
                try:
                    event.destroy()
                except:
                    pass
            self.events.clear()