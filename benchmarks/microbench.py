#!/usr/bin/env python3

import time
import statistics
from typing import List, Dict
import logging

from ..controller import NCCLController, ControllerConfig


class MicrobenchmarkSuite:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Benchmark configuration
        self.message_sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
        self.iterations_per_size = 100
        
    def run_latency_sweep(self, controller: NCCLController) -> Dict:
        results = {}
        
        for msg_size in self.message_sizes:
            latencies = self._benchmark_message_size(controller, msg_size)
            
            results[msg_size] = {
                'mean': statistics.mean(latencies),
                'p50': statistics.median(latencies),
                'p99': latencies[int(0.99 * len(latencies))],
                'p999': latencies[int(0.999 * len(latencies))],
                'count': len(latencies)
            }
            
            self.logger.info(f"Message size {msg_size}: "
                           f"p99={results[msg_size]['p99']*1000:.2f}ms")
        
        return results
    
    def _benchmark_message_size(self, controller: NCCLController, 
                               msg_size: int) -> List[float]:
        latencies = []
        
        for _ in range(self.iterations_per_size):
            # Simulate collective operation
            start_time = time.perf_counter()
            
            # Add realistic delay based on message size
            base_latency = 50e-6  # 50 microseconds
            transfer_time = msg_size / (100e9)  # 100 Gbps
            
            time.sleep(base_latency + transfer_time)
            
            end_time = time.perf_counter()
            latency = end_time - start_time
            
            latencies.append(latency)
            controller.on_collective_end(latency)
        
        return sorted(latencies)
    
    def run_controller_overhead_test(self, controller: NCCLController) -> Dict:
        # Test controller decision overhead
        decision_times = []
        
        for _ in range(1000):
            start = time.perf_counter()
            controller.on_collective_end(0.001)  # 1ms latency
            end = time.perf_counter()
            
            decision_times.append(end - start)
        
        return {
            'mean_overhead_us': statistics.mean(decision_times) * 1e6,
            'p99_overhead_us': sorted(decision_times)[990] * 1e6
        }