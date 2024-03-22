#!/usr/bin/env python3

import os
import sys
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from .monitoring.tail_monitor import TailMonitor
from .algorithms.tdigest import TDigest
from .pacing.token_bucket import TokenBucket
from .pacing.cuda_stagger import CudaStagger
from .hardware.nic_profiler import NicProfiler
from .runtime.process_manager import ProcessManager
from .config.nccl_environment import NcclEnvironment


@dataclass
class ControllerConfig:
    tail_threshold: float = 500e-6
    persistence_windows: int = 3
    window_size: int = 48
    dwell_time: int = 256
    cooldown_time: int = 128
    pacing_min_delay: float = 10e-6
    pacing_max_delay: float = 200e-6
    goodput_budget: float = 0.05


class NCCLController:
    
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.tail_monitor = TailMonitor(
            window_size=config.window_size,
            quantiles=[0.99, 0.999, 0.9999]
        )
        
        self.token_bucket = TokenBucket(
            min_delay=config.pacing_min_delay,
            max_delay=config.pacing_max_delay
        )
        
        self.cuda_stagger = CudaStagger()
        self.nic_profiler = NicProfiler()
        self.process_manager = ProcessManager()
        self.nccl_env = NcclEnvironment()
        
        self.current_mode = "performance"
        self.last_switch_time = 0
        self.consecutive_tail_violations = 0
        self.pacing_enabled = False
        
        # Performance tracking
        self.goodput_history = []
        self.mode_switch_count = 0
        
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    def on_collective_end(self, latency: float) -> None:
        self.tail_monitor.record_latency(latency)
        
        if not self._at_epoch_boundary():
            return
        
        if self._is_cooling_down():
            return
        
        p99 = self.tail_monitor.get_percentile(0.99)
        
        if self._should_switch_to_defensive(p99):
            self._activate_defensive_mode()
        elif self._should_switch_to_performance():
            self._activate_performance_mode()
    
    def _should_switch_to_defensive(self, p99: float) -> bool:
        if p99 > self.config.tail_threshold:
            self.consecutive_tail_violations += 1
        else:
            self.consecutive_tail_violations = 0
        
        return (self.consecutive_tail_violations >= self.config.persistence_windows 
                and self.current_mode == "performance")
    
    def _should_switch_to_performance(self) -> bool:
        return (self.current_mode == "defensive" 
                and self.tail_monitor.is_stable() 
                and self._throughput_acceptable())
    
    def _activate_defensive_mode(self):
        self.logger.info("Switching to defensive mode")
        self.mode_switch_count += 1
        
        nccl_config = {
            'NCCL_ALGO': 'Tree',
            'NCCL_PROTO': 'LL128',
            'NCCL_NCHANNELS': str(max(1, self.nic_profiler.get_optimal_channels() // 2))
        }
        
        self.nccl_env.update_environment(nccl_config)
        self.token_bucket.enable_pacing()
        self.cuda_stagger.enable_staggering()
        
        self.current_mode = "defensive"
        self.last_switch_time = time.time()
        self.pacing_enabled = True
        
        self._relaunch_workers()
    
    def _activate_performance_mode(self):
        self.logger.info("Switching to performance mode")
        self.mode_switch_count += 1
        
        nccl_config = {
            'NCCL_ALGO': 'Ring',
            'NCCL_PROTO': 'Simple',
            'NCCL_NCHANNELS': str(self.nic_profiler.get_optimal_channels())
        }
        
        self.nccl_env.update_environment(nccl_config)
        self.token_bucket.disable_pacing()
        self.cuda_stagger.disable_staggering()
        
        self.current_mode = "performance"
        self.last_switch_time = time.time()
        self.pacing_enabled = False
        
        self._relaunch_workers()
    
    def _relaunch_workers(self):
        start_time = time.time()
        self.process_manager.relaunch_with_env(self.nccl_env.get_current_env())
        relaunch_time = time.time() - start_time
        
        self.logger.info(f"Worker relaunch completed in {relaunch_time*1000:.1f}ms")
        
        # Update goodput tracking
        if hasattr(self, '_last_goodput_measurement'):
            self.goodput_history.append(self._last_goodput_measurement)
    
    def _at_epoch_boundary(self) -> bool:
        return self.tail_monitor.get_sample_count() % 64 == 0
    
    def _is_cooling_down(self) -> bool:
        time_since_switch = time.time() - self.last_switch_time
        min_dwell = self.config.dwell_time / 1000.0
        return time_since_switch < min_dwell
    
    def _throughput_acceptable(self) -> bool:
        if not self.goodput_history:
            return True
        
        # Check if current goodput is within budget
        recent_goodput = sum(self.goodput_history[-8:]) / min(8, len(self.goodput_history))
        baseline_goodput = max(self.goodput_history) if self.goodput_history else recent_goodput
        
        degradation = (baseline_goodput - recent_goodput) / baseline_goodput
        return degradation <= self.config.goodput_budget
    
    def get_stats(self) -> Dict:
        return {
            'mode': self.current_mode,
            'pacing_enabled': self.pacing_enabled,
            'p99_latency': self.tail_monitor.get_percentile(0.99),
            'p999_latency': self.tail_monitor.get_percentile(0.999),
            'sample_count': self.tail_monitor.get_sample_count(),
            'mode_switches': self.mode_switch_count
        }


if __name__ == "__main__":
    config = ControllerConfig()
    controller = NCCLController(config)
    
    controller.logger.info("NCCL Tail Latency Controller started")
    
    # Main control loop would go here in production
    # This is typically integrated with the training framework