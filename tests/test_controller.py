#!/usr/bin/env python3

import pytest
import time
from unittest.mock import Mock, patch

from controller import NCCLController, ControllerConfig


class TestNCCLController:
    
    def test_controller_initialization(self):
        config = ControllerConfig()
        controller = NCCLController(config)
        
        assert controller.current_mode == "performance"
        assert controller.pacing_enabled is False
        assert controller.mode_switch_count == 0
    
    def test_defensive_mode_trigger(self):
        config = ControllerConfig(tail_threshold=100e-6, persistence_windows=2)
        controller = NCCLController(config)
        
        # Mock epoch boundary
        with patch.object(controller, '_at_epoch_boundary', return_value=True):
            with patch.object(controller, '_is_cooling_down', return_value=False):
                with patch.object(controller, '_relaunch_workers'):
                    # Inject high latencies
                    for _ in range(3):
                        controller.on_collective_end(200e-6)  # Above threshold
                    
                    assert controller.current_mode == "defensive"
                    assert controller.pacing_enabled is True
    
    def test_performance_mode_recovery(self):
        config = ControllerConfig()
        controller = NCCLController(config)
        
        # Start in defensive mode
        controller.current_mode = "defensive"
        controller.pacing_enabled = True
        
        with patch.object(controller, '_at_epoch_boundary', return_value=True):
            with patch.object(controller, '_is_cooling_down', return_value=False):
                with patch.object(controller.tail_monitor, 'is_stable', return_value=True):
                    with patch.object(controller, '_throughput_acceptable', return_value=True):
                        with patch.object(controller, '_relaunch_workers'):
                            controller.on_collective_end(50e-6)  # Low latency
                            
                            assert controller.current_mode == "performance"
                            assert controller.pacing_enabled is False
    
    def test_cooldown_prevents_switching(self):
        config = ControllerConfig(dwell_time=1000)  # 1 second
        controller = NCCLController(config)
        
        # Set recent switch time
        controller.last_switch_time = time.time()
        
        with patch.object(controller, '_at_epoch_boundary', return_value=True):
            initial_mode = controller.current_mode
            
            # Try to trigger switch during cooldown
            for _ in range(5):
                controller.on_collective_end(1000e-6)  # High latency
            
            # Mode should not change due to cooldown
            assert controller.current_mode == initial_mode
    
    def test_statistics_collection(self):
        config = ControllerConfig()
        controller = NCCLController(config)
        
        # Record some latencies
        for i in range(10):
            controller.tail_monitor.record_latency(100e-6 + i * 10e-6)
        
        stats = controller.get_stats()
        
        assert 'mode' in stats
        assert 'pacing_enabled' in stats
        assert 'p99_latency' in stats
        assert 'sample_count' in stats
        assert stats['sample_count'] == 10


class TestControllerConfig:
    
    def test_default_config(self):
        config = ControllerConfig()
        
        assert config.tail_threshold == 500e-6
        assert config.persistence_windows == 3
        assert config.window_size == 48
        assert config.goodput_budget == 0.05