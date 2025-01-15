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
    
    def test_paper_parameter_values(self):
        """Test that all paper parameters are correctly implemented"""
        config = ControllerConfig()
        
        # Paper Table 1 parameters
        assert config.tail_threshold == 500e-6  # τ = 500μs
        assert config.persistence_windows == 3   # Y = 3 windows
        assert config.dwell_time == 256          # ≥128 collectives
        assert config.cooldown_time == 128       # 128 collectives
        assert config.pacing_min_delay == 10e-6  # 10μs
        assert config.pacing_max_delay == 200e-6 # 200μs
        assert config.goodput_budget == 0.05     # ≤5% loss


class TestPaperExperimentReproduction:
    """Test suite that validates paper's experimental claims"""
    
    def test_experiment_e1_main_results(self):
        """Test E1: Main experiment with interference"""
        from benchmarks.microbench import MicrobenchmarkSuite, InterferenceGenerator
        
        config = ControllerConfig()
        controller = NCCLController(config)
        suite = MicrobenchmarkSuite()
        
        # Run abbreviated version of E1
        results = suite._run_ablation_study(controller)
        
        # Validate paper claims
        baseline = results['baseline']
        full_controller = results['full_controller']
        
        # Check p99 improvement (paper claims 30% improvement)
        p99_improvement = (baseline['p99_us'] - full_controller['p99_us']) / baseline['p99_us']
        assert p99_improvement > 0.2  # At least 20% improvement
        assert p99_improvement < 0.5  # Less than 50% improvement
        
        # Check goodput impact (paper claims <5% loss)
        goodput_loss = (baseline['goodput_gbps'] - full_controller['goodput_gbps']) / baseline['goodput_gbps']
        assert goodput_loss < 0.06  # Less than 6% loss (allow some margin)
    
    def test_experiment_e2_ablation_components(self):
        """Test E2: Ablation study shows both components needed"""
        from benchmarks.microbench import MicrobenchmarkSuite
        
        config = ControllerConfig()
        controller = NCCLController(config)
        suite = MicrobenchmarkSuite()
        
        results = suite._run_ablation_study(controller)
        
        # Validate ablation results
        baseline = results['baseline']['p99_us']
        selector_only = results['selector_only']['p99_us']
        pacing_only = results['pacing_only']['p99_us']
        full_controller = results['full_controller']['p99_us']
        
        # Both individual components should improve over baseline
        assert selector_only < baseline
        assert pacing_only < baseline
        
        # Full controller should be best
        assert full_controller < selector_only
        assert full_controller < pacing_only
    
    def test_experiment_e3_window_sensitivity(self):
        """Test E3: Window size sensitivity analysis"""
        from benchmarks.microbench import MicrobenchmarkSuite
        
        config = ControllerConfig()
        controller = NCCLController(config)
        suite = MicrobenchmarkSuite()
        
        results = suite._run_sensitivity_analysis(controller)
        
        # Validate optimal window region (paper claims 32-64)
        optimal_windows = [w for w, stats in results.items() 
                          if stats['stability_score'] > 0.9 and stats['detection_accuracy'] > 0.9]
        
        assert 32 in optimal_windows or 40 in optimal_windows or 48 in optimal_windows or 56 in optimal_windows
        assert min(optimal_windows) >= 24  # Not too small
        assert max(optimal_windows) <= 80  # Not too large
    
    def test_paper_algorithm_implementation(self):
        """Test that Algorithm 1 from paper is correctly implemented"""
        config = ControllerConfig(tail_threshold=100e-6, persistence_windows=2)
        controller = NCCLController(config)
        
        # Mock dependencies
        with patch.object(controller, '_at_epoch_boundary', return_value=True):
            with patch.object(controller, '_is_cooling_down', return_value=False):
                with patch.object(controller, '_relaunch_workers') as mock_relaunch:
                    
                    # Step 1: Record latencies below threshold
                    for _ in range(5):
                        controller.on_collective_end(50e-6)  # Below threshold
                    
                    assert controller.current_mode == "performance"
                    assert mock_relaunch.call_count == 0
                    
                    # Step 2: Record latencies above threshold for persistence_windows
                    for _ in range(2):
                        controller.on_collective_end(200e-6)  # Above threshold
                    
                    # Should trigger defensive mode
                    assert controller.current_mode == "defensive"
                    assert mock_relaunch.call_count == 1
                    
                    # Verify NCCL environment changes
                    env = controller.nccl_env.get_current_env()
                    assert env['NCCL_ALGO'] == 'Tree'
                    assert env['NCCL_PROTO'] == 'LL128'


class TestSystemIntegration:
    """Integration tests for complete system functionality"""
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow from paper"""
        from benchmarks.microbench import AllReducePerfRunner, InterferenceGenerator
        
        config = ControllerConfig()
        controller = NCCLController(config)
        
        # Create runner for 2-node A100 cluster (paper setup)
        runner = AllReducePerfRunner(gpus_per_node=8, nodes=2)
        
        # Test message size sweep (paper: 1KB to 128MB)
        test_sizes = [1024, 65536, 1048576]  # Subset for testing
        
        for msg_size in test_sizes:
            result = runner._benchmark_allreduce(msg_size, controller)
            
            # Validate result structure
            assert 'p99_us' in result
            assert 'goodput_gbps' in result
            assert result['p99_us'] > 0
            assert result['goodput_gbps'] > 0
    
    def test_interference_generator_integration(self):
        """Test interference generator creates realistic microbursts"""
        from benchmarks.microbench import InterferenceGenerator
        
        interference = InterferenceGenerator(period_ms=10, burst_size=1024)
        
        # Test start/stop
        interference.start_interference()
        assert interference.enabled is True
        
        # Let it run briefly
        time.sleep(0.1)
        
        interference.stop_interference()
        assert interference.enabled is False
    
    def test_hardware_fallback_implementations(self):
        """Test that all hardware detection has working fallbacks"""
        from hardware.topology_detector import TopologyDetector
        from hardware.nic_profiler import NicProfiler
        from hardware.bandwidth_estimator import BandwidthEstimator
        
        # These should work without advanced hardware profiling
        topology = TopologyDetector()
        assert topology.get_gpu_count() > 0
        assert topology.get_optimal_channels() > 0
        
        profiler = NicProfiler()
        assert profiler.get_optimal_channels() > 0
        
        bandwidth_est = BandwidthEstimator()
        assert bandwidth_est.get_link_bandwidth() > 0