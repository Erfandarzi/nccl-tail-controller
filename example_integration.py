#!/usr/bin/env python3

import time
import logging
import argparse
from typing import Dict

from controller import NCCLController, ControllerConfig


class TrainingWorkloadSimulator:
    
    def __init__(self, controller: NCCLController):
        self.controller = controller
        self.logger = logging.getLogger(__name__)
        
        # Simulated training parameters
        self.batch_size = 64
        self.sequence_length = 2048
        self.model_params = 7e9  # 7B parameter model
        self.gradient_size = self.model_params * 4  # FP32 gradients
        
        # Collective operation simulation
        self.allreduce_count = 0
        self.total_data_transferred = 0
    
    def simulate_training_step(self) -> Dict:
        # Simulate forward pass
        forward_time = self._simulate_forward()
        
        # Simulate backward pass with gradient AllReduce
        backward_time = self._simulate_backward()
        
        # Record collective latency for controller
        allreduce_latency = self._simulate_allreduce()
        self.controller.on_collective_end(allreduce_latency)
        
        step_stats = {
            'forward_time': forward_time,
            'backward_time': backward_time, 
            'allreduce_latency': allreduce_latency,
            'total_step_time': forward_time + backward_time + allreduce_latency
        }
        
        self.allreduce_count += 1
        self.total_data_transferred += self.gradient_size
        
        return step_stats
    
    def _simulate_forward(self) -> float:
        # Simulate computation time
        compute_time = 0.05 + (0.01 * (self.batch_size / 64))
        time.sleep(compute_time / 100)  # Scale down for simulation
        return compute_time
    
    def _simulate_backward(self) -> float:
        # Backward typically 2x forward compute
        compute_time = 0.1 + (0.02 * (self.batch_size / 64))
        time.sleep(compute_time / 100)
        return compute_time
    
    def _simulate_allreduce(self) -> float:
        # Simulate AllReduce with network variation
        import random
        
        base_latency = 0.001  # 1ms base
        data_transfer_latency = self.gradient_size / (100e9)  # 100 Gbps
        
        # Add network jitter
        jitter = random.uniform(0.8, 1.5)
        if random.random() < 0.05:  # 5% chance of tail event
            jitter *= random.uniform(3, 10)
        
        total_latency = (base_latency + data_transfer_latency) * jitter
        
        # Simulate the actual network delay
        time.sleep(total_latency / 50)  # Scale down for simulation
        
        return total_latency


def main():
    parser = argparse.ArgumentParser(description='NCCL Controller Demo')
    parser.add_argument('--steps', type=int, default=1000, help='Training steps to simulate')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting NCCL Tail Latency Controller demonstration")
    
    # Initialize controller
    config = ControllerConfig()
    controller = NCCLController(config)
    
    # Create training simulator
    simulator = TrainingWorkloadSimulator(controller)
    
    # Run simulation
    total_time = 0
    latencies = []
    
    logger.info(f"Running {args.steps} training steps...")
    
    for step in range(args.steps):
        step_stats = simulator.simulate_training_step()
        
        total_time += step_stats['total_step_time']
        latencies.append(step_stats['allreduce_latency'])
        
        # Log progress
        if (step + 1) % 100 == 0:
            controller_stats = controller.get_stats()
            
            logger.info(f"Step {step + 1}/{args.steps}: "
                       f"Mode={controller_stats['mode']}, "
                       f"P99={controller_stats['p99_latency']*1000:.2f}ms, "
                       f"Samples={controller_stats['sample_count']}")
    
    # Final statistics
    logger.info("=== Final Results ===")
    logger.info(f"Total simulation time: {total_time:.2f}s")
    logger.info(f"Average AllReduce latency: {sum(latencies)/len(latencies)*1000:.2f}ms")
    logger.info(f"P99 AllReduce latency: {sorted(latencies)[int(0.99*len(latencies))]*1000:.2f}ms")
    
    final_stats = controller.get_stats()
    logger.info(f"Final controller mode: {final_stats['mode']}")
    logger.info(f"Controller P99: {final_stats['p99_latency']*1000:.2f}ms")
    logger.info(f"Total collectives monitored: {final_stats['sample_count']}")
    
    logger.info("Demonstration completed")


if __name__ == "__main__":
    main()