# NCCL Controller Configuration Guide

## Overview

The NCCL Tail Latency Controller provides host-side optimization for GPU collective operations without requiring network fabric control. This document covers configuration options and deployment considerations.

## Controller Configuration

### Basic Parameters

```python
config = ControllerConfig(
    tail_threshold=500e-6,      # 500μs P99 trigger threshold
    persistence_windows=3,       # Consecutive violations before switch
    window_size=48,             # Monitoring window size (samples)
    dwell_time=256,             # Min collectives before re-evaluation  
    cooldown_time=128,          # Grace period after mode switch
    goodput_budget=0.05         # Max 5% throughput loss tolerance
)
```

### Hardware-Specific Tuning

#### InfiniBand Environments
- Larger buffer sizes (8MB) for bandwidth optimization
- Higher channel counts leverage IB's low latency
- Aggressive pacing parameters for congestion control

#### EFA/Ethernet Environments  
- Conservative buffer sizing (4MB) for stability
- Cross-NIC load balancing enabled
- More conservative pacing to avoid packet loss

### Production Deployment

#### System Requirements
- Linux kernel 4.15+ with cgroup v2 support
- systemd for service management
- CAP_SYS_ADMIN for hardware profiling (optional)
- CUDA 11.0+ for GPU integration

#### Service Configuration
```bash
# Deploy as systemd service
sudo ./scripts/deploy.sh

# Monitor controller status
journalctl -u nccl-controller -f

# Performance tuning
echo 'isolated_cores=0-7' >> /proc/cmdline
systemctl set-property nccl-controller.service CPUAffinity=8-15
```

## Integration Examples

### PyTorch Integration
```python
from nccl_controller import NCCLController, ControllerConfig

# Initialize before distributed training
controller = NCCLController(ControllerConfig())

# In training loop
def train_step():
    start_time = time.time()
    
    # Forward/backward pass
    loss.backward()
    
    # Record collective timing
    collective_time = time.time() - start_time  
    controller.on_collective_end(collective_time)
```

## Monitoring and Debugging

### Controller Statistics
- `mode`: Current operating mode (performance/defensive)
- `p99_latency`: Current 99th percentile tail latency
- `mode_switches`: Total mode transitions (detect oscillation)
- `pacing_enabled`: Whether traffic pacing is active

### Hardware Profiling
When production hardware profiling modules are available:
- PCIe lane utilization and congestion detection
- RDMA queue depth monitoring  
- Fabric topology-aware channel optimization

Note: Advanced hardware profiling requires proprietary libraries not included in this repository.