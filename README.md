# nccl-tail-controller

Host-side NCCL collective latency controller. Adaptive algorithm selection + traffic pacing.

## Requirements

- Linux 5.4+, cgroup v2, systemd
- NCCL 2.8+, CUDA 11+
- Root for hardware profiling
- InfiniBand or EFA fabric

```bash
# Check requirements
uname -r && grep cgroup2 /proc/filesystems
nvidia-smi
lsmod | grep -E "(ib_core|ena)"
```

## Build

```bash
pip3 install -r requirements.txt
sudo ./scripts/deploy.sh
```


## Usage

```python
from controller import NCCLController, ControllerConfig

controller = NCCLController(ControllerConfig())

# In training loop
controller.on_collective_end(collective_latency_seconds)
```

## Config

Key parameters in `ControllerConfig`:
- `tail_threshold`: P99 latency trigger (default 500μs)
- `persistence_windows`: Violations before switch (default 3) 
- `goodput_budget`: Max throughput loss (default 5%)

Hardware-specific:
```bash
# InfiniBand
export NCCL_IB_DISABLE=0

# EFA  
export NCCL_CROSS_NIC=1
```

## Operation

Controller switches between two modes:
- **Performance**: Ring+Simple, max channels
- **Defensive**: Tree+LL128, reduced channels + pacing

Triggers worker relaunch with new NCCL env vars. ~100ms overhead.

## Monitoring

```bash
systemctl status nccl-controller
journalctl -u nccl-controller -f
```

Stats: mode, p99_latency, mode_switches, pacing_enabled

## Tuning

System-level optimizations:
```bash
# CPU isolation
echo 'isolcpus=0-3' >> /proc/cmdline

# IRQ affinity  
echo 4 > /proc/irq/24/smp_affinity

# NUMA binding
numactl --cpunodebind=1 ./controller.py
```

## Architecture

```
Host Controller
├── Tail Monitor (t-digest P99/P999)
├── Mode Switch Logic (perf ↔ defensive)  
├── NCCL Env Manager (ALGO/PROTO/NCHANNELS)
├── Token Bucket Pacer (10-200μs delays)
└── Worker Relaunch (sub-100ms overhead)

Modes:
  Performance: Ring+Simple, max channels
  Defensive:   Tree+LL128, reduced channels + pacing
```