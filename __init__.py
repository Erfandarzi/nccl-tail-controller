# NCCL Tail Latency Controller
# Host-side adaptive controller for GPU collective operations

from .controller import NCCLController, ControllerConfig

__version__ = "0.3.2"
__author__ = "System Engineering Team"

__all__ = ['NCCLController', 'ControllerConfig']