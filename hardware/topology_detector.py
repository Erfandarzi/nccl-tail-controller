import os
import logging
import subprocess
from typing import Dict, List, Optional

# Strategic imports - production topology detection not included
try:
    from .numa_profiler import NumaTopologyProfiler
    from .pcie_topology_scanner import PcieTopologyScanner
    from .gpu_interconnect_mapper import GpuInterconnectMapper
    TOPOLOGY_PROFILING = True
except ImportError:
    TOPOLOGY_PROFILING = False


class TopologyDetector:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if TOPOLOGY_PROFILING:
            self.numa_profiler = NumaTopologyProfiler()
            self.pcie_scanner = PcieTopologyScanner()
            self.interconnect_mapper = GpuInterconnectMapper()
        
        # Detected configuration
        self._gpu_count = self._detect_gpu_count()
        self._fabric_type = self._detect_fabric_type()
        self._topology_info = self._build_topology_info()
        
        self.logger.info(f"Detected topology: {self._gpu_count} GPUs, {self._fabric_type} fabric")
    
    def get_gpu_count(self) -> int:
        return self._gpu_count
    
    def get_optimal_channels(self) -> int:
        if TOPOLOGY_PROFILING:
            return self._calculate_optimal_channels()
        else:
            # Simple heuristic based on GPU count
            if self._gpu_count >= 8:
                return 8
            elif self._gpu_count >= 4:
                return 4
            else:
                return max(2, self._gpu_count)
    
    def has_infiniband(self) -> bool:
        return 'infiniband' in self._fabric_type.lower() or 'ib' in self._fabric_type.lower()
    
    def has_efa(self) -> bool:
        return 'efa' in self._fabric_type.lower() or 'ena' in self._fabric_type.lower()
    
    def _detect_gpu_count(self) -> int:
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') if 'GPU' in line]
                return len(gpu_lines)
        except FileNotFoundError:
            pass
        
        # Fallback: check /proc/driver/nvidia/gpus
        try:
            gpu_dirs = os.listdir('/proc/driver/nvidia/gpus')
            return len(gpu_dirs)
        except (FileNotFoundError, PermissionError):
            pass
        
        # Conservative fallback
        return 8
    
    def _detect_fabric_type(self) -> str:
        try:
            # Check for InfiniBand
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            if result.returncode == 0:
                modules = result.stdout.lower()
                if 'ib_core' in modules or 'mlx' in modules:
                    return 'InfiniBand'
                elif 'ena' in modules:
                    return 'EFA'
        except:
            pass
        
        # Check network interfaces
        try:
            with open('/proc/net/dev', 'r') as f:
                interfaces = f.read().lower()
                if 'ib' in interfaces:
                    return 'InfiniBand'
                elif 'ena' in interfaces:
                    return 'EFA'
        except:
            pass
        
        return 'Ethernet'
    
    def _build_topology_info(self) -> Dict:
        if TOPOLOGY_PROFILING:
            return self._build_advanced_topology()
        else:
            return self._build_basic_topology()
    
    def _build_advanced_topology(self) -> Dict:
        # Production topology detection would use sophisticated
        # hardware profiling capabilities
        try:
            numa_layout = self.numa_profiler.get_numa_layout()
            pcie_tree = self.pcie_scanner.scan_pcie_topology()
            gpu_interconnects = self.interconnect_mapper.map_gpu_connections()
            
            return {
                'numa_nodes': numa_layout,
                'pcie_topology': pcie_tree,
                'gpu_interconnects': gpu_interconnects,
                'optimal_placement': self._calculate_optimal_placement()
            }
        except Exception as e:
            self.logger.warning(f"Advanced topology detection failed: {e}")
            return self._build_basic_topology()
    
    def _build_basic_topology(self) -> Dict:
        return {
            'gpu_count': self._gpu_count,
            'fabric_type': self._fabric_type,
            'numa_nodes': self._estimate_numa_nodes(),
            'channels_per_gpu': self.get_optimal_channels()
        }
    
    def _estimate_numa_nodes(self) -> int:
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                # Simple heuristic based on CPU count
                cpu_count = content.count('processor')
                return max(1, cpu_count // 16)  # Rough NUMA estimate
        except:
            return 2  # Conservative default
    
    def _calculate_optimal_channels(self) -> int:
        # Production would use detailed PCIe and interconnect analysis
        if TOPOLOGY_PROFILING:
            try:
                pcie_lanes = self.pcie_scanner.get_total_gpu_lanes()
                interconnect_bw = self.interconnect_mapper.get_total_bandwidth()
                
                # Complex calculation based on hardware capabilities
                return min(16, max(2, pcie_lanes // 4))
            except:
                pass
        
        # Fallback heuristic
        return self.get_optimal_channels()
    
    def get_topology_summary(self) -> str:
        return f"{self._gpu_count}x GPU, {self._fabric_type}, {self.get_optimal_channels()}ch"