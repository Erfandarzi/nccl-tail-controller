import os
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

from ..hardware.topology_detector import TopologyDetector


@dataclass
class NcclConfig:
    algo: str = "Ring"
    proto: str = "Simple" 
    nchannels: int = 8
    tree_threshold: int = 8192
    min_ctas: int = 32
    max_ctas: int = 32
    buffsize: int = 4194304
    
    # Advanced tuning parameters
    p2p_net_chunksize: int = 131072
    p2p_pcie_chunksize: int = 131072
    cross_nic: int = 0
    nvls_enable: int = 0


class NcclEnvironment:
    
    NCCL_ENV_VARS = {
        'NCCL_ALGO': 'algo',
        'NCCL_PROTO': 'proto', 
        'NCCL_NCHANNELS': 'nchannels',
        'NCCL_TREE_THRESHOLD': 'tree_threshold',
        'NCCL_MIN_CTAS': 'min_ctas',
        'NCCL_MAX_CTAS': 'max_ctas',
        'NCCL_BUFFSIZE': 'buffsize',
        'NCCL_P2P_NET_CHUNKSIZE': 'p2p_net_chunksize',
        'NCCL_P2P_PCIE_CHUNKSIZE': 'p2p_pcie_chunksize',
        'NCCL_CROSS_NIC': 'cross_nic',
        'NCCL_NVLS_ENABLE': 'nvls_enable'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Current configuration
        self.current_config = NcclConfig()
        
        # Hardware topology detection
        self.topology = TopologyDetector()
        
        # Load initial configuration from environment
        self._load_from_environment()
        
        # Apply hardware-specific defaults
        self._apply_hardware_defaults()
    
    def update_environment(self, updates: Dict[str, str]) -> None:
        for env_var, value in updates.items():
            if env_var in self.NCCL_ENV_VARS:
                field_name = self.NCCL_ENV_VARS[env_var]
                
                # Convert to appropriate type
                if field_name in ['nchannels', 'tree_threshold', 'min_ctas', 
                                  'max_ctas', 'buffsize', 'p2p_net_chunksize', 
                                  'p2p_pcie_chunksize', 'cross_nic', 'nvls_enable']:
                    value = int(value)
                
                setattr(self.current_config, field_name, value)
                
                self.logger.debug(f"Updated {field_name} = {value}")
    
    def get_current_env(self) -> Dict[str, str]:
        env_dict = {}
        config_dict = asdict(self.current_config)
        
        for env_var, field_name in self.NCCL_ENV_VARS.items():
            value = config_dict[field_name]
            env_dict[env_var] = str(value)
        
        return env_dict
    
    def apply_performance_mode(self) -> None:
        self.logger.info("Applying performance mode configuration")
        
        self.current_config.algo = "Ring"
        self.current_config.proto = "Simple"
        self.current_config.nchannels = self.topology.get_optimal_channels()
        self.current_config.tree_threshold = 8192
        
        # Optimize for bandwidth
        self.current_config.buffsize = 8 * 1024 * 1024  # 8MB for large transfers
        self.current_config.p2p_net_chunksize = 512 * 1024  # 512KB chunks
    
    def apply_defensive_mode(self) -> None:
        self.logger.info("Applying defensive mode configuration")
        
        self.current_config.algo = "Tree"
        self.current_config.proto = "LL128"
        self.current_config.nchannels = max(1, self.topology.get_optimal_channels() // 2)
        self.current_config.tree_threshold = 2048
        
        # Optimize for latency and stability
        self.current_config.buffsize = 2 * 1024 * 1024  # 2MB for low latency
        self.current_config.p2p_net_chunksize = 128 * 1024  # 128KB chunks
    
    def get_config_summary(self) -> str:
        return f"{self.current_config.algo}+{self.current_config.proto} (ch={self.current_config.nchannels})"
    
    def _load_from_environment(self):
        for env_var, field_name in self.NCCL_ENV_VARS.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if field_name in ['algo', 'proto']:
                        setattr(self.current_config, field_name, env_value)
                    else:
                        setattr(self.current_config, field_name, int(env_value))
                except ValueError:
                    self.logger.warning(f"Invalid environment value for {env_var}: {env_value}")
    
    def _apply_hardware_defaults(self):
        # Apply hardware-specific optimizations
        gpu_count = self.topology.get_gpu_count()
        
        if gpu_count >= 8:
            # Multi-GPU optimizations
            self.current_config.nchannels = min(8, gpu_count)
        else:
            self.current_config.nchannels = max(2, gpu_count)
        
        # Network fabric optimizations
        if self.topology.has_infiniband():
            self.current_config.p2p_net_chunksize = 256 * 1024
            self.current_config.buffsize = 8 * 1024 * 1024  # 8MB for IB
        elif self.topology.has_efa():
            self.current_config.p2p_net_chunksize = 128 * 1024
            self.current_config.cross_nic = 1
            self.current_config.buffsize = 4 * 1024 * 1024  # 4MB for EFA
    
    def validate_config(self) -> List[str]:
        warnings = []
        
        if self.current_config.nchannels > 16:
            warnings.append("NCHANNELS > 16 may cause resource contention")
        
        if self.current_config.buffsize < 1024*1024:
            warnings.append("BUFFSIZE < 1MB may hurt large message performance")
        
        if self.current_config.algo == "Tree" and self.current_config.nchannels > 8:
            warnings.append("Tree algorithm with >8 channels may be suboptimal")
        
        return warnings