import platform
import subprocess
from typing import Dict, Optional


def get_system_info() -> Dict:
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'architecture': platform.machine()
    }
    
    # GPU information
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            info['gpus'] = result.stdout.strip().split('\n')
    except FileNotFoundError:
        info['gpus'] = ['CUDA not available']
    
    # Network interfaces
    try:
        result = subprocess.run(['ip', 'link', 'show'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            # Extract interface names
            interfaces = []
            for line in result.stdout.split('\n'):
                if ':' in line and 'state' in line.lower():
                    iface = line.split(':')[1].strip()
                    interfaces.append(iface)
            info['network_interfaces'] = interfaces
    except FileNotFoundError:
        info['network_interfaces'] = ['Unknown']
    
    return info


def check_requirements() -> Dict[str, bool]:
    requirements = {
        'cuda_available': False,
        'infiniband_available': False,
        'efa_available': False,
        'root_privileges': False
    }
    
    # Check CUDA
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        requirements['cuda_available'] = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    # Check InfiniBand
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'ib_core' in result.stdout:
            requirements['infiniband_available'] = True
    except:
        pass
    
    # Check EFA
    try:
        result = subprocess.run(['modinfo', 'ena'], capture_output=True)
        if result.returncode == 0:
            requirements['efa_available'] = True
    except:
        pass
    
    # Check root privileges
    import os
    requirements['root_privileges'] = os.geteuid() == 0
    
    return requirements