import os
import sys
import time
import signal
import subprocess
import threading
from typing import Dict, List, Optional, Callable
import logging


class ProcessManager:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.current_processes: List[subprocess.Popen] = []
        self.relaunch_count = 0
        self.last_relaunch_time = 0.0
        
        # Process monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitoring_enabled = False
        
        # Relaunch configuration
        self.max_relaunch_time = 5.0  # seconds
        self.graceful_shutdown_timeout = 2.0
        
        self._lock = threading.Lock()
    
    def relaunch_with_env(self, env_vars: Dict[str, str]) -> float:
        start_time = time.time()
        
        with self._lock:
            self.logger.info(f"Initiating worker relaunch (#{self.relaunch_count + 1})")
            
            # Terminate existing processes
            self._terminate_processes()
            
            # Update environment
            updated_env = os.environ.copy()
            updated_env.update(env_vars)
            
            # Launch new processes
            self._launch_workers(updated_env)
            
            self.relaunch_count += 1
            self.last_relaunch_time = time.time()
        
        relaunch_duration = time.time() - start_time
        self.logger.info(f"Relaunch completed in {relaunch_duration*1000:.1f}ms")
        
        return relaunch_duration
    
    def _terminate_processes(self):
        if not self.current_processes:
            return
        
        self.logger.debug(f"Terminating {len(self.current_processes)} worker processes")
        
        # Send SIGTERM first
        for proc in self.current_processes:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        
        # Wait for graceful shutdown
        start_wait = time.time()
        while (time.time() - start_wait) < self.graceful_shutdown_timeout:
            all_dead = True
            for proc in self.current_processes:
                if proc.poll() is None:
                    all_dead = False
                    break
            
            if all_dead:
                break
            
            time.sleep(0.01)
        
        # Force kill if necessary
        for proc in self.current_processes:
            if proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        
        self.current_processes.clear()
    
    def _launch_workers(self, env: Dict[str, str]):
        # In a real implementation, this would launch the actual training workers
        # For demo purposes, we simulate with a simple process
        
        worker_cmd = self._get_worker_command()
        
        try:
            proc = subprocess.Popen(
                worker_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.current_processes.append(proc)
            self.logger.debug(f"Launched worker process PID {proc.pid}")
            
        except Exception as e:
            self.logger.error(f"Failed to launch worker: {e}")
            raise
    
    def _get_worker_command(self) -> List[str]:
        # This would typically be the actual training command
        # For demonstration, return a simple command
        return ["python3", "-c", "import time; time.sleep(3600)"]
    
    def start_monitoring(self, callback: Optional[Callable] = None):
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_processes,
            args=(callback,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Process monitoring started")
    
    def stop_monitoring(self):
        self.monitoring_enabled = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        self.logger.info("Process monitoring stopped")
    
    def _monitor_processes(self, callback: Optional[Callable] = None):
        while self.monitoring_enabled:
            with self._lock:
                dead_processes = []
                
                for proc in self.current_processes:
                    if proc.poll() is not None:
                        dead_processes.append(proc)
                
                if dead_processes:
                    self.logger.warning(f"Detected {len(dead_processes)} dead processes")
                    
                    for proc in dead_processes:
                        self.current_processes.remove(proc)
                    
                    if callback:
                        callback(dead_processes)
            
            time.sleep(0.5)
    
    def get_process_stats(self) -> Dict:
        with self._lock:
            active_count = len(self.current_processes)
            
            # Check process health
            healthy_count = 0
            for proc in self.current_processes:
                if proc.poll() is None:
                    healthy_count += 1
            
            return {
                'active_processes': active_count,
                'healthy_processes': healthy_count,
                'relaunch_count': self.relaunch_count,
                'last_relaunch_time': self.last_relaunch_time,
                'monitoring_enabled': self.monitoring_enabled
            }
    
    def cleanup(self):
        self.logger.info("Cleaning up process manager")
        
        self.stop_monitoring()
        self._terminate_processes()
        
        self.logger.info("Process manager cleanup completed")