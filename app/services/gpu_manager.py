"""
GPU Integration and Resource Management System
============================================

Advanced GPU detection, allocation, and optimization for video processing
with multi-GPU support and intelligent resource management.

Features:
- Multi-GPU detection and management
- Dynamic GPU memory allocation
- Hardware-specific optimization profiles
- Real-time performance monitoring
- Automatic fallback strategies
- Cross-platform GPU support (NVIDIA, Intel, AMD)

Author: Claude Code Enhanced System
Version: 2.0.0
"""

import os
import platform
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import json

from loguru import logger
import psutil


class GPUVendor(Enum):
    """GPU vendor types"""
    NVIDIA = "nvidia"
    INTEL = "intel"
    AMD = "amd"
    UNKNOWN = "unknown"


class GPUCapability(Enum):
    """GPU capabilities for video processing"""
    H264_ENCODE = "h264_encode"
    H264_DECODE = "h264_decode"
    H265_ENCODE = "h265_encode"
    H265_DECODE = "h265_decode"
    AV1_ENCODE = "av1_encode"
    AV1_DECODE = "av1_decode"
    NVENC = "nvenc"
    NVDEC = "nvdec"
    QUICKSYNC = "quicksync"
    VAAPI = "vaapi"
    VCE = "vce"  # AMD Video Coding Engine


@dataclass
class GPUInfo:
    """GPU information and capabilities"""
    id: int
    name: str
    vendor: GPUVendor
    memory_total: int  # MB
    memory_free: int   # MB
    capabilities: List[GPUCapability]
    driver_version: str
    compute_capability: str = ""
    utilization: float = 0.0  # Percentage
    temperature: float = 0.0  # Celsius
    power_usage: float = 0.0  # Watts
    is_available: bool = True
    ffmpeg_device: str = ""  # FFmpeg device string
    
    @property
    def memory_used(self) -> int:
        """Calculate used memory"""
        return self.memory_total - self.memory_free
    
    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage"""
        return (self.memory_used / self.memory_total * 100) if self.memory_total > 0 else 0


@dataclass
class GPUOptimizationProfile:
    """GPU-specific optimization settings"""
    vendor: GPUVendor
    codec_preferences: List[str]
    optimal_batch_size: int
    memory_buffer_mb: int
    max_concurrent_streams: int
    preferred_pixel_format: str
    encoding_preset: str
    quality_settings: Dict[str, Any]
    ffmpeg_extra_args: List[str] = field(default_factory=list)


class GPUDetector:
    """Detect and enumerate available GPUs"""
    
    def __init__(self):
        self.platform = platform.system().lower()
    
    def detect_all_gpus(self) -> List[GPUInfo]:
        """Detect all available GPUs across vendors"""
        all_gpus = []
        
        # Detect NVIDIA GPUs
        nvidia_gpus = self._detect_nvidia_gpus()
        all_gpus.extend(nvidia_gpus)
        
        # Detect Intel GPUs
        intel_gpus = self._detect_intel_gpus()
        all_gpus.extend(intel_gpus)
        
        # Detect AMD GPUs
        amd_gpus = self._detect_amd_gpus()
        all_gpus.extend(amd_gpus)
        
        logger.info(f"Detected {len(all_gpus)} GPU(s): {[gpu.name for gpu in all_gpus]}")
        return all_gpus
    
    def _detect_nvidia_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi"""
        gpus = []
        
        try:
            # Try nvidia-smi
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.free,driver_version,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        try:
                            gpu_info = GPUInfo(
                                id=int(parts[0]),
                                name=parts[1],
                                vendor=GPUVendor.NVIDIA,
                                memory_total=int(parts[2]),
                                memory_free=int(parts[3]),
                                driver_version=parts[4],
                                utilization=float(parts[5]),
                                temperature=float(parts[6]) if len(parts) > 6 and parts[6] != '[Not Supported]' else 0.0,
                                power_usage=float(parts[7]) if len(parts) > 7 and parts[7] != '[Not Supported]' else 0.0,
                                capabilities=self._get_nvidia_capabilities(),
                                ffmpeg_device=f"cuda:{int(parts[0])}"
                            )
                            gpus.append(gpu_info)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse NVIDIA GPU info: {e}")
                
                logger.info(f"Detected {len(gpus)} NVIDIA GPU(s)")
            
        except FileNotFoundError:
            logger.debug("nvidia-smi not found, no NVIDIA GPUs detected")
        except Exception as e:
            logger.warning(f"Error detecting NVIDIA GPUs: {e}")
        
        return gpus
    
    def _detect_intel_gpus(self) -> List[GPUInfo]:
        """Detect Intel GPUs"""
        gpus = []
        
        try:
            # Check for Intel GPU via system info
            if self.platform == "linux":
                # Check for Intel GPU in /sys/class/drm
                import glob
                intel_devices = glob.glob("/sys/class/drm/card*/device/vendor")
                
                for device_path in intel_devices:
                    try:
                        with open(device_path, 'r') as f:
                            vendor_id = f.read().strip()
                        
                        if vendor_id == "0x8086":  # Intel vendor ID
                            # This is an Intel GPU
                            card_path = os.path.dirname(os.path.dirname(device_path))
                            card_name = os.path.basename(card_path)
                            
                            gpu_info = GPUInfo(
                                id=len(gpus),
                                name="Intel Integrated Graphics",
                                vendor=GPUVendor.INTEL,
                                memory_total=self._estimate_intel_memory(),
                                memory_free=self._estimate_intel_memory(),
                                driver_version="Unknown",
                                capabilities=self._get_intel_capabilities(),
                                ffmpeg_device="/dev/dri/renderD128"  # Default Intel device
                            )
                            gpus.append(gpu_info)
                            break
                    
                    except Exception:
                        continue
            
            elif self.platform == "windows":
                # Check Windows registry or WMI for Intel GPU
                try:
                    result = subprocess.run([
                        'wmic', 'path', 'win32_VideoController',
                        'get', 'name,AdapterRAM,DriverVersion',
                        '/format:csv'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines[1:]:  # Skip header
                            if 'intel' in line.lower():
                                parts = line.split(',')
                                if len(parts) >= 3:
                                    gpu_info = GPUInfo(
                                        id=len(gpus),
                                        name=parts[2].strip(),
                                        vendor=GPUVendor.INTEL,
                                        memory_total=int(parts[1]) // (1024 * 1024) if parts[1].strip() else 1024,
                                        memory_free=int(parts[1]) // (1024 * 1024) if parts[1].strip() else 1024,
                                        driver_version=parts[3].strip(),
                                        capabilities=self._get_intel_capabilities(),
                                        ffmpeg_device="qsv"
                                    )
                                    gpus.append(gpu_info)
                                    break
                
                except Exception:
                    pass
            
            if gpus:
                logger.info(f"Detected {len(gpus)} Intel GPU(s)")
        
        except Exception as e:
            logger.warning(f"Error detecting Intel GPUs: {e}")
        
        return gpus
    
    def _detect_amd_gpus(self) -> List[GPUInfo]:
        """Detect AMD GPUs"""
        gpus = []
        
        try:
            if self.platform == "linux":
                # Try rocm-smi for AMD GPUs
                try:
                    result = subprocess.run([
                        'rocm-smi', '--showproductname', '--showmeminfo', 'vram'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        # Parse rocm-smi output
                        lines = result.stdout.strip().split('\n')
                        current_gpu = None
                        
                        for line in lines:
                            if 'GPU[' in line:
                                if 'Product Name' in line:
                                    gpu_name = line.split(':')[-1].strip()
                                    current_gpu = GPUInfo(
                                        id=len(gpus),
                                        name=gpu_name,
                                        vendor=GPUVendor.AMD,
                                        memory_total=4096,  # Default, will be updated
                                        memory_free=4096,
                                        driver_version="Unknown",
                                        capabilities=self._get_amd_capabilities(),
                                        ffmpeg_device=f"vaapi:/dev/dri/renderD12{len(gpus) + 8}"
                                    )
                                    gpus.append(current_gpu)
                
                except FileNotFoundError:
                    # rocm-smi not found, try alternative detection
                    pass
                
                # Fallback: Check for AMD GPU in /sys/class/drm
                import glob
                amd_devices = glob.glob("/sys/class/drm/card*/device/vendor")
                
                for device_path in amd_devices:
                    try:
                        with open(device_path, 'r') as f:
                            vendor_id = f.read().strip()
                        
                        if vendor_id in ["0x1002", "0x1022"]:  # AMD vendor IDs
                            if not gpus:  # Only add if not already detected
                                gpu_info = GPUInfo(
                                    id=len(gpus),
                                    name="AMD Radeon Graphics",
                                    vendor=GPUVendor.AMD,
                                    memory_total=4096,  # Estimate
                                    memory_free=4096,
                                    driver_version="Unknown",
                                    capabilities=self._get_amd_capabilities(),
                                    ffmpeg_device="vaapi:/dev/dri/renderD128"
                                )
                                gpus.append(gpu_info)
                                break
                    
                    except Exception:
                        continue
            
            if gpus:
                logger.info(f"Detected {len(gpus)} AMD GPU(s)")
        
        except Exception as e:
            logger.warning(f"Error detecting AMD GPUs: {e}")
        
        return gpus
    
    def _get_nvidia_capabilities(self) -> List[GPUCapability]:
        """Get NVIDIA GPU capabilities"""
        capabilities = [
            GPUCapability.H264_ENCODE,
            GPUCapability.H264_DECODE,
            GPUCapability.NVENC,
            GPUCapability.NVDEC
        ]
        
        # Test for additional capabilities
        try:
            result = subprocess.run([
                'ffmpeg', '-hide_banner', '-encoders'
            ], capture_output=True, text=True, timeout=5)
            
            if 'h264_nvenc' in result.stdout:
                capabilities.append(GPUCapability.H264_ENCODE)
            if 'hevc_nvenc' in result.stdout:
                capabilities.append(GPUCapability.H265_ENCODE)
        
        except Exception:
            pass
        
        return capabilities
    
    def _get_intel_capabilities(self) -> List[GPUCapability]:
        """Get Intel GPU capabilities"""
        return [
            GPUCapability.H264_ENCODE,
            GPUCapability.H264_DECODE,
            GPUCapability.QUICKSYNC
        ]
    
    def _get_amd_capabilities(self) -> List[GPUCapability]:
        """Get AMD GPU capabilities"""
        return [
            GPUCapability.H264_ENCODE,
            GPUCapability.H264_DECODE,
            GPUCapability.VCE,
            GPUCapability.VAAPI
        ]
    
    def _estimate_intel_memory(self) -> int:
        """Estimate Intel integrated GPU memory"""
        # Intel integrated GPUs share system memory
        total_memory = psutil.virtual_memory().total // (1024 * 1024)  # MB
        # Typically 1/8 to 1/4 of system memory is available to integrated GPU
        return min(total_memory // 8, 2048)  # Cap at 2GB


class GPUResourceManager:
    """Manage GPU resources and allocation"""
    
    def __init__(self):
        self.detector = GPUDetector()
        self.available_gpus: List[GPUInfo] = []
        self.allocation_lock = threading.Lock()
        self.allocated_memory: Dict[int, int] = {}  # gpu_id -> allocated_mb
        self.optimization_profiles = self._create_optimization_profiles()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Initialize GPU detection
        self.refresh_gpu_list()
    
    def refresh_gpu_list(self):
        """Refresh the list of available GPUs"""
        self.available_gpus = self.detector.detect_all_gpus()
        self.allocated_memory = {gpu.id: 0 for gpu in self.available_gpus}
        
        if self.available_gpus:
            logger.info(f"GPU Resource Manager initialized with {len(self.available_gpus)} GPUs")
            for gpu in self.available_gpus:
                logger.info(f"  GPU {gpu.id}: {gpu.name} ({gpu.vendor.value}) - {gpu.memory_total}MB")
        else:
            logger.warning("No GPUs detected, falling back to CPU processing")
    
    def get_best_gpu_for_task(self, 
                             required_memory_mb: int = 512,
                             preferred_vendor: Optional[GPUVendor] = None,
                             required_capabilities: Optional[List[GPUCapability]] = None) -> Optional[GPUInfo]:
        """
        Select the best available GPU for a specific task
        
        Args:
            required_memory_mb: Minimum memory requirement
            preferred_vendor: Preferred GPU vendor
            required_capabilities: Required GPU capabilities
        
        Returns:
            Best available GPU or None if no suitable GPU found
        """
        with self.allocation_lock:
            suitable_gpus = []
            
            for gpu in self.available_gpus:
                if not gpu.is_available:
                    continue
                
                # Check memory availability
                available_memory = gpu.memory_free - self.allocated_memory.get(gpu.id, 0)
                if available_memory < required_memory_mb:
                    continue
                
                # Check vendor preference
                if preferred_vendor and gpu.vendor != preferred_vendor:
                    continue
                
                # Check capabilities
                if required_capabilities:
                    if not all(cap in gpu.capabilities for cap in required_capabilities):
                        continue
                
                suitable_gpus.append(gpu)
            
            if not suitable_gpus:
                return None
            
            # Sort by preference: vendor preference, then available memory, then utilization
            def gpu_score(gpu: GPUInfo) -> Tuple[int, int, float]:
                vendor_score = 0
                if preferred_vendor:
                    vendor_score = 1 if gpu.vendor == preferred_vendor else 0
                
                available_memory = gpu.memory_free - self.allocated_memory.get(gpu.id, 0)
                utilization_score = 100 - gpu.utilization  # Lower utilization is better
                
                return (vendor_score, available_memory, utilization_score)
            
            best_gpu = max(suitable_gpus, key=gpu_score)
            logger.info(f"Selected GPU {best_gpu.id} ({best_gpu.name}) for task")
            return best_gpu
    
    def allocate_gpu_memory(self, gpu_id: int, memory_mb: int) -> bool:
        """Allocate GPU memory for a task"""
        with self.allocation_lock:
            gpu = next((g for g in self.available_gpus if g.id == gpu_id), None)
            if not gpu:
                return False
            
            current_allocated = self.allocated_memory.get(gpu_id, 0)
            if gpu.memory_free - current_allocated < memory_mb:
                return False
            
            self.allocated_memory[gpu_id] = current_allocated + memory_mb
            logger.debug(f"Allocated {memory_mb}MB on GPU {gpu_id}")
            return True
    
    def release_gpu_memory(self, gpu_id: int, memory_mb: int):
        """Release allocated GPU memory"""
        with self.allocation_lock:
            current_allocated = self.allocated_memory.get(gpu_id, 0)
            self.allocated_memory[gpu_id] = max(0, current_allocated - memory_mb)
            logger.debug(f"Released {memory_mb}MB from GPU {gpu_id}")
    
    def get_optimization_profile(self, vendor: GPUVendor) -> Optional[GPUOptimizationProfile]:
        """Get optimization profile for a GPU vendor"""
        return self.optimization_profiles.get(vendor)
    
    def _create_optimization_profiles(self) -> Dict[GPUVendor, GPUOptimizationProfile]:
        """Create GPU vendor-specific optimization profiles"""
        profiles = {}
        
        # NVIDIA optimization profile
        profiles[GPUVendor.NVIDIA] = GPUOptimizationProfile(
            vendor=GPUVendor.NVIDIA,
            codec_preferences=['h264_nvenc', 'hevc_nvenc'],
            optimal_batch_size=4,
            memory_buffer_mb=512,
            max_concurrent_streams=2,
            preferred_pixel_format='yuv420p',
            encoding_preset='p4',  # Balanced preset
            quality_settings={
                'cq': 23,
                'rc': 'vbr',
                'spatial_aq': 1,
                'temporal_aq': 1
            },
            ffmpeg_extra_args=[
                '-gpu', '0',
                '-delay', '0',
                '-b_ref_mode', '1'
            ]
        )
        
        # Intel optimization profile
        profiles[GPUVendor.INTEL] = GPUOptimizationProfile(
            vendor=GPUVendor.INTEL,
            codec_preferences=['h264_qsv', 'hevc_qsv'],
            optimal_batch_size=6,
            memory_buffer_mb=256,
            max_concurrent_streams=3,
            preferred_pixel_format='nv12',
            encoding_preset='balanced',
            quality_settings={
                'global_quality': 23,
                'look_ahead': 1,
                'b_strategy': 1
            },
            ffmpeg_extra_args=[
                '-async_depth', '4',
                '-refs', '3'
            ]
        )
        
        # AMD optimization profile
        profiles[GPUVendor.AMD] = GPUOptimizationProfile(
            vendor=GPUVendor.AMD,
            codec_preferences=['h264_vaapi', 'hevc_vaapi'],
            optimal_batch_size=3,
            memory_buffer_mb=384,
            max_concurrent_streams=2,
            preferred_pixel_format='nv12',
            encoding_preset='medium',
            quality_settings={
                'quality': 23,
                'rc_mode': 'CQP'
            },
            ffmpeg_extra_args=[
                '-vaapi_device', '/dev/dri/renderD128'
            ]
        )
        
        return profiles
    
    def start_monitoring(self, interval: float = 5.0):
        """Start GPU monitoring in background thread"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_gpus,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started GPU monitoring")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped GPU monitoring")
    
    def _monitor_gpus(self, interval: float):
        """Monitor GPU utilization and health"""
        while self._monitoring_active:
            try:
                # Update GPU utilization for NVIDIA GPUs
                for gpu in self.available_gpus:
                    if gpu.vendor == GPUVendor.NVIDIA:
                        self._update_nvidia_stats(gpu)
                
                time.sleep(interval)
            
            except Exception as e:
                logger.warning(f"GPU monitoring error: {e}")
                time.sleep(interval)
    
    def _update_nvidia_stats(self, gpu: GPUInfo):
        """Update NVIDIA GPU statistics"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                f'--id={gpu.id}',
                '--query-gpu=memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                parts = [part.strip() for part in result.stdout.strip().split(',')]
                if len(parts) >= 2:
                    gpu.memory_free = int(parts[0])
                    gpu.utilization = float(parts[1])
                    if len(parts) > 2 and parts[2] != '[Not Supported]':
                        gpu.temperature = float(parts[2])
                    if len(parts) > 3 and parts[3] != '[Not Supported]':
                        gpu.power_usage = float(parts[3])
        
        except Exception:
            pass  # Ignore monitoring errors
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive GPU status report"""
        report = {
            'total_gpus': len(self.available_gpus),
            'available_gpus': len([g for g in self.available_gpus if g.is_available]),
            'vendors': {},
            'memory_usage': {},
            'utilization': {},
            'gpus': []
        }
        
        for gpu in self.available_gpus:
            # Vendor distribution
            vendor_name = gpu.vendor.value
            if vendor_name not in report['vendors']:
                report['vendors'][vendor_name] = 0
            report['vendors'][vendor_name] += 1
            
            # Memory usage
            allocated = self.allocated_memory.get(gpu.id, 0)
            report['memory_usage'][f'gpu_{gpu.id}'] = {
                'total': gpu.memory_total,
                'free': gpu.memory_free,
                'allocated': allocated,
                'usage_percent': gpu.memory_usage_percent
            }
            
            # Utilization
            report['utilization'][f'gpu_{gpu.id}'] = gpu.utilization
            
            # Individual GPU info
            report['gpus'].append({
                'id': gpu.id,
                'name': gpu.name,
                'vendor': vendor_name,
                'memory_total': gpu.memory_total,
                'memory_free': gpu.memory_free,
                'utilization': gpu.utilization,
                'temperature': gpu.temperature,
                'capabilities': [cap.value for cap in gpu.capabilities],
                'is_available': gpu.is_available
            })
        
        return report


# Global GPU resource manager instance
gpu_manager = GPUResourceManager()


def get_gpu_manager() -> GPUResourceManager:
    """Get the global GPU resource manager instance"""
    return gpu_manager


def initialize_gpu_resources(start_monitoring: bool = True) -> GPUResourceManager:
    """Initialize GPU resources and optionally start monitoring"""
    manager = get_gpu_manager()
    manager.refresh_gpu_list()
    
    if start_monitoring and manager.available_gpus:
        manager.start_monitoring()
    
    return manager