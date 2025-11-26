"""NPU (Neural Processing Unit) detection and configuration utilities."""

from typing import Dict, Any, Optional, List, Tuple
import platform
import subprocess


def _detect_external_npu() -> Tuple[bool, Optional[str]]:
    """
    Detect if an external NPU is connected (USB, Thunderbolt, PCIe expansion).
    
    External NPUs include:
    - Intel Movidius Neural Compute Stick (USB)
    - Google Coral Edge TPU (USB/PCIe)
    - Hailo AI acceleration cards (PCIe/M.2)
    - Intel Neural Compute Stick 2 (USB)
    - External AI accelerator boxes via Thunderbolt
    
    Returns:
        Tuple of (is_external: bool, device_name: Optional[str])
    """
    system = platform.system()
    
    try:
        if system == 'Windows':
            # Check for USB-connected NPUs
            cmd = '''
                Get-PnpDevice | Where-Object {
                    $_.FriendlyName -like "*Neural*" -or 
                    $_.FriendlyName -like "*Coral*" -or
                    $_.FriendlyName -like "*Movidius*" -or
                    $_.FriendlyName -like "*Hailo*" -or
                    $_.FriendlyName -like "*Edge TPU*"
                } | Select-Object FriendlyName, InstanceId
            '''
            result = subprocess.run(
                ['powershell', '-Command', cmd],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if any(keyword in line for keyword in ['Coral', 'Movidius', 'Hailo', 'TPU']):
                        # Extract device name
                        if 'Coral' in line:
                            return True, 'Google Coral Edge TPU'
                        elif 'Movidius' in line:
                            return True, 'Intel Movidius NCS'
                        elif 'Hailo' in line:
                            return True, 'Hailo AI Accelerator'
                        else:
                            return True, 'External NPU'
                            
        elif system == 'Linux':
            # Check for USB devices
            try:
                result = subprocess.run(
                    ['lsusb'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False
                )
                
                if result.returncode == 0:
                    output = result.stdout.lower()
                    if 'movidius' in output or 'neural compute stick' in output:
                        return True, 'Intel Movidius NCS'
                    elif 'coral' in output or 'edge tpu' in output:
                        return True, 'Google Coral Edge TPU'
            except FileNotFoundError:
                pass
            
            # Check for PCIe devices (Hailo, Coral M.2)
            try:
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False
                )
                
                if result.returncode == 0:
                    output = result.stdout.lower()
                    if 'hailo' in output:
                        return True, 'Hailo AI Accelerator'
                    elif 'coral' in output:
                        return True, 'Google Coral Edge TPU'
            except FileNotFoundError:
                pass
                
        elif system == 'Darwin':  # macOS
            # Check USB devices
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPUSBDataType'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False
                )
                
                if result.returncode == 0:
                    output = result.stdout.lower()
                    if 'movidius' in output or 'neural compute stick' in output:
                        return True, 'Intel Movidius NCS'
                    elif 'coral' in output or 'edge tpu' in output:
                        return True, 'Google Coral Edge TPU'
            except FileNotFoundError:
                pass
    
    except (subprocess.TimeoutExpired, OSError):
        pass
    
    return False, None


def detect_npu_info() -> Dict[str, Any]:
    """
    Detect available NPU (Neural Processing Unit) information and capabilities.
    
    NPUs are specialized processors for AI/ML workloads found in:
    - Intel Core Ultra processors (Intel AI Boost)
    - AMD Ryzen AI processors
    - Apple Silicon (Neural Engine)
    - Qualcomm Snapdragon (Hexagon NPU)
    - Microsoft's custom NPUs in Surface devices
    - External NPUs (Coral Edge TPU, Movidius NCS, Hailo AI)
    
    Returns:
        Dictionary containing NPU information:
        - available: bool, whether NPU is detected
        - npu_type: str, type of NPU detected
        - device_name: str, name of the NPU
        - backend: str, backend/framework for NPU ('openvino', 'directml', 'coreml', etc.)
        - capabilities: dict with NPU capabilities
        - recommended_device: str, device string to use
        - is_external: bool, whether NPU is external device
        - connection_type: str, connection type if external
    """
    info = {
        'available': False,
        'npu_type': None,
        'device_name': None,
        'backend': None,
        'capabilities': {},
        'recommended_device': None,
        'detection_method': None,
        'is_external': False,
        'connection_type': None,
    }
    
    # Check for external NPU first
    is_external, external_device = _detect_external_npu()
    if is_external and external_device:
        info['available'] = True
        info['is_external'] = True
        info['device_name'] = external_device
        info['npu_type'] = 'External NPU'
        info['connection_type'] = 'USB/PCIe'
        info['detection_method'] = 'External device detection'
        
        # Determine backend based on device type
        if 'Coral' in external_device:
            info['backend'] = 'TensorFlow Lite'
            info['recommended_device'] = 'edge_tpu'
        elif 'Movidius' in external_device:
            info['backend'] = 'OpenVINO'
            info['recommended_device'] = 'openvino'
        elif 'Hailo' in external_device:
            info['backend'] = 'Hailo Runtime'
            info['recommended_device'] = 'hailo'
        else:
            info['backend'] = 'Unknown'
            info['recommended_device'] = 'cpu'
            
        info['capabilities'] = {
            'int8': True,
            'fp16': True,
            'inference_only': True,
            'external': True,
        }
        return info
    
    system = platform.system()
    processor = platform.processor()
    
    # Detect Intel NPU (AI Boost in Core Ultra processors)
    if _detect_intel_npu():
        info['available'] = True
        info['npu_type'] = 'Intel AI Boost'
        info['device_name'] = 'Intel NPU'
        info['backend'] = 'openvino'  # Intel's OpenVINO toolkit
        info['recommended_device'] = 'openvino'
        info['detection_method'] = 'Intel VPU detection'
        info['capabilities'] = {
            'int8': True,
            'fp16': True,
            'inference_only': True,  # Most NPUs are inference-optimized
        }
        return info
    
    # Detect AMD NPU (Ryzen AI)
    if _detect_amd_npu():
        info['available'] = True
        info['npu_type'] = 'AMD Ryzen AI'
        info['device_name'] = 'AMD NPU'
        info['backend'] = 'ryzenai'  # AMD Ryzen AI SDK
        info['recommended_device'] = 'ryzenai'
        info['detection_method'] = 'AMD Ryzen AI detection'
        info['capabilities'] = {
            'int8': True,
            'fp16': True,
            'inference_only': True,
        }
        return info
    
    # Detect Apple Neural Engine (macOS only)
    if system == 'Darwin':
        npu_info = _detect_apple_neural_engine()
        if npu_info['available']:
            info.update(npu_info)
            return info
    
    # Detect DirectML NPU (Windows only)
    if system == 'Windows':
        npu_info = _detect_windows_directml_npu()
        if npu_info['available']:
            info.update(npu_info)
            return info
    
    # No NPU detected
    return info


def _detect_intel_npu() -> bool:
    """
    Detect Intel NPU (VPU/AI Boost).
    
    Intel NPUs are available in:
    - Intel Core Ultra (Meteor Lake and newer)
    - Intel Core 14th gen with AI Boost
    
    Returns:
        True if Intel NPU is detected
    """
    try:
        # Check for Intel VPU via OpenVINO detection
        # This is a placeholder - actual detection would require OpenVINO installed
        import subprocess
        
        # Try to detect via Windows Device Manager (Windows only)
        if platform.system() == 'Windows':
            try:
                result = subprocess.run(
                    ['powershell', '-Command', 
                     'Get-PnpDevice | Where-Object {$_.FriendlyName -like "*NPU*" -or $_.FriendlyName -like "*VPU*" -or $_.FriendlyName -like "*AI Boost*"}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except Exception:
                pass
        
        # Try to import OpenVINO and check for VPU device
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices()
            # Look for VPU or NPU device
            for device in devices:
                if 'VPU' in device or 'NPU' in device:
                    return True
        except ImportError:
            pass
        
        return False
        
    except Exception:
        return False


def _detect_amd_npu() -> bool:
    """
    Detect AMD Ryzen AI NPU.
    
    AMD NPUs are available in:
    - AMD Ryzen 7040 series and newer
    - AMD Ryzen AI processors
    
    Returns:
        True if AMD NPU is detected
    """
    try:
        processor_info = platform.processor().lower()
        
        # Check for AMD Ryzen AI in processor name
        if 'amd' in processor_info and 'ryzen' in processor_info:
            # Try to detect via Windows Device Manager
            if platform.system() == 'Windows':
                try:
                    result = subprocess.run(
                        ['powershell', '-Command', 
                         'Get-PnpDevice | Where-Object {$_.FriendlyName -like "*Ryzen AI*" -or $_.FriendlyName -like "*AMD NPU*"}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return True
                except Exception:
                    pass
        
        # Try to check for AMD Ryzen AI SDK
        try:
            import ryzen_ai
            return True
        except ImportError:
            pass
        
        return False
        
    except Exception:
        return False


def _detect_apple_neural_engine() -> Dict[str, Any]:
    """
    Detect Apple Neural Engine (ANE).
    
    Available on Apple Silicon (M1, M2, M3, A-series chips).
    
    Returns:
        Dictionary with detection info
    """
    info = {
        'available': False,
        'npu_type': None,
        'device_name': None,
        'backend': None,
        'capabilities': {},
        'recommended_device': None,
        'detection_method': None,
    }
    
    try:
        if platform.system() != 'Darwin':
            return info
        
        # Check for Apple Silicon
        machine = platform.machine()
        if machine == 'arm64':
            # Apple Silicon detected
            info['available'] = True
            info['npu_type'] = 'Apple Neural Engine'
            info['device_name'] = 'Apple Neural Engine'
            info['backend'] = 'coreml'  # Core ML framework
            info['recommended_device'] = 'mps'  # Metal Performance Shaders
            info['detection_method'] = 'Apple Silicon detection'
            info['capabilities'] = {
                'int8': True,
                'fp16': True,
                'fp32': True,
                'inference_only': False,  # ANE supports training via Core ML
            }
            
            # Try to get more specific info about the chip
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    chip_name = result.stdout.strip()
                    info['device_name'] = f'Apple Neural Engine ({chip_name})'
            except Exception:
                pass
        
        return info
        
    except Exception:
        return info


def _detect_windows_directml_npu() -> Dict[str, Any]:
    """
    Detect NPU via DirectML on Windows.
    
    DirectML can utilize various NPUs on Windows including:
    - Intel NPU
    - AMD NPU
    - Qualcomm NPU
    
    Returns:
        Dictionary with detection info
    """
    info = {
        'available': False,
        'npu_type': None,
        'device_name': None,
        'backend': None,
        'capabilities': {},
        'recommended_device': None,
        'detection_method': None,
    }
    
    try:
        if platform.system() != 'Windows':
            return info
        
        # Try to import DirectML
        try:
            import torch_directml
            
            # Check if DirectML device is available
            if torch_directml.is_available():
                device_count = torch_directml.device_count()
                if device_count > 0:
                    info['available'] = True
                    info['npu_type'] = 'DirectML NPU'
                    info['device_name'] = torch_directml.device_name(0)
                    info['backend'] = 'directml'
                    info['recommended_device'] = 'privateuseone'  # PyTorch DirectML device
                    info['detection_method'] = 'DirectML detection'
                    info['capabilities'] = {
                        'int8': True,
                        'fp16': True,
                        'fp32': True,
                        'inference_only': False,
                    }
        except ImportError:
            pass
        
        return info
        
    except Exception:
        return info


def print_npu_info(info: Optional[Dict[str, Any]] = None) -> None:
    """
    Print formatted NPU information.
    
    Args:
        info: NPU info dict from detect_npu_info(). If None, will detect automatically.
    """
    if info is None:
        info = detect_npu_info()
    
    print("\n" + "="*70)
    print("NPU (NEURAL PROCESSING UNIT) CONFIGURATION")
    print("="*70)
    
    if not info['available']:
        print("Status: ❌ No NPU detected")
        print("\nNPUs are specialized AI accelerators found in:")
        print("  • Intel Core Ultra processors (AI Boost)")
        print("  • AMD Ryzen AI processors")
        print("  • Apple Silicon (M1/M2/M3 Neural Engine)")
        print("  • Qualcomm Snapdragon (Hexagon NPU)")
        print("\nTo use NPU acceleration:")
        print("  1. Ensure you have compatible hardware")
        print("  2. Install appropriate SDK:")
        print("     - Intel: pip install openvino")
        print("     - AMD: Install Ryzen AI SDK")
        print("     - Apple: Use Core ML (built-in)")
        print("     - Windows: pip install torch-directml")
        return
    
    npu_location = "🔌 External" if info.get('is_external', False) else "💻 Internal"
    print(f"Status: ✅ NPU detected ({npu_location})")
    print(f"NPU Type: {info['npu_type']}")
    print(f"Device Name: {info['device_name']}")
    if info.get('is_external', False) and info.get('connection_type'):
        print(f"Connection: {info['connection_type']}")
    print(f"Backend: {info['backend']}")
    print(f"Detection Method: {info['detection_method']}")
    print(f"Recommended Device: {info['recommended_device']}")
    
    print(f"\nCapabilities:")
    for capability, supported in info['capabilities'].items():
        status = "✅" if supported else "❌"
        print(f"  {status} {capability.upper()}")
    
    print("\n" + "="*70)


def check_accelerator_availability() -> Dict[str, bool]:
    """
    Check availability of all hardware accelerators.
    
    Returns:
        Dictionary with availability of each accelerator type:
        - cuda: NVIDIA GPU
        - mps: Apple Metal (M1/M2/M3)
        - npu: Neural Processing Unit
        - cpu: CPU (always available)
    """
    import torch
    
    availability = {
        'cpu': True,  # Always available
        'cuda': torch.cuda.is_available(),
        'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'npu': detect_npu_info()['available'],
    }
    
    return availability


def get_best_available_device(prefer_npu: bool = False) -> str:
    """
    Get the best available device for training/inference.
    
    Priority (default):
    1. CUDA (if available)
    2. MPS (if available)
    3. NPU (if available)
    4. CPU
    
    With prefer_npu=True:
    1. NPU (if available)
    2. CUDA (if available)
    3. MPS (if available)
    4. CPU
    
    Args:
        prefer_npu: If True, prefer NPU over GPU when both available
    
    Returns:
        Device string to use
    """
    import torch
    
    if prefer_npu:
        npu_info = detect_npu_info()
        if npu_info['available']:
            return npu_info['recommended_device']
    
    # Check for CUDA
    if torch.cuda.is_available():
        return 'cuda'
    
    # Check for Apple MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    
    # Check for NPU (if not already preferred)
    if not prefer_npu:
        npu_info = detect_npu_info()
        if npu_info['available']:
            return npu_info['recommended_device']
    
    # Fallback to CPU
    return 'cpu'


if __name__ == '__main__':
    # Demo: Print NPU information
    print("Detecting NPU...")
    npu_info = detect_npu_info()
    print_npu_info(npu_info)
    
    print("\n\nAll Accelerator Availability:")
    availability = check_accelerator_availability()
    for device, available in availability.items():
        status = "✅" if available else "❌"
        print(f"  {status} {device.upper()}")
    
    print("\n\nRecommended Device:")
    device = get_best_available_device(prefer_npu=False)
    print(f"  Default: {device}")
    device_with_npu = get_best_available_device(prefer_npu=True)
    print(f"  With NPU preference: {device_with_npu}")
