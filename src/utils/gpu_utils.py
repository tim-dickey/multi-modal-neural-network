"""GPU detection and configuration utilities."""

from typing import Dict, Any, Optional, Tuple
import torch
import platform
import subprocess


def _detect_external_gpu(gpu_id: int, gpu_name: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if a GPU is external (eGPU via Thunderbolt, USB-C, etc.).
    
    Args:
        gpu_id: GPU device ID
        gpu_name: Name of the GPU
    
    Returns:
        Tuple of (is_external: bool, connection_type: Optional[str])
        connection_type can be: 'Thunderbolt', 'USB-C', 'PCIe External', 'Unknown', or None
    """
    system = platform.system()
    
    try:
        if system == 'Windows':
            # Use PowerShell to query PCIe device information
            # External GPUs typically show up with specific bus types
            cmd = f'''
                $gpu = Get-PnpDevice -Class Display | Where-Object {{$_.FriendlyName -like "*{gpu_name.split()[0]}*"}} | Select-Object -First 1;
                if ($gpu) {{
                    $parent = Get-PnpDeviceProperty -InstanceId $gpu.InstanceId -KeyName "DEVPKEY_Device_Parent" | Select-Object -ExpandProperty Data;
                    $parentDevice = Get-PnpDevice -InstanceId $parent;
                    $busType = Get-PnpDeviceProperty -InstanceId $parent -KeyName "DEVPKEY_Device_BusTypeGuid" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Data;
                    Write-Output "$($parentDevice.FriendlyName)|$busType";
                }}
            '''
            result = subprocess.run(
                ['powershell', '-Command', cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip().lower()
                # Check for Thunderbolt indicators
                if 'thunderbolt' in output or 'usb4' in output:
                    return True, 'Thunderbolt'
                # Check for USB-C eGPU enclosures
                elif 'usb' in output and 'type-c' in output:
                    return True, 'USB-C'
                # Check for external PCIe indicators
                elif 'external' in output:
                    return True, 'PCIe External'
                    
        elif system == 'Linux':
            # Use lspci to detect Thunderbolt GPUs
            try:
                result = subprocess.run(
                    ['lspci', '-vv'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if 'VGA' in line or 'Display' in line:
                            # Check next few lines for Thunderbolt
                            for j in range(i, min(i+20, len(lines))):
                                check_line = lines[j].lower()
                                if 'thunderbolt' in check_line:
                                    return True, 'Thunderbolt'
                                elif 'external' in check_line:
                                    return True, 'PCIe External'
            except FileNotFoundError:
                pass
                
        elif system == 'Darwin':  # macOS
            # Use system_profiler to detect Thunderbolt eGPUs
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPThunderboltDataType', 'SPDisplaysDataType'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    output = result.stdout.lower()
                    if 'egpu' in output or ('thunderbolt' in output and 'display' in output):
                        return True, 'Thunderbolt'
            except FileNotFoundError:
                pass
    
    except Exception:
        # If detection fails, assume internal GPU
        pass
    
    # Default: assume internal GPU
    return False, None


def detect_gpu_info() -> Dict[str, Any]:
    """
    Detect available GPU information and capabilities.
    Includes detection of external GPUs (eGPUs) via Thunderbolt, USB-C, or other connections.
    
    Returns:
        Dictionary containing GPU information:
        - available: bool, whether CUDA is available
        - device_count: int, number of GPUs
        - devices: list of device information dicts (includes external GPU info)
        - external_gpu_count: int, number of external GPUs detected
        - recommended_device: str, recommended device string ('cuda' or 'cpu')
        - compute_capability: tuple or None
        - memory_info: dict with memory details
    """
    info = {
        'available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
        'external_gpu_count': 0,
        'recommended_device': 'cpu',
        'compute_capability': None,
        'memory_info': {},
        'cuda_version': None,
        'cudnn_version': None,
    }
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Training will use CPU.")
        print("  To enable GPU training:")
        print("  1. Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads")
        print("  2. Reinstall PyTorch with CUDA support:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return info
    
    # CUDA is available
    info['device_count'] = torch.cuda.device_count()
    info['recommended_device'] = 'cuda'
    info['cuda_version'] = torch.version.cuda
    
    if torch.backends.cudnn.is_available():
        info['cudnn_version'] = torch.backends.cudnn.version()
    
    # Get information for each GPU
    for i in range(info['device_count']):
        device_props = torch.cuda.get_device_properties(i)
        
        # Detect if this is an external GPU
        is_external, connection_type = _detect_external_gpu(i, device_props.name)
        
        device_info = {
            'id': i,
            'name': device_props.name,
            'compute_capability': (device_props.major, device_props.minor),
            'total_memory_gb': device_props.total_memory / (1024**3),
            'multi_processor_count': device_props.multi_processor_count,
            'is_external': is_external,
            'connection_type': connection_type,
        }
        
        if is_external:
            info['external_gpu_count'] += 1
        
        # Get current memory usage
        try:
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            mem_free = (device_props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
            
            device_info['memory_allocated_gb'] = mem_allocated
            device_info['memory_reserved_gb'] = mem_reserved
            device_info['memory_free_gb'] = mem_free
        except (RuntimeError, Exception):
            pass
        
        info['devices'].append(device_info)
    
    # Set compute capability for primary device
    if info['devices']:
        info['compute_capability'] = info['devices'][0]['compute_capability']
        info['memory_info'] = {
            'total_gb': info['devices'][0]['total_memory_gb'],
            'free_gb': info['devices'][0].get('memory_free_gb', 0),
        }
    
    return info


def print_gpu_info(info: Optional[Dict[str, Any]] = None) -> None:
    """
    Print formatted GPU information.
    
    Args:
        info: GPU info dict from detect_gpu_info(). If None, will detect automatically.
    """
    if info is None:
        info = detect_gpu_info()
    
    print("\n" + "="*70)
    print("GPU CONFIGURATION")
    print("="*70)
    
    if not info['available']:
        print("Status: ❌ No CUDA GPUs detected")
        print(f"Recommended device: {info['recommended_device']}")
        return
    
    print(f"Status: ✅ CUDA available")
    print(f"CUDA Version: {info['cuda_version']}")
    if info['cudnn_version']:
        print(f"cuDNN Version: {info['cudnn_version']}")
    print(f"Number of GPUs: {info['device_count']}")
    if info['external_gpu_count'] > 0:
        print(f"  └─ External GPUs: {info['external_gpu_count']}")
        print(f"  └─ Internal GPUs: {info['device_count'] - info['external_gpu_count']}")
    print(f"Recommended device: {info['recommended_device']}")
    
    print(f"\n{'GPU Details:'}")
    print("-" * 70)
    
    for device in info['devices']:
        gpu_type = "🔌 External" if device.get('is_external', False) else "💻 Internal"
        print(f"\n  GPU {device['id']}: {device['name']} ({gpu_type})")
        if device.get('is_external', False) and device.get('connection_type'):
            print(f"    Connection: {device['connection_type']}")
        print(f"    Compute Capability: {device['compute_capability'][0]}.{device['compute_capability'][1]}")
        print(f"    Total Memory: {device['total_memory_gb']:.2f} GB")
        if 'memory_free_gb' in device:
            print(f"    Free Memory: {device['memory_free_gb']:.2f} GB")
            print(f"    Allocated Memory: {device['memory_allocated_gb']:.2f} GB")
        print(f"    Multi-Processors: {device['multi_processor_count']}")
        
        # Provide recommendations based on compute capability
        cc = device['compute_capability']
        if cc >= (7, 0):
            print(f"    ✅ Supports mixed precision training (Tensor Cores available)")
        else:
            print(f"    ⚠ Limited mixed precision support (compute capability < 7.0)")
    
    print("\n" + "="*70)


def get_optimal_device(prefer_gpu: bool = True) -> str:
    """
    Get the optimal device string for training.
    
    Args:
        prefer_gpu: If True, prefer GPU when available. Otherwise use CPU.
    
    Returns:
        Device string ('cuda', 'cuda:0', 'cpu', etc.)
    """
    if not prefer_gpu:
        return 'cpu'
    
    info = detect_gpu_info()
    return info['recommended_device']


def configure_device_for_training(
    device: Optional[str] = None,
    gpu_id: Optional[int] = None,
    verbose: bool = True
) -> torch.device:
    """
    Configure and return a torch device for training.
    
    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.). If None, auto-detect.
        gpu_id: Specific GPU ID to use. Overrides device if both specified.
        verbose: Print device information.
    
    Returns:
        torch.device object configured for training.
    """
    # Handle gpu_id override
    if gpu_id is not None:
        if not torch.cuda.is_available():
            if verbose:
                print(f"⚠ GPU {gpu_id} requested but CUDA not available. Using CPU.")
            return torch.device('cpu')
        if gpu_id >= torch.cuda.device_count():
            if verbose:
                print(f"⚠ GPU {gpu_id} requested but only {torch.cuda.device_count()} GPUs available. Using GPU 0.")
            gpu_id = 0
        device = f'cuda:{gpu_id}'
    
    # Auto-detect if not specified
    if device is None:
        device = get_optimal_device()
    
    # Validate device string
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            if verbose:
                print(f"⚠ CUDA device '{device}' requested but CUDA not available. Using CPU.")
            device = 'cpu'
    
    torch_device = torch.device(device)
    
    if verbose:
        print(f"\n✅ Using device: {torch_device}")
        if torch_device.type == 'cuda':
            gpu_info = detect_gpu_info()
            if gpu_info['devices']:
                idx = torch_device.index if torch_device.index is not None else 0
                if idx < len(gpu_info['devices']):
                    dev_info = gpu_info['devices'][idx]
                    print(f"   GPU: {dev_info['name']}")
                    print(f"   Memory: {dev_info['total_memory_gb']:.2f} GB")
    
    return torch_device


def check_mixed_precision_support() -> Dict[str, bool]:
    """
    Check what types of mixed precision training are supported.
    
    Returns:
        Dictionary with support flags for different precision types.
    """
    support = {
        'fp16': False,
        'bf16': False,
        'tf32': False,
    }
    
    if not torch.cuda.is_available():
        return support
    
    # FP16 is generally supported on all CUDA devices
    support['fp16'] = True
    
    # BF16 requires compute capability >= 8.0 (Ampere and newer)
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = (device_props.major, device_props.minor)
    
    if compute_capability >= (8, 0):
        support['bf16'] = True
    
    # TF32 is available on Ampere (8.0) and newer
    if compute_capability >= (8, 0):
        support['tf32'] = True
    
    return support


if __name__ == '__main__':
    # Demo: Print GPU information
    info = detect_gpu_info()
    print_gpu_info(info)
    
    print("\nMixed Precision Support:")
    mp_support = check_mixed_precision_support()
    for precision, supported in mp_support.items():
        status = "✅" if supported else "❌"
        print(f"  {status} {precision.upper()}")
    
    print("\nRecommended Configuration:")
    device = configure_device_for_training(verbose=True)
