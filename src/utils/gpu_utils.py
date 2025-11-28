"""GPU detection and configuration utilities."""

import logging
import platform
import shutil
import subprocess
import re
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _get_nvml_info() -> dict[str, Any]:
    """Attempt to collect GPU information via NVML (pynvml).

    Returns a dict with keys:
        - available: bool
        - driver: str | None
        - devices: list of device info dicts (id, name, total_memory_gb)
    """
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode(errors="ignore")

        count = pynvml.nvmlDeviceGetCount()
        devices: list[dict[str, Any]] = []
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode(errors="ignore")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = float(mem.total) / (1024**3)
            devices.append({"id": idx, "name": name, "total_memory_gb": total_gb})

        try:
            pynvml.nvmlShutdown()
        except (pynvml.NVMLError, OSError):
            # Best-effort cleanup; ignore NVML shutdown errors
            pass

        return {"available": True, "driver": driver, "devices": devices}
    except (ImportError, OSError):
        return {"available": False}


def _detect_external_gpu(gpu_id: int, gpu_name: str) -> tuple[bool, str | None]:
    """
    Detect if a GPU is external (eGPU via Thunderbolt, USB-C, etc.).

    Args:

        gpu_id: GPU device ID
        gpu_name: Name of the GPU

    Returns:

        Tuple of (is_external: bool, connection_type: Optional[str]).
        Examples for `connection_type`: 'Thunderbolt', 'USB-C', or 'PCIe External'.
    """
    system = platform.system()

    try:
        if system == "Windows":
            return _detect_external_gpu_windows(gpu_id, gpu_name)
        if system == "Linux":
            return _detect_external_gpu_linux()
        if system == "Darwin":
            return _detect_external_gpu_darwin()
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        # If detection fails due to OS/subprocess issues, assume internal GPU
        pass

    return False, None


def _detect_external_gpu_windows(gpu_id: int, gpu_name: str) -> tuple[bool, str | None]:
    """Windows-specific external GPU detection using PowerShell."""
    # Build the PowerShell command as a parameterized script and pass the
    # GPU name as an argument to avoid embedding untrusted input directly
    # into the command string. We also sanitize the value to remove any
    # control or metacharacters as a defence-in-depth measure.
    safe_fragment = ""
    try:
        first_token = gpu_name.split()[0] if gpu_name else ""
        # Allow alphanumerics, space, underscore and hyphen only
        safe_fragment = re.sub(r"[^A-Za-z0-9 _-]", "", first_token)
    except Exception:
        safe_fragment = ""

    # PowerShell script accepts a parameter ($name) and uses it in a -like
    # comparison. Passing via -ArgumentList avoids shell interpolation risks.
    cmd_parts = [
        "param($name); ",
        "$gpu = Get-PnpDevice -Class Display | Where-Object { ",
        "$_.FriendlyName -like \"*$name*\" } ",
        "| Select-Object -First 1;",
        " if ($gpu) { ",
        " $parent = Get-PnpDeviceProperty -InstanceId $gpu.InstanceId ",
        "-KeyName 'DEVPKEY_Device_Parent' | Select-Object -ExpandProperty Data;",
        " $parentDevice = Get-PnpDevice -InstanceId $parent;",
        " $busType = Get-PnpDeviceProperty -InstanceId $parent ",
        "-KeyName 'DEVPKEY_Device_BusTypeGuid' -ErrorAction SilentlyContinue ",
        "| Select-Object -ExpandProperty Data;",
        " Write-Output \"$($parentDevice.FriendlyName)|$busType\"; }",
    ]
    cmd = "".join(cmd_parts)

    # Only run PowerShell if available
    if not shutil.which("powershell"):
        return False, None

    result = subprocess.run(
        ["powershell", "-Command", cmd, "-ArgumentList", safe_fragment],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    if result.returncode == 0 and result.stdout.strip():
        output = result.stdout.strip().lower()
        if "thunderbolt" in output or "usb4" in output:
            return True, "Thunderbolt"
        if "usb" in output and "type-c" in output:
            return True, "USB-C"
        if "external" in output:
            return True, "PCIe External"

    return False, None


def _detect_external_gpu_linux() -> tuple[bool, str | None]:
    """Linux-specific external GPU detection using lspci output."""
    if not shutil.which("lspci"):
        return False, None

    result = subprocess.run(
        ["lspci", "-vv"], capture_output=True, text=True, timeout=5, check=False
    )
    if result.returncode == 0:
        lines = result.stdout.split("\n")
        for i, line in enumerate(lines):
            if "VGA" in line or "Display" in line:
                for j in range(i, min(i + 20, len(lines))):
                    check_line = lines[j].lower()
                    if "thunderbolt" in check_line:
                        return True, "Thunderbolt"
                    if "external" in check_line:
                        return True, "PCIe External"
    return False, None


def _detect_external_gpu_darwin() -> tuple[bool, str | None]:
    """macOS-specific external GPU detection using system_profiler."""
    if not shutil.which("system_profiler"):
        return False, None

    result = subprocess.run(
        ["system_profiler", "SPThunderboltDataType", "SPDisplaysDataType"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode == 0:
        output = result.stdout.lower()
        if "egpu" in output or ("thunderbolt" in output and "display" in output):
            return True, "Thunderbolt"
    return False, None


def _query_nvidia_smi() -> list[str]:
    """Return lines from `nvidia-smi -L` when available, else empty list."""
    if not shutil.which("nvidia-smi"):
        return []

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except subprocess.TimeoutExpired:
        # nvidia-smi timed out
        pass
    return []


def _populate_nvidia_smi_info(info: dict[str, Any], parsed: list[str]) -> None:
    """Populate `info` dict with parsed nvidia-smi names.

    This is a separate helper to reduce complexity in callers.
    """
    if not parsed:
        return
    info["nvidia_smi"] = True
    parsed_names: list[str] = []
    for line in parsed:
        parts = line.split(":", 1)
        name = parts[1].strip() if len(parts) > 1 else line
        parsed_names.append(name)
    info["nvidia_gpus"] = parsed_names


def _collect_cuda_devices(info: dict[str, Any]) -> None:
    """Collect per-GPU device properties and memory info into `info` dict."""
    for i in range(info["device_count"]):
        device_props = torch.cuda.get_device_properties(i)
        is_external, connection_type = _detect_external_gpu(i, device_props.name)

        device_info = {
            "id": i,
            "name": device_props.name,
            "compute_capability": (device_props.major, device_props.minor),
            "total_memory_gb": device_props.total_memory / (1024**3),
            "multi_processor_count": device_props.multi_processor_count,
            "is_external": is_external,
            "connection_type": connection_type,
        }

        if is_external:
            info["external_gpu_count"] += 1

        try:
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            mem_free = (device_props.total_memory - torch.cuda.memory_reserved(i)) / (
                1024**3
            )

            device_info["memory_allocated_gb"] = mem_allocated
            device_info["memory_reserved_gb"] = mem_reserved
            device_info["memory_free_gb"] = mem_free
        except RuntimeError:
            # CUDA memory queries can raise runtime errors if device state changes
            pass

        info["devices"].append(device_info)

    if info["devices"]:
        info["compute_capability"] = info["devices"][0]["compute_capability"]
        info["memory_info"] = {
            "total_gb": info["devices"][0]["total_memory_gb"],
            "free_gb": info["devices"][0].get("memory_free_gb", 0),
        }


def detect_gpu_info() -> dict[str, Any]:
    """Detect available GPU information and capabilities.

    Includes basic eGPU detection (Thunderbolt, USB-C, etc.).

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
        "available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "external_gpu_count": 0,
        "recommended_device": "cpu",
        "compute_capability": None,
        "memory_info": {},
        "cuda_version": None,
        "cudnn_version": None,
        "nvidia_smi": False,
        "nvidia_gpus": [],
    }

    # Use module-level helpers for nvidia-smi parsing and population
    parsed = _query_nvidia_smi()
    _populate_nvidia_smi_info(info, parsed)

    # If CUDA not available, try nvidia-smi detection to give user guidance
    # and exit early
    if not torch.cuda.is_available():
        parsed = _query_nvidia_smi()
        _populate_nvidia_smi_info(info, parsed)
        if parsed:
            logger.warning("⚠ CUDA not available in this Python environment;")
            logger.warning("  NVIDIA GPU(s) detected via nvidia-smi.")
            logger.info("  GPUs detected:")
            for name in info["nvidia_gpus"]:
                logger.info("    - %s", name)
            logger.info("  To enable GPU training:")
            logger.info("    1) Install NVIDIA drivers + CUDA toolkit")
            logger.info("    2) Install a CUDA-enabled PyTorch wheel.")
            logger.info("      See: https://pytorch.org/get-started/locally/")
            return info

        logger.warning("⚠ CUDA not available. Training will use CPU.")
        logger.info("  To enable GPU training:")
        logger.info("  1. Install NVIDIA drivers and the CUDA toolkit.")
        logger.info("     See: https://developer.nvidia.com/cuda-downloads")
        logger.info(
            "  2. Install a CUDA-enabled PyTorch build matching your CUDA version."
        )
        logger.info("     See: https://pytorch.org/")
        return info

    # CUDA is available: collect device info
    info["device_count"] = torch.cuda.device_count()
    info["recommended_device"] = "cuda"
    info["cuda_version"] = torch.version.cuda

    if torch.backends.cudnn.is_available():
        info["cudnn_version"] = torch.backends.cudnn.version()

    # Collect per-GPU info using a module-level helper
    _collect_cuda_devices(info)

    # Try to supplement with nvidia-smi names when available (again, in case
    # device properties were collected after initial parse)
    parsed = _query_nvidia_smi()
    _populate_nvidia_smi_info(info, parsed)

    return info


def print_gpu_info(info: dict[str, Any] | None = None) -> None:
    """Print formatted GPU information.

    Args:

        info: GPU info dict from detect_gpu_info(). If None, will detect automatically.
    """
    if info is None:
        info = detect_gpu_info()
    logger.info("\n%s", "=" * 70)
    logger.info("GPU CONFIGURATION")
    logger.info("=" * 70)

    if not info["available"]:
        logger.info("Status: ❌ No CUDA GPUs detected")
        logger.info("Recommended device: %s", info["recommended_device"])
        return

    logger.info("Status: ✅ CUDA available")
    logger.info("CUDA Version: %s", info["cuda_version"])
    if info["cudnn_version"]:
        logger.info("cuDNN Version: %s", info["cudnn_version"])
    logger.info("Number of GPUs: %s", info["device_count"])
    if info["external_gpu_count"] > 0:
        logger.info("  └─ External GPUs: %s", info["external_gpu_count"])
        internal_count = info["device_count"] - info["external_gpu_count"]
        logger.info("  └─ Internal GPUs: %s", internal_count)
    logger.info("Recommended device: %s", info["recommended_device"])

    logger.info("\n%s", "GPU Details:")
    logger.info("%s", "-" * 70)

    for device in info["devices"]:
        gpu_type = "🔌 External" if device.get("is_external", False) else "💻 Internal"
        logger.info("\n  GPU %s: %s (%s)", device["id"], device["name"], gpu_type)
        if device.get("is_external", False) and device.get("connection_type"):
            logger.info("    Connection: %s", device["connection_type"])
        # Avoid long f-string lines by using intermediate variables
        cc = device["compute_capability"]
        logger.info("    Compute Capability: %s.%s", cc[0], cc[1])
        logger.info("    Total Memory: %.2f GB", device["total_memory_gb"])
        if "memory_free_gb" in device:
            logger.info("    Free Memory: %.2f GB", device["memory_free_gb"])
            logger.info("    Allocated Memory: %.2f GB", device["memory_allocated_gb"])
        logger.info("    Multi-Processors: %s", device["multi_processor_count"])

        # Provide recommendations based on compute capability
        cc = device["compute_capability"]
        if cc >= (7, 0):
            logger.info(
                "    ✅ Supports mixed precision training (Tensor Cores available)"
            )
        else:
            logger.info(
                "    ⚠ Limited mixed precision support (compute capability < 7.0)"
            )

    logger.info("\n%s", "=" * 70)


def get_optimal_device(*, prefer_gpu: bool = True) -> str:
    """Get the optimal device string for training.

    Args:

        prefer_gpu: If True, prefer GPU when available. Otherwise use CPU.

    Returns:

        Device string ('cuda', 'cuda:0', 'cpu', etc.)
    """
    if not prefer_gpu:
        return "cpu"

    info = detect_gpu_info()
    # Ensure we return a string (info dict values may be typed as Any)
    return str(info["recommended_device"])


def _handle_gpu_id_override(gpu_id: int, *, verbose: bool = True) -> str:
    """Handle explicit GPU id overrides, returning a device string or 'cpu'."""
    if not torch.cuda.is_available():
        if verbose:
            logger.warning("⚠ GPU %s requested but CUDA not available.", gpu_id)
            logger.info("Using CPU.")
        return "cpu"
    total = torch.cuda.device_count()
    if gpu_id >= total:
        if verbose:
            logger.warning(
                "⚠ GPU %s requested but only %s GPUs available.", gpu_id, total
            )
            logger.info("Using GPU 0.")
        return "cuda:0"
    return f"cuda:{gpu_id}"


def _ensure_valid_device_string(dev_str: str, *, verbose: bool = True) -> str:
    """Validate device string and fall back to CPU if CUDA not available."""
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        if verbose:
            logger.warning(
                "⚠ CUDA device '%s' requested but CUDA not available.", dev_str
            )
            logger.info("Using CPU.")
        return "cpu"
    return dev_str


def _print_gpu_verbose(dev: torch.device) -> None:
    """Print additional GPU info for a selected torch.device."""
    gpu_info = detect_gpu_info()
    if gpu_info["devices"]:
        idx = dev.index if dev.index is not None else 0
        if idx < len(gpu_info["devices"]):
            dev_info = gpu_info["devices"][idx]
            logger.info("   GPU: %s", dev_info["name"])
            logger.info("   Memory: %.2f GB", dev_info["total_memory_gb"])


def _resolve_device_string(
    device: str | None, gpu_id: int | None, *, verbose: bool = True
) -> str:
    """Resolve and validate the device string to use for training.

    Handles gpu_id overrides, auto-detection, and validation.
    """
    # Handle gpu_id override
    if gpu_id is not None:
        return _handle_gpu_id_override(gpu_id, verbose=verbose)

    # Auto-detect if not specified
    if device is None:
        device = get_optimal_device()

    return _ensure_valid_device_string(device, verbose=verbose)


def configure_device_for_training(
    device: str | None = None, gpu_id: int | None = None, *, verbose: bool = True
) -> torch.device:
    """Configure and return a torch device for training.

    Args:

        device: Device string ('cuda', 'cpu', 'cuda:0', etc.). If None, auto-detect.
        gpu_id: Specific GPU ID to use. Overrides device if both specified.
        verbose: Print device information.

    Returns:

        torch.device object configured for training.
    """

    # Resolve and validate the device string using the module-level helper.
    resolved = _resolve_device_string(device, gpu_id, verbose=verbose)

    torch_device = torch.device(resolved)

    if verbose:
        logger.info("\n✅ Using device: %s", torch_device)
        if torch_device.type == "cuda":
            _print_gpu_verbose(torch_device)

    return torch_device


def check_mixed_precision_support() -> dict[str, bool]:
    """Check what types of mixed precision training are supported.

    Returns:
        Dictionary with support flags for different precision types.
    """
    support = {
        "fp16": False,
        "bf16": False,
        "tf32": False,
    }

    if not torch.cuda.is_available():
        return support

    # FP16 is generally supported on all CUDA devices
    support["fp16"] = True

    # BF16 requires compute capability >= 8.0 (Ampere and newer)
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = (device_props.major, device_props.minor)

    if compute_capability >= (8, 0):
        support["bf16"] = True

    # TF32 is available on Ampere (8.0) and newer
    if compute_capability >= (8, 0):
        support["tf32"] = True

    return support


if __name__ == "__main__":
    # Demo: Print GPU information
    info = detect_gpu_info()
    print_gpu_info(info)

    logger.info("\nMixed Precision Support:")
    mp_support = check_mixed_precision_support()
    for precision, supported in mp_support.items():
        status = "✅" if supported else "❌"
        logger.info("  %s %s", status, precision.upper())

    logger.info("\nRecommended Configuration:")
    device = configure_device_for_training(verbose=True)
