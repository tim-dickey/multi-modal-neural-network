"""NPU (Neural Processing Unit) detection and configuration utilities."""

import importlib
import logging
import platform
import shutil
import subprocess
from importlib import util as importlib_util
from typing import Any, Optional
from .subprocess_utils import _safe_subprocess_run as _safe_run

logger = logging.getLogger(__name__)


# Centralized subprocess helper is imported above as `_safe_run`


def _run_powershell_pnp_probe(cmd_body: str, args: list[str] | None = None) -> Optional[subprocess.CompletedProcess]:
    """Run a PowerShell PnP device probe command via :func:`_safe_run`.

    The function accepts the body of the PowerShell command (the part after
    ``-Command``) and wraps it in a safe subprocess call. Callers may pass
    an optional `args` list which will be forwarded to PowerShell via the
    `-ArgumentList` parameter. This avoids embedding untrusted input directly
    into the command string and reduces risk of command injection.
    """
    # Avoid running PowerShell if it's not present on the system.
    if not shutil.which("powershell"):
        # PowerShell not available: return None so callers can treat this
        # as a non-fatal absence of the probe rather than an exception.
        return None

    # Build a static command list and pass arguments via -ArgumentList to
    # avoid shell interpolation. Use -NoProfile and -NonInteractive for
    # a safer non-interactive invocation.
    base = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd_body]
    if args:
        cmd = [*base, "-ArgumentList", *args]
    else:
        cmd = base

    return _safe_run(cmd, capture_output=True, text=True)


def _detect_external_npu() -> tuple[bool, str | None]:
    """Detect if an external NPU is connected (USB, Thunderbolt, PCIe expansion).

    External NPUs include examples such as Coral Edge TPU,
    Movidius NCS, and Hailo cards.

    Returns a tuple (is_external, device_name_or_None).
    """
    system = platform.system()

    try:
        if system == "Windows":
            return _detect_external_npu_windows()
        if system == "Linux":
            return _detect_external_npu_linux()
        if system == "Darwin":
            return _detect_external_npu_darwin()
    except (subprocess.TimeoutExpired, OSError):
        logger.debug("External NPU probe timed out or failed", exc_info=True)

    return False, None


def _detect_external_npu_windows() -> tuple[bool, str | None]:
    """Detect external NPUs on Windows using PowerShell queries."""
    # Build PowerShell command in parts to keep source lines short
    cmd_parts = [
        "Get-PnpDevice | Where-Object {",
        "$_.FriendlyName -like '*Neural*' -or",
        "$_.FriendlyName -like '*Coral*' -or",
        "$_.FriendlyName -like '*Movidius*' -or",
        "$_.FriendlyName -like '*Hailo*' -or",
        "$_.FriendlyName -like '*Edge TPU*'",
        "} | Select-Object FriendlyName, InstanceId",
    ]
    cmd = "\n".join(cmd_parts)

    # Use the centralized PowerShell probe helper which wraps _safe_run
    result = _run_powershell_pnp_probe(cmd)

    if result and result.returncode == 0 and getattr(result, "stdout", None):
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if any(
                keyword in line for keyword in ["Coral", "Movidius", "Hailo", "TPU"]
            ):
                if "Coral" in line:
                    return True, "Google Coral Edge TPU"
                if "Movidius" in line:
                    return True, "Intel Movidius NCS"
                if "Hailo" in line:
                    return True, "Hailo AI Accelerator"
                return True, "External NPU"

    return False, None


def _detect_external_npu_linux() -> tuple[bool, str | None]:
    """Detect external NPUs on Linux using lsusb and lspci."""
    # Check for USB devices (only run probe if the tool exists)
    if shutil.which("lsusb"):
        result = _safe_run(["lsusb"], capture_output=True, text=True)
        if result and result.returncode == 0 and getattr(result, "stdout", None):
            output = result.stdout.lower()
            if "movidius" in output or "neural compute stick" in output:
                return True, "Intel Movidius NCS"
            if "coral" in output or "edge tpu" in output:
                return True, "Google Coral Edge TPU"

    # Check for PCIe devices (Hailo, Coral M.2) when lspci is available
    if shutil.which("lspci"):
        result = _safe_run(["lspci"], capture_output=True, text=True)
        if result and result.returncode == 0 and getattr(result, "stdout", None):
            output = result.stdout.lower()
            if "hailo" in output:
                return True, "Hailo AI Accelerator"
            if "coral" in output:
                return True, "Google Coral Edge TPU"

    return False, None


def _detect_external_npu_darwin() -> tuple[bool, str | None]:
    """Detect external NPUs on macOS using system_profiler."""
    # Only run system_profiler when present to avoid FileNotFoundError
    if shutil.which("system_profiler"):
        result = _safe_run(
            ["system_profiler", "SPUSBDataType"], capture_output=True, text=True
        )

        if result and result.returncode == 0 and getattr(result, "stdout", None):
            output = result.stdout.lower()
            if "movidius" in output or "neural compute stick" in output:
                return True, "Intel Movidius NCS"
            if "coral" in output or "edge tpu" in output:
                return True, "Google Coral Edge TPU"

    return False, None


def detect_npu_info() -> dict[str, Any]:
    """Detect available NPU (Neural Processing Unit) information and capabilities.

    NPUs are specialized processors for AI/ML workloads found in:
    - Intel Core Ultra processors (Intel AI Boost)
    - AMD Ryzen AI processors
    - Apple Silicon (Neural Engine)
    - Qualcomm Snapdragon (Hexagon NPU)
    - Microsoft's custom NPUs in Surface devices
    - External NPUs (Coral Edge TPU, Movidius NCS, Hailo AI)

    Returns:
        Dictionary containing NPU information with keys like:
        - available, npu_type, device_name, backend, capabilities, recommended_device,
          is_external, connection_type
    """
    info: dict[str, Any] = {
        "available": False,
        "npu_type": None,
        "device_name": None,
        "backend": None,
        "capabilities": {},
        "recommended_device": None,
        "detection_method": None,
        "is_external": False,
        "connection_type": None,
    }

    # Check for external NPU first
    is_external, external_device = _detect_external_npu()
    if is_external and external_device:
        external_info = _external_npu_info(external_device)
        if external_info:
            return external_info

    system = platform.system()

    # Check for vendor/platform NPUs (Intel / AMD)
    vendor_info = _vendor_npu_info()
    if vendor_info:
        return vendor_info

    # Detect Apple Neural Engine (macOS only)
    if system == "Darwin":
        apple_info = _detect_apple_neural_engine()
        if apple_info.get("available"):
            info.update(apple_info)
            return info

    # Detect DirectML NPU (Windows only)
    if system == "Windows":
        directml_info = _detect_windows_directml_npu()
        if directml_info.get("available"):
            info.update(directml_info)
            return info

    # No NPU detected
    return info


def _external_npu_info(device_name: str) -> dict[str, Any]:
    """Build the NPU info dict for an external device name."""
    info: dict[str, Any] = {
        "available": True,
        "is_external": True,
        "device_name": device_name,
        "npu_type": "External NPU",
        "connection_type": "USB/PCIe",
        "detection_method": "External device detection",
        "capabilities": {
            "int8": True,
            "fp16": True,
            "inference_only": True,
            "external": True,
        },
    }

    if "Coral" in device_name:
        info["backend"] = "TensorFlow Lite"
        info["recommended_device"] = "edge_tpu"
    elif "Movidius" in device_name:
        info["backend"] = "OpenVINO"
        info["recommended_device"] = "openvino"
    elif "Hailo" in device_name:
        info["backend"] = "Hailo Runtime"
        info["recommended_device"] = "hailo"
    else:
        info["backend"] = "Unknown"
        info["recommended_device"] = "cpu"

    return info


def _vendor_npu_info() -> dict[str, Any] | None:
    """Check for Intel or AMD NPUs and return an info dict when found."""
    if _detect_intel_npu():
        return {
            "available": True,
            "npu_type": "Intel AI Boost",
            "device_name": "Intel NPU",
            "backend": "openvino",
            "recommended_device": "openvino",
            "detection_method": "Intel VPU detection",
            "capabilities": {"int8": True, "fp16": True, "inference_only": True},
        }

    if _detect_amd_npu():
        return {
            "available": True,
            "npu_type": "AMD Ryzen AI",
            "device_name": "AMD NPU",
            "backend": "ryzenai",
            "recommended_device": "ryzenai",
            "detection_method": "AMD Ryzen AI detection",
            "capabilities": {"int8": True, "fp16": True, "inference_only": True},
        }

    return None


def _detect_intel_npu() -> bool:
    """Detect Intel NPU (VPU/AI Boost).

    Returns True if an Intel VPU/NPU device appears available.
    """
    try:
        # Check for Intel VPU via OpenVINO detection
        # This is a placeholder - actual detection would require OpenVINO installed

        # Try to detect via Windows Device Manager (Windows only)
        if platform.system() == "Windows":
            try:
                ps_cmd = (
                    "Get-PnpDevice | Where-Object {"
                    '\n  $_.FriendlyName -like "*NPU*" -or'
                    '\n  $_.FriendlyName -like "*VPU*" -or'
                    '\n  $_.FriendlyName -like "*AI Boost*"'
                    "\n}"
                )
                result = _run_powershell_pnp_probe(ps_cmd)
                if result and result.returncode == 0 and getattr(result, "stdout", None) and result.stdout.strip():
                    return True
            except (subprocess.TimeoutExpired, OSError):
                # PowerShell not available or timed out
                pass

        # Try to import OpenVINO and check for VPU device via importlib
        try:
            if importlib_util.find_spec("openvino") is not None:
                ov = importlib.import_module("openvino")
                core = ov.Core()
                devices = core.available_devices()
                # Look for VPU or NPU device
                for dev in devices:
                    if "VPU" in dev or "NPU" in dev:
                        return True
        except (ImportError, AttributeError, RuntimeError, OSError) as _err:
            # Any failure during the OpenVINO probe is non-fatal; log for debugging
            logger.debug("OpenVINO probe failed: %s", _err, exc_info=True)

        return False

    except (OSError, RuntimeError):
        return False


def _detect_amd_npu() -> bool:
    """Detect AMD Ryzen AI NPU.

    Returns True if AMD Ryzen AI appears available on this system.
    """
    try:
        processor_info = platform.processor().lower()

        # Check for AMD Ryzen AI in processor name and Windows Device Manager probe
        if (
            "amd" in processor_info
            and "ryzen" in processor_info
            and platform.system() == "Windows"
        ):
            try:
                ps_cmd = (
                    "Get-PnpDevice | Where-Object {"
                    '\n  $_.FriendlyName -like "*Ryzen AI*" -or'
                    '\n  $_.FriendlyName -like "*AMD NPU*"'
                    "\n}"
                )
                result = _run_powershell_pnp_probe(ps_cmd)
                if result and result.returncode == 0 and getattr(result, "stdout", None) and result.stdout.strip():
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass

        # Try to check for AMD Ryzen AI SDK via importlib
        try:
            if importlib_util.find_spec("ryzen_ai") is not None:
                return True
        except (ImportError, AttributeError, RuntimeError, OSError) as _err:
            logger.debug(
                "DirectML/torch_directml probe failed: %s",
                _err,
                exc_info=True,
            )

        return False

    except (OSError, RuntimeError):
        return False


def _detect_apple_neural_engine() -> dict[str, Any]:
    """Detect Apple Neural Engine (ANE) on Apple Silicon devices.

    Returns a dictionary with detection info; keys mirror detect_npu_info.
    """
    info: dict[str, Any] = {
        "available": False,
        "npu_type": None,
        "device_name": None,
        "backend": None,
        "capabilities": {},
        "recommended_device": None,
        "detection_method": None,
    }

    try:
        if platform.system() != "Darwin":
            return info

        # Check for Apple Silicon
        machine = platform.machine()
        if machine == "arm64":
            # Apple Silicon detected
            info["available"] = True
            info["npu_type"] = "Apple Neural Engine"
            info["device_name"] = "Apple Neural Engine"
            info["backend"] = "coreml"  # Core ML framework
            info["recommended_device"] = "mps"  # Metal Performance Shaders
            info["detection_method"] = "Apple Silicon detection"
            info["capabilities"] = {
                "int8": True,
                "fp16": True,
                "fp32": True,
                "inference_only": False,  # ANE supports training via Core ML
            }

            # Try to get more specific info about the chip when `sysctl` exists
            if shutil.which("sysctl"):
                try:
                    result = _safe_run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result and result.returncode == 0 and getattr(result, "stdout", None):
                        chip_name = result.stdout.strip()
                        info["device_name"] = f"Apple Neural Engine ({chip_name})"
                except subprocess.TimeoutExpired:
                    # sysctl timed out
                    pass

        return info

    except (OSError, RuntimeError):
        return info


def _detect_windows_directml_npu() -> dict[str, Any]:
    """Detect NPU via DirectML on Windows and return info dict.

    If DirectML is available and reports devices, returns an info dict.
    """
    info: dict[str, Any] = {
        "available": False,
        "npu_type": None,
        "device_name": None,
        "backend": None,
        "capabilities": {},
        "recommended_device": None,
        "detection_method": None,
    }

    try:
        if platform.system() != "Windows":
            return info

        # Try to import DirectML
        try:
            if importlib_util.find_spec("torch_directml") is not None:
                torch_directml = importlib.import_module("torch_directml")
                if torch_directml.is_available():
                    device_count = torch_directml.device_count()
                    if device_count > 0:
                        info["available"] = True
                        info["npu_type"] = "DirectML NPU"
                        info["device_name"] = torch_directml.device_name(0)
                        info["backend"] = "directml"
                        info["recommended_device"] = (
                            "privateuseone"  # PyTorch DirectML device
                        )
                        info["detection_method"] = "DirectML detection"
                        info["capabilities"] = {
                            "int8": True,
                            "fp16": True,
                            "fp32": True,
                            "inference_only": False,
                        }
        except (ImportError, AttributeError, RuntimeError, OSError):
            pass

        return info

    except (ImportError, OSError, RuntimeError):
        return info


def log_npu_info(info: dict[str, Any] | None = None, *, verbose: bool = True) -> None:
    """Log formatted NPU information.

    Args:

        info: NPU info dict from :func:`detect_npu_info`. If ``None``, will detect
            automatically.
        verbose: If ``False``, returns early without logging detailed info. This
            is useful for programmatic callers that want to suppress console
            output.
    """
    if info is None:
        info = detect_npu_info()

    if not verbose:
        return

    logger.info("NPU (NEURAL PROCESSING UNIT) CONFIGURATION")

    if not info.get("available"):
        logger.info("Status: ❌ No NPU detected")
        logger.info(
            "NPUs are specialized AI accelerators found in: Intel/AMD/Apple/Qualcomm"
        )
        logger.info("To use NPU acceleration: ensure compatible hardware")
        logger.info("and SDKs (openvino/ryzen_ai/coreml/torch-directml)")
        return

    npu_location = "External" if info.get("is_external", False) else "Internal"
    logger.info("Status: ✅ NPU detected (%s)", npu_location)
    logger.info("NPU Type: %s", info.get("npu_type"))
    logger.info("Device Name: %s", info.get("device_name"))
    if info.get("is_external", False) and info.get("connection_type"):
        logger.info("Connection: %s", info.get("connection_type"))
    logger.info("Backend: %s", info.get("backend"))
    logger.info("Detection Method: %s", info.get("detection_method"))
    logger.info("Recommended Device: %s", info.get("recommended_device"))

    logger.info("Capabilities:")
    for capability, supported in info.get("capabilities", {}).items():
        status = "✅" if supported else "❌"
        logger.info("  %s %s", status, capability.upper())


def print_npu_info(info: dict[str, Any] | None = None) -> None:
    """Backward-compatible shim for the legacy ``print_npu_info`` API.

    This calls :func:`log_npu_info` and emits a deprecation warning via the
    module logger. Callers should prefer :func:`log_npu_info` and may set
    ``verbose=False`` to suppress console output.
    """
    logger.warning(
        "`print_npu_info` is deprecated — use `log_npu_info(info, verbose=True)`"
    )
    # Keep legacy behavior: verbose logging
    return log_npu_info(info=info, verbose=True)


def check_accelerator_availability() -> dict[str, bool]:
    """Check availability of all hardware accelerators.

    Returns a dict with boolean availability for: cpu, cuda, mps, npu.
    """
    try:
        import torch  # noqa: PLC0415

        cuda_avail = torch.cuda.is_available()
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except (ImportError, OSError, RuntimeError) as _err:
        # Torch not installed or import failed; report accelerators as unavailable
        logger.debug(
            "torch not available when checking accelerators: %s",
            _err,
            exc_info=True,
        )
        cuda_avail = False
        mps_avail = False

    return {
        "cpu": True,  # Always available
        "cuda": cuda_avail,
        "mps": mps_avail,
        "npu": detect_npu_info().get("available", False),
    }


def get_best_available_device(*, prefer_npu: bool = False) -> str:
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
    import torch  # noqa: PLC0415

    if prefer_npu:
        npu_local = detect_npu_info()
        if npu_local.get("available"):
            return str(npu_local.get("recommended_device"))

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Check for Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    # Check for NPU (if not already preferred)
    if not prefer_npu:
        npu_local = detect_npu_info()
        if npu_local.get("available"):
            return str(npu_local.get("recommended_device"))

    # Fallback to CPU
    return "cpu"


if __name__ == "__main__":
    # Demo: Log NPU information
    logger.info("Detecting NPU...")
    npu_info = detect_npu_info()
    # Use the new logging API to avoid the deprecation shim warning
    log_npu_info(npu_info, verbose=True)

    logger.info("All Accelerator Availability:")
    avail_map = check_accelerator_availability()
    for device, available in avail_map.items():
        avail_status = "✅" if available else "❌"
        logger.info("  %s %s", avail_status, device.upper())

    logger.info("Recommended Device:")
    device = get_best_available_device(prefer_npu=False)
    logger.info("  Default: %s", device)
    device_with_npu = get_best_available_device(prefer_npu=True)
    logger.info("  With NPU preference: %s", device_with_npu)
