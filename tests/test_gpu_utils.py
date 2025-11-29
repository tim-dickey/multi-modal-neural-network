import types
import subprocess
import torch

from src.utils import gpu_utils
import shutil


def test_detect_gpu_info_no_cuda_but_nvidia_smi(monkeypatch):
    # Simulate CUDA not available but nvidia-smi present
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gpu_utils, "_query_nvidia_smi", lambda: ["GPU 0: FakeGPU"])

    info = gpu_utils.detect_gpu_info()
    assert info["available"] is False
    assert info["nvidia_gpus"] == ["FakeGPU"] or info["nvidia_smi"] is True


def test_detect_gpu_info_with_cuda(monkeypatch):
    # Simulate CUDA available with one device and simple properties
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    class FakeProps:
        name = "FakeGPU"
        major = 7
        minor = 5
        total_memory = 8 * 1024 ** 3
        multi_processor_count = 4

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps)
    # Ensure cuDNN not available
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: False)

    info = gpu_utils.detect_gpu_info()
    assert info["available"] is True
    assert info["device_count"] == 1
    assert info["recommended_device"] == "cuda"


def test_populate_nvidia_smi_info_parses_names():
    info = {"nvidia_smi": False, "nvidia_gpus": []}
    parsed = ["GPU 0: FakeGPU", "GPU 1: OtherGPU"]
    gpu_utils._populate_nvidia_smi_info(info, parsed)
    assert info["nvidia_smi"] is True
    assert info["nvidia_gpus"] == ["FakeGPU", "OtherGPU"]


def test_get_nvml_info_monkeypatched(monkeypatch):
    # Create a fake pynvml module
    class FakeMem:
        def __init__(self, total):
            self.total = total

    class FakeNV:
        def nvmlInit(self):
            return None

        def nvmlSystemGetDriverVersion(self):
            return b"525.60"

        def nvmlDeviceGetCount(self):
            return 1

        def nvmlDeviceGetHandleByIndex(self, idx):
            return object()

        def nvmlDeviceGetName(self, handle):
            return b"FakeGPU"

        def nvmlDeviceGetMemoryInfo(self, handle):
            return FakeMem(8 * 1024 ** 3)

        def nvmlShutdown(self):
            return None

    import sys

    monkeypatch.setitem(sys.modules, "pynvml", FakeNV())
    info = gpu_utils._get_nvml_info()
    assert info.get("available") is True
    assert "driver" in info and info["driver"]


def test_check_mixed_precision_support(monkeypatch):
    # Simulate CUDA available and an Ampere device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    class FakeProps:
        major = 8
        minor = 0

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps)
    mp = gpu_utils.check_mixed_precision_support()
    assert mp["fp16"] is True
    assert mp["bf16"] is True
    assert mp["tf32"] is True


def test_collect_cuda_devices_handles_memory_errors(monkeypatch):
    # Simulate device with properties but memory methods raising
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    class FakeProps:
        name = "FakeGPU"
        major = 7
        minor = 0
        total_memory = 4 * 1024 ** 3
        multi_processor_count = 2

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps)

    def raise_runtime_err(i):
        raise RuntimeError("cuda failed")

    monkeypatch.setattr(torch.cuda, "memory_allocated", raise_runtime_err)
    info = {"device_count": 1, "devices": [], "external_gpu_count": 0}
    gpu_utils._collect_cuda_devices(info)
    assert info["devices"]


def test_query_nvidia_smi_parses(monkeypatch):
    # Ensure shutil.which reports nvidia-smi exists
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/nvidia-smi" if x == "nvidia-smi" else None)

    class CP:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", lambda *a, **k: CP("GPU 0: FakeGPU\nGPU 1: OtherGPU"))
    parsed = gpu_utils._query_nvidia_smi()
    assert isinstance(parsed, list)
    assert "GPU 0: FakeGPU" in parsed


def test_handle_gpu_id_override_and_ensure_valid(monkeypatch):
    # No CUDA available: override returns cpu
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    res = gpu_utils._handle_gpu_id_override(1, verbose=False)
    assert res == "cpu"

    # CUDA available but gpu_id out of range
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    res2 = gpu_utils._handle_gpu_id_override(5, verbose=False)
    assert res2 == "cuda:0"

    # ensure valid device string when cuda not available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_utils._ensure_valid_device_string("cuda:0", verbose=False) == "cpu"


def test_resolve_device_string_gpu_id(monkeypatch):
    # When gpu_id specified and cuda not available, should return cpu
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_utils._resolve_device_string(None, 0, verbose=False) == "cpu"


def test_print_gpu_info_no_cuda(monkeypatch, caplog):
    # Call print_gpu_info with a no-GPU info dict
    info = {"available": False, "recommended_device": "cpu"}
    gpu_utils.print_gpu_info(info)
    # Should have logged 'No CUDA GPUs detected'
    assert any("No CUDA GPUs detected" in r.message for r in caplog.records)


def test_print_gpu_info_with_devices(monkeypatch, caplog):
    # Simulate detect_gpu_info returning devices and external count
    info = {
        "available": True,
        "recommended_device": "cuda",
        "cuda_version": "11.8",
        "cudnn_version": None,
        "device_count": 1,
        "external_gpu_count": 1,
        "devices": [
            {
                "id": 0,
                "name": "FakeGPU",
                "compute_capability": (7, 5),
                "total_memory_gb": 8.0,
                "multi_processor_count": 16,
                "is_external": True,
                "connection_type": "Thunderbolt",
            }
        ],
    }

    gpu_utils.print_gpu_info(info)
    assert any("CUDA available" in r.message or "GPU 0" in r.message for r in caplog.records)


def test_get_optimal_device_prefers_gpu(monkeypatch):
    # When prefer_gpu False should return cpu
    assert gpu_utils.get_optimal_device(prefer_gpu=False) == "cpu"


def test_print_gpu_verbose(monkeypatch, caplog):
    # Monkeypatch detect_gpu_info to return a device list
    monkeypatch.setattr(gpu_utils, "detect_gpu_info", lambda: {"devices": [{"name": "X", "total_memory_gb": 4.0}]})
    dev = type("D", (), {"index": 0})()
    gpu_utils._print_gpu_verbose(dev)
    assert any("GPU" in r.message or "Memory" in r.message for r in caplog.records)


def test_detect_external_gpu_windows(monkeypatch):
    # Simulate Windows and PowerShell output indicating Thunderbolt
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: True)

    class CP:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", lambda *a, **k: CP("ParentDevice|Thunderbolt"))

    is_ext, conn = gpu_utils._detect_external_gpu_windows(0, "FakeGPU")
    assert is_ext is True
    assert conn == "Thunderbolt"


def test_detect_external_gpu_linux(monkeypatch):
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: True)

    class CP:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    # lspci output with VGA line followed by thunderbolt indicator
    out = "00:02.0 VGA compatible controller: Fake Vendor\n\tSubsystem: Something\n\tThunderbolt: present"
    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", lambda *a, **k: CP(out))

    is_ext, conn = gpu_utils._detect_external_gpu_linux()
    assert is_ext is True
    assert conn == "Thunderbolt"


def test_detect_external_gpu_darwin(monkeypatch):
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: True)

    class CP:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", lambda *a, **k: CP("eGPU connected via Thunderbolt"))

    is_ext, conn = gpu_utils._detect_external_gpu_darwin()
    assert is_ext is True
    assert conn == "Thunderbolt"


def test_get_nvml_info_import_failure(monkeypatch):
    # Simulate ImportError for pynvml
    import sys

    monkeypatch.setitem(sys.modules, "pynvml", None)
    info = gpu_utils._get_nvml_info()
    assert info.get("available") is False


def test_collect_cuda_devices_external_flag(monkeypatch):
    # Simulate CUDA device properties and external detection
    monkeypatch.setattr(gpu_utils.torch.cuda, "device_count", lambda: 1)

    class FakeProps:
        name = "FakeGPU"
        major = 7
        minor = 5
        total_memory = 8 * 1024 ** 3
        multi_processor_count = 16

    monkeypatch.setattr(gpu_utils.torch.cuda, "get_device_properties", lambda i: FakeProps)
    monkeypatch.setattr(gpu_utils, "_detect_external_gpu", lambda i, n: (True, "Thunderbolt"))
    monkeypatch.setattr(gpu_utils.torch.cuda, "memory_allocated", lambda i: 0)
    monkeypatch.setattr(gpu_utils.torch.cuda, "memory_reserved", lambda i: 0)

    info = {"device_count": 1, "devices": [], "external_gpu_count": 0}
    gpu_utils._collect_cuda_devices(info)
    assert info["external_gpu_count"] == 1
    assert info["devices"][0]["is_external"] is True


def test_configure_device_for_training_cuda(monkeypatch):
    # Test configure_device_for_training with CUDA available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(gpu_utils, "detect_gpu_info", lambda: {
        "available": True,
        "recommended_device": "cuda",
        "devices": [{"name": "FakeGPU", "total_memory_gb": 8.0}]
    })
    dev = gpu_utils.configure_device_for_training(verbose=True)
    assert dev.type == "cuda"


def test_configure_device_for_training_cpu(monkeypatch):
    # Test configure_device_for_training with explicit cpu
    dev = gpu_utils.configure_device_for_training(device="cpu", verbose=True)
    assert dev.type == "cpu"


def test_configure_device_for_training_gpu_id(monkeypatch):
    # Test configure_device_for_training with explicit gpu_id
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(gpu_utils, "detect_gpu_info", lambda: {
        "available": True,
        "devices": [{"name": "GPU0", "total_memory_gb": 8.0}, {"name": "GPU1", "total_memory_gb": 8.0}]
    })
    dev = gpu_utils.configure_device_for_training(gpu_id=1, verbose=True)
    assert dev.type == "cuda"
    assert dev.index == 1


def test_detect_external_gpu_windows_usbc(monkeypatch):
    # Test USB-C detection path
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: True)

    class CP:
        returncode = 0
        stdout = "ParentDevice|usb type-c"

    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", lambda *a, **k: CP())
    is_ext, conn = gpu_utils._detect_external_gpu_windows(0, "FakeGPU")
    assert is_ext is True
    assert conn == "USB-C"


def test_detect_external_gpu_windows_pcie_external(monkeypatch):
    # Test PCIe External detection path
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: True)

    class CP:
        returncode = 0
        stdout = "External GPU Box"

    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", lambda *a, **k: CP())
    is_ext, conn = gpu_utils._detect_external_gpu_windows(0, "FakeGPU")
    assert is_ext is True
    assert conn == "PCIe External"


def test_detect_external_gpu_windows_no_powershell(monkeypatch):
    # When PowerShell not available
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: None)
    is_ext, conn = gpu_utils._detect_external_gpu_windows(0, "FakeGPU")
    assert is_ext is False
    assert conn is None


def test_detect_external_gpu_linux_no_lspci(monkeypatch):
    # When lspci not available
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: None)
    is_ext, conn = gpu_utils._detect_external_gpu_linux()
    assert is_ext is False
    assert conn is None


def test_detect_external_gpu_linux_pcie_external(monkeypatch):
    # Test PCIe External detection on Linux
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: True)

    class CP:
        returncode = 0
        stdout = "00:02.0 VGA compatible controller: NVIDIA\n\tFlags: External enclosure"

    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", lambda *a, **k: CP())
    is_ext, conn = gpu_utils._detect_external_gpu_linux()
    assert is_ext is True
    assert conn == "PCIe External"


def test_detect_external_gpu_darwin_no_profiler(monkeypatch):
    # When system_profiler not available
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: None)
    is_ext, conn = gpu_utils._detect_external_gpu_darwin()
    assert is_ext is False
    assert conn is None


def test_detect_external_gpu_exception_handling(monkeypatch):
    # Test exception handling in _detect_external_gpu
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Windows")

    def raise_error(*a, **k):
        raise OSError("test error")

    monkeypatch.setattr(gpu_utils, "_detect_external_gpu_windows", raise_error)
    is_ext, conn = gpu_utils._detect_external_gpu(0, "FakeGPU")
    assert is_ext is False
    assert conn is None


def test_query_nvidia_smi_no_binary(monkeypatch):
    # When nvidia-smi not found
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda x: None)
    result = gpu_utils._query_nvidia_smi()
    assert result == []


def test_query_nvidia_smi_timeout(monkeypatch):
    # When nvidia-smi times out
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda x: "/usr/bin/nvidia-smi")

    def timeout_run(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 3)

    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", timeout_run)
    result = gpu_utils._query_nvidia_smi()
    assert result == []


def test_detect_gpu_info_cuda_with_cudnn(monkeypatch):
    # Test detect_gpu_info with cuDNN available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    class FakeProps:
        name = "FakeGPU"
        major = 8
        minor = 0
        total_memory = 8 * 1024 ** 3
        multi_processor_count = 64

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps)
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 8600)
    monkeypatch.setattr(gpu_utils, "_detect_external_gpu", lambda i, n: (False, None))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i: 0)

    info = gpu_utils.detect_gpu_info()
    assert info["available"] is True
    assert info["cudnn_version"] == 8600


def test_print_gpu_info_with_cudnn(monkeypatch, caplog):
    # Test print_gpu_info with cuDNN version
    info = {
        "available": True,
        "recommended_device": "cuda",
        "cuda_version": "12.1",
        "cudnn_version": 8900,
        "device_count": 1,
        "external_gpu_count": 0,
        "devices": [
            {
                "id": 0,
                "name": "FakeGPU",
                "compute_capability": (8, 9),
                "total_memory_gb": 24.0,
                "multi_processor_count": 128,
                "memory_free_gb": 20.0,
                "memory_allocated_gb": 1.0,
                "is_external": False,
                "connection_type": None,
            }
        ],
    }
    gpu_utils.print_gpu_info(info)
    assert any("cuDNN" in r.message for r in caplog.records)


def test_print_gpu_info_low_compute_capability(monkeypatch, caplog):
    # Test print_gpu_info with low compute capability (< 7.0)
    info = {
        "available": True,
        "recommended_device": "cuda",
        "cuda_version": "10.2",
        "cudnn_version": None,
        "device_count": 1,
        "external_gpu_count": 0,
        "devices": [
            {
                "id": 0,
                "name": "OldGPU",
                "compute_capability": (6, 1),
                "total_memory_gb": 4.0,
                "multi_processor_count": 10,
                "is_external": False,
                "connection_type": None,
            }
        ],
    }
    gpu_utils.print_gpu_info(info)
    assert any("Limited mixed precision" in r.message for r in caplog.records)


def test_check_mixed_precision_no_cuda(monkeypatch):
    # Test check_mixed_precision_support when CUDA not available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mp = gpu_utils.check_mixed_precision_support()
    assert mp["fp16"] is False
    assert mp["bf16"] is False
    assert mp["tf32"] is False


def test_check_mixed_precision_old_gpu(monkeypatch):
    # Test check_mixed_precision_support on older GPU (< 8.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    class FakeProps:
        major = 7
        minor = 5

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps)
    mp = gpu_utils.check_mixed_precision_support()
    assert mp["fp16"] is True
    assert mp["bf16"] is False
    assert mp["tf32"] is False


def test_check_mixed_precision_exception(monkeypatch):
    # Test check_mixed_precision_support with exception
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def raise_error(i):
        raise RuntimeError("GPU error")

    monkeypatch.setattr(torch.cuda, "get_device_properties", raise_error)
    mp = gpu_utils.check_mixed_precision_support()
    assert mp["fp16"] is True
    assert mp["bf16"] is False


def test_handle_gpu_id_override_valid(monkeypatch):
    # Test valid gpu_id override
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    result = gpu_utils._handle_gpu_id_override(1, verbose=False)
    assert result == "cuda:1"


def test_ensure_valid_device_string_non_cuda(monkeypatch):
    # Test _ensure_valid_device_string with non-cuda device
    result = gpu_utils._ensure_valid_device_string("cpu", verbose=False)
    assert result == "cpu"


def test_resolve_device_string_auto_detect(monkeypatch):
    # Test _resolve_device_string with auto-detection
    monkeypatch.setattr(gpu_utils, "get_optimal_device", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = gpu_utils._resolve_device_string(None, None, verbose=False)
    assert result == "cuda"


def test_print_gpu_verbose_empty_devices(monkeypatch, caplog):
    # Test _print_gpu_verbose with no devices
    monkeypatch.setattr(gpu_utils, "detect_gpu_info", lambda: {"devices": []})
    dev = type("D", (), {"index": 0, "type": "cuda"})()
    gpu_utils._print_gpu_verbose(dev)
    # Should not crash, but may not log anything


def test_print_gpu_verbose_index_out_of_range(monkeypatch, caplog):
    # Test _print_gpu_verbose when device index > available devices
    monkeypatch.setattr(gpu_utils, "detect_gpu_info", lambda: {"devices": [{"name": "GPU0", "total_memory_gb": 8.0}]})
    dev = type("D", (), {"index": 5})()
    gpu_utils._print_gpu_verbose(dev)
    # Should not crash


def test_detect_external_gpu_windows_usb4(monkeypatch):
    # Test USB4 detection path on Windows
    monkeypatch.setattr(gpu_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda name: True)

    class CP:
        returncode = 0
        stdout = "ParentDevice|USB4 Host"

    monkeypatch.setattr(gpu_utils, "_safe_subprocess_run", lambda *a, **k: CP())
    is_ext, conn = gpu_utils._detect_external_gpu_windows(0, "FakeGPU")
    assert is_ext is True
    assert conn == "Thunderbolt"


def test_get_nvml_info_with_string_returns(monkeypatch):
    # Test NVML info when driver/name are already strings
    import sys

    class FakeMem:
        total = 8 * 1024 ** 3

    class FakeNV:
        NVMLError = Exception

        def nvmlInit(self):
            pass

        def nvmlSystemGetDriverVersion(self):
            return "535.104"  # Return string, not bytes

        def nvmlDeviceGetCount(self):
            return 1

        def nvmlDeviceGetHandleByIndex(self, idx):
            return object()

        def nvmlDeviceGetName(self, handle):
            return "NVIDIA RTX 4090"  # Return string, not bytes

        def nvmlDeviceGetMemoryInfo(self, handle):
            return FakeMem()

        def nvmlShutdown(self):
            pass

    monkeypatch.setitem(sys.modules, "pynvml", FakeNV())
    info = gpu_utils._get_nvml_info()
    assert info["available"] is True
    assert info["driver"] == "535.104"


def test_populate_nvidia_smi_info_empty():
    # Test _populate_nvidia_smi_info with empty list
    info = {"nvidia_smi": False, "nvidia_gpus": []}
    gpu_utils._populate_nvidia_smi_info(info, [])
    assert info["nvidia_smi"] is False
    assert info["nvidia_gpus"] == []

