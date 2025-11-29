"""Device detection and configuration for training."""

import logging
from typing import Any, Dict, Optional

import torch


class DeviceManager:
    """Manages device detection and configuration for training.

    Handles automatic detection of GPU, NPU, MPS devices and
    provides a unified interface for device configuration.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device_override: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize device manager.

        Args:
            config: Configuration dictionary with 'hardware' section
            device_override: Optional device string to override config
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._device: Optional[torch.device] = None

        self._configure_device(device_override)

    @property
    def device(self) -> torch.device:
        """Get the configured device."""
        if self._device is None:
            raise RuntimeError("Device not configured")
        return self._device

    @property
    def device_type(self) -> str:
        """Get the device type string (cuda, cpu, mps, etc.)."""
        return self.device.type

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self.device_type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self.device_type == "cpu"

    @property
    def is_mps(self) -> bool:
        """Check if using Apple MPS."""
        return self.device_type == "mps"

    def _configure_device(self, device_override: Optional[str]) -> None:
        """Configure the device based on config and available hardware."""
        from ..utils.gpu_utils import configure_device_for_training, detect_gpu_info
        from ..utils.npu_utils import detect_npu_info, get_best_available_device

        if device_override is not None:
            self._device = torch.device(
                device_override if torch.cuda.is_available() else "cpu"
            )
            self.logger.info("Using override device: %s", self._device)
            return

        hardware_config = self.config.get("hardware", {})
        device_config = hardware_config.get("device", "auto")
        gpu_id = hardware_config.get("gpu_id")
        prefer_npu = hardware_config.get("prefer_npu", False)

        # Handle "auto" device selection
        if device_config == "auto":
            device_config = get_best_available_device(prefer_npu=prefer_npu)
            self._log_detected_device(device_config)

        # Handle NPU device strings
        if device_config in ["npu", "openvino", "ryzenai", "privateuseone"]:
            self.logger.info(
                "NPU detected (%s), using CPU for training. "
                "NPU can be used for optimized inference via ONNX export.",
                device_config,
            )
            self._device = torch.device("cpu")
        elif device_config == "mps":
            self._device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self._device = configure_device_for_training(
                device=device_config, gpu_id=gpu_id, verbose=False
            )

    def _log_detected_device(self, device_config: str) -> None:
        """Log information about the detected device."""
        from ..utils.gpu_utils import detect_gpu_info
        from ..utils.npu_utils import detect_npu_info

        if device_config == "cuda":
            gpu_info = detect_gpu_info()
            if gpu_info["available"]:
                self.logger.info(
                    "Using CUDA GPU: %s", gpu_info["devices"][0]["name"]
                )
        elif device_config in ["openvino", "ryzenai", "mps", "privateuseone"]:
            npu_info = detect_npu_info()
            self.logger.info("Using NPU: %s", npu_info.get("device_name"))
        else:
            self.logger.warning(
                "No GPU or NPU detected. Training will use CPU. "
                "For GPU support, install PyTorch with CUDA. "
                "For NPU support, install an appropriate SDK "
                "(OpenVINO, DirectML, etc.)"
            )

    def move_to_device(self, tensor_or_module: Any) -> Any:
        """Move a tensor or module to the configured device.

        Args:
            tensor_or_module: PyTorch tensor or nn.Module

        Returns:
            The tensor/module on the configured device
        """
        if hasattr(tensor_or_module, "to"):
            return tensor_or_module.to(self.device)
        return tensor_or_module

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in a batch dictionary to the configured device.

        Args:
            batch: Dictionary containing tensors

        Returns:
            Dictionary with tensors moved to device
        """
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    def create_grad_scaler(self) -> Optional[torch.amp.GradScaler]:
        """Create a GradScaler for mixed precision training if applicable.

        Returns:
            GradScaler for CUDA devices, None otherwise
        """
        if self.is_cuda:
            # Use unified torch.amp.GradScaler API (PyTorch 2.4+)
            return torch.amp.GradScaler("cuda")
        return None

    def get_autocast_context(self, dtype: Optional[torch.dtype] = None):
        """Get the appropriate autocast context for mixed precision.

        Args:
            dtype: Optional dtype for autocast (default: bfloat16)

        Returns:
            Autocast context manager
        """
        if dtype is None:
            dtype = torch.bfloat16

        # Use the unified torch.amp.autocast API (PyTorch 2.0+)
        return torch.amp.autocast(device_type=self.device_type, dtype=dtype)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        """String representation of device manager."""
        return f"DeviceManager(device={self._device})"
