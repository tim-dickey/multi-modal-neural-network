# Multi-Modal Neural Network with Double-Loop Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-446%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](htmlcov/index.html)

This repository contains an open-source implementation of a multi-modal small neural network that incorporates double-loop learning mechanisms and integrates with external APIs for computational knowledge enhancement. The system is designed to train on consumer-grade hardware while maintaining acceptable performance and accuracy.

## Features

- **Multi-Modal Architecture**: Supports vision (images) and text modalities with early fusion.
- **Double-Loop Learning**: Implements meta-learning for structural adaptation during training.
- **API Integration Framework**: Extensible framework for external knowledge sources (Wolfram Alpha, etc.).
- **Consumer Hardware Optimized**: Designed for single GPU systems (8-16GB VRAM, 16-32GB RAM).
- **Parameter Efficient**: Total parameters capped at 100-500 million.
- **Full Type Safety**: Complete type annotations with mypy compliance across all 23 source files. Zero type errors with strict static analysis ensuring runtime reliability and enhanced developer experience.
- **Production Ready**: Comprehensive configuration management and environment variable support.
- **Hardware Acceleration**: Automatic detection and support for NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Apple Silicon (MPS), and NPUs (Intel AI Boost, AMD Ryzen AI, Apple Neural Engine).
- **External Device Support**: Detects and utilizes external GPUs (eGPU via Thunderbolt/USB-C) and external NPUs (Coral Edge TPU, Intel Movidius NCS, Hailo AI).
- **Flexible Device Configuration**: Auto-detection of optimal hardware or manual device selection with comprehensive fallback handling.

## Installation

### Prerequisites

- Python 3.10+
- **GPU Support (Optional)**:
  - NVIDIA: CUDA 12.1+ with RTX 3060 (12GB) or better
  - AMD: ROCm 5.7+ with RX 6700 XT (12GB) or better
  - Apple: M1/M2/M3 with Metal Performance Shaders (MPS)
- **NPU Support (Optional)**:
  - Intel AI Boost (Meteor Lake/Lunar Lake)
  - AMD Ryzen AI (7040/8040 series)
  - Apple Neural Engine (M1/M2/M3)
  - Qualcomm Hexagon NPU (Snapdragon X)
- **CPU**: Works on CPU-only systems (slower training)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tim-dickey/multi-modal-neural-network.git
   cd multi-modal-neural-network
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Set up development tools:
   ```bash
   # Install pre-commit hooks for code quality
   pip install pre-commit
   pre-commit install
   
   # Verify installation
   make test  # Run tests
   make lint  # Check code quality
   ```

5. (Optional) Install with Poetry:
   ```bash
   poetry install
   ```

## Quick Start

1. **Check your hardware** (detects internal and external devices):
   ```python
   # Check GPU availability (including eGPU via Thunderbolt/USB-C)
   from src.utils.gpu_utils import detect_gpu_info, print_gpu_info
   info = detect_gpu_info()
   print_gpu_info(info)
   
   # Shows: GPU count, memory, external GPU detection, connection type
   if info['external_gpu_count'] > 0:
       print(f"External GPUs detected: {info['external_gpu_count']}")
   
   # Check NPU availability (including external NPUs like Coral Edge TPU)
   from src.utils.npu_utils import detect_npu_info, print_npu_info
   npu_info = detect_npu_info()
   print_npu_info(npu_info)
   
   # Shows: NPU type, backend, internal/external status
   ```

2. Configure your environment:
   ```bash
   cp configs/default.yaml configs/my_config.yaml
   # Edit my_config.yaml with your settings
   ```

3. Set up environment variables (see [Environment Setup](#environment-setup))

4. Run the getting started notebook:
   ```bash
   jupyter notebook notebooks/01_getting_started.ipynb
   ```

5. Train the model:
   ```python
   from src.training.trainer import Trainer
   trainer = Trainer(config_path="configs/my_config.yaml")
   trainer.train()
   ```

## Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual keys
# WOLFRAM_API_KEY=your_wolfram_alpha_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here  # For future integrations
```

**Important**: Never commit `.env` files to version control. They are automatically ignored by `.gitignore`.

## Project Structure

```
multi-modal-neural-network/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .env.example                    # Environment variable template
├── .gitignore                      # Comprehensive ignore patterns
├── configs/
│   └── default.yaml               # Default configuration
├── src/
│   ├── models/                    # Core model components (fully typed)
│   │   ├── multi_modal_model.py
│   │   ├── vision_encoder.py
│   │   ├── text_encoder.py
│   │   ├── fusion_layer.py
│   │   ├── double_loop_controller.py
│   │   └── heads.py
│   ├── training/                  # Training infrastructure
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── losses.py
│   │   └── checkpointing.py
│   ├── data/                      # Data processing pipeline
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   └── streaming.py
│   ├── integrations/              # API integration framework
│   │   ├── __init__.py
│   │   ├── wolfram_alpha.py       # Wolfram Alpha integration
│   │   ├── validators.py          # Response validation
│   │   └── knowledge_injection.py # Knowledge injection logic
│   ├── evaluation/                # Evaluation and benchmarking
│   │   ├── metrics.py
│   │   ├── benchmarks.py
│   │   └── api_comparison.py      # API-based evaluation
│   └── utils/                     # Utilities and helpers
│       ├── config.py              # Configuration management
│       ├── logging.py             # Logging utilities
│       ├── profiling.py           # Performance profiling
│       ├── gpu_utils.py           # GPU detection and configuration
│       └── npu_utils.py           # NPU detection and configuration
├── notebooks/
│   ├── 01_getting_started.ipynb   # Setup and basic usage
│   ├── 02_training.ipynb          # Training workflows
│   └── 03_evaluation.ipynb        # Evaluation and analysis
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
│   ├── GPU_TRAINING.md            # GPU configuration guide
│   └── NPU_TRAINING.md            # NPU configuration guide
└── examples/                      # Usage examples
```
## API Integration Framework

The project includes a flexible API integration framework designed for external knowledge sources:

### Current Integrations

- **Wolfram Alpha**: Symbolic computation and mathematical verification
  - API key required: `WOLFRAM_API_KEY`
  - Used for ground truth validation and computational knowledge injection

### Future Integrations

The framework is designed to easily accommodate additional APIs:

- **OpenAI GPT**: Text generation and reasoning augmentation
- **Google PaLM**: Multimodal understanding enhancement
- **Hugging Face Inference**: Specialized model access
- **Custom APIs**: Domain-specific knowledge sources

### Adding New API Integrations

1. Create a new module in `src/integrations/`:
   ```python
   # src/integrations/new_api.py
   from src.integrations.base import APIIntegration

   class NewAPIIntegration(APIIntegration):
       def __init__(self, api_key: str, config: dict):
           super().__init__(api_key, config)

       def query(self, prompt: str) -> dict:
           # Implementation here
           pass
   ```

2. Add configuration to `configs/default.yaml`:
   ```yaml
   new_api:
     api_key: "${NEW_API_KEY}"
     endpoint: "https://api.example.com"
     timeout: 30
   ```

3. Update environment variables in `.env.example`

## Configuration

The model is configured via YAML files in the `configs/` directory. Key parameters include:

- Model architecture (layer counts, dimensions, heads)
- Training hyperparameters (learning rates, batch sizes)
- Double-loop controller settings
- API integration configurations
- Hardware optimization settings

See `configs/default.yaml` for a complete example.

### Hardware Configuration

The system automatically detects and configures available hardware accelerators. Configure in `configs/default.yaml`:

```yaml
hardware:
  device: "auto"        # Auto-detect best device
  # OR specify manually:
  # device: "cuda"      # NVIDIA GPU
  # device: "mps"       # Apple Silicon
  # device: "npu"       # Neural Processing Unit
  # device: "cpu"       # CPU fallback
  
  gpu_id: null          # Specify GPU index for multi-GPU systems (e.g., 0, 1)
  prefer_npu: false     # Prefer NPU over GPU when both available
```

**Device Options:**
- `"auto"`: Automatically selects the best available device (GPU > NPU > CPU)
- `"cuda"` or `"cuda:0"`: NVIDIA GPU (specify index for multi-GPU)
- `"mps"`: Apple Silicon Neural Engine
- `"npu"`: Generic NPU (Intel AI Boost, AMD Ryzen AI, etc.)
- `"openvino"`: Intel AI Boost via OpenVINO
- `"ryzenai"`: AMD Ryzen AI
- `"cpu"`: CPU-only mode

**Hardware Detection (includes external devices):**
```python
from src.utils.gpu_utils import detect_gpu_info
from src.utils.npu_utils import check_accelerator_availability, get_best_available_device

# Detect all GPUs (internal + external eGPU)
gpu_info = detect_gpu_info()
print(f"Total GPUs: {gpu_info['device_count']}")
print(f"External GPUs: {gpu_info['external_gpu_count']}")

# Check what accelerators are available
availability = check_accelerator_availability()
print(f"CUDA (NVIDIA GPU): {availability['cuda']}")
print(f"MPS (Apple Silicon): {availability['mps']}")
print(f"NPU (Internal/External): {availability['npu']}")

# Get recommended device
device = get_best_available_device(prefer_npu=False)
print(f"Recommended: {device}")
```

**External Device Support:**
- **External GPUs (eGPU)**: Automatically detected via Thunderbolt 3/4, USB-C, or external PCIe
  - Shows connection type and performance characteristics
  - Works with all major eGPU enclosures (Razer Core, Sonnet, Akitio, etc.)
- **External NPUs**: Detects USB/PCIe AI accelerators
  - Google Coral Edge TPU (USB/M.2/PCIe)
  - Intel Movidius Neural Compute Stick 2
  - Hailo-8 AI Accelerator (PCIe)

For detailed hardware setup guides:
- **GPU Training**: See [docs/GPU_TRAINING.md](docs/GPU_TRAINING.md) - includes eGPU setup
- **NPU Training**: See [docs/NPU_TRAINING.md](docs/NPU_TRAINING.md) - includes external NPU setup

## Dataset Selection

You can assemble training/validation/test sets from multiple datasets declaratively in `configs/default.yaml` using the `data.datasets` list. Example:

```yaml
data:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  datasets:
    - name: multimodal_core
      type: multimodal
      data_dir: ./data/multimodal
      splits: {train: 0.8, val: 0.1, test: 0.1}
      enabled: true
    - name: captions_aux
      type: coco_captions
      root: ./data/coco/images
      ann_file: ./data/coco/annotations/captions_train2017.json
      splits: {train: 1.0}
      use_in: [train]
      enabled: true
```

Key fields:

- `type`: One of `multimodal`, `coco_captions`, `imagenet` (mapped to internal dataset classes).
- `splits`: Mapping of split name to ratio; must sum to 1.0. Omit for implicit `{train: 1.0}`.
- `use_in`: Optional restriction of which splits this dataset contributes to.
- `enabled`: Toggle inclusion without deleting entry.

Disable a dataset:

```yaml
    - name: captions_aux
      type: coco_captions
      # ...
      enabled: false
```

Programmatic usage inside notebooks or scripts:

```python
from src.utils.config import load_config
from src.data import build_dataloaders

config = load_config("configs/default.yaml")
train_loader, val_loader, test_loader = build_dataloaders(config)
print(len(train_loader), len(val_loader or []), len(test_loader or []))
```

If `data.datasets` is present, the `Trainer` automatically uses the selector; otherwise it falls back to legacy single-dataset keys (`train_dataset`, `val_dataset`).

## Training

### Hardware Requirements

**Minimum (GPU Training):**
- GPU: NVIDIA RTX 3060 12GB or AMD RX 6700 XT 12GB
- CPU: 6-core / 12-thread
- RAM: 16GB

**Recommended (GPU Training):**
- GPU: NVIDIA RTX 4070 12GB or RTX 3080 16GB
- CPU: 8-core / 16-thread
- RAM: 32GB

**CPU-Only Training:**
- CPU: 8-core / 16-thread or better
- RAM: 32GB+
- Note: Training will be significantly slower (10-50x)

**NPU Inference (After Training):**
- NPU: Intel AI Boost, AMD Ryzen AI, Apple Neural Engine, or Qualcomm Hexagon
- RAM: 16GB+
- Note: NPUs are optimized for inference, not training. Train on GPU/CPU, then export to ONNX for NPU deployment.

**External Device Training/Inference:**
- eGPU: Any desktop GPU in Thunderbolt 3/4 or USB-C enclosure
  - Thunderbolt bandwidth: 40 Gbps (expect 10-25% slower than internal)
  - Supports both training and inference
- External NPU: Coral Edge TPU, Intel Movidius NCS2, Hailo-8
  - USB 3.0/PCIe connection
  - Inference only (export to ONNX/TFLite first)
  - Ideal for prototyping edge deployments

### Supported Hardware

**NVIDIA GPUs (CUDA):**
- RTX 40 Series: 4090, 4080, 4070 (Ada Lovelace)
- RTX 30 Series: 3090, 3080, 3070, 3060 (Ampere)
- RTX 20 Series: 2080 Ti, 2070 (Turing)
- GTX 16 Series: 1660 Ti (Turing)
- Data Center: A100, A40, V100, T4

**AMD GPUs (ROCm):**
- RX 7000 Series: 7900 XTX, 7900 XT (RDNA 3)
- RX 6000 Series: 6900 XT, 6800 XT, 6700 XT (RDNA 2)
- Instinct: MI250, MI100

**Apple Silicon (MPS):**
- M3 Max, M3 Pro, M3
- M2 Ultra, M2 Max, M2 Pro, M2
- M1 Ultra, M1 Max, M1 Pro, M1

**Internal NPUs (Inference):**
- Intel: AI Boost (Meteor Lake, Lunar Lake) - ~10 TOPS
- AMD: Ryzen AI (Phoenix, Hawk Point) - ~10-16 TOPS
- Apple: Neural Engine (M1/M2/M3) - up to 15.8 TOPS
- Qualcomm: Hexagon NPU (Snapdragon X Elite/Plus) - ~45 TOPS

**External NPUs (Inference):**
- Google Coral Edge TPU (USB/M.2/PCIe) - 4 TOPS, ~$25-75
- Intel Movidius Neural Compute Stick 2 (USB) - ~1 TOPS, ~$70-100
- Hailo-8 AI Accelerator (PCIe/M.2) - 26 TOPS, ~$200-300

**External GPUs (eGPU Enclosures):**
- Thunderbolt 3/4: Razer Core X, Sonnet eGFX, Akitio Node
- Compatible with any desktop GPU (NVIDIA/AMD)
- Expect 10-25% performance reduction vs internal GPU

### Training Command

```bash
python -m src.training.trainer --config configs/default.yaml
```

Expected training time: 100-200 hours on minimum hardware.

## Evaluation

Run benchmarks:
```bash
python -m src.evaluation.benchmarks --config configs/default.yaml
```

Compare with API knowledge:
```bash
python -m src.evaluation.api_comparison --config configs/default.yaml
```

## Development

### Type Safety

The codebase maintains **complete type safety** with comprehensive mypy integration. All 23 source files pass strict static type checking with zero type errors.

#### Type Checking Features
- **Strict mypy Configuration**: Python 3.10+ support with comprehensive type checking rules
- **Complete Type Coverage**: 100% type annotations across the entire codebase
- **Type Stubs**: Full type stub support for all major dependencies (PyTorch, Transformers, etc.)
- **Protocol Usage**: Proper typing protocols for interface definitions and polymorphism
- **Generic Types**: Extensive use of Union, Optional, Dict, and custom generic types

#### Type Safety Benefits
- **Runtime Reliability**: Prevents type-related runtime errors through static analysis
- **Enhanced IDE Support**: Full IntelliSense, autocomplete, and refactoring capabilities
- **Documentation**: Type annotations serve as inline documentation for function signatures
- **Maintainability**: Easier code maintenance and refactoring with type guarantees
- **Developer Experience**: Better error messages and debugging capabilities

#### Running Type Checks
```bash
# Check entire codebase
mypy src/ --show-error-codes

# Check specific file
mypy src/models/multi_modal_model.py --show-error-codes

# Use cache for faster subsequent runs
mypy src/ --cache-dir /tmp/mypy_cache --show-error-codes
```

#### Type Checking Configuration
The type checking is configured in `pyproject.toml` with strict settings including:
- `disallow_untyped_defs`: All functions must have type annotations
- `disallow_incomplete_defs`: All parameters must be typed
- `no_implicit_optional`: Optional types must be explicit
- `warn_return_any`: Any return types are flagged as warnings
- `strict_equality`: Strict type equality checking

### Testing

Run the test suite:
```bash
# Quick test run (using make)
make test

# Run all tests with coverage
make test-cov

# Run tests with pytest directly
pytest tests/

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run integration tests
pytest tests/test_integration.py -v
```

**Test Coverage:** The project maintains **93% test coverage** (446 tests) across all modules.

### Code Quality

We use automated code quality tools with pre-commit hooks:

```bash
# Install pre-commit hooks (one-time setup)
pip install pre-commit
pre-commit install

# Run all quality checks
make lint

# Format code (using make)
make format

# Manual formatting
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
flake8 src/ tests/

# Security scan
bandit -r src/
```

### CI/CD

The project uses GitHub Actions for continuous integration:
- **Multi-version testing**: Python 3.11, 3.12, 3.13
- **Coverage reporting**: Automatic coverage reports on PRs
- **Dependency caching**: Fast CI builds with pip caching

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes with full type annotations
4. Add tests for new functionality
5. Ensure all tests pass and type checking succeeds
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{multi-modal-neural-network,
  title={Multi-Modal Neural Network with Double-Loop Learning},
  author={Tim Dickey},
  year={2025},
  url={https://github.com/tim-dickey/multi-modal-neural-network}
}
```

## Acknowledgments

- Built with PyTorch and Hugging Face Transformers
- Wolfram Alpha for symbolic computation
- Community contributors