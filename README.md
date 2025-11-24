# Multi-Modal Neural Network with Double-Loop Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains an open-source implementation of a multi-modal small neural network that incorporates double-loop learning mechanisms and integrates with Wolfram Alpha for computational knowledge enhancement. The system is designed to train on consumer-grade hardware while maintaining acceptable performance and accuracy.

## Features

- **Multi-Modal Architecture**: Supports vision (images) and text modalities with early fusion.
- **Double-Loop Learning**: Implements meta-learning for structural adaptation during training.
- **Wolfram Alpha Integration**: Provides symbolic computation for ground truth verification.
- **Consumer Hardware Optimized**: Designed for single GPU systems (8-16GB VRAM, 16-32GB RAM).
- **Parameter Efficient**: Total parameters capped at 100-500 million.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ or ROCm 5.7+
- NVIDIA RTX 3060 (12GB) or equivalent AMD GPU

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-modal-neural-net.git
   cd multi-modal-neural-net
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install with Poetry:
   ```bash
   poetry install
   ```

## Quick Start

1. Configure your environment:
   ```bash
   cp configs/default.yaml configs/my_config.yaml
   # Edit my_config.yaml with your settings
   ```

2. Run the getting started notebook:
   ```bash
   jupyter notebook notebooks/01_getting_started.ipynb
   ```

3. Train the model:
   ```python
   from src.training.trainer import Trainer
   trainer = Trainer(config_path="configs/my_config.yaml")
   trainer.train()
   ```

## Project Structure

```
multi-modal-neural-net/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── src/
│   ├── models/
│   │   ├── vision_encoder.py
│   │   ├── text_encoder.py
│   │   ├── fusion_layer.py
│   │   ├── double_loop_controller.py
│   │   └── heads.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── losses.py
│   │   └── checkpointing.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   └── streaming.py
│   ├── integrations/
│   │   ├── wolfram_alpha.py
│   │   ├── validators.py
│   │   └── knowledge_injection.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmarks.py
│   │   └── wolfram_comparison.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── profiling.py
├── notebooks/
│   ├── 01_getting_started.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── tests/
├── docs/
└── examples/
```

## Configuration

The model is configured via YAML files in the `configs/` directory. Key parameters include:

- Model architecture (layer counts, dimensions, heads)
- Training hyperparameters (learning rates, batch sizes)
- Double-loop controller settings
- Wolfram Alpha API configuration
- Hardware optimization settings

See `configs/default.yaml` for a complete example.

## Training

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3060 12GB or AMD RX 6700 XT 12GB
- CPU: 6-core / 12-thread
- RAM: 16GB

**Recommended:**
- GPU: NVIDIA RTX 4070 12GB or RTX 3080 16GB
- CPU: 8-core / 16-thread
- RAM: 32GB

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

## Wolfram Alpha Integration

To use Wolfram Alpha features:

1. Obtain an API key from [Wolfram Alpha Developer Portal](https://developer.wolframalpha.com/)
2. Set the environment variable:
   ```bash
   export WOLFRAM_API_KEY="your_api_key_here"
   ```
3. Configure in your YAML:
   ```yaml
   wolfram:
     api_key: "${WOLFRAM_API_KEY}"
   ```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{multi-modal-neural-net,
  title={Multi-Modal Neural Network with Double-Loop Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/multi-modal-neural-net}
}
```

## Acknowledgments

- Built with PyTorch and Hugging Face Transformers
- Wolfram Alpha for symbolic computation
- Community contributors