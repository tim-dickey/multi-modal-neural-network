# Multi-Modal Neural Network with Double-Loop Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

This repository contains an open-source implementation of a multi-modal small neural network that incorporates double-loop learning mechanisms and integrates with external APIs for computational knowledge enhancement. The system is designed to train on consumer-grade hardware while maintaining acceptable performance and accuracy.

## Features

- **Multi-Modal Architecture**: Supports vision (images) and text modalities with early fusion.
- **Double-Loop Learning**: Implements meta-learning for structural adaptation during training.
- **API Integration Framework**: Extensible framework for external knowledge sources (Wolfram Alpha, etc.).
- **Consumer Hardware Optimized**: Designed for single GPU systems (8-16GB VRAM, 16-32GB RAM).
- **Parameter Efficient**: Total parameters capped at 100-500 million.
- **Full Type Safety**: Complete type annotations with mypy compliance.
- **Production Ready**: Comprehensive configuration management and environment variable support.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ or ROCm 5.7+
- NVIDIA RTX 3060 (12GB) or equivalent AMD GPU

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

4. (Optional) Install with Poetry:
   ```bash
   poetry install
   ```

## Quick Start

1. Configure your environment:
   ```bash
   cp configs/default.yaml configs/my_config.yaml
   # Edit my_config.yaml with your settings
   ```

2. Set up environment variables (see [Environment Setup](#environment-setup))

3. Run the getting started notebook:
   ```bash
   jupyter notebook notebooks/01_getting_started.ipynb
   ```

4. Train the model:
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
│       └── profiling.py           # Performance profiling
├── notebooks/
│   ├── 01_getting_started.ipynb   # Setup and basic usage
│   ├── 02_training.ipynb          # Training workflows
│   └── 03_evaluation.ipynb        # Evaluation and analysis
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
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

Compare with API knowledge:
```bash
python -m src.evaluation.api_comparison --config configs/default.yaml
```

## Development

### Type Safety

The codebase maintains full type safety with mypy. Core model files are fully annotated.

Run type checking:
```bash
mypy src/ --show-error-codes
```

### Testing

Run the test suite:
```bash
pytest tests/
```

### Code Quality

Format code:
```bash
black src/ tests/
isort src/ tests/
```

Lint code:
```bash
flake8 src/ tests/
```

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