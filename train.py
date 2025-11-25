"""Example training script for multi-modal neural network."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training import Trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train multi-modal neural network')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(
        config_path=args.config,
        resume_from=args.resume,
        device=args.device
    )
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
