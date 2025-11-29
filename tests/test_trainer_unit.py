import types
import torch
import logging
from pathlib import Path

from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager


class DummyMetricsLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name

    def log_epoch(self, epoch, train_metrics, val_metrics):
        return None


def make_simple_trainer(tmp_path):
    # Bypass __init__ heavy behavior by creating object and setting attributes
    t = object.__new__(Trainer)

    # Minimal config
    t.config = {"training": {"max_epochs": 1, "inner_lr": 1e-3, "log_interval": 1},
                "paths": {"output_dir": str(tmp_path / "outputs")}}

    # Device
    t.device = torch.device("cpu")
    # Logger used by Trainer
    t.logger = logging.getLogger("trainer")

    # Simple model producing logits for 3 classes; accept keyword args used by Trainer
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # images will be (batch, 1, 2, 2) -> flattened size 4
            self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))

        def forward(self, images=None, input_ids=None, attention_mask=None):
            return {"logits": self.net(images)}

    t.model = SimpleModel()
    # Ensure model attributes expected by Trainer
    t.model.use_double_loop = False

    # Criterion and optimizer
    t.criterion = torch.nn.CrossEntropyLoss()
    t.optimizer = torch.optim.SGD(t.model.parameters(), lr=1e-3)

    # Dummy scheduler with state_dict and step
    t.scheduler = types.SimpleNamespace(state_dict=lambda: {}, step=lambda: None)

    # Metrics and wandb
    t.metrics_logger = DummyMetricsLogger(log_dir=str(tmp_path / "logs"), experiment_name="x")
    t.wandb_logger = None

    # Training state
    t.current_epoch = 0
    t.global_step = 0
    t.best_val_loss = float("inf")

    # Paths
    t.output_dir = tmp_path / "outputs"
    t.checkpoint_dir = tmp_path / "checkpoints"
    t.output_dir.mkdir(parents=True, exist_ok=True)
    t.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint manager (added for decomposed trainer)
    t.checkpoint_manager = CheckpointManager(
        checkpoint_dir=t.checkpoint_dir,
        logger=t.logger,
    )

    return t


def test_normalize_train_step_and_validate(tmp_path):
    t = make_simple_trainer(tmp_path)

    # Create a single batch loader with images of shape (batch, C, H, W) -> map to flattened
    batch = {"images": torch.randn(2, 1, 2, 2), "labels": torch.tensor([0, 1])}
    t.train_loader = [batch]
    t.val_loader = [batch]

    # Test _normalize_batch
    nb = t._normalize_batch({"image": batch["images"], "label": batch["labels"]})
    assert "images" in nb and "labels" in nb

    # Run train_epoch
    metrics = t.train_epoch(0)
    assert "loss" in metrics and "accuracy" in metrics

    # Run train_step
    loss, acc = t.train_step(batch)
    assert loss is not None

    # Validate should return metrics
    v = t.validate()
    assert isinstance(v["loss"], float)

    # Test save_checkpoint to path
    path = str(tmp_path / "ckpt.pt")
    t.save_checkpoint(path=path, epoch=0, step=1, is_best=True)
    assert Path(path).exists()
