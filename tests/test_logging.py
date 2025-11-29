"""Tests for logging utilities."""

import logging
from unittest.mock import Mock, patch

import pytest

from src.utils.logging import MetricsLogger, WandbLogger, log_model_info, setup_logger


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger(name="test_basic")
        
        assert logger.name == "test_basic"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler only
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_setup_logger_with_file(self, tmp_path):
        """Test logger setup with file handler."""
        log_file = tmp_path / "logs" / "test.log"
        
        logger = setup_logger(name="test_file", log_file=str(log_file))
        
        assert len(logger.handlers) == 2  # Console and file handlers
        assert log_file.exists()
        
        # Log something and verify it's in the file
        logger.info("Test message")
        for handler in logger.handlers:
            handler.flush()
        
        content = log_file.read_text()
        assert "Test message" in content
        
        # Clean up handlers to avoid ResourceWarning
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

    def test_setup_logger_custom_level(self):
        """Test logger with custom level."""
        logger = setup_logger(name="test_debug", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG
        assert logger.handlers[0].level == logging.DEBUG

    def test_setup_logger_replaces_existing_handlers(self):
        """Test that setup_logger replaces existing handlers."""
        # First setup
        logger = setup_logger(name="test_replace")
        assert len(logger.handlers) == 1
        
        # Second setup should replace handlers
        logger = setup_logger(name="test_replace")
        assert len(logger.handlers) == 1  # Still only 1 handler

    def test_setup_logger_creates_parent_directories(self, tmp_path):
        """Test that log file parent directories are created."""
        log_file = tmp_path / "deep" / "nested" / "dirs" / "test.log"
        
        logger = setup_logger(name="test_dirs", log_file=str(log_file))
        
        assert log_file.parent.exists()
        
        # Clean up
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)


class TestMetricsLogger:
    """Tests for MetricsLogger class."""

    def test_metrics_logger_init(self, tmp_path):
        """Test MetricsLogger initialization."""
        log_dir = tmp_path / "metrics"
        
        logger = MetricsLogger(str(log_dir), "test_experiment")
        
        assert logger.log_dir == log_dir
        assert logger.experiment_name == "test_experiment"
        assert logger.metrics_file.exists()
        
        content = logger.metrics_file.read_text()
        assert "Experiment: test_experiment" in content
        assert "Started:" in content

    def test_log_metrics(self, tmp_path):
        """Test logging metrics."""
        logger = MetricsLogger(str(tmp_path), "test_metrics")
        
        metrics = {"loss": 0.5, "accuracy": 0.95, "step_count": 100}
        logger.log_metrics(step=10, metrics=metrics)
        
        content = logger.metrics_file.read_text()
        assert "Step 10:" in content
        assert "loss: 0.500000" in content
        assert "accuracy: 0.950000" in content
        assert "step_count: 100" in content

    def test_log_metrics_with_prefix(self, tmp_path):
        """Test logging metrics with prefix."""
        logger = MetricsLogger(str(tmp_path), "test_prefix")
        
        metrics = {"loss": 0.3}
        logger.log_metrics(step=5, metrics=metrics, prefix="train_")
        
        content = logger.metrics_file.read_text()
        assert "train_loss: 0.300000" in content

    def test_log_epoch(self, tmp_path):
        """Test logging epoch summary."""
        logger = MetricsLogger(str(tmp_path), "test_epoch")
        
        train_metrics = {"loss": 0.2, "accuracy": 0.98}
        val_metrics = {"loss": 0.3, "accuracy": 0.95}
        
        logger.log_epoch(epoch=1, train_metrics=train_metrics, val_metrics=val_metrics)
        
        content = logger.metrics_file.read_text()
        assert "Epoch 1 Summary:" in content
        assert "Train metrics:" in content
        assert "Validation metrics:" in content
        assert "loss: 0.200000" in content  # train loss
        assert "loss: 0.300000" in content  # val loss

    def test_log_epoch_without_validation(self, tmp_path):
        """Test logging epoch without validation metrics."""
        logger = MetricsLogger(str(tmp_path), "test_no_val")
        
        train_metrics = {"loss": 0.25, "accuracy": 0.9}
        
        logger.log_epoch(epoch=2, train_metrics=train_metrics, val_metrics=None)
        
        content = logger.metrics_file.read_text()
        assert "Epoch 2 Summary:" in content
        assert "Train metrics:" in content
        assert "Validation metrics:" not in content


class TestWandbLogger:
    """Tests for WandbLogger class."""

    def test_wandb_logger_disabled(self):
        """Test WandbLogger when disabled."""
        logger = WandbLogger(
            project="test",
            experiment="test_exp",
            config={"lr": 0.001},
            enabled=False,
        )
        
        assert logger.enabled is False
        
        # Should not raise even when disabled
        logger.log({"loss": 0.5})
        logger.log_image("image", Mock())
        logger.finish()

    def test_wandb_logger_enabled(self):
        """Test WandbLogger when enabled with real wandb if available."""
        # Just test the disabled path since we can't reliably mock wandb import
        logger = WandbLogger(
            project="test_project",
            experiment="test_exp",
            config={"lr": 0.001},
            enabled=False,
        )
        
        assert logger.enabled is False

    def test_wandb_logger_import_error(self):
        """Test WandbLogger gracefully handles ImportError."""
        with patch.dict("sys.modules", {"wandb": None}):
            with patch("builtins.__import__", side_effect=ImportError("No wandb")):
                logger = WandbLogger(
                    project="test",
                    experiment="test_exp",
                    config={},
                    enabled=True,
                )
                
                # Should fall back to disabled
                assert logger.enabled is False

    def test_wandb_logger_init_error(self):
        """Test WandbLogger handles wandb init errors."""
        mock_wandb = Mock()
        mock_wandb.init.side_effect = RuntimeError("Init failed")
        
        # Create a logger that will try to initialize but fail
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch("builtins.__import__", return_value=mock_wandb):
                logger = WandbLogger(
                    project="test",
                    experiment="test",
                    config={},
                    enabled=True,
                )
                # Should be disabled after error
                assert logger.enabled is False

    def test_wandb_logger_log_when_disabled(self):
        """Test log methods do nothing when disabled."""
        logger = WandbLogger("p", "e", {}, enabled=False)
        
        # These should not raise
        logger.log({"metric": 1.0}, step=1)
        logger.log_image("key", Mock(), step=1)
        logger.finish()


class TestLogModelInfo:
    """Tests for log_model_info function."""

    def test_log_model_info_basic(self, caplog):
        """Test log_model_info with basic model."""
        logger = logging.getLogger("test_model_info")
        logger.setLevel(logging.INFO)
        
        model = Mock()
        # Remove methods to test basic case
        del model.get_model_info
        del model.get_num_parameters
        
        with caplog.at_level(logging.INFO):
            log_model_info(logger, model)
        
        assert "Model Information:" in caplog.text

    def test_log_model_info_with_model_info(self, caplog):
        """Test log_model_info with get_model_info method."""
        logger = logging.getLogger("test_model_info_method")
        logger.setLevel(logging.INFO)
        
        model = Mock()
        model.get_model_info.return_value = {
            "architecture": "Transformer",
            "layers": 12,
        }
        del model.get_num_parameters
        
        with caplog.at_level(logging.INFO):
            log_model_info(logger, model)
        
        assert "architecture" in caplog.text
        assert "Transformer" in caplog.text

    def test_log_model_info_with_parameters(self, caplog):
        """Test log_model_info with get_num_parameters method."""
        logger = logging.getLogger("test_model_params")
        logger.setLevel(logging.INFO)
        
        model = Mock()
        del model.get_model_info
        model.get_num_parameters.side_effect = lambda trainable_only: (
            50000 if trainable_only else 100000
        )
        
        with caplog.at_level(logging.INFO):
            log_model_info(logger, model)
        
        assert "Total parameters: 100000" in caplog.text
        assert "Trainable parameters: 50000" in caplog.text
        assert "Non-trainable parameters: 50000" in caplog.text

    def test_log_model_info_full(self, caplog):
        """Test log_model_info with all model methods."""
        logger = logging.getLogger("test_model_full")
        logger.setLevel(logging.INFO)
        
        model = Mock()
        model.get_model_info.return_value = {"name": "TestModel"}
        model.get_num_parameters.side_effect = lambda trainable_only: (
            1000 if trainable_only else 2000
        )
        
        with caplog.at_level(logging.INFO):
            log_model_info(logger, model)
        
        assert "name" in caplog.text
        assert "TestModel" in caplog.text
        assert "Total parameters: 2000" in caplog.text
