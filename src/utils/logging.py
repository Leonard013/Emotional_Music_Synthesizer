"""
Logging utilities for the whistle-to-music synthesizer.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time
from datetime import datetime


def setup_logging(log_level: str = "INFO",
                  log_file: Optional[str] = None,
                  log_format: Optional[str] = None,
                  include_timestamp: bool = True) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        include_timestamp: Whether to include timestamp in log format
        
    Returns:
        Configured logger
    """
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Default log format
    if log_format is None:
        if include_timestamp:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            log_format = '%(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[]
    )
    
    # Get root logger
    logger = logging.getLogger()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """Custom logger for training progress."""
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 experiment_name: Optional[str] = None):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Setup loggers
        self.train_logger = self._setup_file_logger(
            "training", 
            self.experiment_dir / "training.log"
        )
        self.eval_logger = self._setup_file_logger(
            "evaluation", 
            self.experiment_dir / "evaluation.log"
        )
        
        # Training metrics
        self.metrics_history = []
        self.start_time = time.time()
    
    def _setup_file_logger(self, name: str, log_file: Path) -> logging.Logger:
        """Setup file logger."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_training_step(self, 
                         epoch: int, 
                         step: int, 
                         loss: float, 
                         metrics: Dict[str, float]):
        """Log training step."""
        elapsed_time = time.time() - self.start_time
        
        log_message = (
            f"Epoch {epoch}, Step {step} | "
            f"Loss: {loss:.4f} | "
            f"Time: {elapsed_time:.2f}s"
        )
        
        # Add metrics
        for key, value in metrics.items():
            log_message += f" | {key}: {value:.4f}"
        
        self.train_logger.info(log_message)
        
        # Store metrics
        self.metrics_history.append({
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'metrics': metrics,
            'timestamp': time.time()
        })
    
    def log_evaluation(self, 
                      epoch: int, 
                      val_loss: float, 
                      val_metrics: Dict[str, float]):
        """Log evaluation results."""
        log_message = f"Epoch {epoch} Evaluation | Val Loss: {val_loss:.4f}"
        
        for key, value in val_metrics.items():
            log_message += f" | Val {key}: {value:.4f}"
        
        self.eval_logger.info(log_message)
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.train_logger.info("Model Information:")
        for key, value in model_info.items():
            self.train_logger.info(f"  {key}: {value}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        self.train_logger.info("Training Configuration:")
        self._log_dict(config, indent=2)
    
    def _log_dict(self, d: Dict[str, Any], indent: int = 0):
        """Log dictionary with indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                self.train_logger.info("  " * indent + f"{key}:")
                self._log_dict(value, indent + 1)
            else:
                self.train_logger.info("  " * indent + f"{key}: {value}")
    
    def save_metrics_summary(self):
        """Save metrics summary to file."""
        import json
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_training_time': time.time() - self.start_time,
            'total_steps': len(self.metrics_history),
            'final_metrics': self.metrics_history[-1] if self.metrics_history else None,
            'metrics_history': self.metrics_history
        }
        
        summary_file = self.experiment_dir / "metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.train_logger.info(f"Metrics summary saved to {summary_file}")


class ProgressLogger:
    """Logger for progress tracking."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress logger.
        
        Args:
            total: Total number of items to process
            description: Description of the process
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.logger = get_logger("progress")
    
    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress."""
        self.current += increment
        
        # Calculate progress
        progress = self.current / self.total
        elapsed_time = time.time() - self.start_time
        
        if self.current > 0:
            eta = elapsed_time * (self.total - self.current) / self.current
        else:
            eta = 0
        
        # Log progress
        log_message = (
            f"{self.description}: {self.current}/{self.total} "
            f"({progress:.1%}) | "
            f"Elapsed: {elapsed_time:.1f}s | "
            f"ETA: {eta:.1f}s"
        )
        
        if message:
            log_message += f" | {message}"
        
        self.logger.info(log_message)
    
    def finish(self, message: Optional[str] = None):
        """Finish progress logging."""
        total_time = time.time() - self.start_time
        
        log_message = (
            f"{self.description} completed: {self.total} items in {total_time:.1f}s"
        )
        
        if message:
            log_message += f" | {message}"
        
        self.logger.info(log_message)


def log_system_info(logger: Optional[logging.Logger] = None):
    """Log system information."""
    if logger is None:
        logger = get_logger("system")
    
    import platform
    import torch
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def log_model_summary(model, logger: Optional[logging.Logger] = None):
    """Log model summary."""
    if logger is None:
        logger = get_logger("model")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Summary:")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Log model architecture
    logger.info("Model Architecture:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name}: {module_params:,} parameters")
