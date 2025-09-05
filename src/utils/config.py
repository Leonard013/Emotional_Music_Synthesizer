"""
Configuration management utilities.
Handles loading, saving, and validating configuration files.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import os


@dataclass
class Config:
    """Base configuration class."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def to_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def load_config(config_path: Union[str, Path], 
                config_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        config_type: Type of configuration ('yaml', 'json', 'auto')
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Auto-detect file type if not specified
    if config_type is None:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_type = 'yaml'
        elif config_path.suffix.lower() == '.json':
            config_type = 'json'
        else:
            raise ValueError(f"Unsupported configuration file type: {config_path.suffix}")
    
    # Load configuration
    if config_type.lower() == 'yaml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_type.lower() == 'json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration type: {config_type}")
    
    return config


def save_config(config: Dict[str, Any], 
                config_path: Union[str, Path],
                config_type: Optional[str] = None):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        config_type: Type of configuration ('yaml', 'json', 'auto')
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect file type if not specified
    if config_type is None:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_type = 'yaml'
        elif config_path.suffix.lower() == '.json':
            config_type = 'json'
        else:
            config_type = 'yaml'  # Default to YAML
    
    # Save configuration
    if config_type.lower() == 'yaml':
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif config_type.lower() == 'json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration type: {config_type}")


def merge_configs(base_config: Dict[str, Any], 
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any], 
                   required_keys: Optional[list] = None,
                   config_schema: Optional[Dict] = None) -> bool:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        config_schema: Configuration schema for validation
        
    Returns:
        True if configuration is valid
    """
    # Check required keys
    if required_keys:
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Required configuration key missing: {key}")
    
    # Validate against schema if provided
    if config_schema:
        _validate_schema(config, config_schema)
    
    return True


def _validate_schema(config: Dict[str, Any], schema: Dict[str, Any]):
    """Validate configuration against schema."""
    for key, expected_type in schema.items():
        if key in config:
            if not isinstance(config[key], expected_type):
                raise ValueError(f"Configuration key '{key}' should be of type {expected_type}, got {type(config[key])}")


def get_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'model.d_model')
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    value: Any):
    """
    Set configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'model.d_model')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def load_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load configuration values from environment variables.
    
    Args:
        config: Base configuration
        
    Returns:
        Configuration with environment variables loaded
    """
    env_config = config.copy()
    
    # Common environment variables
    env_mappings = {
        'WMS_MODEL_PATH': 'model.model_path',
        'WMS_DEVICE': 'model.device',
        'WMS_BATCH_SIZE': 'training.batch_size',
        'WMS_LEARNING_RATE': 'training.learning_rate',
        'WMS_OUTPUT_DIR': 'output.output_dir',
        'WMS_MAESTRO_DIR': 'data.maestro_dir',
        'WMS_USE_WANDB': 'logging.use_wandb',
        'WMS_WANDB_PROJECT': 'logging.wandb_project'
    }
    
    for env_var, config_path in env_mappings.items():
        if env_var in os.environ:
            set_config_value(env_config, config_path, os.environ[env_var])
    
    return env_config


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'model': {
            'vocab_size': 2000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 1024,
            'conditioning_dim': 16,
            'dropout': 0.1,
            'use_conditioning': True,
            'use_musical_embedding': True
        },
        'training': {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'num_epochs': 100,
            'warmup_steps': 1000,
            'max_grad_norm': 1.0,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'mixed_precision': True,
            'early_stopping_patience': 10
        },
        'data': {
            'maestro_dir': 'data/maestro',
            'cache_dir': 'data/cache',
            'max_files_per_split': {
                'train': None,
                'validation': 50,
                'test': 50
            },
            'num_workers': 4,
            'pin_memory': True
        },
        'logging': {
            'use_wandb': False,
            'wandb_project': 'whistle-music-synthesizer',
            'log_every': 100,
            'save_every': 10,
            'eval_every': 5
        },
        'output': {
            'output_dir': './outputs',
            'save_best_model': True,
            'save_checkpoints': True
        },
        'device': 'auto'
    }
