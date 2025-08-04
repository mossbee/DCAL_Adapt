import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class TwinConfig:
    """
    Configuration class for twin verification
    """
    
    # Training device
    device: str = "cuda"
    
    # Pair sampling ratios
    same_person_ratio: float = 0.5
    twin_pairs_ratio: float = 0.3
    non_twin_ratio: float = 0.2
    
    # Model configuration
    backbone: str = "deit_tiny_patch16_224"
    embedding_dim: int = 512
    top_ratio: float = 0.35
    learnable_threshold: bool = True
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    save_frequency: int = 1
    num_workers: int = 4
    
    # Loss weights
    verification_loss_weight: float = 1.0
    triplet_loss_weight: float = 0.1
    use_dynamic_weighting: bool = True
    
    # Loss parameters
    verification_margin: float = 0.0
    triplet_margin: float = 0.3
    twin_triplet_margin: float = 0.1
    distance_metric: str = "cosine"
    
    # Progressive training
    progressive_training: bool = True
    phase1_epochs: int = 30
    phase2_epochs: int = 40
    phase3_epochs: int = 30
    
    # Evaluation
    evaluation_thresholds: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Paths
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Check device
        if self.device not in ["cuda", "cpu"]:
            errors.append(f"Invalid device: {self.device}. Must be 'cuda' or 'cpu'")
            
        # Check ratios are positive
        if self.same_person_ratio <= 0:
            errors.append(f"Same person ratio must be positive, got {self.same_person_ratio}")
        if self.twin_pairs_ratio <= 0:
            errors.append(f"Twin pairs ratio must be positive, got {self.twin_pairs_ratio}")
        if self.non_twin_ratio <= 0:
            errors.append(f"Non-twin ratio must be positive, got {self.non_twin_ratio}")
            
        # Check positive values
        if self.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            errors.append(f"Learning rate must be positive, got {self.learning_rate}")
        if self.num_epochs <= 0:
            errors.append(f"Number of epochs must be positive, got {self.num_epochs}")
            
        # Check embedding dimension
        if self.embedding_dim <= 0:
            errors.append(f"Embedding dimension must be positive, got {self.embedding_dim}")
            
        # Check top ratio
        if not (0.0 < self.top_ratio <= 1.0):
            errors.append(f"Top ratio must be between 0 and 1, got {self.top_ratio}")
            
        # Check distance metric
        if self.distance_metric not in ["cosine", "euclidean"]:
            errors.append(f"Invalid distance metric: {self.distance_metric}")
            
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'device': self.device,
            'same_person_ratio': self.same_person_ratio,
            'twin_pairs_ratio': self.twin_pairs_ratio,
            'non_twin_ratio': self.non_twin_ratio,
            'backbone': self.backbone,
            'embedding_dim': self.embedding_dim,
            'top_ratio': self.top_ratio,
            'learnable_threshold': self.learnable_threshold,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'save_frequency': self.save_frequency,
            'num_workers': self.num_workers,
            'verification_loss_weight': self.verification_loss_weight,
            'triplet_loss_weight': self.triplet_loss_weight,
            'use_dynamic_weighting': self.use_dynamic_weighting,
            'verification_margin': self.verification_margin,
            'triplet_margin': self.triplet_margin,
            'twin_triplet_margin': self.twin_triplet_margin,
            'distance_metric': self.distance_metric,
            'progressive_training': self.progressive_training,
            'phase1_epochs': self.phase1_epochs,
            'phase2_epochs': self.phase2_epochs,
            'phase3_epochs': self.phase3_epochs,
            'evaluation_thresholds': self.evaluation_thresholds,
            'save_dir': self.save_dir,
            'log_dir': self.log_dir
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TwinConfig':
        """Create from dictionary"""
        return cls(**config_dict)


class ConfigManager:
    """
    Configuration manager for twin verification
    """
    
    # Default configurations for different scenarios
    DEFAULT_CONFIG = TwinConfig()
    
    CPU_CONFIG = TwinConfig(
        device="cpu",
        batch_size=16,  # Smaller batch size for CPU
        num_workers=0,  # No multiprocessing for CPU
        progressive_training=False  # Disable progressive training for CPU
    )
    
    GPU_CONFIG = TwinConfig(
        device="cuda",
        batch_size=32,
        num_workers=4,
        progressive_training=True
    )
    
    FAST_CONFIG = TwinConfig(
        device="cuda",
        batch_size=16,
        num_epochs=50,
        progressive_training=False,
        phase1_epochs=15,
        phase2_epochs=20,
        phase3_epochs=15
    )
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path]) -> TwinConfig:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            TwinConfig object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
                
        config = TwinConfig.from_dict(config_dict)
        
        if not config.validate():
            raise ValueError("Configuration validation failed")
            
        return config
        
    @classmethod
    def save_config(cls, config: TwinConfig, config_path: Union[str, Path]):
        """
        Save configuration to file
        
        Args:
            config: TwinConfig object
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
                
    @classmethod
    def get_preset_config(cls, preset: str) -> TwinConfig:
        """
        Get preset configuration
        
        Args:
            preset: Preset name ('default', 'cpu', 'gpu', 'fast')
            
        Returns:
            TwinConfig object
        """
        presets = {
            'default': cls.DEFAULT_CONFIG,
            'cpu': cls.CPU_CONFIG,
            'gpu': cls.GPU_CONFIG,
            'fast': cls.FAST_CONFIG
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available presets: {list(presets.keys())}")
            
        return presets[preset]
        
    @classmethod
    def create_config(cls, 
                     device: str = "cuda",
                     batch_size: Optional[int] = None,
                     num_epochs: Optional[int] = None,
                     **kwargs) -> TwinConfig:
        """
        Create configuration with custom parameters
        
        Args:
            device: Device to use
            batch_size: Batch size (auto-adjusted for device if None)
            num_epochs: Number of epochs
            **kwargs: Additional configuration parameters
            
        Returns:
            TwinConfig object
        """
        # Start with appropriate preset
        if device == "cpu":
            config = cls.CPU_CONFIG
        else:
            config = cls.GPU_CONFIG
            
        # Override with provided parameters
        if batch_size is not None:
            config.batch_size = batch_size
        if num_epochs is not None:
            config.num_epochs = num_epochs
            
        # Override with additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
                
        if not config.validate():
            raise ValueError("Configuration validation failed")
            
        return config


def create_default_config_files():
    """Create default configuration files"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # Create YAML config
    yaml_config = ConfigManager.get_preset_config('gpu')
    ConfigManager.save_config(yaml_config, config_dir / "twin_config.yaml")
    
    # Create CPU config
    cpu_config = ConfigManager.get_preset_config('cpu')
    ConfigManager.save_config(cpu_config, config_dir / "twin_config_cpu.yaml")
    
    # Create fast config
    fast_config = ConfigManager.get_preset_config('fast')
    ConfigManager.save_config(fast_config, config_dir / "twin_config_fast.yaml")
    
    print("Default configuration files created in 'configs/' directory:")
    print("  - twin_config.yaml (GPU optimized)")
    print("  - twin_config_cpu.yaml (CPU optimized)")
    print("  - twin_config_fast.yaml (Fast training)")


def load_twin_config(config_path: Optional[Union[str, Path]] = None,
                    preset: Optional[str] = None,
                    **kwargs) -> TwinConfig:
    """
    Load twin configuration
    
    Args:
        config_path: Path to configuration file
        preset: Preset name ('default', 'cpu', 'gpu', 'fast')
        **kwargs: Additional configuration parameters
        
    Returns:
        TwinConfig object
    """
    if config_path is not None:
        # Load from file
        config = ConfigManager.load_config(config_path)
    elif preset is not None:
        # Load preset
        config = ConfigManager.get_preset_config(preset)
    else:
        # Use default
        config = ConfigManager.get_preset_config('default')
        
    # Override with additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter: {key}")
            
    if not config.validate():
        raise ValueError("Configuration validation failed")
        
    return config


if __name__ == "__main__":
    # Create default configuration files
    create_default_config_files() 