"""Configuration system using dataclasses and YAML support."""
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_root: Path = field(default_factory=lambda: Path("./data"))
    csv_path: Path = field(default_factory=lambda: Path("./data/metadata.csv"))
    image_dir: Path = field(default_factory=lambda: Path("./data/images"))
    image_size: int = 224
    target_col: str = "target"
    image_id_cols: List[str] = field(default_factory=lambda: ["image_name"])
    max_samples: Optional[int] = None
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        self.dataset_root = Path(self.dataset_root)
        self.csv_path = Path(self.csv_path)
        self.image_dir = Path(self.image_dir)


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    rotation_degrees: float = 25.0
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    color_jitter_brightness: float = 0.15
    color_jitter_contrast: float = 0.15
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.02


@dataclass
class ImbalanceConfig:
    """Class imbalance handling configuration."""
    method: str = "weighted_sampler"  # weighted_sampler, smote, mixup, cutmix, none
    weighted_sampler_enabled: bool = True
    smote_enabled: bool = False
    smote_k_neighbors: int = 5
    mixup_enabled: bool = False
    mixup_alpha: float = 1.0
    cutmix_enabled: bool = False
    cutmix_alpha: float = 1.0


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: str = "weighted_bce"  # bce, weighted_bce, focal, cb_focal
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    cb_focal_beta: float = 0.9999


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "efficientnet_b0"  # efficientnet_b0, convnext_tiny, vit_tiny, swin_tiny
    pretrained: bool = True
    num_classes: int = 1  # Binary classification


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer_type: str = "adamw"  # adamw, sgd
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    scheduler_type: str = "cosine"  # cosine, step, exponential, none
    cosine_t_max: int = 20
    step_gamma: float = 0.1
    step_step_size: int = 10


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    num_epochs: int = 20
    num_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: int = 5
    early_stopping_metric: str = "auprc"  # auprc, roc_auc, f1
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class ThresholdOptimizationConfig:
    """Threshold optimization configuration."""
    enabled: bool = True
    methods: List[str] = field(default_factory=lambda: ["youden", "f1", "clinical"])
    clinical_fn_cost_ratio: float = 1.0  # Cost ratio FN / FP


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    threshold: float = 0.5
    save_predictions: bool = True
    compute_gradcam: bool = True
    compute_error_analysis: bool = True
    error_analysis_n_samples: int = 20


@dataclass
class OutputConfig:
    """Output paths configuration."""
    root_dir: Path = field(default_factory=lambda: Path("./outputs"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    figures_dir: Path = field(default_factory=lambda: Path("figures"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self) -> None:
        """Resolve relative paths."""
        self.root_dir = Path(self.root_dir)
        for attr in ["checkpoints_dir", "logs_dir", "figures_dir", "results_dir"]:
            path = Path(getattr(self, attr))  # Convert to Path first
            if not path.is_absolute():
                path = self.root_dir / path
            setattr(self, attr, path)


@dataclass
class Config:
    """Main configuration container."""
    seed: int = 42
    device: str = "cuda"  # cuda or cpu
    project_name: str = "skin-cancer-classification"
    models_cache_dir: Optional[Path] = None  # For offline HPC: path to cached models

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    threshold_optimization: ThresholdOptimizationConfig = field(
        default_factory=ThresholdOptimizationConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from nested dictionary."""
        # Extract nested configs
        dataset_dict = config_dict.pop("dataset", {})
        augmentation_dict = config_dict.pop("augmentation", {})
        imbalance_dict = config_dict.pop("imbalance", {})
        loss_dict = config_dict.pop("loss", {})
        model_dict = config_dict.pop("model", {})
        optimizer_dict = config_dict.pop("optimizer", {})
        scheduler_dict = config_dict.pop("scheduler", {})
        training_dict = config_dict.pop("training", {})
        threshold_opt_dict = config_dict.pop("threshold_optimization", {})
        evaluation_dict = config_dict.pop("evaluation", {})
        outputs_dict = config_dict.pop("outputs", {})

        # Create nested configs
        return cls(
            **config_dict,
            dataset=DatasetConfig(**dataset_dict) if dataset_dict else DatasetConfig(),
            augmentation=AugmentationConfig(**augmentation_dict) if augmentation_dict else AugmentationConfig(),
            imbalance=ImbalanceConfig(**imbalance_dict) if imbalance_dict else ImbalanceConfig(),
            loss=LossConfig(**loss_dict) if loss_dict else LossConfig(),
            model=ModelConfig(**model_dict) if model_dict else ModelConfig(),
            optimizer=OptimizerConfig(**optimizer_dict) if optimizer_dict else OptimizerConfig(),
            scheduler=SchedulerConfig(**scheduler_dict) if scheduler_dict else SchedulerConfig(),
            training=TrainingConfig(**training_dict) if training_dict else TrainingConfig(),
            threshold_optimization=ThresholdOptimizationConfig(**threshold_opt_dict)
            if threshold_opt_dict
            else ThresholdOptimizationConfig(),
            evaluation=EvaluationConfig(**evaluation_dict) if evaluation_dict else EvaluationConfig(),
            outputs=OutputConfig(**outputs_dict) if outputs_dict else OutputConfig(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if self.models_cache_dir is not None:
            self.models_cache_dir = Path(self.models_cache_dir)
