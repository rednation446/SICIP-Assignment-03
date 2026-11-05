from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # model
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128

    # data
    toxicity_threshold: float = 0.5
    train_size: int = 40000
    train_toxic_ratio: float = 0.3
    val_size: int = 5000
    test_size: int = 10000
    tune_train_size: int = 8000

    # frozen baseline
    embed_batch_size: int = 64
    logreg_C: float = 1.0

    # fine-tuning defaults (used before tuning)
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # hyperparameter search
    n_trials: int = 12
    search_space: dict = field(default_factory=lambda: {
        "learning_rate": (1e-5, 6e-5),
        "batch_size": [16, 32],
        "epochs": [1, 2, 3],
        "weight_decay": (0.0, 0.1),
        "warmup_ratio": (0.0, 0.2),
    })

    # interpretability
    shap_max_evals: int = 300

    seed: int = 42
    figures_dir: Path = PROJECT_ROOT / "figures"
    results_dir: Path = PROJECT_ROOT / "results"
    models_dir: Path = PROJECT_ROOT / "saved_models"

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v.relative_to(PROJECT_ROOT))
        return d


def get_config(preset: str = "full") -> Config:
    """
    full  : the setting used for the reported results (GPU recommended)
    quick : a much smaller run that finishes on a laptop CPU
    """
    cfg = Config()
    if preset == "quick":
        cfg.train_size = 3000
        cfg.val_size = 800
        cfg.test_size = 1500
        cfg.tune_train_size = 1500
        cfg.n_trials = 4
        cfg.epochs = 1
        cfg.search_space["epochs"] = [1, 2]
        cfg.shap_max_evals = 200
    elif preset != "full":
        raise ValueError(f"Unknown preset: {preset}")

    for d in (cfg.figures_dir, cfg.results_dir, cfg.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
