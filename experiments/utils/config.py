from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    is_pretrained: bool
    max_seq_len: int | None = None
    config_name: str | None = None
    tokenizer_name: str | None = None


class DataConfig(BaseModel):
    train: str
    val: str
    test: str
    use_2d_seq: bool = False


class TrainingConfig(BaseModel):
    num_train_epochs: int
    batch_size: int
    warmup_steps: int
    logging_steps: int
    logging_dir: str
    learning_rate: float
    loss_type: str
    num_workers: int = 1


class ExperimentConfig(BaseModel):
    config_file: str | Path
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, config_file: str | Path) -> "ExperimentConfig":
        """Load a configuration from a YAML file."""
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        config_dict["config_file"] = config_file

        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation of a ExperimentConfig object."""
        items = []
        config_dict = self.model_dump()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value_repr = "\n".join(f"    {k}: {v}" for k, v in value.items())
                items.append(f"{key}:\n{value_repr}")
            else:
                items.append(f"{key}: {value!r}")

        config_items = "\n".join(items)

        return f"ExperimentConfig({config_items})"
