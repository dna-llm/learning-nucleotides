import glob
import logging
from unittest.mock import MagicMock

import pytest

from experiment_one.utils import (
    ExperimentConfig,
    load_model,
    load_tokenizer,
    load_trainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml_config(config_file: str) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(config_file)


@pytest.mark.parametrize("config_file", glob.glob("tests/configs/*.yml"))
def test_load_config(config_file):
    logger.info(f"Testing configuration loading for {config_file}")
    try:
        config = load_yaml_config(config_file)
        assert isinstance(config, ExperimentConfig)
        logger.info(f"Successfully loaded configuration for {config_file}")
    except Exception as e:
        logger.error(f"Failed to load config {config_file}: {e}")
        pytest.fail(f"Failed to load config {config_file}: {e}")


@pytest.mark.parametrize("config_file", glob.glob("tests/configs/*.yml"))
def test_load_model(config_file):
    logger.info(f"Testing model loading for {config_file}")
    try:
        config = load_yaml_config(config_file)
        model = load_model(**config.model.model_dump())
        assert model is not None
        logger.info(f"Successfully loaded model for {config_file}")
    except Exception as e:
        logger.error(f"Failed to load model for config {config_file}: {e}")
        pytest.fail(f"Failed to load model for config {config_file}: {e}")


@pytest.mark.parametrize("config_file", glob.glob("tests/configs/*.yml"))
def test_load_trainer(config_file):
    logger.info(f"Testing trainer initialization for {config_file}")
    try:
        config = load_yaml_config(config_file)
        model = load_model(**config.model.model_dump())
        tokenizer = load_tokenizer(config.model.tokenizer_name)
        trainer = load_trainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="output_dir",
            train_dataset=MagicMock(),
            eval_dataset=MagicMock(),
            num_train_epochs=config.training.num_train_epochs,
            train_batch_size=config.training.batch_size,
            eval_batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            logging_steps=config.training.logging_steps,
            loss_type=config.training.loss_type,
        )
        assert trainer is not None
        logger.info(f"Successfully initialized trainer for {config_file}")
    except Exception as e:
        logger.error(f"Failed to initialize trainer for config {config_file}: {e}")
        pytest.fail(f"Failed to initialize trainer for config {config_file}: {e}")
