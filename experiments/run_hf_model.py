import argparse
import logging
import os

import torch
from huggingface_hub import HfApi
from utils import (
    ExperimentConfig,
    check_missing_checkpoints,
    create_repo_with_retry,
    format_param_count,
    load_datasets,
    load_model,
    load_tokenizer,
    load_trainer,
    upload_checkpoints,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_model(config_file: str) -> None:
    cfg = ExperimentConfig.from_yaml(config_file)
    model = load_model(**cfg.model.model_dump())
    print(model)
    model_param_count = format_param_count(sum(p.numel() for p in model.parameters()))
    output_dir = f"virus-{cfg.model.name}-{model_param_count}-{cfg.model.max_seq_len}-{cfg.training.loss_type}"
    api = HfApi()
    repo_name = f"DNA-LLM/{output_dir}"
    api_token = os.getenv("HF_API_TOKEN")
    repo_url = create_repo_with_retry(api, repo_name, api_token)
    if repo_url is None:
        logger.error("Exiting due to repository creation failure.")
        return
    logger.info(f"Repository created at {repo_url}")
    logger.info("Loading datasets...")
    tokenizer = load_tokenizer(cfg.model.tokenizer_name)
    ds_train, ds_val, ds_test = load_datasets(
        train_path=cfg.data.train,
        val_path=cfg.data.val,
        test_path=cfg.data.test,
        is_pretrained=cfg.model.is_pretrained,
        use_2d_seq=cfg.data.use_2d_seq,
    )
    logger.info(
        f"Training Data: {ds_train}"
    )
    logger.info(
        f"Training model: {cfg.model.name}-{model_param_count} with loss: {cfg.training.loss_type}"
    )

    trainer = load_trainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        num_train_epochs=cfg.training.num_train_epochs,
        train_batch_size=cfg.training.batch_size,
        eval_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        num_workers=cfg.training.num_workers,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=cfg.training.logging_steps,
        loss_type=cfg.training.loss_type,
        is_pretrained=cfg.model.is_pretrained,
        fp16=True,
        ddp_find_unused_parameters=False,
    )
    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed successfully")
    logger.info("Pushing final model to Hub")
    trainer.push_to_hub()
    logger.info("Model pushed to 🤗 HF Hub successfully")
    logger.info("Starting eval")
    eval_results = trainer.evaluate(eval_dataset=ds_test["train"])
    logger.info(f"Evaluation results: {eval_results}")

    # upload any checkpoints that failed to upload during training
    missing_checkpoints = check_missing_checkpoints(api, repo_name, output_dir)
    if missing_checkpoints:
        logger.info(
            f"Found {len(missing_checkpoints)} checkpoints that failed to upload during training."
        )
        logger.info("Attempting to upload missing checkpoints...")
        upload_checkpoints(output_dir, repo_name)
    else:
        logger.info("All checkpoints were successfully uploaded during training.")

    del trainer
    torch.cuda.empty_cache()
    logger.info("Training and eval completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training for a HuggingFace model.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    run_model(args.config_file)
