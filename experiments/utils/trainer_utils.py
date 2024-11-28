import os

import torch
from datasets import DatasetDict, load_dataset
from losses import (
    ComplementLoss,
    Headless,
    StandardLoss,
  #  TwoDRepLoss,
    VAELoss,
    Discrete_CE_Loss
 #   TwoDRepL2CELoss
)
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .model_utils import (
    load_denseformer,
    load_evo,
    load_pythia,
    load_vae,
    load_wavelet,
    load_discrete
)
from .adopt import ADOPT


def load_model(name: str, **kwargs) -> torch.nn.Module:
    model_loaders = {
        "pythia": load_pythia,
        "denseformer": load_denseformer,
        "evo": load_evo,
        "wavelet": load_wavelet,
        "vae": load_vae,
        "discrete": load_discrete
    }

    return model_loaders[name.lower()](**kwargs)


def load_loss(loss_type: str) -> Trainer:
    losses = {
        "complement": ComplementLoss,
        "cross_entropy": StandardLoss,
        "headless": Headless,
     #   "two_d": TwoDRepLoss,
        "vae_loss": VAELoss,
     #   "two_d_ce": TwoDRepL2CELoss, 
        "discrete_ce_loss": Discrete_CE_Loss,
    }

    return losses[loss_type]


def pad_input_ids(example, max_length: int = 1000):
    input_ids = example["input_ids"]
    if isinstance(input_ids, str):
        input_ids = [int(x) for x in input_ids.split()]
    elif not isinstance(input_ids, list):
        input_ids = input_ids.tolist()

    length = min(len(input_ids), max_length)
    padded = [0] * max_length
    padded[:length] = input_ids[:length]
    return {"input_ids": padded}


def load_datasets(
    train_path: str,
    val_path: str,
    test_path: str,
    is_pretrained: bool = True,
    use_2d_seq: bool = False,
    remove_columns:bool = True,
) -> tuple[DatasetDict, DatasetDict, DatasetDict]:
    train = load_dataset(train_path)
    val = load_dataset(val_path)
    test = load_dataset(test_path)

    def filter_bad_2d(example):
        if "2D_Sequence_Interpolated" in example:
            return sum(example["2D_Sequence_Interpolated"][0]) != 0
        else:
            return False
    new_ds = []
    for ds in [train, val, test]:
        ds.set_format(type="torch")
        print(f"Before filtering: {len(ds['train'])}")
        sequence = "input_ids"
        if use_2d_seq:
            ds = ds.filter(filter_bad_2d)
            print(f"After filtering: {len(ds['train'])}")
            sequence = "2D_Sequence_Interpolated"
        if remove_columns: 
            ds = ds.select_columns(['id','input_ids', 'attention_mask'])
            ds.set_format(type="torch")
            new_ds.append(ds)
        if not is_pretrained:
            ds["train"] = ds["train"].select_columns(["id", sequence])
            ds = ds.map(pad_input_ids, remove_columns=ds["train"].column_names)

    return train, val, test


def tokenize_function(examples, tokenizer, max_length: int):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def load_trainer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer | None,
    output_dir: str,
    train_dataset: DatasetDict,
    eval_dataset: DatasetDict,
    num_train_epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    logging_steps: int,
    loss_type: str,
    num_workers: int,
    is_pretrained: bool = True,
    **kwargs,
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        logging_dir="./logs",
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=2,
        report_to=[],
        remove_unused_columns=False,#is_pretrained or
        push_to_hub=True,
        hub_strategy="all_checkpoints",
        hub_model_id=f"DNA-LLM/{output_dir}",
        hub_token=os.getenv("HF_API_TOKEN"),
        hub_private_repo=True,
        hub_always_push=True,
        dataloader_pin_memory=True,
        **kwargs,
    )
    loss_fn = load_loss(loss_type)
    optimizer = ADOPT(model.parameters(), lr=training_args.learning_rate)
    trainer = loss_fn(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=eval_dataset["train"],
        tokenizer=tokenizer,
        optimizers=(optimizer, None)
    )
    return trainer
