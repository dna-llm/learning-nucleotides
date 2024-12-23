import contextlib
import os
from typing import Optional

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, HfFileSystem, create_repo
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
MAX_LENGTH = 1024
DEVICE = "cuda"


def find_last_checkpoint(repo_id: str) -> Optional[str]:
    """Find the name of the last checkpoint directory in a HuggingFace Model repository."""
    fs = HfFileSystem()
    all_files = fs.ls(repo_id, detail=False)
    checkpoint_dirs = [file for file in all_files if file.startswith("checkpoint-")]

    if not checkpoint_dirs:
        return None

    return max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))


def get_model_tokenizer(repo_id: str):
    """Get the model and tokenizer from a HuggingFace repository."""
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    except Exception as err:
        last_checkpoint = find_last_checkpoint(repo_id)
        if last_checkpoint is None:
            raise FileNotFoundError(f"No checkpoints found in {repo_id}.") from err
        checkpoint_dir = last_checkpoint.split("/")[-1]
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, subfolder=checkpoint_dir, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, subfolder=checkpoint_dir, trust_remote_code=True
        )

    return model, tokenizer


def process_dataset(row, model, tokenizer):
    """Process a single row of the dataset."""
    sequence_chunks = [
        row["seq"][i : i + MAX_LENGTH] for i in range(0, len(row["seq"]), MAX_LENGTH)
    ]
    labels_chunks = [
        row["labels"][i : i + MAX_LENGTH] for i in range(0, len(row["labels"]), MAX_LENGTH)
    ]

    embeddings_chunks = []
    for sequence in sequence_chunks:
        tokens = tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            embedding = model(tokens["input_ids"].to(DEVICE)).past_key_values[0][0]
        embedding = embedding.cpu().numpy()
        embeddings_chunks.append(
            embedding.reshape(embedding.shape[2], embedding.shape[1] * embedding.shape[3])
        )

    return {
        "embeddings": embeddings_chunks,
        "labels": labels_chunks,
        "length_embeddings": [len(emb) for emb in embeddings_chunks],
    }


def main():
    api = HfApi()

    models = api.list_models(author="DNA-LLM")
    datasets = api.list_datasets(author="DNA-LLM")

    all_models = [m.id for m in models if "diffusion" not in m.id]
    benchmarks = [b.id for b in datasets if "benchmark" in b.id]

    for trained_model_repo in all_models:
        model, tokenizer = get_model_tokenizer(trained_model_repo)
        model = model.to(DEVICE)

        for benchmark in benchmarks:
            if "gene_finding" in benchmark:
                dataset_name = f"{benchmark}_{trained_model_repo.split('/')[-1]}"
                with contextlib.suppress(Exception):
                    create_repo(dataset_name, private=True, repo_type="dataset")
                for k in tqdm(range(60)):
                    if k > 58:
                        continue 
                    ds = load_dataset(benchmark, split=f"train[{k * 100}:{(k + 1) * 100}]")
                    ds = ds.map(
                        lambda row, model=model, tokenizer=tokenizer: process_dataset(
                            row, model, tokenizer
                        ),
                        batched=False,
                    )
                    df = pd.DataFrame(ds)
                    df = (
                        df.drop(columns=["seq"])
                        .explode(["labels", "embeddings"])
                        .explode(["labels", "embeddings"])
                        .reset_index()
                    )
                    df["length_embeddings"] = df.embeddings.str.len()

                    parquet_file = f"part_{k}.parquet"
                    df.to_parquet(parquet_file)

                    api.upload_file(
                        path_or_fileobj=parquet_file,
                        path_in_repo=f"data/part_{k}.parquet",
                        repo_id=dataset_name,
                        repo_type="dataset",
                        create_pr=False,
                    )

                    os.remove(parquet_file)
                    print(
                        f"Processed and uploaded: section {k} of {benchmark} for the model {trained_model_repo}"
                    )


if __name__ == "__main__":
    main()
