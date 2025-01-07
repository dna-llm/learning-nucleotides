import json
import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

api = HfApi()
models = api.list_models(author="DNA-LLM")
datasets = api.list_datasets(author="DNA-LLM")
MAX_LENGTH = 1024
TEST_SIZE = 0.15
RANDOM_STATE = 42
DEVICE = "cuda"
all_models = []
for m in models:
    all_models.append(m.id)

benchmarks = []
for b in datasets:
    dataset = b.id
    if "benchmark" in dataset:
        benchmarks.append(b.id)


def find_last_checkpoint(repo_id: str) -> str:
    """
    Find the name of the last checkpoint directory in a HuggingFace Model repository.

    Args:
        repo_id (str): The repository ID on HuggingFace Hub.

    Returns:
        str: The path to the last checkpoint directory.
    """
    fs = HfFileSystem()
    all_files = fs.ls(repo_id, detail=False)
    checkpoint_dirs = [file for file in all_files if "checkpoint-" in file]
    if not checkpoint_dirs:
        try:
            return None
        except FileNotFoundError:
            raise FileNotFoundError(f"No checkpoints found in {repo_id}.") from None
    # Sort checkpoint directories to get the last checkpoint
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
    return checkpoint_dirs[0]


def download_trainer_state_from_checkpoint(
    repo_id: str, checkpoint_dir: str, filename="trainer_state.json"
) -> str:
    """
    Download the trainer state file from a specific checkpoint in a HuggingFace repository.

    Args:
        repo_id (str): The repository ID on HuggingFace Hub.
        checkpoint_dir (str): The checkpoint directory to download from.
        filename (str): The name of the trainer state file.

    Returns:
        str: The path to the downloaded file.
    """
    if checkpoint_dir:
        # file_path = f"{checkpoint_dir}/{filename}"
        print(checkpoint_dir.split("/")[-1])
        os.system(f"mkdir {repo_id}/")
        return hf_hub_download(
            repo_id=repo_id,
            subfolder=checkpoint_dir.split("/")[-1],
            filename=filename,
            local_dir="trainer_state.json",
        )
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir="trainer_state.json")


def load_trainer_state_from_file(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trainer state not found at {filepath}")

    with open(filepath, "r", encoding="utf-8") as file:
        trainer_state_dict = json.load(file)

    return trainer_state_dict


def combine_training_logs_from_repos(repo_ids: list, benchmarks: list) -> list:
    data_list = []

    for repo_id in repo_ids:
        try:
            _trainer_state_dict = load_trainer_state_from_file(repo_id)
        except FileNotFoundError:
            last_checkpoint = find_last_checkpoint(repo_id)
            trainer_state_filepath = download_trainer_state_from_checkpoint(
                repo_id, last_checkpoint
            )
            _trainer_state_dict = load_trainer_state_from_file(trainer_state_filepath)

        try:
            model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
            model = model.to(DEVICE)
            tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
            model_dict = {}
            model_dict["model"] = repo_id
            num_params = model.num_parameters()
            model_dict["num_params"] = num_params
            for benchmark in benchmarks:
                if "gene_finding" in benchmark:
                    ds = load_dataset(benchmark)
                    embeddings_big = []
                    labels = []
                    df = pd.DataFrame(ds["train"])
                    for _idx, row in df.iterrows():
                        sequence_small = [
                            row["seq"][i : i + MAX_LENGTH]
                            for i in range(0, len(row["seq"]), MAX_LENGTH)
                        ]
                        labels_small = [
                            row["labels"][i : i + MAX_LENGTH]
                            for i in range(0, len(row["seq"]), MAX_LENGTH)
                        ]
                        sequence_small = [
                            tokenizer(sequence, return_tensors="pt") for sequence in sequence_small
                        ]
                        embeddings_small = [
                            model(sequence["input_ids"].to("cuda"))
                            .past_key_values[0][0]
                            .detach()
                            .cpu()
                            for sequence in sequence_small
                        ]
                        embedding_shapes = [emb.shape for emb in embeddings_small]
                        embeddings_small = [
                            emb.reshape(shape[2], shape[1] * shape[3])
                            for emb, shape in zip(embeddings_small, embedding_shapes, strict=False)
                        ]
                        embeddings_big.extend(embeddings_small)
                        labels.extend(labels_small)

                    embeddings_big = [
                        embedding for sublist in embeddings_big for embedding in sublist
                    ]
                    labels = [label for sublist in labels for label in sublist]
                    X_train, X_test, y_train, y_test = train_test_split(
                        embeddings_big, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
                    )
                    clf = HistGradientBoostingClassifier(max_iter=2).fit(X_train, y_train)
                    pred = clf.predict(X_test)
                    score = matthews_corrcoef(y_test, pred)
                    model_dict[benchmark + "_MCC"] = score
                    model_dict[benchmark + "_accuracy"] = clf.score(X_test, y_test)
                    print(model_dict)
                    data_list.append(model_dict)

        except Exception as e:
            print(f"An error occurred: {e}")
    data_to_be_saved = str(data_list)
    with open("benchmarks.txt", "w", encoding="utf-8") as text:
        text.write(data_to_be_saved)
    return data_list


def main(repo_ids: list, benchmarks: list):
    data_list = combine_training_logs_from_repos(repo_ids, benchmarks)
    return data_list


if __name__ == "__main__":
    main(all_models, benchmarks)
