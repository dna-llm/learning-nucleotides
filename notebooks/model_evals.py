# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from transformers import AutoModelForCausalLM

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

    with open(filepath, "r") as file:
        trainer_state_dict = json.load(file)

    return trainer_state_dict


def combine_training_logs_from_repos(repo_ids: list) -> list:
    data_list = []

    for _repo_id in repo_ids:
        repo_id = _repo_id
        try:
            print(repo_id)
            trainer_state_dict = load_trainer_state_from_file(repo_id)
        except FileNotFoundError:
            last_checkpoint = find_last_checkpoint(repo_id)
            trainer_state_filepath = download_trainer_state_from_checkpoint(
                repo_id, last_checkpoint
            )
            trainer_state_dict = load_trainer_state_from_file(trainer_state_filepath)
        try:
            # Load the model from the Hugging Face repository
            model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
            print(model)
            print(repo_id)

            # Get the number of model parameters
            num_params = model.num_parameters()

        except Exception:
            num_params = "Unknown"
        if repo_id == "DNA-LLM/virus_pythia_31_1024_headless":
            num_params = 4743680
        if repo_id == "/content/pythia_14_ph_trainer_state.json":
            repo_id = "DNA-LLM/virus_pythia_14_1024_ph"
        if repo_id == "/content/pythia_31_ph_trainer_state.json":
            repo_id = "DNA-LLM/virus_pythia_31_1024_ph"
            num_params = 4743680
        log_history = trainer_state_dict["log_history"]
        df = pd.DataFrame(log_history)

        # Extract param type and loss type from repo ID
        repo_parts = repo_id.split("-")
        param_type = repo_parts[-3]
        if num_params == "Unknown":
            num_params = float(repo_parts[-3].replace("M", "")) * 1_000_000
        if param_type == "1024":
            param_type = repo_parts[-4]
        if param_type == "2d":
            param_type = repo_id.split("1024")[0].split("_")[-2]
        loss_type = repo_parts[-1]
        if loss_type == "entropy":
            loss_type = "cross_entropy"
        if loss_type == "GaussianPlusCE":
            loss_type = "2d_representation_GaussianPlusCE"
        if loss_type == "MSEPlusCE":
            loss_type = "2d_representation_MSEPlusCE"
        if loss_type == "d":
            loss_type = "2d"

        df["param_type"] = param_type
        df["loss_type"] = loss_type
        df_interp = pd.DataFrame()
        df_interp["epoch_interp"] = np.linspace(0, 0.9, num=101)
        df_interp["num_params"] = num_params

        df_interp["loss_interp"] = np.interp(
            df_interp["epoch_interp"].to_list(), df["epoch"].to_list(), df["loss"].to_list()
        )
        df_interp["param_type"] = param_type
        df_interp["loss_type"] = loss_type
        df_interp["model_type"] = repo_parts[1]
        model_type = repo_parts[1]
        if model_type == "LLM/virus":
            df_interp["model_type"] = repo_parts[2]

        print(df_interp)
        data_list.append(df_interp)

    return data_list


def plot_training_loss(data_list: list):
    if len(data_list) > 0:
        combined_df = pd.concat(data_list, ignore_index=True)
        combined_df.to_csv("training_data.csv", index=False)

        plt.figure(figsize=(10, 6))
        for param_type in combined_df["param_type"].unique():
            for loss_type in combined_df["loss_type"].unique():
                filtered_data = combined_df[
                    (combined_df["param_type"] == param_type)
                    & (combined_df["loss_type"] == loss_type)
                ]
                if not filtered_data.empty:
                    plt.plot(
                        filtered_data["epoch_interp"],
                        filtered_data["loss_interp"],
                        label=f"{param_type} - {loss_type}",
                    )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss by Param Type and Loss Type")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data entered. Exiting.")
# %%

def main(repo_ids: list):
    api = HfApi()
    models = api.list_models(author="DNA-LLM")
    repo_ids = []
    for m in models:
        if "diffusion" not in m.id:
            repo_ids.append(m.id)

    data_list = combine_training_logs_from_repos(repo_ids)
    plot_training_loss(data_list)


if __name__ == "__main__":
    main()
