import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_repo_with_retry(api: HfApi, repo_name: str, api_token: str, max_retries: int = 3):
    max_retries = int(max_retries)

    def _create_with_retry():
        for attempt in range(max_retries):
            try:
                repo = api.create_repo(repo_name, token=api_token, private=True, exist_ok=True)
                repo_url = (
                    repo.url
                    if hasattr(repo, "url")
                    else repo.clone_url
                    if hasattr(repo, "clone_url")
                    else None
                )
                if not repo_url:
                    raise ValueError("Failed to get repository URL")
                logger.info(f"Repository created or already exists at {repo_url}")
                return repo_url
            except (KeyError, AttributeError, ValueError) as e:
                logger.error(f"Error when creating repo (attempt {attempt + 1}/{max_retries}): {e}")
            except HfHubHTTPError as e:
                logger.error(
                    f"HTTP error when creating repo (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if e.response.status_code == 401:
                    logger.error("Authentication failed. Please check your API token.")
                    return None
            except Exception as e:
                logger.error(
                    f"Unexpected error when creating repo (attempt {attempt + 1}/{max_retries}): {e}"
                )

            if attempt + 1 < max_retries:
                time.sleep(2**attempt)  # Exponential backoff

        logger.error("Failed to create repository after multiple attempts.")
        return None

    if not torch.distributed.is_initialized() or dist.get_rank() == 0:
        repo_url = _create_with_retry()
    else:
        repo_url = None

    if torch.distributed.is_initialized():
        object_list = [repo_url]
        dist.broadcast_object_list(object_list, src=0)
        repo_url = object_list[0]

    return repo_url


def check_missing_checkpoints(api: HfApi, repo_id: str, output_dir: str) -> list[str]:
    local_checkpoints = {f.name for f in Path(output_dir).glob("*.pt")}
    remote_files = {file.filename for file in api.list_repo_files(repo_id)}
    missing_checkpoints = [
        ckpt for ckpt in local_checkpoints if f"checkpoints/{ckpt}" not in remote_files
    ]
    return missing_checkpoints


def get_existing_checkpoints(api: HfApi, repo_id: str) -> set:
    try:
        files = api.list_repo_files(repo_id=repo_id)
        checkpoints = set()
        for file in files:
            parts = Path(file).parts
            if parts and parts[0].startswith("checkpoint-"):
                checkpoints.add(parts[0])

        return checkpoints
    except Exception as e:
        logger.error(f"Failed to list repository contents: {e}")
        return set()


def upload_checkpoints(model_dir: str, repo_id: str) -> None:
    api = HfApi()
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise ValueError("HF_API_TOKEN environment variable is not set.")
    try:
        logger.info(f"Authenticating with Hugging Face for repo: {repo_id}")
        api.whoami(token=token)
        logger.info("Authentication successful")
        logger.info(f"Beginning upload process for {model_dir} to {repo_id}")
        response = api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        logger.info("Upload completed successfully")
        return response
    except Exception as e:
        logger.error(f"An error occurred during the upload process: {e!s}")
        raise


def upload_model(model_dir: str) -> None:
    model_dir = Path(model_dir)
    logger.info(f"Beginning upload process for {model_dir} to Hugging Face...")
    model_name = str(model_dir.name)
    repo_id = f"DNA-LLM/{model_name}"
    upload_checkpoints(model_dir=model_dir, repo_id=repo_id)
    logger.info("All checkpoints uploaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload missing model checkpoints to HuggingFace.")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to folder containing all models to upload.",
    )
    args = parser.parse_args()
    upload_model(args.model_dir)
